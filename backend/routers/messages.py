
"""
Messages Router — Real-time WebSocket chat
GET  /api/messages/conversations     → list conversations
GET  /api/messages/thread/{user_id}/{other_user_id}  → get message thread
POST /api/messages                   → send message (REST fallback)
WS   /api/messages/ws/{user_id}      → WebSocket live chat
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_
from typing import Dict, List
import json
from services.notification_service import create_notification
from models.database import get_db
from models.models import Message, User, Follow, EmotionLog, Notification, AgentDecision
from schemas.schemas import MessageCreate, MessageOut
from routers.auth import get_current_user, get_optional_user
from ai.pipeline.analyzer import analyze_text
from ai.agents.orchestrator import run_agents, EmotionSnapshot
from routers.posts import get_user_risk_history, get_emotion_history

router = APIRouter()


async def are_mutual_followers(db: AsyncSession, user_a_id: int, user_b_id: int) -> bool:
    """True only if each user follows the other."""
    result = await db.execute(
        select(Follow).where(
            or_(
                and_(Follow.follower_id == user_a_id, Follow.following_id == user_b_id),
                and_(Follow.follower_id == user_b_id, Follow.following_id == user_a_id),
            )
        )
    )
    follows = result.scalars().all()
    a_follows_b = any(f.follower_id == user_a_id and f.following_id == user_b_id for f in follows)
    b_follows_a = any(f.follower_id == user_b_id and f.following_id == user_a_id for f in follows)
    return a_follows_b and b_follows_a


async def _analyze_and_log_message(sender_id: int, content: str, db: AsyncSession):
    """
    Shared DM analysis path — mirrors routers/posts.py's post-creation
    pipeline (risk history -> analyze_text -> EmotionLog -> agent check ->
    AgentDecision) so a DM saying something acutely concerning can trigger
    the same intervention path a post would, not just a silently-stored
    risk_score nobody acts on.

    Scoped to the SENDER — a message reflects the sender's state, same as
    EmotionLog(source="message") already being keyed on user_id=sender.

    Returns the PipelineResult so callers can still populate the Message
    row and websocket payload as before.
    """
    risk_history = await get_user_risk_history(sender_id, db)
    pipeline = analyze_text(content, risk_history)

    log = EmotionLog(
        user_id=sender_id,
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        risk_score=pipeline.risk_score,
        source="message",
    )
    db.add(log)

    history = await get_emotion_history(sender_id, db)
    current_snap = EmotionSnapshot(
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        risk_score=pipeline.risk_score,
        source="message",
    )
    agent_report = run_agents(current_snap, history)

    db.add(AgentDecision(
        user_id=sender_id,
        risk_level=agent_report.risk_level,
        decision=agent_report.decision,
        intervention=agent_report.intervention,
        rag_suggestion=agent_report.rag_suggestion,
        metadata_json=agent_report.metadata,
    ))
    log.agent_action = agent_report.decision

    return pipeline


# ── WebSocket Connection Manager ──────────────────────────────

class ConnectionManager:
    def __init__(self):
        # user_id → list of websockets
        self.active: Dict[int, List[WebSocket]] = {}

    async def connect(self, user_id: int, websocket: WebSocket):
        await websocket.accept()
        if user_id not in self.active:
            self.active[user_id] = []
        self.active[user_id].append(websocket)

    def disconnect(self, user_id: int, websocket: WebSocket):
        if user_id in self.active:
            self.active[user_id].remove(websocket)
            if not self.active[user_id]:
                del self.active[user_id]

    async def send_to_user(self, user_id: int, data: dict):
        """Send message to a specific user if they are online."""
        if user_id in self.active:
            message_str = json.dumps(data)
            for ws in self.active[user_id]:
                try:
                    await ws.send_text(message_str)
                except Exception:
                    pass

    def is_online(self, user_id: int) -> bool:
        return user_id in self.active and len(self.active[user_id]) > 0


manager = ConnectionManager()


# ── WebSocket Endpoint ────────────────────────────────────────

@router.websocket("/ws/{user_id}")
async def websocket_chat(
    websocket: WebSocket,
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Real-time WebSocket chat endpoint.
    Messages are analyzed by AI pipeline on the fly.
    """
    await manager.connect(user_id, websocket)
    try:
        while True:
            # Receive message from client
            raw = await websocket.receive_text()
            data = json.loads(raw)

            receiver_id = data.get("receiver_id")
            content = data.get("content", "").strip()

            if not content or not receiver_id:
                continue

            # Verify users exist
            sender = await db.get(User, user_id)
            receiver = await db.get(User, receiver_id)
            if not sender or not receiver:
                continue

            # Only allow messaging between mutual followers
            if not await are_mutual_followers(db, user_id, receiver_id):
                await manager.send_to_user(user_id, {
                    "type": "error",
                    "detail": "You can only message users who follow you back.",
                    "receiver_id": receiver_id,
                })
                continue

            # Run AI pipeline silently (risk-history-aware, agent-checked)
            pipeline = await _analyze_and_log_message(user_id, content, db)

            # Save message
            msg = Message(
                sender_id=user_id,
                receiver_id=receiver_id,
                content=content,
                sentiment=pipeline.sentiment,
                emotion=pipeline.emotion,
                risk_score=pipeline.risk_score,
            )
            db.add(msg)

            await db.commit()
            await db.refresh(msg)
            await create_notification(
             db=db,
             user_id=receiver_id,
             from_user_id=user_id,
             notification_type="message",
             message=f"{sender.username} sent you a message",
            )

            # Build response payload
            payload = {
                "id": msg.id,
                "sender_id": user_id,
                "receiver_id": receiver_id,
                "content": content,
                "created_at": msg.created_at.isoformat(),
                "sentiment": pipeline.sentiment,
                "emotion": pipeline.emotion,
                "risk_score": pipeline.risk_score,
                "sender": {
                    "username": sender.username,
                    "avatar_url": sender.avatar_url,
                    "display_name": sender.display_name,
                },
            }

            # Send to sender (confirmation)
            await manager.send_to_user(user_id, {
                "type": "message_sent",
                **payload,
            })

            # Send to receiver (if online)
            await manager.send_to_user(receiver_id, {
                "type": "new_message",
                **payload,
            })

    except WebSocketDisconnect:
        manager.disconnect(user_id, websocket)


# ── REST Endpoints ────────────────────────────────────────────

@router.post("", response_model=MessageOut, status_code=201)
async def send_message(
    body: MessageCreate,
    current_user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """REST fallback for sending messages."""
    sender_id = body.sender_id
    if current_user and current_user.id != sender_id:
        raise HTTPException(status_code=403, detail="Sender mismatch with authenticated user")

    sender = await db.get(User, sender_id)
    if not sender:
        raise HTTPException(status_code=404, detail="Sender not found")

    receiver = await db.get(User, body.receiver_id)
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")

    pipeline = await _analyze_and_log_message(sender_id, body.content, db)

    msg = Message(
        sender_id=sender_id,
        receiver_id=body.receiver_id,
        content=body.content,
        sentiment=pipeline.sentiment,
        emotion=pipeline.emotion,
        risk_score=pipeline.risk_score,
    )
    db.add(msg)

    await db.commit()
    await db.refresh(msg)
    await create_notification(
     db=db,
     user_id=body.receiver_id,
     from_user_id=sender.id,
     notification_type="message",
     message=f"{sender.username} sent you a message",
    )

    # Notify receiver via WebSocket if online — payload must match the
    # websocket path's shape so the client handles both identically.
    await manager.send_to_user(body.receiver_id, {
        "type": "new_message",
        "id": msg.id,
        "sender_id": sender.id,
        "receiver_id": body.receiver_id,
        "content": body.content,
        "created_at": msg.created_at.isoformat(),
        "sentiment": msg.sentiment,
        "emotion": msg.emotion,
        "risk_score": msg.risk_score,
        "sender": {
            "username": sender.username,
            "avatar_url": sender.avatar_url,
        },
    })

    return msg


@router.get("/conversations")
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get list of conversations with latest message."""
    result = await db.execute(
        select(Message)
        .where(
            or_(
                Message.sender_id == current_user.id,
                Message.receiver_id == current_user.id,
            )
        )
        .order_by(Message.created_at.desc())
    )
    messages = result.scalars().all()

    # Group by conversation partner
    conversations = {}
    for msg in messages:
        partner_id = msg.receiver_id if msg.sender_id == current_user.id else msg.sender_id
        if partner_id not in conversations:
            partner = await db.get(User, partner_id)
            conversations[partner_id] = {
                "user": {
                    "id": partner.id,
                    "username": partner.username,
                    "display_name": partner.display_name,
                    "avatar_url": partner.avatar_url,
                },
                "last_message": {
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                    "is_mine": msg.sender_id == current_user.id,
                    "sentiment": msg.sentiment,
                },
                "is_online": manager.is_online(partner_id),
                "unread_count": 0,
            }

    return list(conversations.values())


@router.get("/thread/{user_id}/{other_user_id}", response_model=list[MessageOut])
async def get_thread(
    user_id: int,
    other_user_id: int,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Get message thread between two users."""
    result = await db.execute(
        select(Message)
        .where(
            or_(
                and_(
                    Message.sender_id == user_id,
                    Message.receiver_id == other_user_id,
                ),
                and_(
                    Message.sender_id == other_user_id,
                    Message.receiver_id == user_id,
                ),
            )
        )
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    messages = result.scalars().all()

    # Mark as read for the requested user
    for msg in messages:
        if msg.receiver_id == user_id and not msg.is_read:
            msg.is_read = True
    await db.commit()

    return messages


@router.get("/online-status/{user_id}")
async def get_online_status(user_id: int):
    return {"user_id": user_id, "is_online": manager.is_online(user_id)}