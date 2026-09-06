"""
Chat module — FastAPI routes.
Mount with: app.include_router(router, prefix="/chat", tags=["chat"])
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import models.models as models
import schemas.chat as schemas

from models.database import get_db         # adjust to your project
from routers.auth import get_current_user    # adjust to your project



router = APIRouter()


# ---------- Conversations ----------

@router.post("/conversations", response_model=schemas.ConversationOut)
def create_conversation(
    payload: schemas.ConversationCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    all_ids = set(payload.participant_ids) | {current_user.id}
    convo = models.Conversation(
        is_group=payload.is_group or len(all_ids) > 2,
        title=payload.title,
        created_by=current_user.id,
    )
    participants = db.query(models.User).filter(models.User.id.in_(all_ids)).all()  # adjust model import
    convo.participants = participants
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo


@router.get("/conversations", response_model=List[schemas.ConversationOut])
def list_conversations(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    return current_user.conversations  # via the participants relationship


@router.post("/conversations/{conversation_id}/accept")
def accept_request(conversation_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    convo = db.query(models.Conversation).get(conversation_id)
    if not convo:
        raise HTTPException(404, "Conversation not found")
    convo.request_status = models.ConversationRequestStatus.accepted
    db.commit()
    return {"status": "accepted"}


@router.post("/conversations/{conversation_id}/block")
def block_request(conversation_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    convo = db.query(models.Conversation).get(conversation_id)
    if not convo:
        raise HTTPException(404, "Conversation not found")
    convo.request_status = models.ConversationRequestStatus.blocked
    db.commit()
    return {"status": "blocked"}


# ---------- Messages ----------

@router.post("/messages", response_model=schemas.MessageOut)
def send_message(
    payload: schemas.MessageCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    message = models.Message(
        conversation_id=payload.conversation_id,
        sender_id=current_user.id,
        type=payload.type,
        content=payload.content,
        media_url=payload.media_url,
        shared_post_id=payload.shared_post_id,
        shared_reel_id=payload.shared_reel_id,
        reply_to_id=payload.reply_to_id,
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    # Run AI analysis off the request path so sending a message stays fast.
    if payload.type == schemas.MessageType.text and payload.content:
        background_tasks.add_task(_analyze_message_task, message.id)

    return message


def _analyze_message_task(message_id: int):
    from app.database import SessionLocal  # adjust import
    db = SessionLocal()
    try:
        message = db.query(models.Message).get(message_id)
        if not message:
            return
        analysis = ai_analysis.analyze_and_store_message(db, message)

        pattern = (
            db.query(models.ConversationRiskPattern)
            .filter_by(conversation_id=message.conversation_id, user_id=message.sender_id)
            .first()
        )
        if pattern and ai_analysis.should_trigger_intervention(pattern):
            ai_analysis.mark_intervention_triggered(db, pattern)
            # hook into your existing wellness-nudge mechanism here, e.g.:
            # trending_wellness.trigger_gentle_checkin(message.sender_id)
    finally:
        db.close()


@router.get("/conversations/{conversation_id}/messages", response_model=List[schemas.MessageOut])
def get_messages(
    conversation_id: int,
    limit: int = 50,
    before_id: int | None = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    q = db.query(models.Message).filter(
        models.Message.conversation_id == conversation_id,
        models.Message.is_deleted_for_sender.is_(False) | (models.Message.sender_id != current_user.id),
    )
    if before_id:
        q = q.filter(models.Message.id < before_id)
    return q.order_by(models.Message.id.desc()).limit(limit).all()


@router.delete("/messages/{message_id}")
def delete_message(message_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    message = db.query(models.Message).get(message_id)
    if not message or message.sender_id != current_user.id:
        raise HTTPException(404, "Message not found")
    message.is_deleted_for_sender = True
    db.commit()
    return {"status": "deleted"}


@router.post("/messages/{message_id}/unsend")
def unsend_message(message_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    message = db.query(models.Message).get(message_id)
    if not message or message.sender_id != current_user.id:
        raise HTTPException(404, "Message not found")
    message.is_unsent = True
    message.content = None
    message.media_url = None
    db.commit()
    return {"status": "unsent"}


@router.post("/messages/{message_id}/reactions")
def react_to_message(
    message_id: int,
    payload: schemas.ReactionCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    reaction = models.MessageReaction(message_id=message_id, user_id=current_user.id, emoji=payload.emoji)
    db.add(reaction)
    db.commit()
    return {"status": "reacted"}


# ---------- Risk pattern (surface to moderation / wellness dashboard) ----------

@router.get("/conversations/{conversation_id}/risk-pattern", response_model=schemas.RiskPatternOut)
def get_risk_pattern(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    pattern = (
        db.query(models.ConversationRiskPattern)
        .filter_by(conversation_id=conversation_id, user_id=current_user.id)
        .first()
    )
    if not pattern:
        raise HTTPException(404, "No pattern data yet")
    return schemas.RiskPatternOut(
        negative_ratio=pattern.negative_ratio,
        trend_slope=pattern.trend_slope,
        consecutive_negative_days=pattern.consecutive_negative_days,
        intervention_count=pattern.intervention_count,
        should_intervene=ai_analysis.should_trigger_intervention(pattern),
    )