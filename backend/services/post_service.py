"""
Post Service — business logic for post creation and retrieval.
Decouples the router layer from direct DB + AI calls.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from datetime import datetime

from models.models import Post, User, EmotionLog, AgentDecision, PostMedia
from schemas.schemas import PostCreate, PipelineResult
from services.topic_utils import extract_topics
from ai.pipeline.analyzer import analyze_text
from ai.agents.orchestrator import run_agents, EmotionSnapshot


async def get_user_risk_history(user_id: int, db: AsyncSession, limit: int = 20) -> list[float]:
    """Fetch recent risk scores for the LSTM temporal context window."""
    result = await db.execute(
        select(EmotionLog.risk_score)
        .where(EmotionLog.user_id == user_id)
        .order_by(EmotionLog.timestamp.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return list(reversed(rows))


async def get_emotion_snapshots(user_id: int, db: AsyncSession, limit: int = 20) -> list[EmotionSnapshot]:
    """Fetch recent emotion logs as structured snapshots for the agentic pipeline."""
    result = await db.execute(
        select(EmotionLog)
        .where(EmotionLog.user_id == user_id)
        .order_by(EmotionLog.timestamp.desc())
        .limit(limit)
    )
    logs = result.scalars().all()
    return [
        EmotionSnapshot(
            sentiment_score=log.sentiment_score,
            emotion=log.emotion,
            emotion_score=log.emotion_score,
            risk_score=log.risk_score,
            source=log.source,
        )
        for log in reversed(logs)
    ]


async def create_post_with_ai(
    body: PostCreate,
    db: AsyncSession,
) -> Post:
    """
    Full pipeline:
      1. Run NLP analysis (sentiment, emotion, sarcasm, LSTM risk)
      2. Persist post with AI annotations
      3. Log emotion snapshot
      4. Run agentic pipeline
      5. Persist agent decision
    Returns the saved Post ORM object.
    """
    user = await db.get(User, body.user_id)
    if not user:
        raise ValueError(f"User {body.user_id} not found")

    # 1 — AI pipeline
    risk_history = await get_user_risk_history(body.user_id, db)
    pipeline: PipelineResult = analyze_text(body.content, risk_history)

    # 2 — Persist post
    post = Post(
        user_id=body.user_id,
        content=body.content,
        image_url=body.image_url or "",
        video_url=body.video_url or "",
        is_reel=body.is_reel,
        sentiment=pipeline.sentiment,
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        sarcasm=pipeline.sarcasm,
        sarcasm_score=pipeline.sarcasm_score,
        risk_score=pipeline.risk_score,
        feed_score=pipeline.feed_score,
        topics=extract_topics(body.content, pipeline.emotion, body.location or ""),
    )
    db.add(post)
    await db.flush()

    media_items = []
    if body.image_url:
        media_items.append(PostMedia(post_id=post.id, media_type="image", url=body.image_url, position=0))
    if body.video_url:
        media_items.append(PostMedia(post_id=post.id, media_type="video", url=body.video_url, position=len(media_items)))
    db.add_all(media_items)

    # 3 — Emotion log
    log = EmotionLog(
        user_id=body.user_id,
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        risk_score=pipeline.risk_score,
        source="post",
    )
    db.add(log)

    # 4 — Agentic pipeline
    history = await get_emotion_snapshots(body.user_id, db)
    current_snap = EmotionSnapshot(
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        risk_score=pipeline.risk_score,
        source="post",
    )
    agent_report = run_agents(current_snap, history)

    # 5 — Persist decision + update log action label
    db.add(AgentDecision(
        user_id=body.user_id,
        risk_level=agent_report.risk_level,
        decision=agent_report.decision,
        intervention=agent_report.intervention,
        rag_suggestion=agent_report.rag_suggestion,
        metadata_json=agent_report.metadata,
    ))
    log.agent_action = agent_report.decision

    await db.commit()
    await db.refresh(post)
    post.author = user
    return post


async def fetch_user_posts(user_id: int, db: AsyncSession, limit: int = 20) -> list[Post]:
    """Retrieve a user's posts ordered by recency."""
    result = await db.execute(
        select(Post)
        .options(selectinload(Post.media), selectinload(Post.author))
        .where(Post.user_id == user_id)
        .order_by(Post.created_at.desc())
        .limit(limit)
    )
    posts = result.scalars().all()
    user = await db.get(User, user_id)
    for p in posts:
        p.author = user
    return posts
