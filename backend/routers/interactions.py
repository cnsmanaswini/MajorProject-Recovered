"""
Interactions Router — likes, comments, shares, negative feedback
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.database import get_db
from models.models import Post, Like, Comment, User, Report, AgentDecision, NotInterested, EmotionLog
from schemas.schemas import (
    InteractionCreate, CommentCreate, CommentOut,
    ReportCreate, ImpressionCreate,
)
from ai.pipeline.analyzer import analyze_text
from ai.agents.orchestrator import run_agents, EmotionSnapshot
from services.algorithm import (
    update_user_interests, record_impression, log_like_behavioral_signal,
)
from routers.posts import get_user_risk_history, get_emotion_history

router = APIRouter()
logger = logging.getLogger("mindgram.interactions")


@router.post("")
async def record_interaction(body: InteractionCreate, db: AsyncSession = Depends(get_db)):
    post = await db.get(Post, body.post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if body.action == "like":
        # Check for existing like to prevent duplicates
        existing_like = await db.execute(
            select(Like).where(
                Like.post_id == body.post_id,
                Like.user_id == body.user_id
            )
        )
        if existing_like.scalar_one_or_none():
            # Already liked, return success without creating duplicate
            return {"status": "ok", "action": "already_liked", "post_id": body.post_id}

        like = Like(post_id=body.post_id, user_id=body.user_id)
        db.add(like)
        post.likes_count += 1
        await db.commit()
        await update_user_interests(body.user_id, post.emotion, db, weight=0.3)
        await log_like_behavioral_signal(body.user_id, post, db)
        return {"status": "ok", "action": body.action, "post_id": body.post_id}

    elif body.action == "unlike":
        result = await db.execute(
            select(Like).where(Like.post_id == body.post_id, Like.user_id == body.user_id)
        )
        like = result.scalar_one_or_none()
        if like:
            await db.delete(like)
            post.likes_count = max(0, post.likes_count - 1)
            await db.commit()
            return {"status": "ok", "action": body.action, "post_id": body.post_id}
        else:
            # Not liked, nothing to unlike
            return {"status": "ok", "action": "not_liked", "post_id": body.post_id}

    elif body.action == "not_interested":
        # Strong explicit negative signal — deliberate user choice, so it
        # gets a bigger weight than the passive skip-rate penalty.
        existing = await db.execute(
            select(NotInterested).where(
                NotInterested.user_id == body.user_id,
                NotInterested.post_id == body.post_id,
            )
        )
        if not existing.scalar_one_or_none():
            db.add(NotInterested(
                user_id=body.user_id,
                post_id=body.post_id,
                author_id=post.user_id,
            ))
            await db.commit()
        await update_user_interests(body.user_id, post.emotion, db, weight=-0.5)
        return {"status": "ok", "action": body.action, "post_id": body.post_id}

    await db.commit()
    return {"status": "ok", "action": body.action, "post_id": body.post_id}


@router.post("/comment", response_model=CommentOut, status_code=201)
async def add_comment(body: CommentCreate, db: AsyncSession = Depends(get_db)):
    """
    Comments now run through the same silent AI pipeline as posts:
    sentiment/emotion/sarcasm/numbness -> LSTM-refined risk -> feed_score,
    logged to EmotionLog(source="comment") and evaluated by the agent
    orchestrator, same as post creation in routers/posts.py.

    A comment gets its own risk history (the commenter's, not the post
    author's) since the pipeline tracks the commenter's mental-health
    trajectory, not the post they happen to be commenting on.
    """
    post = await db.get(Post, body.post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    risk_history = await get_user_risk_history(body.user_id, db)
    pipeline = analyze_text(body.content, risk_history)

    comment = Comment(
        post_id=body.post_id,
        user_id=body.user_id,
        content=body.content,
        sentiment=pipeline.sentiment,
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        sarcasm=pipeline.sarcasm,
        sarcasm_score=pipeline.sarcasm_score,
        risk_score=pipeline.risk_score,
        feed_score=pipeline.feed_score,
    )
    db.add(comment)
    post.comments_count += 1

    # Log emotion for the commenter (not the post author)
    log = EmotionLog(
        user_id=body.user_id,
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        risk_score=pipeline.risk_score,
        source="comment",
    )
    db.add(log)

    # Run agentic pipeline on the commenter's trajectory
    history = await get_emotion_history(body.user_id, db)
    current_snap = EmotionSnapshot(
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        risk_score=pipeline.risk_score,
        source="comment",
    )
    agent_report = run_agents(current_snap, history)

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
    await db.refresh(comment)
    comment.user = await db.get(User, body.user_id)

    # Interest signal uses the COMMENT's own emotion, not the post's —
    # what the commenter expressed is the better signal for what content
    # they're drawn to write about, and matches the like-endpoint's use
    # of the post's own emotion for the same reasoning.
    await update_user_interests(body.user_id, pipeline.emotion, db, weight=0.15)

    return comment


@router.get("/comments/{post_id}", response_model=list[CommentOut])
async def get_comments(post_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Comment).where(Comment.post_id == post_id).order_by(Comment.created_at.asc())
    )
    comments = result.scalars().all()
    for c in comments:
        c.user = await db.get(User, c.user_id)
    return comments


@router.post("/report", status_code=201)
async def report_post(body: ReportCreate, db: AsyncSession = Depends(get_db)):
    """
    Records a report, applies a strong negative interest signal for the
    reporter, and — for self_harm reports specifically — re-runs the full
    analysis pipeline on the post and flags the AUTHOR's risk pipeline.
    A self-harm report is fresh human signal the original at-post-time
    analysis may have missed, so it earns its own pass rather than trusting
    the score computed when the post was created. This is about the
    author's wellbeing, not the reporter's feed preferences, so it writes
    to AgentDecision (keyed on post.user_id), not the reporter's interests
    beyond the standard negative-interest signal below.
    """
    post = await db.get(Post, body.post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    report = Report(
        post_id=body.post_id,
        reporter_id=body.user_id,
        reason=body.reason,
        details=body.details or "",
    )
    db.add(report)
    await db.commit()

    # Reporter's own interest signal — stronger than not_interested, since
    # a report is a stronger statement than passive disinterest.
    await update_user_interests(body.user_id, post.emotion, db, weight=-0.8)

    if body.reason == "self_harm":
        pipeline = analyze_text(
            post.content,
            likes_count=post.likes_count,
            comments_count=post.comments_count,
        )
        # Never lower an existing risk_score based on a re-analysis —
        # only a report-triggered increase should move this value here.
        post.risk_score = max(post.risk_score, pipeline.risk_score)
        post.sentiment = pipeline.sentiment
        post.sentiment_score = pipeline.sentiment_score

        risk_tier = (
            "critical" if pipeline.risk_score >= 0.85 else
            "high" if pipeline.risk_score >= 0.60 else
            "moderate" if pipeline.risk_score >= 0.35 else
            "low"
        )
        db.add(AgentDecision(
            user_id=post.user_id,
            risk_level=risk_tier,
            decision="review",
            intervention="user_reported_self_harm",
            metadata_json={
                "reported_post_id": post.id,
                "reporter_id": body.user_id,
                "report_reason": body.reason,
            },
        ))

    await db.commit()
    return {"status": "ok", "action": "report", "post_id": body.post_id}


@router.post("/impression")
async def post_impression(body: ImpressionCreate, db: AsyncSession = Depends(get_db)):
    """
    Records dwell time for a feed impression. skipped/skip-rate evaluation
    happens server-side inside record_impression() — see services/algorithm.py.
    """
    post = await db.get(Post, body.post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    await record_impression(body.user_id, body.post_id, post.emotion, body.dwell_ms, db)
    return {"status": "ok"}