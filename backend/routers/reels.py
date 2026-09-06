import re
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, or_
from sqlalchemy.orm import selectinload

from models.database import get_db
from models.models import User, Post, Like, Comment, Follow
from models.reels import ReelView, SavedReel, NotInterested, Report, ShareEvent
from schemas.reels import (
    ReelOut, FeedResponse, CommentCreate, CommentOut, ViewEventIn,
    ShareIn, NotInterestedIn, ReportIn,
)
from services.reel_recommendation import rank_reels_for_user, build_user_signals
from routers.auth import get_current_user

router = APIRouter(prefix="/api/reels", tags=["reels"])

HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@(\w+)")


def _extract_tags(caption: Optional[str]):
    if not caption:
        return [], []
    return HASHTAG_RE.findall(caption), MENTION_RE.findall(caption)


def _video_url(reel: Post) -> str:
    if (reel.video_url or "").strip():
        return reel.video_url
    for m in getattr(reel, "media", None) or []:
        if m.media_type == "video" and m.url:
            return m.url
    return ""


async def _serialize(reel: Post, current_user: User, db: AsyncSession) -> ReelOut:
    caption = reel.content or ""
    hashtags, mentions = _extract_tags(caption)
    like_row = await db.execute(
        select(Like).filter_by(post_id=reel.id, user_id=current_user.id)
    )
    is_liked = like_row.scalar_one_or_none() is not None
    saved_row = await db.execute(
        select(SavedReel).filter_by(reel_id=reel.id, user_id=current_user.id)
    )
    is_saved = saved_row.scalar_one_or_none() is not None
    follow_row = await db.execute(
        select(Follow).filter_by(follower_id=current_user.id, following_id=reel.user_id)
    )
    is_following = follow_row.scalar_one_or_none() is not None

    author = reel.author
    share_count = (await db.execute(select(ShareEvent).filter_by(reel_id=reel.id))).scalars().all()
    save_count = (await db.execute(select(SavedReel).filter_by(reel_id=reel.id))).scalars().all()
    view_count = (await db.execute(select(ReelView).filter_by(reel_id=reel.id))).scalars().all()
    comment_count = reel.comments_count if hasattr(reel, "comments_count") else 0

    return ReelOut(
        id=reel.id,
        caption=caption,
        hashtags=hashtags,
        mentions=mentions,
        video_url=_video_url(reel),
        thumbnail_url=getattr(reel, "image_url", None) or None,
        duration_seconds=None,
        created_at=reel.created_at,
        creator={
            "id": author.id if author else reel.user_id,
            "username": author.username if author else "",
            "profile_picture_url": author.avatar_url if author else None,
            "is_following": is_following,
        },
        audio=None,
        like_count=reel.likes_count or 0,
        comment_count=comment_count or 0,
        share_count=len(share_count),
        save_count=len(save_count),
        view_count=len(view_count),
        is_liked=is_liked,
        is_saved=is_saved,
        sentiment=getattr(reel, "sentiment", None),
        emotion=getattr(reel, "emotion", None),
        risk_score=getattr(reel, "risk_score", None),
        rank_score=getattr(reel, "rank_score", None),
        rank_reason=getattr(reel, "rank_reason", None),
    )


@router.get("/feed", response_model=FeedResponse)
async def get_reel_feed(
    limit: int = Query(10, le=30),
    exclude_ids: Optional[str] = Query(None, description="comma-separated reel IDs already seen this session"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ni = await db.execute(select(NotInterested).filter_by(user_id=current_user.id))
    not_interested_ids = {r.reel_id for r in ni.scalars().all()}
    seen_ids = set()
    if exclude_ids:
        seen_ids = {int(x) for x in exclude_ids.split(",") if x.strip().isdigit()}

    excluded = not_interested_ids | seen_ids

    q = (
        select(Post)
        .options(selectinload(Post.author), selectinload(Post.media))
        .filter(Post.is_reel == True)  # noqa: E712
        .order_by(desc(Post.created_at))
        .limit(200)
    )
    if excluded:
        q = q.filter(~Post.id.in_(excluded))
    candidates = (await db.execute(q)).scalars().all()

    follows = await db.execute(select(Follow).filter_by(follower_id=current_user.id))
    following_ids = {f.following_id for f in follows.scalars().all()}
    user_signals = build_user_signals(db, current_user.id)

    ranked = rank_reels_for_user(candidates, following_ids, user_signals)[:limit]

    return FeedResponse(
        reels=[await _serialize(r, current_user, db) for r in ranked],
        next_cursor=str(ranked[-1].id) if ranked else None,
    )


@router.get("/search", response_model=FeedResponse)
async def search_reels(
    q: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    like_pattern = f"%{q}%"
    query = (
        select(Post)
        .join(User, Post.user_id == User.id)
        .options(selectinload(Post.author), selectinload(Post.media))
        .filter(Post.is_reel == True)  # noqa: E712
        .filter(
            or_(
                Post.content.ilike(like_pattern),
                User.username.ilike(like_pattern),
            )
        )
        .order_by(desc(Post.created_at))
        .limit(30)
    )
    rows = (await db.execute(query)).scalars().unique().all()
    return FeedResponse(
        reels=[await _serialize(r, current_user, db) for r in rows],
        next_cursor=None,
    )


@router.get("/audio/{audio_id}/reels", response_model=FeedResponse)
async def reels_using_audio(audio_id: int, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    del audio_id
    return FeedResponse(reels=[], next_cursor=None)


@router.post("/{reel_id}/like")
async def toggle_like(reel_id: int, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    reel = (await db.execute(select(Post).filter_by(id=reel_id, is_reel=True))).scalar_one_or_none()
    if not reel:
        raise HTTPException(404, "Reel not found")

    existing = (await db.execute(select(Like).filter_by(post_id=reel_id, user_id=current_user.id))).scalar_one_or_none()
    if existing:
        await db.delete(existing)
        await db.commit()
        return {"liked": False}
    db.add(Like(post_id=reel_id, user_id=current_user.id))
    await db.commit()
    return {"liked": True}


@router.post("/{reel_id}/comments", response_model=CommentOut)
async def add_comment(reel_id: int, payload: CommentCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    reel = (await db.execute(select(Post).filter_by(id=reel_id, is_reel=True))).scalar_one_or_none()
    if not reel:
        raise HTTPException(404, "Reel not found")
    comment = Comment(
        post_id=reel_id,
        user_id=current_user.id,
        content=payload.text,
    )
    db.add(comment)
    await db.commit()
    await db.refresh(comment)
    return CommentOut(
        id=comment.id,
        user={"id": current_user.id, "username": current_user.username, "profile_picture_url": current_user.avatar_url},
        text=comment.content,
        created_at=comment.created_at,
        reply_count=0,
    )


@router.get("/{reel_id}/comments", response_model=list[CommentOut])
async def list_comments(reel_id: int, db: AsyncSession = Depends(get_db)):
    top_level = (await db.execute(
        select(Comment).filter_by(post_id=reel_id).order_by(Comment.created_at.asc())
    )).scalars().all()
    out = []
    for c in top_level:
        user = await db.get(User, c.user_id)
        if not user:
            continue
        out.append(CommentOut(
            id=c.id,
            user={"id": user.id, "username": user.username, "profile_picture_url": user.avatar_url},
            text=c.content,
            created_at=c.created_at,
            reply_count=0,
        ))
    return out


@router.post("/{reel_id}/save")
async def toggle_save(reel_id: int, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    existing = (await db.execute(
        select(SavedReel).filter_by(reel_id=reel_id, user_id=current_user.id)
    )).scalar_one_or_none()
    if existing:
        await db.delete(existing)
        await db.commit()
        return {"saved": False}
    db.add(SavedReel(reel_id=reel_id, user_id=current_user.id))
    await db.commit()
    return {"saved": True}


@router.get("/saved", response_model=FeedResponse)
async def list_saved(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    saved = (await db.execute(
        select(SavedReel).filter_by(user_id=current_user.id).order_by(desc(SavedReel.created_at))
    )).scalars().all()
    reels = []
    for s in saved:
        reel = await db.get(Post, s.reel_id)
        if reel:
            reels.append(reel)
    return FeedResponse(reels=[await _serialize(r, current_user, db) for r in reels], next_cursor=None)


@router.post("/{reel_id}/share")
async def share_reel(reel_id: int, payload: ShareIn, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    db.add(ShareEvent(reel_id=reel_id, user_id=current_user.id, shared_to_user_id=payload.shared_to_user_id, method=payload.method))
    await db.commit()
    return {"shared": True}


@router.post("/{reel_id}/view")
async def log_view(reel_id: int, payload: ViewEventIn, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    db.add(ReelView(
        reel_id=reel_id, user_id=current_user.id,
        watch_seconds=payload.watch_seconds, reel_duration=payload.reel_duration,
        completed=payload.completed,
    ))
    await db.commit()
    return {"logged": True}


@router.post("/{reel_id}/not-interested")
async def mark_not_interested(reel_id: int, payload: NotInterestedIn, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    exists = (await db.execute(
        select(NotInterested).filter_by(reel_id=reel_id, user_id=current_user.id)
    )).scalar_one_or_none()
    if not exists:
        db.add(NotInterested(reel_id=reel_id, user_id=current_user.id, reason=payload.reason))
        await db.commit()
    return {"hidden": True}


@router.post("/{reel_id}/report")
async def report_reel(reel_id: int, payload: ReportIn, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    db.add(Report(reporter_id=current_user.id, reel_id=reel_id, reason=payload.reason, details=payload.details))
    await db.commit()
    return {"reported": True}


@router.post("/follow/{creator_id}")
async def toggle_follow(creator_id: int, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    if creator_id == current_user.id:
        raise HTTPException(400, "Cannot follow yourself")
    existing = (await db.execute(
        select(Follow).filter_by(follower_id=current_user.id, following_id=creator_id)
    )).scalar_one_or_none()
    if existing:
        await db.delete(existing)
        await db.commit()
        return {"following": False}
    db.add(Follow(follower_id=current_user.id, following_id=creator_id))
    await db.commit()
    return {"following": True}
