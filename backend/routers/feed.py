"""
Feed Router
GET /api/feed/         → personalized feed
GET /api/feed/explore  → discover new content
GET /api/feed/reels    → reels feed

NOTE: Route order matters here. FastAPI/Starlette match routes in the
order they're registered, not by specificity. All static-path routes
(/trending, /explore, /reels, /stories) MUST be registered before the
dynamic /{user_id} route — otherwise a request to e.g. /explore gets
caught by /{user_id} first, Pydantic tries to coerce "explore" to an
int, and you get a 422 instead of ever reaching the real handler.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, exists
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta

from models.database import get_db
from models.models import Post, User, Story, PostMedia
from schemas.schemas import PostOut
from routers.auth import get_current_user, get_optional_user
from services.algorithm import build_feed, attach_like_status, get_trending_topics
from models.models import User as UserModel

router = APIRouter()


@router.get("", response_model=list[PostOut])
async def get_feed(
    limit: int = Query(default=20, le=50),
    offset: int = Query(default=0),
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Personalized feed using Instagram-like algorithm.
    """

    posts = await build_feed(
        user_id=current_user.id,
        db=db,
        limit=limit,
        offset=offset,
    )
    await attach_like_status(posts, current_user.id, db)
    return posts


# ── Static routes registered BEFORE /{user_id} ─────────────────

@router.get("/trending")
async def get_trending(
    limit: int = Query(default=10, le=20),
    db: AsyncSession = Depends(get_db),
):
    return await get_trending_topics(db, limit=limit)


@router.get("/explore", response_model=list[PostOut])
async def get_explore(
    limit: int = Query(default=30, le=50),
    offset: int = Query(default=0, ge=0),
    current_user: UserModel = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Explore page — posts AND reels from public accounts,
    regardless of whether the current user follows them.

    (Previously this excluded reels via is_reel == False and had no
    privacy filter at all, which let private accounts' posts leak in.
    Fixed: reels included, private accounts excluded.)
    """

    cutoff = datetime.utcnow() - timedelta(days=14)

    result = await db.execute(
        select(Post)
        .join(User, User.id == Post.user_id)
        .options(selectinload(Post.media), selectinload(Post.author))
        .where(
            Post.created_at >= cutoff,
            User.is_private == False,
        )
        .order_by(
            Post.feed_score.desc(),
            Post.likes_count.desc(),
            Post.id.desc()  # tiebreaker so pagination is stable when scores/likes tie
        )
        .offset(offset)
        .limit(limit)
    )

    posts = result.scalars().all()

    # Load authors
    for p in posts:
        p.author = await db.get(User, p.user_id)
        if not (p.video_url or "").strip() and p.media:
            for m in p.media:
                if m.media_type == "video" and m.url:
                    p.video_url = m.url
                    break

    if current_user:
        await attach_like_status(posts, current_user.id, db)

    return posts


@router.get("/reels", response_model=list[PostOut])
async def get_reels(
    limit: int = Query(default=10, le=30),
    current_user: UserModel = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Dedicated Reels feed.
    Only reels are returned.
    """

    cutoff = datetime.utcnow() - timedelta(days=365)

    has_video_media = exists().where(
        PostMedia.post_id == Post.id,
        PostMedia.media_type == "video",
    )

    result = await db.execute(
        select(Post)
        .options(selectinload(Post.media), selectinload(Post.author))
        .where(
            or_(
                Post.is_reel == True,  # noqa: E712
                Post.video_url != "",
                has_video_media,
            ),
            Post.created_at >= cutoff,
        )
        .order_by(Post.feed_score.desc())
        .limit(limit)
    )

    posts = result.scalars().all()

    for p in posts:
        p.author = await db.get(User, p.user_id)
        if not (p.video_url or "").strip() and p.media:
            for m in p.media:
                if m.media_type == "video" and m.url:
                    p.video_url = m.url
                    break
    if current_user:
        await attach_like_status(posts, current_user.id, db)
    return posts


@router.get("/stories")
async def get_stories(
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get stories from followed users.
    Stories expire after 24 hours.
    """

    from models.models import Follow

    result = await db.execute(
        select(Follow.following_id)
        .where(Follow.follower_id == current_user.id)
    )

    following_ids = [r[0] for r in result.fetchall()]
    following_ids.append(current_user.id)

    cutoff = datetime.utcnow() - timedelta(hours=24)

    stories_result = await db.execute(
        select(Story)
        .where(
            Story.user_id.in_(following_ids),
            Story.created_at >= cutoff,
        )
        .order_by(Story.created_at.desc())
    )

    stories = stories_result.scalars().all()

    stories_by_user = {}

    for story in stories:

        uid = story.user_id

        if uid not in stories_by_user:

            user = await db.get(User, uid)

            stories_by_user[uid] = {
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "display_name": user.display_name,
                    "avatar_url": user.avatar_url,
                },
                "stories": [],
            }

        stories_by_user[uid]["stories"].append({
            "id": story.id,
            "image_url": story.image_url,
            "video_url": story.video_url,
            "text": story.text,
            "created_at": story.created_at,
        })

    return list(stories_by_user.values())


# ── Dynamic route registered LAST ──────────────────────────────

@router.get("/{user_id}", response_model=list[PostOut])
async def get_feed_for_user(
    user_id: int,
    limit: int = Query(default=20, le=50),
    offset: int = Query(default=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Personalized feed for a user by id. This endpoint is used by tests and allows
    feed retrieval without bearer authentication.
    """

    posts = await build_feed(
        user_id=user_id,
        db=db,
        limit=limit,
        offset=offset,
    )
    await attach_like_status(posts, user_id, db)
    return posts