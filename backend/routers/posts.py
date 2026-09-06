"""
Posts Router
POST /api/posts          → create post
GET  /api/posts/{id}     → get post
DELETE /api/posts/{id}   → delete post
POST /api/posts/{id}/like → like/unlike post
"""
from services.cloudinary_service import UPLOAD_ROOT
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import Optional

from models.database import get_db
from models.models import (
    Post,
    User,
    EmotionLog,
    AgentDecision,
    Like,
    PostMedia,
)

from schemas.schemas import PostOut, UserOut
from routers.auth import get_current_user, get_optional_user
from services.algorithm import (
    attach_like_status, update_user_interests, log_like_behavioral_signal,
)
from services.topic_utils import extract_topics

from ai.pipeline.analyzer import analyze_text
from ai.agents.orchestrator import run_agents, EmotionSnapshot
from services.cloudinary_service import upload_image, upload_video
from services.notification_service import create_notification

router = APIRouter()


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

async def get_user_risk_history(
    user_id: int,
    db: AsyncSession,
) -> list[float]:

    result = await db.execute(
        select(EmotionLog.risk_score)
        .where(EmotionLog.user_id == user_id)
        .order_by(EmotionLog.timestamp.desc())
        .limit(20)
    )

    rows = result.scalars().all()
    return list(reversed(rows))


async def get_emotion_history(
    user_id: int,
    db: AsyncSession,
) -> list[EmotionSnapshot]:

    result = await db.execute(
        select(EmotionLog)
        .where(EmotionLog.user_id == user_id)
        .order_by(EmotionLog.timestamp.desc())
        .limit(20)
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


# --------------------------------------------------
# Create Post
# --------------------------------------------------

@router.post("", response_model=PostOut, status_code=201)
async def create_post(
    request: Request,
    content: str = Form(default=""),
    location: str = Form(default=""),
    is_reel: bool = Form(default=False),
    images: Optional[list[UploadFile]] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        body = await request.json()
        content = body.get("content", content or "")
        location = body.get("location", location or "")
        is_reel = body.get("is_reel", is_reel)

    # Author is always the authenticated user — never trust a client-supplied
    # user_id, or anyone could post as anyone else just by changing that field.
    user = current_user

    image_files = [file for file in (images or []) if file and file.filename]
    if image and image.filename:
        image_files.append(image)

    if len(image_files) > 10:
        raise HTTPException(status_code=400, detail="You can add up to 10 photos per post")

    if image_files and video and video.filename:
        raise HTTPException(status_code=400, detail="Use either a photo carousel or one video, not both")

    if not content and not image_files and not (video and video.filename):
        raise HTTPException(
            status_code=400,
            detail="Post must have content, image or video",
        )

    uploaded_media = []

    for position, image_file in enumerate(image_files):
        result = await upload_image(
            image_file,
            folder="mindgram/posts",
        )
        uploaded_media.append({
            "media_type": "image",
            "url": result["url"],
            "public_id": result["public_id"],
            "position": position,
        })

    if video and video.filename:
        result = await upload_video(
            video,
            folder="mindgram/reels",
        )
        uploaded_media.append({
            "media_type": "video",
            "url": result["url"],
            "public_id": result["public_id"],
            "position": 0,
        })
        is_reel = True

    first_image = next((m for m in uploaded_media if m["media_type"] == "image"), None)
    first_video = next((m for m in uploaded_media if m["media_type"] == "video"), None)

    # Run AI pipeline on caption (or a neutral default for image-only posts)
    text_to_analyze = content.strip() or (
        "shared a photo carousel"
        if len(image_files) > 1
        else "shared a photo"
        if first_image
        else "shared a video"
        if first_video
        else "photo post"
    )

    media_url_for_analysis = (
        first_image["url"] if first_image
        else first_video["url"] if first_video
        else None
    )

    # analyze_text/_load_image needs an actual file path for local uploads —
    # the "/uploads/..." url is only servable over HTTP, not a real disk location.
    media_source_for_analysis = media_url_for_analysis
    if media_source_for_analysis and media_source_for_analysis.startswith("/uploads/"):
        media_source_for_analysis = str(UPLOAD_ROOT / media_source_for_analysis.removeprefix("/uploads/"))

    risk_history = await get_user_risk_history(user.id, db)
    pipeline = analyze_text(
        text_to_analyze,
        risk_history,
        media_source=None,   # ← text-only analysis, media analysis disabled
        original_content=content,
    )

    # Create post
    post = Post(
        user_id=user.id,
        content=content,
        image_url=first_image["url"] if first_image else "",
        video_url=first_video["url"] if first_video else "",
        image_public_id=first_image["public_id"] if first_image else "",
        video_public_id=first_video["public_id"] if first_video else "",
        is_reel=is_reel,
        location=location,
        sentiment=pipeline.sentiment,
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        sarcasm=pipeline.sarcasm,
        sarcasm_score=pipeline.sarcasm_score,
        risk_score=pipeline.risk_score,
        feed_score=pipeline.feed_score,
        topics=extract_topics(content, pipeline.emotion, location),
    )

    db.add(post)
    await db.flush()  # assigns post.id, needed before creating PostMedia rows below

    for item in uploaded_media:
        db.add(PostMedia(
            post_id=post.id,
            media_type=item["media_type"],
            url=item["url"],
            public_id=item["public_id"],
            position=item["position"],
        ))

    log = EmotionLog(
        user_id=user.id,
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        risk_score=pipeline.risk_score,
        source="post",
    )

    db.add(log)

    user.posts_count = (
        user.posts_count or 0
    ) + 1

    history = await get_emotion_history(
        user.id,
        db,
    )

    current_snap = EmotionSnapshot(
        sentiment_score=pipeline.sentiment_score,
        emotion=pipeline.emotion,
        emotion_score=pipeline.emotion_score,
        risk_score=pipeline.risk_score,
        source="post",
    )

    agent_report = run_agents(
        current_snap,
        history,
    )

    db.add(
        AgentDecision(
            user_id=user.id,
            risk_level=agent_report.risk_level,
            decision=agent_report.decision,
            intervention=agent_report.intervention,
            rag_suggestion=agent_report.rag_suggestion,
            metadata_json=agent_report.metadata,
        )
    )

    log.agent_action = agent_report.decision

    await db.commit()

    await db.refresh(post, attribute_names=["media"])

    post.author = user

    return post
# --------------------------------------------------
# Get User Posts
# --------------------------------------------------

@router.get("/user/{user_id}", response_model=list[PostOut])
async def get_user_posts(
    user_id: int,
    limit: int = 20,
    offset: int = 0,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Post)
        .options(selectinload(Post.media), selectinload(Post.author))
        .where(Post.user_id == user_id)
        .order_by(Post.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    posts = result.scalars().all()
    user = await db.get(User, user_id)
    for p in posts:
        p.author = user
    if current_user:
        await attach_like_status(posts, current_user.id, db)
    return posts
# --------------------------------------------------
# Get Single Post
# --------------------------------------------------

@router.get("/{post_id}", response_model=PostOut)
async def get_post(
    post_id: int,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(Post)
        .options(selectinload(Post.media), selectinload(Post.author))
        .where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(
            status_code=404,
            detail="Post not found",
        )

    post.author = await db.get(User, post.user_id)
    if current_user:
        await attach_like_status([post], current_user.id, db)

    return post


# --------------------------------------------------
# Delete Post
# --------------------------------------------------

@router.delete("/{post_id}")
async def delete_post(
    post_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(Post)
        .options(selectinload(Post.media), selectinload(Post.author))
        .where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(
            status_code=404,
            detail="Post not found",
        )

    if post.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Not your post",
        )

    for media in post.media:
        delete_asset(media.public_id, resource_type=media.media_type)

    if not post.media and post.image_public_id:
        delete_asset(post.image_public_id, resource_type="image")
    if not post.media and post.video_public_id:
        delete_asset(post.video_public_id, resource_type="video")

    await db.delete(post)

    current_user.posts_count = max(
        0,
        (current_user.posts_count or 1) - 1,
    )

    await db.commit()

    return {
        "status": "deleted",
    }


# --------------------------------------------------
# Like / Unlike Post
# --------------------------------------------------

@router.post("/{post_id}/like")
async def toggle_like(
    post_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):

    post = await db.get(Post, post_id)

    if not post:
        raise HTTPException(
            status_code=404,
            detail="Post not found",
        )

    result = await db.execute(
        select(Like).where(
            Like.post_id == post_id,
            Like.user_id == current_user.id,
        )
    )

    existing = result.scalar_one_or_none()

    if existing:

        await db.delete(existing)

        post.likes_count = max(
            0,
            post.likes_count - 1,
        )

        action = "unliked"

    else:

        like = Like(
            post_id=post_id,
            user_id=current_user.id,
        )

        db.add(like)

        post.likes_count += 1

        action = "liked"

        await update_user_interests(
            current_user.id,
            post.emotion,
            db,
        )

        if post.user_id != current_user.id:

            await create_notification(
                db=db,
                user_id=post.user_id,
                from_user_id=current_user.id,
                notification_type="like",
                message=f"{current_user.username} liked your post",
                post_id=post_id,
            )

    await db.commit()

    if action == "liked":
        await log_like_behavioral_signal(current_user.id, post, db)


    return {
        "status": action,
        "is_liked": action == "liked",
        "likes_count": post.likes_count,
    }
    

@router.get("/{post_id}/likes", response_model=list[UserOut])
async def get_post_likers(
    post_id: int,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List of users who liked this post, most recent first."""
    post = await db.get(Post, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    result = await db.execute(
        select(User)
        .join(Like, Like.user_id == User.id)
        .where(Like.post_id == post_id)
        .order_by(Like.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()