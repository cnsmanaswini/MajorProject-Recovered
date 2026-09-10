"""
Users Router
GET    /api/users/{username}         → get profile
POST   /api/users/{id}/follow        → follow/unfollow
GET    /api/users/{id}/followers     → get followers
GET    /api/users/{id}/following     → get following
PUT    /api/users/me                 → update profile
POST   /api/users/me/avatar          → upload avatar
GET    /api/users/search             → search users
"""

from datetime import datetime
import bcrypt
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from typing import Optional
from pydantic import BaseModel, EmailStr

from models.database import get_db
from models.models import User, Follow, Post, Notification
from routers.auth import get_current_user, get_optional_user
from services.cloudinary_service import upload_avatar

router = APIRouter()


class UserCreateRequest(BaseModel):
    username: str
    display_name: str
    email: Optional[EmailStr] = None
    password: Optional[str] = None


# ── Create User ───────────────────────────────────────────────

@router.post("", status_code=201)
async def create_user(body: UserCreateRequest, db: AsyncSession = Depends(get_db)):
    # Ensure username is unique
    result = await db.execute(select(User).where(User.username == body.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already taken")

    email = body.email
    if email is None:
        base_email = f"{body.username}@mindgram.test"
        email = base_email
        counter = 1
        while True:
            result = await db.execute(select(User).where(User.email == email))
            if not result.scalar_one_or_none():
                break
            email = f"{body.username}+{counter}@mindgram.test"
            counter += 1

    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        username=body.username,
        email=email,
        display_name=body.display_name or body.username,
        hashed_password=bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode() if body.password else None,
        avatar_url=f"https://api.dicebear.com/9.x/avataaars/svg?seed={body.username}",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
        "email": user.email,
    }


@router.get("")
async def list_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).limit(50))
    users = result.scalars().all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "avatar_url": u.avatar_url,
            "bio": u.bio,
        }
        for u in users
    ]


# Static paths MUST be registered before /{username_or_id}.
# Otherwise GET /users/me and /users/search are caught by the int path,
# FastAPI returns 422, and the profile page shows "User not found".

@router.get("/search")
async def search_users(
    q: str,
    db: AsyncSession = Depends(get_db),
):
    """Search users by username or display name."""
    result = await db.execute(
        select(User).where(
            or_(
                User.username.ilike(f"%{q}%"),
                User.display_name.ilike(f"%{q}%"),
            )
        ).limit(20)
    )
    users = result.scalars().all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "avatar_url": u.avatar_url,
            "followers_count": u.followers_count,
        }
        for u in users
    ]


@router.get("/me")
async def get_my_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current user's full profile."""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "username_changed_at": current_user.username_changed_at,
        "display_name": current_user.display_name,
        "avatar_url": current_user.avatar_url,
        "bio": current_user.bio,
        "email": current_user.email,
        "followers_count": current_user.followers_count,
        "following_count": current_user.following_count,
        "posts_count": current_user.posts_count,
        "is_private": current_user.is_private,
    }


@router.get("/{username_or_id}")
async def get_profile(
    username_or_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a user's public profile by username or numeric id."""
    user = None
    if username_or_id.isdigit():
        user = await db.get(User, int(username_or_id))
    if user is None:
        result = await db.execute(
            select(User).where(User.username == username_or_id)
        )
        user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if current user follows this user, and vice versa
    is_following = False
    follows_you = False
    if current_user:
        follow_result = await db.execute(
            select(Follow).where(
                Follow.follower_id == current_user.id,
                Follow.following_id == user.id,
            )
        )
        is_following = follow_result.scalar_one_or_none() is not None

        back_result = await db.execute(
            select(Follow).where(
                Follow.follower_id == user.id,
                Follow.following_id == current_user.id,
            )
        )
        follows_you = back_result.scalar_one_or_none() is not None

    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
        "bio": user.bio,
        "followers_count": user.followers_count,
        "following_count": user.following_count,
        "posts_count": user.posts_count,
        "is_following": is_following,
        "follows_you": follows_you,
        "is_private": user.is_private,
    }


# ── Follow / Unfollow ─────────────────────────────────────────

@router.post("/{user_id}/follow")
async def toggle_follow(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Follow or unfollow a user."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")

    target = await db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if already following
    result = await db.execute(
        select(Follow).where(
            Follow.follower_id == current_user.id,
            Follow.following_id == user_id,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        # Unfollow
        await db.delete(existing)
        target.followers_count = max(0, (target.followers_count or 1) - 1)
        current_user.following_count = max(0, (current_user.following_count or 1) - 1)
        action = "unfollowed"
    else:
        # Follow
        db.add(Follow(
            follower_id=current_user.id,
            following_id=user_id,
        ))
        target.followers_count = (target.followers_count or 0) + 1
        current_user.following_count = (current_user.following_count or 0) + 1
        action = "followed"

        # Send notification
        db.add(Notification(
            user_id=user_id,
            from_user_id=current_user.id,
            type="follow",
            message=f"{current_user.username} started following you",
        ))

    await db.commit()
    return {
        "status": action,
        "followers_count": target.followers_count,
        "following_count": current_user.following_count,
    }


# ── Followers / Following Lists ───────────────────────────────

@router.get("/{user_id}/followers")
async def get_followers(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Follow).where(Follow.following_id == user_id)
    )
    follows = result.scalars().all()
    users = []
    for f in follows:
        u = await db.get(User, f.follower_id)
        if u:
            users.append({
                "id": u.id,
                "username": u.username,
                "display_name": u.display_name,
                "avatar_url": u.avatar_url,
            })
    return users


@router.get("/{user_id}/following")
async def get_following(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Follow).where(Follow.follower_id == user_id)
    )
    follows = result.scalars().all()
    users = []
    for f in follows:
        u = await db.get(User, f.following_id)
        if u:
            users.append({
                "id": u.id,
                "username": u.username,
                "display_name": u.display_name,
                "avatar_url": u.avatar_url,
            })
    return users


# ── Update Profile ────────────────────────────────────────────

@router.put("/me")
async def update_profile(
    display_name: str = Form(default=None),
    bio: str = Form(default=None),
    is_private: bool = Form(default=None),
    username: str = Form(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if display_name is not None:
        current_user.display_name = display_name
    if bio is not None:
        current_user.bio = bio
    if is_private is not None:
        current_user.is_private = is_private

    if username is not None and username != current_user.username:
        if current_user.username_changed_at is not None:
            days_since = (datetime.utcnow() - current_user.username_changed_at).days
            if days_since < 15:
                days_left = 15 - days_since
                raise HTTPException(
                    status_code=429,
                    detail=f"You can change your username again in {days_left} day(s).",
                )

        existing = await db.execute(
            select(User).where(User.username == username, User.id != current_user.id)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Username already taken")

        current_user.username = username
        current_user.username_changed_at = datetime.utcnow()

    await db.commit()
    await db.refresh(current_user)
    return {
        "id": current_user.id,
        "username": current_user.username,
        "display_name": current_user.display_name,
        "bio": current_user.bio,
        "avatar_url": current_user.avatar_url,
        "is_private": current_user.is_private,
        "username_changed_at": current_user.username_changed_at,
    }

# ── Upload Avatar ─────────────────────────────────────────────

@router.post("/me/avatar")
async def update_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await upload_avatar(file, current_user.username)
    current_user.avatar_url = result["url"]
    await db.commit()
    return {"avatar_url": result["url"]}


# ── Notifications ─────────────────────────────────────────────

@router.get("/me/notifications")
async def get_notifications(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Notification)
        .where(Notification.user_id == current_user.id)
        .order_by(Notification.created_at.desc())
        .limit(30)
    )
    notifications = result.scalars().all()

    items = []
    for n in notifications:
        from_user = await db.get(User, n.from_user_id) if n.from_user_id else None
        items.append({
            "id": n.id,
            "type": n.type,
            "message": n.message,
            "is_read": n.is_read,
            "created_at": n.created_at,
            "from_user": {
                "username": from_user.username,
                "avatar_url": from_user.avatar_url,
            } if from_user else None,
        })

    # Mark all as read
    for n in notifications:
        n.is_read = True
    await db.commit()

    return items