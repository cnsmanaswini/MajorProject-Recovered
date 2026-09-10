from services.notification_service import create_notification
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.models import User, Follow


# --------------------------------------------------
# Helper Function
# --------------------------------------------------

async def update_follow_counts(db: AsyncSession, user1_id: int, user2_id: int):
    """
    Update followers_count and following_count
    """

    # Update user1 following count
    following_result = await db.execute(
        select(Follow).where(
            Follow.follower_id == user1_id,
            Follow.status == "accepted"
        )
    )

    following = following_result.scalars().all()

    user1 = await db.get(User, user1_id)

    if user1:
        user1.following_count = len(following)

    # Update user2 followers count
    followers_result = await db.execute(
        select(Follow).where(
            Follow.following_id == user2_id,
            Follow.status == "accepted"
        )
    )

    followers = followers_result.scalars().all()

    user2 = await db.get(User, user2_id)

    if user2:
        user2.followers_count = len(followers)

    await db.commit()


# --------------------------------------------------
# Follow User
# --------------------------------------------------

async def follow_user(
    db: AsyncSession,
    follower_id: int,
    following_id: int
):

    if follower_id == following_id:
        return {
            "success": False,
            "message": "You cannot follow yourself."
        }

    # Check if target user exists
    target_user = await db.get(User, following_id)

    if not target_user:
        return {
            "success": False,
            "message": "User not found."
        }

    # Already following?
    result = await db.execute(
        select(Follow).where(
            Follow.follower_id == follower_id,
            Follow.following_id == following_id
        )
    )

    existing = result.scalars().first()

    if existing:
        return {
            "success": False,
            "message": "Already followed or requested."
        }

    # Private account?
    if target_user.is_private:
        status = "requested"
    else:
        status = "accepted"

    follow = Follow(
        follower_id=follower_id,
        following_id=following_id,
        status=status
    )

    db.add(follow)

    await db.commit()
    if status == "requested":
     await create_notification(
        db=db,
        user_id=following_id,
        from_user_id=follower_id,
        notification_type="follow_request",
        message="sent you a follow request",
     )
    await db.refresh(follow)
    

    if status == "accepted":
        await update_follow_counts(
            db,
            follower_id,
            following_id
        )
        
        
    await create_notification(
    db=db,
    user_id=following_id,
    from_user_id=follower_id,
    notification_type="follow",
    message="started following you",
    )

    return {
        "success": True,
        "status": status,
        "message": (
            "Follow request sent."
            if status == "requested"
            else "Following user."
        ),
        "data": follow
    }


# --------------------------------------------------
# Unfollow User
# --------------------------------------------------

async def unfollow_user(
    db: AsyncSession,
    follower_id: int,
    following_id: int
):

    result = await db.execute(
        select(Follow).where(
            Follow.follower_id == follower_id,
            Follow.following_id == following_id
        )
    )

    follow = result.scalars().first()

    if not follow:
        return {
            "success": False,
            "message": "Follow relationship not found."
        }

    await db.delete(follow)

    await db.commit()

    await update_follow_counts(
        db,
        follower_id,
        following_id
    )

    return {
        "success": True,
        "message": "User unfollowed successfully."
    }
    
    
    
    
    # --------------------------------------------------
# Accept Follow Request
# --------------------------------------------------

async def accept_follow_request(
    db: AsyncSession,
    follower_id: int,
    following_id: int
):

    result = await db.execute(
        select(Follow).where(
            Follow.follower_id == follower_id,
            Follow.following_id == following_id,
            Follow.status == "requested"
        )
    )

    follow = result.scalars().first()

    if not follow:
        return {
            "success": False,
            "message": "Follow request not found."
        }

    follow.status = "accepted"

    await db.commit()
    await db.refresh(follow)

    await update_follow_counts(
        db,
        follower_id,
        following_id
    )

    return {
        "success": True,
        "message": "Follow request accepted.",
        "data": follow
    }


# --------------------------------------------------
# Reject Follow Request
# --------------------------------------------------

async def reject_follow_request(
    db: AsyncSession,
    follower_id: int,
    following_id: int
):

    result = await db.execute(
        select(Follow).where(
            Follow.follower_id == follower_id,
            Follow.following_id == following_id,
            Follow.status == "requested"
        )
    )

    follow = result.scalars().first()

    if not follow:
        return {
            "success": False,
            "message": "Follow request not found."
        }

    await db.delete(follow)
    await db.commit()

    return {
        "success": True,
        "message": "Follow request rejected."
    }


# --------------------------------------------------
# Get Followers
# --------------------------------------------------

async def get_followers(
    db: AsyncSession,
    user_id: int
):

    result = await db.execute(
        select(User)
        .join(Follow, User.id == Follow.follower_id)
        .where(
            Follow.following_id == user_id,
            Follow.status == "accepted"
        )
    )

    followers = result.scalars().all()

    return [
        {
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "avatar_url": u.avatar_url,
        }
        for u in followers
    ]


# --------------------------------------------------
# Get Following
# --------------------------------------------------

async def get_following(
    db: AsyncSession,
    user_id: int
):

    result = await db.execute(
        select(User)
        .join(Follow, User.id == Follow.following_id)
        .where(
            Follow.follower_id == user_id,
            Follow.status == "accepted"
        )
    )

    following = result.scalars().all()

    return [
        {
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "avatar_url": u.avatar_url,
        }
        for u in following
    ]


# --------------------------------------------------
# Check Follow Status
# --------------------------------------------------

async def get_follow_status(
    db: AsyncSession,
    follower_id: int,
    following_id: int
):

    result = await db.execute(
        select(Follow).where(
            Follow.follower_id == follower_id,
            Follow.following_id == following_id
        )
    )

    follow = result.scalars().first()

    if not follow:
        return {
            "status": "not_following"
        }

    return {
        "status": follow.status
    }