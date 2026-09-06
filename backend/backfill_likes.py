"""
One-off fix: seed_from_pexels.py forgot to set likes_count/comments_count,
so the 100 posts it created are stuck at 0/0. This backfills those in
place — no new posts, no re-running the pexels script needed.

Scope: updates every Post currently at likes_count=0 AND comments_count=0.
That's a reasonable heuristic for "seeded but never got numbers set" (a
real post that's been up a while with genuinely zero engagement is
possible but unlikely in a dev DB). If you want to protect your OWN
account from this, pass --exclude-user-id.

Usage (from backend/, venv active):
    python backfill_likes.py
    python backfill_likes.py --exclude-user-id 1
"""

import argparse
import asyncio
import random

from sqlalchemy import select

from models.database import AsyncSessionLocal
from models.models import Post


async def backfill(exclude_user_id: int | None):
    async with AsyncSessionLocal() as db:
        query = select(Post).where(Post.likes_count == 0, Post.comments_count == 0)
        if exclude_user_id is not None:
            query = query.where(Post.user_id != exclude_user_id)
        result = await db.execute(query)
        posts = result.scalars().all()

        if not posts:
            print("Nothing to backfill — no posts at 0/0.")
            return

        for post in posts:
            post.likes_count = random.randint(5, 480)
            post.comments_count = random.randint(0, 40)
            if post.is_reel:
                post.views_count = random.randint(50, 2000)

        await db.commit()
        print(f"Backfilled likes/comments on {len(posts)} posts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-user-id", type=int, default=None,
                         help="Skip this user's own posts (e.g. your real account)")
    args = parser.parse_args()
    asyncio.run(backfill(args.exclude_user_id))