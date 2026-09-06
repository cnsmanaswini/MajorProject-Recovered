"""
seed_demo_users.py
Creates ~15 public demo accounts (bio + avatar only, NO posts) so
seed_from_tweeteval.py / seed_from_pexels.py have real other-user accounts
to attribute dataset-sourced, real-pipeline-analyzed posts to.

This is deliberately posts-free — no hand-written captions, no fake
sentiment/emotion. If you want these accounts to have content, run the
other two seed scripts after this one; every post they create goes
through your actual analyze_text() pipeline.

Safe to re-run: usernames are unique, existing accounts are left as-is.

Run from the backend/ folder:
    python seed_demo_users.py
"""

import asyncio
import random
from ai.pipeline.loader import preload_models
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import AsyncSessionLocal
from models.models import User

random.seed(42)

# (username, display_name, bio)
ACCOUNTS = [
    ("maya.wanders", "Maya Chen", "wandering with a camera 📸 // Bay Area"),
    ("kofi_builds", "Kofi Owusu", "furniture maker. sawdust in my coffee."),
    ("elena.codes", "Elena Petrova", "backend engineer, forever debugging"),
    ("raj.eats", "Raj Malhotra", "home cook, chasing the perfect dal"),
    ("noor.sketches", "Noor Haddad", "sketchbook nomad ✏️"),
    ("leo_hikes", "Leo Fischer", "PNW trails, 30 peaks and counting"),
    ("amara.reads", "Amara Johnson", "book a day keeps reality away"),
    ("theo_runs", "Theo Nakamura", "marathon #4 training log"),
    ("priya.bakes", "Priya Nair", "sourdough obsessed, weekend baker"),
    ("sam.shoots", "Sam Okafor", "film photography, mostly 35mm"),
    ("zara.plants", "Zara Kowalski", "plant mom to 40+ and counting 🌱"),
    ("arjun.trains", "Arjun Mehta", "powerlifting, one plate at a time"),
    ("wren.travels", "Wren Alsop", "slow travel, one city a month"),
    ("dev.paints", "Dev Krishnan", "weekend watercolors"),
    ("lina.brews", "Lina Andersson", "third-wave coffee, home roaster"),
]


async def get_or_create_user(db: AsyncSession, username: str, display_name: str, bio: str) -> User:
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if user:
        return user
    user = User(
        username=username,
        display_name=display_name,
        email=f"{username}@example.com",
        hashed_password="seed_dummy_hash",  # not a real login
        avatar_url=f"https://i.pravatar.cc/150?u={username}",
        bio=bio,
        is_private=False,
        is_verified=random.random() < 0.2,
    )
    db.add(user)
    await db.flush()
    return user


async def seed():
    preload_models()
    async with AsyncSessionLocal() as db:
        created = 0
        for username, display_name, bio in ACCOUNTS:
            existing_check = await db.execute(select(User).where(User.username == username))
            was_new = existing_check.scalar_one_or_none() is None
            await get_or_create_user(db, username, display_name, bio)
            if was_new:
                created += 1

        await db.commit()
        print(f"✅ {created} new demo accounts created "
              f"({len(ACCOUNTS) - created} already existed).")
        print("Now run seed_from_tweeteval.py and/or seed_from_pexels.py "
              "to give them real, pipeline-analyzed posts.")


if __name__ == "__main__":
    asyncio.run(seed())