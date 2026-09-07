"""
seed_wellness.py
Creates one dedicated "wellness_daily" account with ~20 genuinely calm,
positive posts — real Pexels nature/calm imagery + hand-written captions
that naturally contain the wellness keywords _matches_wellness_content()
in services/algorithm.py looks for (mindfulness, self-care, calm, gratitude,
breathe, balance, etc.).

Why this exists:
    _load_wellness_candidates() in algorithm.py requires posts that are
    sentiment="positive", risk_score <= 0.25, feed_score >= 0.45, AND match
    a wellness keyword in content/topics. Real TweetEval/Pexels-caption
    posts almost never satisfy all four at once, so the wellness-injection
    feature has had an empty candidate pool this whole session. This
    script exists purely to give it something to actually inject.

Captions are hand-written (not pulled from TweetEval) because we need the
literal keyword match — but they still go through the REAL analyze_text()
pipeline, so sentiment/emotion/risk_score are genuine model output, not
hardcoded. If a caption doesn't score as calm/positive/low-risk, that's the
model's real read of it, not something faked.

No blank posts here — every post gets both image + caption, per your
earlier "no blank posts" preference.

Usage (from backend/, venv active, needs PEXELS_API_KEY in .env same as
seed_from_pexels.py / seed_nature_photos.py):
    python seed_wellness.py
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

import asyncio
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from ai.pipeline.loader import preload_models                       # noqa: E402
from models.database import AsyncSessionLocal                       # noqa: E402
from models.models import User, Post, PostMedia, EmotionLog          # noqa: E402
from ai.pipeline.analyzer import analyze_text                        # noqa: E402
from services.topic_utils import extract_topics                      # noqa: E402
from services.cloudinary_service import _save_local, UPLOAD_ROOT     # noqa: E402
from sqlalchemy import select                                        # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession                      # noqa: E402


PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

WELLNESS_ACCOUNT = ("wellness_daily", "Daily Wellness", "small moments of calm, one post at a time 🌿")

# Each caption naturally contains at least one keyword from
# WELLNESS_CONTENT_KEYWORDS in services/algorithm.py, phrased like a real
# person would post it (not keyword-stuffed).
CAPTIONS_AND_QUERIES = [
    ("Taking five minutes this morning just to breathe and reset before the day starts.", "sunrise beach calm"),
    ("Practicing a little mindfulness on my walk today — noticing the small things.", "quiet forest path"),
    ("Gratitude journal entry #1: grateful for slow mornings like this one.", "coffee journal morning"),
    ("Learning that rest is productive too. Choosing calm over chaos today.", "peaceful lake"),
    ("A short breathing exercise before bed has honestly changed my sleep.", "candle dark room calm"),
    ("Self-care Sunday: tea, a good book, and zero notifications.", "cup of tea cozy"),
    ("Finding balance between work and rest has been the real win this month.", "yoga mat sunrise"),
    ("Reflection time: proud of how far I've come, even on the hard days.", "journal writing desk"),
    ("Sitting in the garden for ten quiet minutes did more for me than I expected.", "garden flowers peaceful"),
    ("A little mental health check-in: I'm doing okay, and that's enough today.", "meditation sunrise"),
    ("Grateful for this view and this moment of stillness.", "mountain view calm"),
    ("Some days self-care just means an early night and a warm blanket.", "cozy blanket window"),
    ("Working on staying present instead of rushing through the day.", "slow morning light"),
    ("Breathing in the fresh air on this trail — exactly the reset I needed.", "hiking trail calm"),
    ("A calm evening walk, no destination, just movement and quiet.", "evening walk park"),
    ("Practicing resilience means being gentle with myself when things are slow.", "peaceful sunset"),
    ("Journaling about what actually brought me peace this week.", "notebook coffee calm"),
    ("A reminder to myself: balance over burnout, always.", "quiet beach chair"),
    ("Grateful for friends who make space for real conversations and calm evenings.", "friends calm conversation"),
    ("Ending today with gratitude for the small, quiet wins.", "sunset reflection calm"),
]


async def search_pexels(client: httpx.AsyncClient, query: str, per_page: int = 5):
    headers = {"Authorization": PEXELS_API_KEY}
    resp = await client.get(
        "https://api.pexels.com/v1/search",
        headers=headers,
        params={"query": query, "per_page": per_page, "orientation": "square"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    photos = data.get("photos") or []
    if not photos:
        return None
    photo = photos[0]
    return {"download_url": photo["src"]["medium"], "ext": ".jpg", "photographer": photo["photographer"]}


async def download_and_save_local(client: httpx.AsyncClient, item: dict) -> dict:
    resp = await client.get(item["download_url"], timeout=60)
    resp.raise_for_status()
    return _save_local(resp.content, folder="posts", extension=item["ext"])


def resolve_disk_path(url: str) -> str:
    return str(UPLOAD_ROOT / url.removeprefix("/uploads/"))


async def get_or_create_wellness_user(db: AsyncSession) -> User:
    username, display_name, bio = WELLNESS_ACCOUNT
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if user:
        return user
    user = User(
        username=username,
        display_name=display_name,
        email=f"{username}@example.com",
        hashed_password="seed_dummy_hash",
        avatar_url=f"https://i.pravatar.cc/150?u={username}",
        bio=bio,
        is_private=False,
        is_verified=True,
    )
    db.add(user)
    await db.flush()
    return user


async def seed():
    if not PEXELS_API_KEY:
        print("❌ PEXELS_API_KEY not set. Get a free key at "
              "https://www.pexels.com/api/ and add it to backend/.env")
        return

    preload_models()

    async with AsyncSessionLocal() as db:
        user = await get_or_create_wellness_user(db)
        print(f"✅ Using account @{user.username} (id={user.id})")

        created = 0
        async with httpx.AsyncClient() as client:
            for caption, query in CAPTIONS_AND_QUERIES:
                try:
                    found = await search_pexels(client, query)
                    if not found:
                        print(f"  ⚠️  no image found for \"{query}\", skipping")
                        continue
                    saved = await download_and_save_local(client, found)
                except Exception as e:
                    print(f"  ⚠️  image step failed for \"{query}\": {e}")
                    continue

                risk_result = await db.execute(
                    select(EmotionLog.risk_score)
                    .where(EmotionLog.user_id == user.id)
                    .order_by(EmotionLog.timestamp.desc()).limit(20)
                )
                risk_history = list(reversed(risk_result.scalars().all()))

                pipeline = analyze_text(
                    caption,
                    risk_history,
                    media_source=resolve_disk_path(saved["url"]),
                    original_content=caption,
                )

                post = Post(
                    user_id=user.id,
                    content=caption,
                    image_url=saved["url"],
                    image_public_id=saved["public_id"],
                    is_reel=False,
                    sentiment=pipeline.sentiment,
                    sentiment_score=pipeline.sentiment_score,
                    emotion=pipeline.emotion,
                    emotion_score=pipeline.emotion_score,
                    sarcasm=pipeline.sarcasm,
                    sarcasm_score=pipeline.sarcasm_score,
                    risk_score=pipeline.risk_score,
                    feed_score=pipeline.feed_score,
                    topics=extract_topics(caption, pipeline.emotion, ""),
                )
                db.add(post)
                await db.flush()
                db.add(PostMedia(
                    post_id=post.id,
                    media_type="image",
                    url=saved["url"],
                    public_id=saved["public_id"],
                    position=0,
                ))
                db.add(EmotionLog(
                    user_id=user.id,
                    sentiment_score=pipeline.sentiment_score,
                    emotion=pipeline.emotion,
                    risk_score=pipeline.risk_score,
                ))
                created += 1

                qualifies = (
                    pipeline.sentiment == "positive"
                    and pipeline.risk_score <= 0.25
                    and pipeline.feed_score >= 0.45
                )
                print(
                    f"  ({created}/{len(CAPTIONS_AND_QUERIES)}) "
                    f"sentiment={pipeline.sentiment:9s} emotion={pipeline.emotion:8s} "
                    f"risk={pipeline.risk_score:.3f} feed_score={pipeline.feed_score:.3f} "
                    f"wellness_eligible={qualifies}  \"{caption[:50]}\""
                )

                if created % 5 == 0:
                    await db.commit()

        await db.commit()
        print(f"\n✅ Seeded {created}/{len(CAPTIONS_AND_QUERIES)} wellness posts for @{user.username}.")


if __name__ == "__main__":
    asyncio.run(seed())