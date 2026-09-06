"""
seed_nature_photos.py
Seeds ~150 real nature photos (Pexels) as posts, each paired with a real
TweetEval caption. Classification is TEXT-ONLY — media_source is
deliberately left None, so CLIP/image analysis never runs. A nature
photo (mountains, forest, ocean...) has no facial expression or scene-
affect for CLIP to meaningfully read, so the caption alone drives the
sentiment/emotion/risk output.

Emotion output is restricted to 5 labels (joy, sadness, anger, fear,
neutral) via the EMOTION_LABELS collapse already applied in analyzer.py
(disgust -> anger, surprise -> fear). Sarcasm is tracked separately as
its own boolean field, same as every other post in this project.

Run from the backend/ folder (needs PEXELS_API_KEY in .env, same as
seed_from_pexels.py):
    python seed_nature_photos.py
    python seed_nature_photos.py --count 150
"""

import argparse
import asyncio
import os
import random
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
from services.cloudinary_service import _save_local                  # noqa: E402
from dataset_utils import fetch_captions                             # noqa: E402
from sqlalchemy import select                                        # noqa: E402


PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Nature-only, still deliberately UNLABELED for emotion — the caption
# text is what actually drives classification, not the scene content.
NATURE_QUERIES = [
    "mountain landscape", "forest trees", "ocean waves", "desert dunes",
    "waterfall", "autumn leaves", "snowy mountains", "sunrise field",
    "sunset sky", "lake reflection", "meadow flowers", "canyon rocks",
    "tropical beach", "pine forest fog", "river valley", "coral reef",
    "northern lights", "cherry blossoms", "rolling hills", "starry night sky",
]


async def search_pexels_photos(client: httpx.AsyncClient, query: str, per_page: int = 8):
    headers = {"Authorization": PEXELS_API_KEY}
    resp = await client.get(
        "https://api.pexels.com/v1/search",
        headers=headers,
        params={"query": query, "per_page": per_page, "orientation": "landscape"},
        timeout=30,
    )
    resp.raise_for_status()
    return [
        {"download_url": p["src"]["medium"], "photographer": p["photographer"]}
        for p in (resp.json().get("photos") or [])
    ]


async def seed(count: int):
    if not PEXELS_API_KEY:
        print("❌ PEXELS_API_KEY not set. Add it to backend/.env")
        return

    preload_models()

    captions = await fetch_captions(count, min_len=10, max_len=140)
    print(f"✅ Pulled {len(captions)} real captions (text-only classification).")

    async with AsyncSessionLocal() as db:
        users = (await db.execute(select(User))).scalars().all()
        if not users:
            print("❌ No users in DB — run seed_demo_users.py first.")
            return

        result_cache: dict[str, list] = {}
        cache_index: dict[str, int] = {}
        created = 0

        async with httpx.AsyncClient() as client:
            for i in range(count):
                query = NATURE_QUERIES[i % len(NATURE_QUERIES)]
                if query not in result_cache:
                    try:
                        result_cache[query] = await search_pexels_photos(client, query)
                        cache_index[query] = 0
                    except Exception as e:
                        print(f"  ⚠️  search failed on \"{query}\": {e}")
                        result_cache[query] = []
                        cache_index[query] = 0

                results = result_cache[query]
                idx = cache_index[query]
                if idx >= len(results):
                    print(f"  (ran out of distinct photos for \"{query}\", skipping)")
                    continue
                found = results[idx]
                cache_index[query] += 1

                try:
                    resp = await client.get(found["download_url"], timeout=60)
                    resp.raise_for_status()
                    saved = _save_local(resp.content, folder="posts", extension=".jpg")
                except Exception as e:
                    print(f"  ⚠️  download failed on \"{query}\": {e}")
                    continue

                user = users[i % len(users)]
                caption = captions[i % len(captions)] if captions else ""

                risk_result = await db.execute(
                    select(EmotionLog.risk_score)
                    .where(EmotionLog.user_id == user.id)
                    .order_by(EmotionLog.timestamp.desc()).limit(20)
                )
                risk_history = list(reversed(risk_result.scalars().all()))

                # media_source deliberately omitted -- text-only classification,
                # no CLIP call, per this script's whole purpose.
                pipeline = analyze_text(
                    caption,
                    risk_history,
                    original_content=caption,
                )

                post = Post(
                    likes_count=random.randint(5, 400),
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
                    post_id=post.id, media_type="image",
                    url=saved["url"], public_id=saved["public_id"], position=0,
                ))
                db.add(EmotionLog(
                    user_id=user.id, sentiment_score=pipeline.sentiment_score,
                    emotion=pipeline.emotion, risk_score=pipeline.risk_score,
                ))
                created += 1
                print(f"  ({created}/{count}) user={user.username:15s} "
                      f"emotion={pipeline.emotion:8s} sarcasm={pipeline.sarcasm!s:5s} "
                      f"risk={pipeline.risk_score:.3f}  query=\"{query}\"")

                if created % 20 == 0:
                    await db.commit()

        await db.commit()
        print(f"\n✅ Seeded {created}/{count} nature-photo posts, text-only classified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150)
    args = parser.parse_args()
    asyncio.run(seed(args.count))