"""
Seed the DB with real PHOTOS + VIDEOS (not just text) run through the
REAL AI pipeline — including the CLIP-based image/video analysis path,
not just text sentiment.

Where the media comes from:
    Pexels (https://pexels.com) — free stock photo/video API, licensed
    for any use, no attribution legally required. Curated professional
    stock footage, not scraped social content.
    Free key: https://www.pexels.com/api/  → add to backend/.env:
        PEXELS_API_KEY=your_key_here

Where the captions come from:
    dataset_utils.fetch_captions() — same real TweetEval source as
    seed_from_tweeteval.py and seed_comments.py. Nothing hand-written
    here anymore.

Why some posts have NO caption:
    Real posts are often media with zero text. Half the posts here are
    left blank on purpose so your `text_is_empty` media-fallback branch
    in analyzer.py actually gets exercised, not just the captioned path.

Getting real volume (not the same clip repeated):
    Pexels' "top result" for a query is deterministic — this pulls
    PER_PAGE distinct ranked results per query instead of just the top
    one, so a modest query list still yields lots of different media.

Usage (from backend/, venv active):
    python seed_from_pexels.py
    python seed_from_pexels.py --count 100 --video-ratio 0.35
        (~100 posts total, ~35 of them reels/videos, rest photos)
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
import os
print("GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))

from ai.pipeline.loader import preload_models                       # noqa: E402
from models.database import AsyncSessionLocal                       # noqa: E402
from models.models import User, Post, PostMedia, EmotionLog          # noqa: E402
from ai.pipeline.analyzer import analyze_text                        # noqa: E402
from services.topic_utils import extract_topics                      # noqa: E402
from services.cloudinary_service import _save_local, UPLOAD_ROOT     # noqa: E402
from dataset_utils import fetch_captions                             # noqa: E402
from sqlalchemy import select                                        # noqa: E402


PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Deliberately NEUTRAL scene queries — not pre-labeled as "happy"/"sad".
# The pipeline decides the emotion for itself from the actual media.
SCENE_QUERIES = [
    "sunrise beach", "empty office desk", "rain window", "friends laughing",
    "city street night", "quiet library", "mountain hiking trail",
    "coffee shop morning", "storm clouds", "birthday party",
    "person running", "ocean waves", "candle dark room", "concert crowd",
    "autumn park walk", "busy kitchen", "train station", "farmers market",
    "campfire night", "empty classroom", "traffic jam", "garden flowers",
    "snowy street", "rooftop skyline", "quiet bedroom",
]


async def search_pexels(client: httpx.AsyncClient, query: str, want_video: bool, per_page: int = 8):
    """Returns a LIST of up to per_page distinct results, not just the top one."""
    headers = {"Authorization": PEXELS_API_KEY}
    if want_video:
        url = "https://api.pexels.com/v1/videos/search"
        params = {"query": query, "per_page": per_page, "size": "small"}
    else:
        url = "https://api.pexels.com/v1/search"
        params = {"query": query, "per_page": per_page, "orientation": "square"}

    resp = await client.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = []
    if want_video:
        for video in data.get("videos") or []:
            files = sorted(video["video_files"], key=lambda f: f.get("width") or 9999)
            file = next((f for f in files if (f.get("width") or 0) >= 360), files[-1])
            results.append({"kind": "video", "download_url": file["link"], "ext": ".mp4",
                             "photographer": video["user"]["name"]})
    else:
        for photo in data.get("photos") or []:
            results.append({"kind": "image", "download_url": photo["src"]["medium"], "ext": ".jpg",
                             "photographer": photo["photographer"]})
    return results


async def download_and_save_local(client: httpx.AsyncClient, item: dict) -> dict:
    resp = await client.get(item["download_url"], timeout=60)
    resp.raise_for_status()
    return _save_local(resp.content, folder="posts", extension=item["ext"])


def resolve_disk_path(url: str) -> str:
    return str(UPLOAD_ROOT / url.removeprefix("/uploads/"))


def build_work_queue(count: int, video_ratio: float) -> list[tuple[str, bool]]:
    n_video = round(count * video_ratio)
    n_photo = count - n_video
    queue = (
        [(SCENE_QUERIES[i % len(SCENE_QUERIES)], True) for i in range(n_video)] +
        [(SCENE_QUERIES[i % len(SCENE_QUERIES)], False) for i in range(n_photo)]
    )
    random.shuffle(queue)
    return queue


async def seed(count: int, video_ratio: float):
    if not PEXELS_API_KEY:
        print("❌ PEXELS_API_KEY not set. Get a free key at "
              "https://www.pexels.com/api/ and add it to backend/.env")
        return

    preload_models()

    captions = await fetch_captions(count, min_len=10, max_len=140)
    print(f"✅ Pulled {len(captions)} real captions for the media posts.")

    queue = build_work_queue(count, video_ratio)

    result_cache: dict[tuple[str, bool], list] = {}
    cache_index: dict[tuple[str, bool], int] = {}

    async with AsyncSessionLocal() as db:
        users = (await db.execute(select(User))).scalars().all()
        if not users:
            print("❌ No users in DB — run seed_demo_users.py first.")
            return

        created = 0
        async with httpx.AsyncClient() as client:
            for i, (query, want_video) in enumerate(queue):
                key = (query, want_video)
                if key not in result_cache:
                    try:
                        result_cache[key] = await search_pexels(client, query, want_video)
                        cache_index[key] = 0
                    except Exception as e:
                        print(f"  ⚠️  search failed on \"{query}\": {e}")
                        result_cache[key] = []
                        cache_index[key] = 0

                results = result_cache[key]
                idx = cache_index[key]
                if idx >= len(results):
                    print(f"  (ran out of distinct results for \"{query}\", skipping)")
                    continue
                found = results[idx]
                cache_index[key] += 1

                try:
                    saved = await download_and_save_local(client, found)
                except Exception as e:
                    print(f"  ⚠️  download failed on \"{query}\": {e}")
                    continue

                user = users[i % len(users)]
                use_caption = (i % 2 == 0) and captions
                caption = captions[i % len(captions)] if use_caption else ""
                text_to_analyze = caption.strip() or (
                    "shared a video" if found["kind"] == "video" else "shared a photo"
                )

                risk_result = await db.execute(
                    select(EmotionLog.risk_score)
                    .where(EmotionLog.user_id == user.id)
                    .order_by(EmotionLog.timestamp.desc()).limit(20)
                )
                risk_history = list(reversed(risk_result.scalars().all()))

                pipeline = analyze_text(
                    text_to_analyze,
                    risk_history,
                    media_source=resolve_disk_path(saved["url"]),
                    original_content=caption,
                )

                post = Post(
                    user_id=user.id,
                    content=caption,
                    image_url=saved["url"] if found["kind"] == "image" else "",
                    video_url=saved["url"] if found["kind"] == "video" else "",
                    image_public_id=saved["public_id"] if found["kind"] == "image" else "",
                    video_public_id=saved["public_id"] if found["kind"] == "video" else "",
                    is_reel=(found["kind"] == "video"),
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
                    media_type=found["kind"],
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
                print(f"  [{found['kind']:5s}] ({created}/{count}) user={user.username:15s} "
                      f"emotion={pipeline.emotion:10s} risk={pipeline.risk_score:.3f}  "
                      f"query=\"{query}\"")

                if created % 10 == 0:
                    await db.commit()

        await db.commit()
        print(f"\n✅ Seeded {created}/{count} media posts (photos + reels) from Pexels + TweetEval.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--video-ratio", type=float, default=0.35)
    args = parser.parse_args()
    asyncio.run(seed(args.count, args.video_ratio))