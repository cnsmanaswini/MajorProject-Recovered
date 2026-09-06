"""
Seed the DB with real captions run through the REAL AI pipeline.

Where the captions come from:
    TweetEval (Barbieri et al., cardiffnlp) — an academic Twitter emotion
    benchmark, checked into GitHub as plain text. Real short-form social
    text, not hand-written by us.
    https://github.com/cardiffnlp/tweeteval

Why not just label them with TweetEval's own emotion labels:
    Because the point is to demo YOUR pipeline, not theirs — every caption
    below goes through your actual `analyze_text()` (sentiment, emotion,
    sarcasm, numbness, risk, LSTM temporal risk). The emotion/risk you see
    on each seeded post is genuinely model-decided on your machine, not
    copied from the dataset's labels.

Usage (from backend/, with venv active, server NOT required to be running):
    python scripts/seed_from_tweeteval.py
    python scripts/seed_from_tweeteval.py --count 80 --user-id 1
    python scripts/seed_from_tweeteval.py --sequence-per-user 12  # see notes below

Notes on --sequence-per-user:
    To actually exercise the LSTM's "gradual" temporal behavior (not just a
    single risk score per post in isolation), this script assigns captions
    to a SEQUENCE per user — each post's risk_history includes the ones
    seeded just before it for that same user, same as real usage. Passing
    a higher --sequence-per-user gives each user a longer run of posts,
    which is what you want for Phase 4 (showing risk trend + wellness
    injection kicking in over a run of negative posts).
"""

import argparse
import asyncio
import random
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai.pipeline.loader import preload_models
from models.database import AsyncSessionLocal          # noqa: E402
from models.models import User, Post, EmotionLog        # noqa: E402
from ai.pipeline.analyzer import analyze_text                  # noqa: E402
from services.topic_utils import extract_topics                # noqa: E402
from sqlalchemy import select                            # noqa: E402


TWEETEVAL_URLS = [
    "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/train_text.txt",
    "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/test_text.txt",
]

# Cheap profanity/slur/political-flame filter — not exhaustive, just enough
# to keep a portfolio demo clean. Add to this list if something slips through.
BLOCKLIST_PATTERN = re.compile(
    r"\b(fuck|shit|bitch|nigg|cunt|rape|kill\s?your|nazi|hitler)\w*",
    re.IGNORECASE,
)


def clean_caption(raw: str) -> str | None:
    text = raw.strip()
    if not text:
        return None

    # Strip @user mentions and bare URLs — TweetEval anonymizes handles as
    # literal "@user", not useful in a demo feed.
    text = re.sub(r"@user\b", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 15 or len(text) > 220:
        return None
    if BLOCKLIST_PATTERN.search(text):
        return None
    # Drop lines that are basically just hashtag soup (low real content).
    hashtag_chars = sum(len(w) for w in text.split() if w.startswith("#"))
    if hashtag_chars > len(text) * 0.4:
        return None

    return text


async def fetch_captions(count: int) -> list[str]:
    seen, cleaned = set(), []
    async with httpx.AsyncClient(timeout=30) as client:
        for url in TWEETEVAL_URLS:
            resp = await client.get(url)
            resp.raise_for_status()
            for line in resp.text.splitlines():
                c = clean_caption(line)
                if c and c not in seen:
                    seen.add(c)
                    cleaned.append(c)

    random.shuffle(cleaned)
    if len(cleaned) < count:
        print(f"⚠️  Only found {len(cleaned)} clean captions after filtering "
              f"(asked for {count}). Using all of them.")
        return cleaned
    return cleaned[:count]


async def seed(count: int, sequence_per_user: int, only_user_id: int | None):
    preload_models()
    captions = await fetch_captions(count)
    print(f"✅ Pulled {len(captions)} clean real captions from TweetEval.")

    async with AsyncSessionLocal() as db:
        if only_user_id:
            users = [await db.get(User, only_user_id)]
            if not users[0]:
                print(f"❌ No user with id={only_user_id}")
                return
        else:
            result = await db.execute(select(User))
            users = result.scalars().all()
            if not users:
                print("❌ No users in DB — create at least one account first.")
                return

        # Chunk captions into runs per user, so each user gets a real
        # SEQUENCE (needed for the LSTM's temporal risk to mean anything).
        idx = 0
        created = 0
        for user in users:
            chunk = captions[idx: idx + sequence_per_user]
            idx += sequence_per_user
            if not chunk:
                break

            risk_history: list[float] = []
            for caption in chunk:
                pipeline = analyze_text(
                    caption,
                    risk_history,
                    original_content=caption,
                )
                risk_history.append(pipeline.risk_score)
                risk_history = risk_history[-20:]

                post = Post(
                    user_id=user.id,
                    content=caption,
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

                db.add(EmotionLog(
                    user_id=user.id,
                    sentiment_score=pipeline.sentiment_score,
                    emotion=pipeline.emotion,
                    risk_score=pipeline.risk_score,
                ))
                created += 1
                print(f"  user={user.username:20s} "
                      f"emotion={pipeline.emotion:10s} "
                      f"sentiment={pipeline.sentiment:8s} "
                      f"risk={pipeline.risk_score:.3f}  "
                      f"\"{caption[:50]}\"")

            if idx >= len(captions):
                break

        await db.commit()
        print(f"\n✅ Seeded {created} posts across {min(len(users), (idx // sequence_per_user) + 1)} users.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100,
                         help="How many clean captions to pull total")
    parser.add_argument("--sequence-per-user", type=int, default=10,
                         help="Posts per user, in order — needed so the LSTM sees a real run")
    parser.add_argument("--user-id", type=int, default=None,
                         help="Seed only this user instead of all users")
    args = parser.parse_args()

    asyncio.run(seed(args.count, args.sequence_per_user, args.user_id))