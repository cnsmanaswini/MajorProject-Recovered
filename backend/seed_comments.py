"""
seed_comments.py
Adds real comments (from the same TweetEval source as your other seed
scripts) to existing posts, each analyzed through your actual
analyze_text() pipeline — not hand-typed, not random placeholder text.

Comments are shorter/simpler than posts (no risk_history threading —
your Comment model doesn't track a per-comment temporal risk sequence,
it mirrors Post's single-item fields), so this calls analyze_text with
an empty risk_history per comment.

Run from the backend/ folder, AFTER you have posts to comment on:
    python seed_comments.py
    python seed_comments.py --max-per-post 5
"""

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai.pipeline.loader import preload_models     # noqa: E402
from models.database import AsyncSessionLocal      # noqa: E402
from models.models import User, Post, Comment       # noqa: E402
from ai.pipeline.analyzer import analyze_text        # noqa: E402
from dataset_utils import fetch_captions             # noqa: E402
from sqlalchemy import select, func                  # noqa: E402

random.seed(7)


async def seed(max_per_post: int, caption_pool_size: int):
    preload_models()

    captions = await fetch_captions(caption_pool_size, min_len=5, max_len=140)
    print(f"✅ Pulled {len(captions)} real short lines for comments.")

    async with AsyncSessionLocal() as db:
        users = (await db.execute(select(User))).scalars().all()
        posts = (await db.execute(select(Post))).scalars().all()

        if not users or not posts:
            print("❌ Need at least some users and posts already seeded first.")
            return

        total_comments = 0
        for post in posts:
            n = random.randint(0, max_per_post)
            if n == 0:
                continue

            # Don't let someone comment on their own post — mirrors real
            # social behavior and keeps author checks in the frontend sane.
            possible_commenters = [u for u in users if u.id != post.user_id] or users

            for _ in range(n):
                text = random.choice(captions)
                commenter = random.choice(possible_commenters)

                pipeline = analyze_text(text, [], original_content=text)

                comment = Comment(
                    post_id=post.id,
                    user_id=commenter.id,
                    content=text,
                    sentiment=pipeline.sentiment,
                    sentiment_score=pipeline.sentiment_score,
                    emotion=pipeline.emotion,
                    emotion_score=pipeline.emotion_score,
                    sarcasm=pipeline.sarcasm,
                    sarcasm_score=pipeline.sarcasm_score,
                    risk_score=pipeline.risk_score,
                    feed_score=pipeline.feed_score,
                    likes=random.randint(0, 30),
                )
                db.add(comment)
                total_comments += 1

            post.comments_count = n  # keep the denormalized counter honest

        await db.commit()
        print(f"\n✅ Seeded {total_comments} real, pipeline-analyzed comments "
              f"across {len(posts)} posts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-post", type=int, default=4,
                         help="Upper bound on comments per post (randomized 0..N)")
    parser.add_argument("--pool-size", type=int, default=150,
                         help="How many distinct real comment lines to pull")
    args = parser.parse_args()
    asyncio.run(seed(args.max_per_post, args.pool_size))
    