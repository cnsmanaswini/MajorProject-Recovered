"""
compute_user_feed_profiles.py
build_feed() in services/algorithm.py personalizes the home feed using
each user's AgentDecision.risk_level and UserInterest scores — but
nothing creates those rows for seeded (or real) users. Without them,
every user silently falls back to the same default (low risk, no
interest weighting), so the feed can't actually differentiate users.

This script computes each user's REAL average sentiment/risk/emotion
from their own EmotionLog rows (already written, real pipeline output,
from every other seed script) and writes the AgentDecision +
UserInterest rows build_feed() needs to act on it.

Run this LAST, after all your other seed scripts (it needs EmotionLog
rows to already exist):
    python compute_user_feed_profiles.py
"""

import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.database import AsyncSessionLocal              # noqa: E402
from models.models import User, EmotionLog, AgentDecision, UserInterest  # noqa: E402
from sqlalchemy import select, delete                        # noqa: E402


# Matches the thresholds build_feed() itself uses:
# risk_map = {"low": 0.1, "moderate": 0.45, "high": 0.70, "critical": 0.92}
def bucket_risk(avg_risk: float) -> str:
    if avg_risk < 0.275:
        return "low"
    if avg_risk < 0.575:
        return "moderate"
    if avg_risk < 0.81:
        return "high"
    return "critical"


DECISION_BY_LEVEL = {
    "low": ("monitor", "", ""),
    "moderate": ("gentle_prompt", "Gently prompt user with a wellness check-in.",
                 "Take a moment today to write 3 things you're grateful for."),
    "high": ("supportive_checkin", "Surface a supportive check-in and reduce heavy content.",
             "Consider reaching out to someone you trust today."),
    "critical": ("priority_support", "Prioritize wellness resources in the feed.",
                 "If you're struggling, support is available — you don't have to carry this alone."),
}


async def compute():
    async with AsyncSessionLocal() as db:
        users = (await db.execute(select(User))).scalars().all()
        if not users:
            print("❌ No users in DB.")
            return

        # Clear any previous run's rows for these users so re-running
        # doesn't stack up duplicate AgentDecision/UserInterest history.
        user_ids = [u.id for u in users]
        await db.execute(delete(AgentDecision).where(AgentDecision.user_id.in_(user_ids)))
        await db.execute(delete(UserInterest).where(UserInterest.user_id.in_(user_ids)))

        updated = 0
        for user in users:
            logs = (await db.execute(
                select(EmotionLog).where(EmotionLog.user_id == user.id)
            )).scalars().all()

            if not logs:
                continue  # no real data for this user yet — leave them at build_feed's default

            avg_risk = sum(l.risk_score for l in logs) / len(logs)
            avg_sentiment = sum(l.sentiment_score for l in logs) / len(logs)
            risk_level = bucket_risk(avg_risk)
            decision, intervention, rag_suggestion = DECISION_BY_LEVEL[risk_level]

            db.add(AgentDecision(
                user_id=user.id,
                risk_level=risk_level,
                decision=decision,
                intervention=intervention,
                rag_suggestion=rag_suggestion,
            ))

            # UserInterest: how often each emotion shows up in THIS user's
            # own posts, normalized to a 0-1 score per emotion. This is a
            # proxy for "what emotional territory this user's content
            # lives in" — real derived data, not assigned.
            emotion_counts = Counter(l.emotion for l in logs)
            total = sum(emotion_counts.values())
            for emotion, count in emotion_counts.items():
                db.add(UserInterest(
                    user_id=user.id,
                    emotion=emotion,
                    score=round(count / total, 4),
                ))

            updated += 1
            print(f"  {user.username:15s} avg_sentiment={avg_sentiment:+.3f} "
                  f"avg_risk={avg_risk:.3f} -> risk_level={risk_level:10s} "
                  f"top_emotion={emotion_counts.most_common(1)[0][0]}")

        await db.commit()
        print(f"\n✅ Computed feed-personalization profiles for {updated}/{len(users)} users.")
        print("build_feed() will now actually differentiate between users' home feeds.")


if __name__ == "__main__":
    asyncio.run(compute())