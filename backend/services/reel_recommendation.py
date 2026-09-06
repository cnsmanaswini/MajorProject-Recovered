# backend/services/reel_recommendation.py
"""
Reel feed ranking: interest + recency + relationship + engagement,
with a wellness-aware smoothing layer that reuses MindGram's existing
sentiment/emotion pipeline instead of duplicating it.

Call `rank_reels_for_user(db, user_id, candidate_reels)` from your feed
router. It returns the same reels sorted, each annotated with
`rank_score` and a human-readable `rank_reason`.
"""

import math
from datetime import datetime, timezone
from typing import List, Dict, Any


# ---- Tunable weights -------------------------------------------------
W_ENGAGEMENT = 0.30
W_RECENCY = 0.15
W_RELATIONSHIP = 0.20
W_AFFINITY = 0.20      # user's historical interest in this creator/hashtag/audio
W_WELLNESS = 0.15      # smooths away from a spiral of consistently negative content


def _engagement_score(reel) -> float:
    """Normalize engagement using a log curve so viral outliers don't dominate."""
    likes = getattr(reel, "like_count", None) or getattr(reel, "likes_count", 0) or 0
    comments = getattr(reel, "comment_count", None) or getattr(reel, "comments_count", 0) or 0
    shares = getattr(reel, "share_count", None) or getattr(reel, "shares_count", 0) or 0
    saves = getattr(reel, "save_count", 0) or 0
    weighted = likes * 1.0 + comments * 2.0 + shares * 3.0 + saves * 2.5
    return min(1.0, math.log1p(weighted) / math.log1p(2000))  # cap at ~2000 weighted actions


def _recency_score(created_at: datetime, half_life_hours: float = 30.0) -> float:
    age_hours = (datetime.now(timezone.utc) - created_at.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    return 0.5 ** (age_hours / half_life_hours)


def _relationship_score(user_id: int, creator_id: int, following_ids: set) -> float:
    return 1.0 if creator_id in following_ids else 0.0


def _affinity_score(reel, user_signals: Dict[str, Any]) -> float:
    """user_signals holds aggregated history: favorite hashtags, audios, creators
    the user engages with repeatedly. Pass this in precomputed for the batch."""
    score = 0.0
    creator_engagement = user_signals.get("creator_engagement", {}).get(reel.user_id, 0)
    score += min(0.5, creator_engagement / 10)

    reel_hashtags = set(getattr(reel, "hashtags", []) or [])
    fav_hashtags = user_signals.get("favorite_hashtags", set())
    if reel_hashtags & fav_hashtags:
        score += 0.3

    audio_id = getattr(reel, "audio_id", None)
    if audio_id and audio_id in user_signals.get("favorite_audio_ids", set()):
        score += 0.2

    return min(1.0, score)


def _wellness_adjustment(reel, user_signals: Dict[str, Any]) -> float:
    """
    Gentle, gradual nudge — NOT a hard filter. If a user's recent trajectory
    (from your LSTM temporal risk scoring) shows rising negative affect, mildly
    deprioritize additional heavy/negative content and mildly favor supportive
    or neutral content, rather than abruptly changing their feed.
    """
    user_risk_trend = user_signals.get("risk_trend", 0.0)  # 0 = stable, 1 = concerning upward trend
    reel_sentiment = getattr(reel, "sentiment", None)
    reel_risk = getattr(reel, "risk_score", 0.0) or 0.0

    if user_risk_trend <= 0.2:
        return 0.5  # neutral, no adjustment needed

    # scale the nudge by how concerning the trend is, capped so it's gradual
    nudge_strength = min(0.4, user_risk_trend * 0.4)

    if reel_sentiment == "negative" or reel_risk > 0.6:
        return 0.5 - nudge_strength
    elif reel_sentiment == "positive" or reel_risk < 0.3:
        return 0.5 + nudge_strength
    return 0.5


def _build_reason(scores: Dict[str, float], reel, following: bool) -> str:
    top = max(scores, key=scores.get)
    reasons = {
        "engagement": "Popular right now — lots of likes, comments and shares",
        "recency": "Recently posted",
        "relationship": (
            f"From @{getattr(getattr(reel, 'creator', None) or getattr(reel, 'author', None), 'username', 'user')}, who you follow"
            if following else "From an account you interact with"
        ),
        "affinity": "Similar to content you've engaged with before",
        "wellness": "Chosen to keep your feed balanced",
    }
    return reasons.get(top, "Recommended for you")


def rank_reels_for_user(candidate_reels: List[Any], following_ids: set, user_signals: Dict[str, Any]) -> List[Any]:
    """
    candidate_reels: list of Reel ORM/DTO objects (must have like_count, comment_count,
        share_count, save_count, created_at, user_id, audio_id, hashtags, sentiment, risk_score)
    following_ids: set of user IDs the current user follows
    user_signals: precomputed dict — see _affinity_score / _wellness_adjustment
    """
    scored = []
    for reel in candidate_reels:
        following = reel.user_id in following_ids
        component_scores = {
            "engagement": _engagement_score(reel),
            "recency": _recency_score(reel.created_at),
            "relationship": _relationship_score(None, reel.user_id, following_ids),
            "affinity": _affinity_score(reel, user_signals),
            "wellness": _wellness_adjustment(reel, user_signals),
        }
        total = (
            component_scores["engagement"] * W_ENGAGEMENT
            + component_scores["recency"] * W_RECENCY
            + component_scores["relationship"] * W_RELATIONSHIP
            + component_scores["affinity"] * W_AFFINITY
            + component_scores["wellness"] * W_WELLNESS
        )
        reel.rank_score = round(total, 4)
        reel.rank_reason = _build_reason(component_scores, reel, following)
        scored.append(reel)

    scored.sort(key=lambda r: r.rank_score, reverse=True)
    return scored


def build_user_signals(db, user_id: int) -> Dict[str, Any]:
    """
    Aggregate a user's history into the signals dict ranking needs.
    Wire this to real queries against your Like/Comment/ReelView/risk-score tables.
    Sketch below — replace with your actual ORM queries.
    """
    # Example sketch (replace with real SQLAlchemy queries):
    #
    # creator_engagement = (
    #     db.query(Post.user_id, func.count(Like.id))
    #     .join(Like, Like.post_id == Post.id)
    #     .filter(Like.user_id == user_id)
    #     .group_by(Post.user_id).all()
    # )
    # favorite_hashtags = derive from recently-liked/commented reels' hashtags
    # favorite_audio_ids = derive from ReelView table, audios with watch_seconds > 80% duration
    # risk_trend = latest value from your LSTM temporal risk scoring service (already built)

    return {
        "creator_engagement": {},
        "favorite_hashtags": set(),
        "favorite_audio_ids": set(),
        "risk_trend": 0.0,
    }