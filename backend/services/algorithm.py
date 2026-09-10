"""
MindGram Feed Algorithm
Instagram-inspired ranking + Silent AI mental health layer

Ranking factors:
1. Interest Score    → based on what you engage with
2. Relationship Score → how often you interact with someone
3. Recency           → newer posts ranked higher
4. Engagement        → likes + comments + shares
5. Novelty           → boost unseen authors/emotions
6. Trending Topics   → boost posts aligned with hot platform topics
7. Silent AI Layer   → suppress negative content for at-risk users
                     → promote positive content gently

Negative feedback:
- Not Interested  → strong explicit negative interest signal + hide post/author
- Report          → strong negative interest + (self_harm) author risk flag
- Skip rate       → weak implicit signal, only applied after enough
                     rolling history exists for a user/emotion pair

Diversity:
- MMR re-ranking after scoring to avoid back-to-back same author/emotion
"""

import logging
import math
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from collections import defaultdict

from services.topic_utils import aggregate_trending_scores, topic_overlap_score
from ai.agents.orchestrator import run_agents, EmotionSnapshot
from models.models import (
    Post, User, Like, Comment, Follow,
    AgentDecision, EmotionLog, UserInterest, ImpressionLog,
    NotInterested, Report,
)

logger = logging.getLogger("mindgram.algorithm")


# ── Constants ─────────────────────────────────────────────────

RECENCY_HALF_LIFE_HOURS = 48.0
RISK_SUPPRESSION_THRESHOLD = 0.40   # midpoint of the ramp below, not a hard cutoff
RISK_RAMP_WIDTH = 0.15              # how gradual the ramp is; bigger = softer/slower transition

async def attach_like_status(posts, user_id: int, db: AsyncSession):
    """Mutates posts in-place, setting `is_liked` based on whether
    `user_id` has a Like row for each post. One query for the whole batch."""
    if not posts:
        return posts
    post_ids = [p.id for p in posts]
    result = await db.execute(
        select(Like.post_id).where(
            Like.post_id.in_(post_ids),
            Like.user_id == user_id,
        )
    )
    liked_ids = {row[0] for row in result.all()}
    for p in posts:
        p.is_liked = p.id in liked_ids
    return posts
    
WEIGHTS = {
    "interest":     0.22,
    "relationship": 0.18,
    "recency":      0.13,
    "engagement":   0.10,
    "novelty":      0.09,
    "trending":     0.10,
    "ai_quality":   0.08,
}

# Trending topics
TRENDING_WINDOW_HOURS = 48

# Skip-rate tracking
SKIP_DWELL_THRESHOLD_MS = 1500   # under 1.5s on screen counts as a skip
SKIP_WINDOW = 20                 # need this many impressions before evaluating
SKIP_RATE_THRESHOLD = 0.70       # skip rate above this triggers a penalty
SKIP_PENALTY = -0.2

# Novelty / diversity
NOVELTY_WINDOW_DAYS = 14
MMR_LAMBDA = 0.72                # relevance vs diversity trade-off
AUTHOR_SUPPRESS_PENALTY = 0.35   # applied when author was marked not interested
REPORTED_POST_PENALTY = 0.50

# How many EmotionLog entries to pull for the agent orchestrator's history
EMOTION_HISTORY_LIMIT = 20


# ── Individual Scoring Functions ──────────────────────────────

def recency_score(created_at: datetime) -> float:
    """Exponential decay — newer posts score higher."""
    age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600.0
    return round(math.exp(-age_hours / RECENCY_HALF_LIFE_HOURS), 4)


def engagement_score(post: Post) -> float:
    """Normalized engagement score."""
    score = (
        post.likes_count * 1.0 +
        post.comments_count * 2.0 +
        post.shares_count * 3.0 +
        post.views_count * 0.1
    )
    return min(1.0, score / 200.0)


def interest_score(post: Post, user_interests: dict) -> float:
    """
    How likely is this user to be interested in this post?
    Based on what emotions/content they've engaged with before.
    """
    if not user_interests:
        return 0.5  # neutral for new users

    emotion = post.emotion or "neutral"
    return user_interests.get(emotion, 0.3)


def relationship_score(post: Post, interaction_counts: dict) -> float:
    """
    How close is this user to the post author?
    Based on number of past interactions.
    """
    count = interaction_counts.get(post.user_id, 0)
    return min(1.0, count / 20.0)


def novelty_score(
    post: Post,
    seen_post_ids: set[int],
    author_impression_counts: dict,
    emotion_impression_counts: dict,
) -> float:
    """
    Higher when the user hasn't recently seen this post, author, or emotion.
    """
    if post.id in seen_post_ids:
        return 0.0

    author_visits = author_impression_counts.get(post.user_id, 0)
    emotion = post.emotion or "neutral"
    emotion_visits = emotion_impression_counts.get(emotion, 0)

    author_novelty = 1.0 / (1.0 + author_visits * 0.35)
    emotion_novelty = 1.0 / (1.0 + emotion_visits * 0.25)
    return round(author_novelty * 0.6 + emotion_novelty * 0.4, 4)


def trending_topic_score(
    post: Post,
    trending_topics: dict[str, float],
    user_topic_affinity: dict[str, float],
) -> float:
    """
    How relevant is this post to currently trending topics,
    with a small boost when topics match the user's past engagement.
    """
    post_topics = post.topics or []
    if not post_topics:
        return 0.0

    platform_score = topic_overlap_score(post_topics, trending_topics)
    personal_score = topic_overlap_score(post_topics, user_topic_affinity)
    return round(platform_score * 0.7 + personal_score * 0.3, 4)


def negative_feedback_penalty(
    post: Post,
    suppressed_authors: set[int],
    reported_post_ids: set[int],
    skip_rates: dict[str, float],
) -> float:
    """
    Penalize posts tied to explicit or implicit negative signals.
    skip_rates maps emotion -> rolling skip rate in [0, 1].
    """
    penalty = 0.0

    if post.id in reported_post_ids:
        penalty += REPORTED_POST_PENALTY
    if post.user_id in suppressed_authors:
        penalty += AUTHOR_SUPPRESS_PENALTY

    emotion = post.emotion or "neutral"
    skip_rate = skip_rates.get(emotion, 0.0)
    if skip_rate >= SKIP_RATE_THRESHOLD:
        penalty += min(0.25, (skip_rate - SKIP_RATE_THRESHOLD) * 0.8)

    return round(penalty, 4)


def diversity_rerank(scored_posts: list[dict], limit: int) -> list[Post]:
    """
    Greedy MMR re-ranking — reduces consecutive posts from the same
    author or emotion while preserving relevance.
    """
    if not scored_posts:
        return []

    selected: list[dict] = []
    remaining = list(scored_posts)

    while remaining and len(selected) < limit:
        best_idx = 0
        best_mmr = float("-inf")

        for idx, item in enumerate(remaining):
            relevance = item["rank_score"]
            if not selected:
                similarity = 0.0
            else:
                same_author = sum(
                    1 for s in selected if s["post"].user_id == item["post"].user_id
                )
                same_emotion = sum(
                    1 for s in selected
                    if (s["post"].emotion or "neutral") == (item["post"].emotion or "neutral")
                )
                similarity = min(
                    1.0,
                    (same_author * 0.65 + same_emotion * 0.35) / max(len(selected), 1),
                )
            mmr = MMR_LAMBDA * relevance - (1.0 - MMR_LAMBDA) * similarity
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        selected.append(remaining.pop(best_idx))

    return [item["post"] for item in selected]


def suppression_intensity(user_risk: float) -> float:
    """
    Smooth 0..1 ramp centered on RISK_SUPPRESSION_THRESHOLD, replacing what used
    to be a hard >= cutoff. A sigmoid means a user drifting from 0.35 -> 0.45
    sees the mix shift a little at a time across refreshes instead of the feed
    visibly flipping character the moment they cross 0.40 — the whole point of
    this being a *silent* layer is that there's no perceptible switch flip.

    intensity ≈ 0   well below threshold  (no adjustment)
    intensity = 0.5 exactly at threshold
    intensity ≈ 1   well above threshold  (full suppression/boost effect)
    """
    z = (user_risk - RISK_SUPPRESSION_THRESHOLD) / RISK_RAMP_WIDTH
    return 1.0 / (1.0 + math.exp(-z))


def silent_ai_adjustment(
    post: Post,
    user_risk: float,
) -> float:
    """
    Silent mental health adjustment.
    For at-risk users:
      - Suppress high-risk negative content
      - Boost positive content
      - Boost wellness content specifically
    Effect strength ramps in gradually via suppression_intensity() rather than
    switching on at a fixed risk value. User never sees this happening.
    """
    intensity = suppression_intensity(user_risk)
    if intensity < 0.07:
        return 0.0  # negligible this far below threshold — skip the work

    adjustment = 0.0

    # Suppress high-risk negative content (any emotion, still gated on risk_score
    # so a "sad" post that isn't actually risky can still get through — that's
    # the "a bit of sad" you don't want fully wiped out).
    if post.risk_score > 0.5:
        penalty = (post.risk_score - 0.5) * intensity * 0.5
        adjustment -= penalty

    # Boost calm joy/neutral content specifically (not just "positive sentiment",
    # which doesn't reliably track the emotion label).
    if post.emotion in ("joy", "neutral") and post.risk_score < 0.2:
        boost = intensity * 0.2
        adjustment += boost

    # Wellness content gets an additional boost for at-risk users — this is
    # separate from the joy/neutral boost above so explicitly-curated wellness
    # posts (meditation, calming imagery, support resources) rise higher in
    # the feed than generic positive content when the user needs them most.
    if _matches_wellness_content(post):
        wellness_boost = intensity * WELLNESS_BOOST
        adjustment += wellness_boost

    return adjustment
# ── Main Ranking Function ─────────────────────────────────────

def compute_rank(
    post: Post,
    user_risk: float,
    user_interests: dict,
    interaction_counts: dict,
    seen_post_ids: set[int],
    author_impression_counts: dict,
    emotion_impression_counts: dict,
    suppressed_authors: set[int],
    reported_post_ids: set[int],
    skip_rates: dict[str, float],
    trending_topics: dict[str, float],
    user_topic_affinity: dict[str, float],
) -> float:
    """
    Compute final rank score for a post.
    Combines all signals with weights.
    """
    scores = {
        "interest":     interest_score(post, user_interests),
        "relationship": relationship_score(post, interaction_counts),
        "recency":      recency_score(post.created_at),
        "engagement":   engagement_score(post),
        "novelty":      novelty_score(
            post, seen_post_ids, author_impression_counts, emotion_impression_counts
        ),
        "trending":     trending_topic_score(post, trending_topics, user_topic_affinity),
        "ai_quality":   post.feed_score,
    }

    rank = sum(scores[k] * WEIGHTS[k] for k in scores)
    rank += silent_ai_adjustment(post, user_risk)
    rank -= negative_feedback_penalty(
        post, suppressed_authors, reported_post_ids, skip_rates
    )

    return round(max(0.0, min(1.0, rank)), 4)


# ── Wellness Content Injection ────────────────────────────────

WELLNESS_CONTENT_KEYWORDS = {
    "wellness", "self-care", "self care", "mindfulness", "mental health",
    "calm", "peace", "gratitude", "support", "resilience", "balance",
    "breathe", "breathing", "relax", "reflection",
}
WELLNESS_TOPIC_KEYWORDS = {
    "wellness", "self-care", "self care", "mindfulness", "mental health",
    "calm", "peace", "gratitude", "support", "resilience", "balance",
}
WELLNESS_RECENCY_DAYS = 21
WELLNESS_POOL_LIMIT = 50
WELLNESS_INJECTION_LIMIT = 2
WELLNESS_RISK_SCORE_MAX = 0.65  # raised from 0.25 — calm imagery often scores 0.3-0.6 on CLIP
WELLNESS_MIN_FEED_SCORE = 0.0   # wellness eligibility is governed by safety (risk <= 0.65), not engagement score
WELLNESS_MIN_GAP = 4    # posts between injections once intensity is ~maxed
WELLNESS_MAX_GAP = 12   # posts between injections right as intensity starts
WELLNESS_BOOST = 0.25   # ranking boost for wellness posts shown to high-risk users


def should_inject_wellness(user_risk: float, position: int) -> bool:
    """
    Decide if we should inject a wellness post at this feed position.
    Moderate-risk users (0.25+) get occasional wellness posts with frequency
    scaling up as risk increases. Prevents feed domination by wellness content.
    """
    intensity = suppression_intensity(user_risk)
    if intensity < 0.25:  # roughly user_risk < 0.32 — very low risk, no injection
        return False

    # Scale gap from 20 (low/moderate risk) down to 4 (very high risk)
    # At user_risk=0.29 (intensity≈0.32): gap ≈ 18-19 (1 wellness per ~18 posts)
    # At user_risk=0.40 (intensity=0.50): gap ≈ 12 (1 per ~12 posts)
    # At user_risk=0.70 (intensity≈0.95): gap ≈ 4 (1 per ~4 posts)
    min_gap = 4
    max_gap = 20
    normalized_intensity = max(0.0, min(1.0, (intensity - 0.25) / 0.75))
    gap = round(max_gap - (max_gap - min_gap) * normalized_intensity)
    gap = max(min_gap, gap)
    return position % gap == 0


def _matches_wellness_content(post: Post) -> bool:
    """
    Check if a post is wellness-oriented.
    Primary signal: explicit is_wellness flag (set at seed time or by admins).
    Fallback: heuristic keyword match for organically-posted wellness content.
    """
    # Explicit wellness flag takes precedence — seeded wellness posts, curated
    # content, or admin-approved wellness accounts all set this directly.
    if getattr(post, "is_wellness", False):
        return True

    # Fallback heuristic for organic content that wasn't explicitly tagged
    text = (post.content or "").lower()
    if any(keyword in text for keyword in WELLNESS_CONTENT_KEYWORDS):
        return True

    topics = post.topics or []
    for topic in topics:
        if not topic:
            continue
        topic_text = topic.lower()
        if any(keyword in topic_text for keyword in WELLNESS_TOPIC_KEYWORDS):
            return True

    return False


async def _load_wellness_candidates(
    user_id: int,
    db: AsyncSession,
    exclude_post_ids: set[int],
) -> list[Post]:
    """
    Fetch wellness content for injection into high-risk user feeds.

    Wellness identification is separated from general safety filtering:
      - is_wellness=True OR keyword match identifies wellness content
      - Moderate risk scores (0.3-0.65) are acceptable for calming imagery
      - Neutral sentiment is acceptable — wellness isn't about forced positivity
      - BUT: genuinely dangerous content (risk > 0.65, or hard safety signals)
        still gets filtered regardless of wellness flag

    This two-stage approach means a candle/meditation photo that CLIP scores
    at 0.4-0.5 can still be wellness content, but a literal self-harm image
    marked wellness=True would still be blocked by the safety ceiling.
    """
    cutoff = datetime.utcnow() - timedelta(days=WELLNESS_RECENCY_DAYS)

    # Stage 1: broad wellness candidate pool, relaxed sentiment/risk thresholds
    query = select(Post).options(selectinload(Post.media), selectinload(Post.author)).where(
        Post.created_at >= cutoff,
        Post.user_id != user_id,
        Post.risk_score <= WELLNESS_RISK_SCORE_MAX,  # 0.65 — blocks genuinely dangerous content
        Post.feed_score >= WELLNESS_MIN_FEED_SCORE,  # 0.40 — basic quality floor
    )
    if exclude_post_ids:
        query = query.where(Post.id.notin_(exclude_post_ids))

    result = await db.execute(
        query.order_by(func.random()).limit(WELLNESS_POOL_LIMIT)  # randomize to vary selection
    )

    # Stage 2: filter to posts that match wellness heuristics or have explicit flag
    candidates = [post for post in result.scalars().all() if _matches_wellness_content(post)]
    return candidates[:WELLNESS_INJECTION_LIMIT]


def _inject_wellness_posts(
    posts: list[Post],
    wellness_posts: list[Post],
    user_risk: float,
) -> list[Post]:
    """
    Inject wellness posts into the ranked feed at dynamic positions.
    Ensures no consecutive wellness posts to maintain natural feed flow.
    """
    if not wellness_posts:
        return posts

    # Only inject for users with intensity >= 0.25 (roughly user_risk >= 0.32)
    intensity = suppression_intensity(user_risk)
    if intensity < 0.25:
        return posts

    existing_ids = {post.id for post in posts}
    injected: list[Post] = []
    candidate_index = 0
    last_was_wellness = False

    for position, post in enumerate(posts, start=1):
        # Check if this position should have a wellness injection
        should_inject = (
            candidate_index < len(wellness_posts)
            and should_inject_wellness(user_risk, position)
            and not last_was_wellness  # prevent consecutive wellness posts
        )

        if should_inject:
            candidate = wellness_posts[candidate_index]
            candidate_index += 1
            if candidate.id not in existing_ids:
                candidate.is_wellness = True
                injected.append(candidate)
                last_was_wellness = True
                # Continue to next regular post
                injected.append(post)
                last_was_wellness = False
                continue

        injected.append(post)
        last_was_wellness = getattr(post, 'is_wellness', False) or _matches_wellness_content(post)

    return injected


# ── Feed Context Loaders ──────────────────────────────────────

async def _load_novelty_context(user_id: int, db: AsyncSession):
    """Recent impressions used for novelty scoring."""
    cutoff = datetime.utcnow() - timedelta(days=NOVELTY_WINDOW_DAYS)
    result = await db.execute(
        select(ImpressionLog.post_id, ImpressionLog.emotion, Post.user_id)
        .join(Post, ImpressionLog.post_id == Post.id)
        .where(
            ImpressionLog.user_id == user_id,
            ImpressionLog.created_at >= cutoff,
        )
    )

    seen_post_ids: set[int] = set()
    author_counts: dict[int, int] = defaultdict(int)
    emotion_counts: dict[str, int] = defaultdict(int)

    for post_id, emotion, author_id in result.fetchall():
        seen_post_ids.add(post_id)
        author_counts[author_id] += 1
        if emotion:
            emotion_counts[emotion] += 1

    return seen_post_ids, author_counts, emotion_counts


async def _load_negative_feedback(user_id: int, db: AsyncSession):
    """Posts/authors to suppress and rolling skip rates per emotion."""
    ni_result = await db.execute(
        select(NotInterested.post_id, NotInterested.author_id)
        .where(NotInterested.user_id == user_id)
    )
    hidden_post_ids: set[int] = set()
    suppressed_authors: set[int] = set()
    for post_id, author_id in ni_result.fetchall():
        hidden_post_ids.add(post_id)
        suppressed_authors.add(author_id)

    report_result = await db.execute(
        select(Report.post_id).where(Report.reporter_id == user_id)
    )
    reported_post_ids = {row[0] for row in report_result.fetchall()}

    skip_result = await db.execute(
        select(ImpressionLog.emotion, ImpressionLog.skipped)
        .where(ImpressionLog.user_id == user_id)
        .order_by(ImpressionLog.created_at.desc())
        .limit(SKIP_WINDOW * 5)
    )
    emotion_skips: dict[str, list[bool]] = defaultdict(list)
    for emotion, skipped in skip_result.fetchall():
        key = emotion or "neutral"
        if len(emotion_skips[key]) < SKIP_WINDOW:
            emotion_skips[key].append(skipped)

    skip_rates = {
        emotion: sum(flags) / len(flags)
        for emotion, flags in emotion_skips.items()
        if len(flags) >= SKIP_WINDOW
    }

    return hidden_post_ids, suppressed_authors, reported_post_ids, skip_rates


async def _load_trending_topics(db: AsyncSession) -> dict[str, float]:
    """Platform-wide trending topic scores from recent engagement velocity."""
    cutoff = datetime.utcnow() - timedelta(hours=TRENDING_WINDOW_HOURS)
    result = await db.execute(
        select(
            Post.topics,
            Post.likes_count,
            Post.comments_count,
            Post.shares_count,
            Post.created_at,
        ).where(Post.created_at >= cutoff)
    )
    rows = result.fetchall()
    return aggregate_trending_scores(rows)


async def _load_user_topic_affinity(user_id: int, db: AsyncSession) -> dict[str, float]:
    """Topics from posts the user has liked or commented on."""
    affinity: dict[str, float] = defaultdict(float)

    like_result = await db.execute(
        select(Post.topics)
        .join(Like, Like.post_id == Post.id)
        .where(Like.user_id == user_id)
    )
    for (topics,) in like_result.fetchall():
        if topics:
            for topic in topics:
                affinity[topic] += 1.0

    comment_result = await db.execute(
        select(Post.topics)
        .join(Comment, Comment.post_id == Post.id)
        .where(Comment.user_id == user_id)
    )
    for (topics,) in comment_result.fetchall():
        if topics:
            for topic in topics:
                affinity[topic] += 0.6

    if not affinity:
        return {}

    peak = max(affinity.values())
    return {topic: round(score / peak, 4) for topic, score in affinity.items()}


async def get_trending_topics(db: AsyncSession, limit: int = 10) -> list[dict]:
    """Public helper for API — top trending topics with scores."""
    scores = await _load_trending_topics(db)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [{"topic": topic, "score": score} for topic, score in ranked[:limit]]


# ── Feed Builder ──────────────────────────────────────────────

async def build_feed(
    user_id: int,
    db: AsyncSession,
    limit: int = 20,
    offset: int = 0,
) -> list[Post]:
    """
    Build the personalized feed for a user.

    Steps:
    1. Get user's risk level (silent)
    2. Get user's interests from engagement history
    3. Get user's relationship scores
    4. Fetch candidate posts (from followed users + explore)
    5. Rank all posts
    6. Return top N
    """

    # 1 — Get user risk level
    agent_result = await db.execute(
        select(AgentDecision)
        .where(AgentDecision.user_id == user_id)
        .order_by(AgentDecision.timestamp.desc())
        .limit(1)
    )
    agent_row = agent_result.scalar_one_or_none()
    risk_map = {"low": 0.1, "moderate": 0.45, "high": 0.70, "critical": 0.92}
    user_risk = risk_map.get(agent_row.risk_level, 0.1) if agent_row else 0.1

    # 2 — Get user interests
    interest_result = await db.execute(
        select(UserInterest).where(UserInterest.user_id == user_id)
    )
    interests = interest_result.scalars().all()
    user_interests = {i.emotion: i.score for i in interests}

    # 3 — Get interaction counts (who user interacts with most)
    following_result = await db.execute(
        select(Follow.following_id)
        .where(Follow.follower_id == user_id)
    )
    following_ids = [r[0] for r in following_result.fetchall()]

    # Count likes on their posts as relationship signal
    interaction_counts = defaultdict(int)
    if following_ids:
        like_result = await db.execute(
            select(Post.user_id, func.count(Like.id))
            .join(Like, Like.post_id == Post.id)
            .where(Like.user_id == user_id)
            .group_by(Post.user_id)
        )
        for author_id, count in like_result.fetchall():
            interaction_counts[author_id] = count

    # 3b — Novelty context + negative feedback
    seen_post_ids, author_impression_counts, emotion_impression_counts = (
        await _load_novelty_context(user_id, db)
    )
    hidden_post_ids, suppressed_authors, reported_post_ids, skip_rates = (
        await _load_negative_feedback(user_id, db)
    )
    trending_topics = await _load_trending_topics(db)
    user_topic_affinity = await _load_user_topic_affinity(user_id, db)

    # 4 — Fetch candidate posts
    cutoff = datetime.utcnow() - timedelta(days=7)

    # Posts from followed users
    if following_ids:
        followed_result = await db.execute(
            select(Post)
            .options(selectinload(Post.media), selectinload(Post.author))
            .where(
                Post.user_id.in_(following_ids),
                Post.created_at >= cutoff,
            )
            .order_by(func.random())  # randomize to prevent same order every refresh
            .limit(100)
        )
        followed_posts = followed_result.scalars().all()
    else:
        followed_posts = []

    # Explore posts (from non-followed users)
    explore_result = await db.execute(
        select(Post)
        .options(selectinload(Post.media), selectinload(Post.author))
        .where(
            Post.user_id != user_id,
            Post.created_at >= cutoff,
        )
        .order_by(func.random())  # randomize to prevent same order every refresh
        .limit(50)
    )
    explore_posts = explore_result.scalars().all()

    # Combine, deduplicate, and drop explicit negative-feedback posts
    seen_ids = set()
    all_posts = []
    for p in followed_posts + explore_posts:
        if p.id in seen_ids:
            continue
        if p.id in hidden_post_ids or p.id in reported_post_ids:
            continue
        seen_ids.add(p.id)
        all_posts.append(p)

    # Load authors
    author_cache = {}
    for p in all_posts:
        if p.user_id not in author_cache:
            author_cache[p.user_id] = await db.get(User, p.user_id)
        p.author = author_cache[p.user_id]

    # 5 — Rank posts, then apply diversity re-ranking
    scored_posts = [
        {
            "post": p,
            "rank_score": compute_rank(
                p,
                user_risk,
                user_interests,
                interaction_counts,
                seen_post_ids,
                author_impression_counts,
                emotion_impression_counts,
                suppressed_authors,
                reported_post_ids,
                skip_rates,
                trending_topics,
                user_topic_affinity,
            ),
        }
        for p in all_posts
    ]
    scored_posts.sort(key=lambda item: item["rank_score"], reverse=True)

    ranked_posts = diversity_rerank(scored_posts, limit=len(scored_posts))

    wellness_candidates = await _load_wellness_candidates(
        user_id,
        db,
        exclude_post_ids={p.id for p in ranked_posts},
    )
    ranked_posts = _inject_wellness_posts(ranked_posts, wellness_candidates, user_risk)

    for post in ranked_posts:
        if not hasattr(post, "is_wellness"):
            post.is_wellness = False

    # 6 — Return paginated results
    return ranked_posts[offset:offset + limit]


async def update_user_interests(
    user_id: int,
    emotion: str,
    db: AsyncSession,
    weight: float = 0.1,
):
    """
    Update user interest scores when they engage with content.
    Called for likes/comments (positive weight) and not_interested/
    reports/skip-rate penalties (negative weight).
    """
    result = await db.execute(
        select(UserInterest).where(
            UserInterest.user_id == user_id,
            UserInterest.emotion == emotion,
        )
    )
    interest = result.scalar_one_or_none()

    if interest:
        # Exponential moving average — clamp to [0, 1] so repeated negative
        # weights can't push the score negative (interest_score() assumes
        # scores live in [0, 1] as a lookup default of 0.3/0.5 implies).
        interest.score = max(0.0, min(1.0, interest.score * 0.9 + weight))
        interest.updated_at = datetime.utcnow()
    else:
        db.add(UserInterest(
            user_id=user_id,
            emotion=emotion,
            score=max(0.0, min(1.0, weight)),
        ))

    await db.commit()


# ── Skip / Impression Tracking ────────────────────────────────

async def record_impression(
    user_id: int,
    post_id: int,
    emotion: str,
    dwell_ms: int,
    db: AsyncSession,
):
    """
    Log a feed impression with dwell time, and — once enough data exists
    for this user/emotion pair — apply an interest penalty if the rolling
    skip rate crosses threshold.

    Called once per post per feed render, when the post leaves viewport
    (frontend sends dwell_ms via IntersectionObserver or similar).

    `skipped` is computed here server-side from dwell_ms rather than
    trusting a client-sent boolean, since that's gameable and less
    accurate than raw dwell time.

    Waiting for a full SKIP_WINDOW of history before evaluating avoids
    overreacting to a handful of fast scrolls (e.g. a hurried session)
    being misread as durable disinterest in that emotion category.
    """
    skipped = dwell_ms < SKIP_DWELL_THRESHOLD_MS

    db.add(ImpressionLog(
        user_id=user_id,
        post_id=post_id,
        emotion=emotion,
        dwell_ms=dwell_ms,
        skipped=skipped,
    ))
    await db.commit()

    result = await db.execute(
        select(ImpressionLog.skipped)
        .where(
            ImpressionLog.user_id == user_id,
            ImpressionLog.emotion == emotion,
        )
        .order_by(ImpressionLog.created_at.desc())
        .limit(SKIP_WINDOW)
    )
    recent = [r[0] for r in result.fetchall()]

    if len(recent) >= SKIP_WINDOW:
        skip_rate = sum(recent) / len(recent)
        if skip_rate >= SKIP_RATE_THRESHOLD:
            await update_user_interests(user_id, emotion, db, weight=SKIP_PENALTY)


# ── Behavioral Signal Helpers (non-text interactions) ─────────

async def _get_emotion_history_for_agent(user_id: int, db: AsyncSession) -> list[EmotionSnapshot]:
    """
    Self-contained EmotionLog history loader for the agent orchestrator.
    Deliberately NOT imported from routers/posts.py — that module imports
    from this one (services.algorithm), so importing back would create a
    circular import. This is a duplicate of routers.posts.get_emotion_history
    by necessity; if that query's shape ever changes, mirror the change here.
    """
    result = await db.execute(
        select(EmotionLog)
        .where(EmotionLog.user_id == user_id)
        .order_by(EmotionLog.timestamp.desc())
        .limit(EMOTION_HISTORY_LIMIT)
    )
    logs = result.scalars().all()
    return [
        EmotionSnapshot(
            sentiment_score=log.sentiment_score,
            emotion=log.emotion,
            emotion_score=log.emotion_score,
            risk_score=log.risk_score,
            source=log.source,
        )
        for log in reversed(logs)
    ]


async def log_like_behavioral_signal(liker_id: int, post: Post, db: AsyncSession):
    """
    Likes have no text of their own to run through analyze_text, so instead
    of re-analyzing, this treats "user liked this post" as a behavioral
    signal about the LIKER: it logs the liked post's existing emotion/risk
    scores into the liker's own EmotionLog trail (source="like") and lets
    the agent orchestrator evaluate it against their trajectory. Repeatedly
    engaging with high-risk content is itself a pattern worth catching,
    separate from the post author's own risk pipeline.

    Shared by both like endpoints (routers/interactions.py's
    record_interaction and routers/posts.py's toggle_like) so the behavior
    is consistent regardless of which one the frontend calls.

    Best-effort: logs and swallows errors rather than failing the like
    itself, since a like succeeding is more important than this signal.
    """
    try:
        history = await _get_emotion_history_for_agent(liker_id, db)
        current_snap = EmotionSnapshot(
            sentiment_score=post.sentiment_score,
            emotion=post.emotion,
            emotion_score=post.emotion_score,
            risk_score=post.risk_score,
            source="like",
        )
        log = EmotionLog(
            user_id=liker_id,
            sentiment_score=post.sentiment_score,
            emotion=post.emotion,
            emotion_score=post.emotion_score,
            risk_score=post.risk_score,
            source="like",
        )
        db.add(log)

        agent_report = run_agents(current_snap, history)
        db.add(AgentDecision(
            user_id=liker_id,
            risk_level=agent_report.risk_level,
            decision=agent_report.decision,
            intervention=agent_report.intervention,
            rag_suggestion=agent_report.rag_suggestion,
            metadata_json=agent_report.metadata,
        ))
        log.agent_action = agent_report.decision
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("Like behavioral-signal logging failed for user_id=%s", liker_id)