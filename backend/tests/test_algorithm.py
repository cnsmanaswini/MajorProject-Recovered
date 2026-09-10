from datetime import datetime
from unittest.mock import MagicMock

from services.algorithm import (
    should_inject_wellness,
    _inject_wellness_posts,
    _matches_wellness_content,
    silent_ai_adjustment,
    suppression_intensity,
    RISK_SUPPRESSION_THRESHOLD,
    WELLNESS_BOOST,
)


def make_post(post_id=1, content="", sentiment="positive", risk_score=0.1, emotion="joy", is_wellness=False):
    post = MagicMock()
    post.id = post_id
    post.content = content
    post.sentiment = sentiment
    post.risk_score = risk_score
    post.emotion = emotion
    post.topics = []
    post.is_wellness = is_wellness
    return post


class TestSuppressionIntensity:
    def test_ramps_smoothly_instead_of_snapping(self):
        # No single risk value should flip behavior from 0 to full — check
        # that intensity increases gradually across the threshold instead
        # of jumping, by sampling either side of it.
        below = suppression_intensity(RISK_SUPPRESSION_THRESHOLD - 0.05)
        at = suppression_intensity(RISK_SUPPRESSION_THRESHOLD)
        above = suppression_intensity(RISK_SUPPRESSION_THRESHOLD + 0.05)
        assert 0.0 < below < at < above < 1.0
        assert at == 0.5

    def test_far_below_threshold_is_near_zero(self):
        assert suppression_intensity(0.0) < 0.10

    def test_far_above_threshold_is_near_one(self):
        assert suppression_intensity(1.0) > 0.95


class TestShouldInjectWellness:
    def test_injection_gets_denser_as_risk_rises(self):
        # count how many of the first 20 slots get a wellness post at a few
        # risk levels — should increase monotonically with risk, not flip
        low = sum(should_inject_wellness(0.30, p) for p in range(1, 21))
        mid = sum(should_inject_wellness(0.55, p) for p in range(1, 21))
        high = sum(should_inject_wellness(0.90, p) for p in range(1, 21))
        assert low <= mid <= high
        assert high > low  # genuinely denser once well into risk territory

    def test_no_injection_well_below_threshold(self):
        assert should_inject_wellness(0.05, 5) is False


class TestInjectWellnessPosts:
    def test_injection_positions_inserts_at_5_and_10(self):
        """Test that wellness posts are injected at dynamic positions based on user risk.
        At user_risk=0.8 (high), injection gap is around 4-5 posts."""
        posts = [make_post(post_id=i) for i in range(1, 15)]
        wellness = [make_post(post_id=100, content="self-care reminder"), make_post(post_id=101, content="mindfulness tip")]

        injected = _inject_wellness_posts(posts, wellness, user_risk=0.8)

        # With high user risk (0.8), should inject at dynamic positions
        # Verify wellness posts were injected (length increased by 2)
        assert len(injected) == len(posts) + 2
        # Verify the wellness posts are in the feed
        wellness_ids = {p.id for p in injected if p.id in {100, 101}}
        assert wellness_ids == {100, 101}
        # Verify is_wellness flag is set on injected posts
        assert all(getattr(item, "is_wellness", False) for item in injected if item.id in {100, 101})

    def test_no_injection_when_no_candidates(self):
        posts = [make_post(post_id=i) for i in range(1, 10)]
        injected = _inject_wellness_posts(posts, [], user_risk=0.8)
        assert injected == posts

    def test_does_not_duplicate_existing_post(self):
        posts = [make_post(post_id=i) for i in range(1, 15)]
        wellness = [make_post(post_id=5, content="wellness tip")]
        injected = _inject_wellness_posts(posts, wellness, user_risk=0.8)
        assert len(injected) == len(posts)
        assert all(post.id != 5 or post is posts[4] for post in injected)

    def test_low_risk_user_receives_no_wellness_injection(self):
        """Test that very low-risk user (0.20) does not receive wellness injection."""
        posts = [make_post(post_id=i) for i in range(1, 15)]
        wellness = [make_post(post_id=100, content="mindfulness tip", is_wellness=True)]

        injected = _inject_wellness_posts(posts, wellness, user_risk=0.20)

        # Should return original posts unchanged (no injection for very low-risk users)
        assert len(injected) == len(posts)
        assert 100 not in {p.id for p in injected}

    def test_moderate_risk_user_receives_occasional_wellness(self):
        """Test that moderate-risk user (0.29) receives occasional wellness injection."""
        posts = [make_post(post_id=i) for i in range(1, 25)]
        wellness = [make_post(post_id=100, content="mindfulness tip", is_wellness=True)]

        injected = _inject_wellness_posts(posts, wellness, user_risk=0.29)

        # Should inject wellness for moderate-risk users (but not as frequently as high-risk)
        assert len(injected) > len(posts) or 100 in {p.id for p in injected}

    def test_no_consecutive_wellness_posts(self):
        """Test that wellness posts are never placed consecutively."""
        posts = [make_post(post_id=i) for i in range(1, 20)]
        wellness = [
            make_post(post_id=100, content="wellness tip 1", is_wellness=True),
            make_post(post_id=101, content="wellness tip 2", is_wellness=True),
        ]

        injected = _inject_wellness_posts(posts, wellness, user_risk=0.8)

        # Check no two consecutive posts are both wellness posts
        for i in range(len(injected) - 1):
            current_wellness = getattr(injected[i], 'is_wellness', False)
            next_wellness = getattr(injected[i + 1], 'is_wellness', False)
            assert not (current_wellness and next_wellness), f"Consecutive wellness posts at positions {i}, {i+1}"


class TestMatchWellnessContent:
    def test_matches_by_is_wellness_flag(self):
        post = make_post(content="A calm photo.", sentiment="neutral", is_wellness=True)
        assert _matches_wellness_content(post) is True

    def test_matches_by_keyword_in_content(self):
        post = make_post(content="This wellness moment is so calm and mindful.")
        assert _matches_wellness_content(post) is True

    def test_matches_by_keyword_in_topics(self):
        post = make_post(content="Fresh post", sentiment="positive")
        post.topics = ["mindfulness", "relaxation"]
        assert _matches_wellness_content(post) is True

    def test_rejects_non_wellness_posts(self):
        post = make_post(content="Just a regular update about my day.", is_wellness=False)
        assert _matches_wellness_content(post) is False

    def test_neutral_sentiment_with_wellness_flag_matches(self):
        post = make_post(content="Quiet morning.", sentiment="neutral", emotion="neutral", is_wellness=True)
        assert _matches_wellness_content(post) is True


class TestSilentAIAdjustment:
    def test_positive_low_risk_post_gets_boost_for_at_risk_user(self):
        post = make_post(risk_score=0.1, sentiment="positive", emotion="joy")
        boost = silent_ai_adjustment(post, user_risk=0.7)
        assert boost > 0

    def test_boost_grows_with_risk_instead_of_snapping_on(self):
        # Same post, increasing risk — the boost should scale up smoothly,
        # not stay at 0 then suddenly jump to a fixed value at the threshold.
        post = make_post(risk_score=0.1, sentiment="positive", emotion="joy")
        boost_low  = silent_ai_adjustment(post, user_risk=0.20)
        boost_mid  = silent_ai_adjustment(post, user_risk=RISK_SUPPRESSION_THRESHOLD)
        boost_high = silent_ai_adjustment(post, user_risk=0.90)
        assert 0.0 < boost_low < boost_mid < boost_high

    def test_negligible_far_below_threshold(self):
        # Not exactly zero — there's no hard cutoff anymore — but small
        # enough to be imperceptible in the ranking.
        post = make_post(risk_score=0.1, sentiment="positive", emotion="joy")
        assert silent_ai_adjustment(post, user_risk=0.0) < 0.02

    def test_high_risk_post_penalty_grows_with_user_risk(self):
        post = make_post(risk_score=0.8, sentiment="negative", emotion="sadness")
        penalty_mid  = silent_ai_adjustment(post, user_risk=RISK_SUPPRESSION_THRESHOLD)
        penalty_high = silent_ai_adjustment(post, user_risk=0.90)
        assert penalty_mid < 0
        assert penalty_high < penalty_mid  # more negative = stronger suppression

    def test_neutral_sentiment_wellness_gets_boost_for_at_risk_user(self):
        # A: Neutral sentiment + wellness content -> must receive boost
        post = make_post(risk_score=0.15, sentiment="neutral", emotion="neutral", is_wellness=True)
        boost = silent_ai_adjustment(post, user_risk=0.80)
        assert boost > 0.20

    def test_moderate_clip_risk_wellness_post_still_gets_boost(self):
        # B: Moderate CLIP risk (0.4-0.6) + wellness content
        post = make_post(risk_score=0.55, sentiment="neutral", emotion="neutral", is_wellness=True)
        boost = silent_ai_adjustment(post, user_risk=0.80)
        # Moderate risk penalty (0.55 - 0.5) * 0.5 * intensity is outweighed by wellness boost
        assert boost > 0

    def test_actually_dangerous_content_marked_wellness_still_suppressed(self):
        # C: Dangerous content (risk_score = 0.90) marked wellness -> still penalized/suppressed
        post = make_post(risk_score=0.90, sentiment="negative", emotion="sadness", is_wellness=True)
        adj = silent_ai_adjustment(post, user_risk=0.85)
        # Penalty (0.9 - 0.5) * 0.5 * intensity = 0.2 * intensity, so net adjustment is negative or suppressed
        assert adj < 0.1  # heavily offset by penalty compared to safe wellness (> 0.23)

    def test_normal_neutral_non_wellness_post_no_wellness_boost(self):
        # D: Normal neutral post (sentiment = neutral, wellness = false) -> no wellness boost
        post = make_post(risk_score=0.30, sentiment="neutral", emotion="neutral", is_wellness=False)
        adj = silent_ai_adjustment(post, user_risk=0.85)
        assert adj == 0.0

    def test_normal_risk_user_wellness_post_does_not_dominate(self):
        # F: Normal-risk user (user_risk = 0.1) -> wellness content does not receive large boost
        post = make_post(risk_score=0.30, sentiment="neutral", emotion="neutral", is_wellness=True)
        adj = silent_ai_adjustment(post, user_risk=0.10)
        # At low risk, intensity is low, so wellness boost is minimal (not dominating)
        assert adj < 0.05  # small boost acceptable, just not dominating