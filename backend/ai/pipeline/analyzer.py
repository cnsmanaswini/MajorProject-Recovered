"""
MindGram Core AI Pipeline (all-in-one)
-----------------------------------------
Everything lives in this one file:
  1. Image analysis (CLIP zero-shot similarity)
  2. Video analysis (frame sampling, reuses #1 per frame)
  3. Engagement analysis (liked-post risk signal, reduced weight)
  4. Core text pipeline: sentiment → emotion → sarcasm → numbness →
     LSTM risk → feed score, blending in #1/#2 when media is attached.
"""

import os
import logging
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in lightweight environments
    torch = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency in lightweight environments
    Image = None

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:  # pragma: no cover - optional dependency in lightweight environments
    CLIPModel = None
    CLIPProcessor = None

from schemas.schemas import PipelineResult
from ai.pipeline.loader import get_model
from ai.pipeline.numbness_detector import detect_numbness_signal
from ai.pipeline.risk_detector import detect_depression_suicide_risk, RiskTier

logger = logging.getLogger("mindgram.pipeline")


# =============================================================================
# 1. IMAGE ANALYSIS (CLIP zero-shot similarity)
# =============================================================================
# Same philosophy as numbness_detector.py: instead of hardcoded visual rules
# (brittle -- "dark image = risky" is nonsense), we embed the image with CLIP
# and compare it against two sets of natural-language prompts -- one spanning
# a sentiment axis (distressing -> uplifting), one spanning a risk axis
# (neutral/everyday -> self-harm/crisis imagery) -- and use similarity to
# place the image on each axis. Generalizes far better than keyword/color
# heuristics, and is explainable: you can log which prompt scored highest.

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

SENTIMENT_POSITIVE_PROMPTS = [
    "a joyful, uplifting photo or illustration",
    "a bright, happy scene with friends or family",
    "a celebratory or cheerful moment",
    "a peaceful, calming photo or illustration",
]
SENTIMENT_NEGATIVE_PROMPTS = [
    "a dark, sad, or distressing photo or illustration",
    "a lonely or isolated scene",
    "an image conveying grief or despair",
    "a bleak or hopeless looking photo or illustration",
]

# Risk axis -- deliberately clinical/descriptive rather than graphic, since
# these prompts only need to be semantically close enough for CLIP's
# embedding space to pick up the pattern, not explicit.
RISK_HIGH_PROMPTS = [
    "a photo or illustration depicting self-harm or a suicide-related scene",
    "a photo or illustration showing means of self-harm, such as pills, blades, or rope",
    "a farewell-themed or memorial-style photo or illustration",
    "a photo or illustration of a person in visible physical distress or crisis",
]
RISK_NEUTRAL_PROMPTS = [
    "an everyday photo or illustration of daily life",
    "a neutral photo or illustration with no cause for concern",
    "a normal social media picture or illustration",
]

# Emotion prompts -- mirrors EMOTION_LABELS (section 4) so image-derived
# emotion can blend with text-derived emotion using the same label space.
# This exists because blend_media_signal previously only touched
# sentiment_score/risk_score, leaving `emotion` stuck on whatever the text
# classifier returned (often "neutral" for empty/thin captions) even when
# the attached image clearly shows a different emotion (e.g. crying eyes
# tagged "neutral" because there was no caption to read).
EMOTION_PROMPTS = {
    "anger": [
        "a photo or illustration showing anger or rage",
        "an angry facial expression or scene",
    ],
    "disgust": ["a photo or illustration showing disgust or revulsion"],
    "fear": [
        "a photo or illustration showing fear or terror",
        "a frightened facial expression",
    ],
    "joy": [
        "a photo or illustration showing joy or happiness",
        "a smiling, joyful facial expression",
    ],
    "neutral": [
        "a photo or illustration with a neutral, unremarkable expression or scene",
    ],
    "sadness": [
        "a photo or illustration showing sadness or crying",
        "a person crying with visible tears running down their face",
        "tearful, sorrowful, or downturned eyes",
    ],
    "surprise": ["a photo or illustration showing surprise or shock"],
}


def _clip_emotion(image: Image.Image) -> tuple[str, float]:
    """
    Returns (emotion_label, confidence) for an image using CLIP zero-shot
    similarity against EMOTION_PROMPTS, in the same label space as the text
    emotion classifier (EMOTION_LABELS, section 4) so the two can be blended.
    """
    all_prompts: list[str] = []
    prompt_to_label: list[str] = []
    for label, prompts in EMOTION_PROMPTS.items():
        for p in prompts:
            all_prompts.append(p)
            prompt_to_label.append(label)

    sims = _clip_similarities(image, all_prompts)

    # Take the max similarity per label (a label can have multiple prompts).
    best_per_label: dict[str, float] = {}
    for label, score in zip(prompt_to_label, sims):
        if label not in best_per_label or score > best_per_label[label]:
            best_per_label[label] = score

    top_label = max(best_per_label, key=best_per_label.get)
    top_score = round(best_per_label[top_label], 4)
    return top_label, top_score


@lru_cache(maxsize=1)
def _get_clip():
    """Lazily loads and caches the CLIP model + processor (loaded once per process)."""
    if CLIPModel is None or CLIPProcessor is None or torch is None:
        raise RuntimeError(
            "CLIP support requires transformers and torch to be installed. "
            "Install them to use image analysis."
        )
    logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    return model, processor


def _load_image(image_source: str) -> Image.Image:
    """
    Loads an image from a local path or URL into a PIL Image.
    image_source: local file path, or a direct image URL (e.g. Cloudinary URL).
    """
    if image_source.startswith("http://") or image_source.startswith("https://"):
        import requests
        from io import BytesIO
        resp = requests.get(image_source, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_source).convert("RGB")


def _clip_similarities(image: Image.Image, prompts: list[str]) -> list[float]:
    """Returns a softmax similarity distribution between the image and each prompt."""
    if CLIPModel is None or CLIPProcessor is None or torch is None:
        raise RuntimeError(
            "CLIP support requires transformers and torch to be installed. "
            "Install them to use image analysis."
        )
    model, processor = _get_clip()
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)[0]
    return probs.tolist()


def analyze_image(image_source: str) -> dict:
    """
    Returns {"sentiment_score": float in [-1, 1], "risk_score": float in [0, 1]}
    for a single image, using CLIP zero-shot similarity against the prompt
    sets above.
    """
    image = _load_image(image_source)

    pos_sims = _clip_similarities(image, SENTIMENT_POSITIVE_PROMPTS)
    neg_sims = _clip_similarities(image, SENTIMENT_NEGATIVE_PROMPTS)
    sentiment_score = round(max(pos_sims) - max(neg_sims), 4)
    sentiment_score = max(-1.0, min(1.0, sentiment_score))

    risk_sims = _clip_similarities(image, RISK_HIGH_PROMPTS)
    neutral_sims = _clip_similarities(image, RISK_NEUTRAL_PROMPTS)
    risk_score = round(max(0.0, max(risk_sims) - 0.3 * max(neutral_sims)), 4)
    risk_score = max(0.0, min(1.0, risk_score))

    emotion, emotion_score = _clip_emotion(image)

    logger.debug(
        f"Image analysis -> source={image_source}, sentiment_score={sentiment_score}, "
        f"risk_score={risk_score}, emotion={emotion}({emotion_score})"
    )

    return {
        "sentiment_score": sentiment_score,
        "risk_score": risk_score,
        "emotion": emotion,
        "emotion_score": emotion_score,
    }


# =============================================================================
# 2. VIDEO ANALYSIS (frame sampling, reuses analyze_image() above)
# =============================================================================
FRAME_SAMPLE_INTERVAL_SEC = 2.0   # how often to grab a frame
MAX_FRAMES_SAMPLED = 15           # hard cap so long videos don't blow up inference time


@dataclass
class VideoAnalysisResult:
    sentiment_score: float
    risk_score: float
    emotion: str
    emotion_score: float
    frames_sampled: int
    per_frame_risk: list[float] = field(default_factory=list)


def _sample_frame_timestamps(duration_sec: float) -> list[float]:
    if duration_sec <= 0:
        return [0.0]
    timestamps = []
    t = 0.0
    while t < duration_sec and len(timestamps) < MAX_FRAMES_SAMPLED:
        timestamps.append(t)
        t += FRAME_SAMPLE_INTERVAL_SEC
    return timestamps or [0.0]


def analyze_video(video_source: str) -> VideoAnalysisResult:
    """
    video_source: local file path, or a direct video URL cv2 can open.
    NOTE: adaptive-streaming URLs (HLS/DASH) won't open directly -- use
    Cloudinary's direct file URL, or download locally first.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for video analysis. Install opencv-python."
        ) from exc

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
        raise ValueError(f"Unable to open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration_sec = (frame_count / fps) if fps > 0 else 0.0
    timestamps = _sample_frame_timestamps(duration_sec)

    sentiment_scores: list[float] = []
    risk_scores: list[float] = []
    emotions: list[str] = []
    emotion_scores: list[float] = []
    tmp_paths: list[str] = []

    try:
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            success, frame = cap.read()
            if not success:
                logger.warning(f"Frame read failed at t={t}s for {video_source}, skipping.")
                continue

            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            cv2.imwrite(tmp_path, frame)
            tmp_paths.append(tmp_path)

            try:
                frame_result = analyze_image(tmp_path)
                sentiment_scores.append(frame_result["sentiment_score"])
                risk_scores.append(frame_result["risk_score"])
                emotions.append(frame_result["emotion"])
                emotion_scores.append(frame_result["emotion_score"])
            except Exception as e:
                logger.warning(f"Frame analysis failed at t={t}s for {video_source}: {e}")
                continue
    finally:
        cap.release()
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    if not risk_scores:
        logger.warning(f"No frames successfully analyzed for {video_source}; returning neutral result.")
        return VideoAnalysisResult(
            sentiment_score=0.0, risk_score=0.0, emotion="neutral", emotion_score=0.0, frames_sampled=0
        )

    aggregated_sentiment = round(sum(sentiment_scores) / len(sentiment_scores), 4)
    aggregated_risk = round(max(risk_scores), 4)

    # Emotion aggregation: mode across frames (most frequent label), with
    # ties broken by highest average confidence for that label. A simple
    # majority vote generalizes better than "emotion of the riskiest frame"
    # since a single odd frame shouldn't dominate the overall read.
    label_confidences: dict[str, list[float]] = {}
    for label, score in zip(emotions, emotion_scores):
        label_confidences.setdefault(label, []).append(score)
    aggregated_emotion = max(
        label_confidences,
        key=lambda lbl: (len(label_confidences[lbl]), sum(label_confidences[lbl]) / len(label_confidences[lbl])),
    )
    aggregated_emotion_score = round(
        sum(label_confidences[aggregated_emotion]) / len(label_confidences[aggregated_emotion]), 4
    )

    logger.debug(
        f"Video analysis -> {len(risk_scores)} frames sampled, "
        f"mean_sentiment={aggregated_sentiment}, max_risk={aggregated_risk}, "
        f"emotion={aggregated_emotion}({aggregated_emotion_score})"
    )

    return VideoAnalysisResult(
        sentiment_score=aggregated_sentiment,
        risk_score=aggregated_risk,
        emotion=aggregated_emotion,
        emotion_score=aggregated_emotion_score,
        frames_sampled=len(risk_scores),
        per_frame_risk=risk_scores,
    )


VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".m4v", ".avi"}


def is_video_source(media_source: str) -> bool:
    """Routes image vs. video in blend_media_signal below."""
    lower = media_source.lower().split("?")[0]
    return any(lower.endswith(ext) for ext in VIDEO_EXTENSIONS)


def analyze_media(media_source: str) -> dict:
    """
    Single entry point: routes to image or video analysis based on file
    extension, returns a uniform {"sentiment_score", "risk_score"} dict.
    """
    if is_video_source(media_source):
        result = analyze_video(media_source)
        return {
            "sentiment_score": result.sentiment_score,
            "risk_score": result.risk_score,
            "emotion": result.emotion,
            "emotion_score": result.emotion_score,
        }
    return analyze_image(media_source)


# =============================================================================
# 3. ENGAGEMENT ANALYSIS (liked-post risk signal, reduced weight)
# =============================================================================
LIKE_ENGAGEMENT_WEIGHT = 0.4          # per project direction: reduced, not full weight
MAX_SINGLE_LIKE_CONTRIBUTION = 0.5    # cap so a single like can never approach CRITICAL


@dataclass
class EngagementSignal:
    source_post_risk_score: float
    weighted_risk_contribution: float
    engagement_weight: float
    signal_type: str = "liked_post"


def compute_engagement_signal(
    liked_post_result: PipelineResult,
    engagement_weight: float = LIKE_ENGAGEMENT_WEIGHT,
    repeated_engagement_count: int = 1,
) -> EngagementSignal:
    """
    Computes the risk contribution a like adds to the LIKER's own risk
    history, using the liked post's already-computed PipelineResult (no
    need to re-run the full pipeline -- the author's post was already scored).
    """
    base_contribution = liked_post_result.risk_score * engagement_weight

    if repeated_engagement_count > 1:
        scale = min(1.5, 1.0 + 0.1 * (repeated_engagement_count - 1))
        base_contribution *= scale

    weighted_contribution = round(min(MAX_SINGLE_LIKE_CONTRIBUTION, base_contribution), 4)

    logger.debug(
        f"Engagement signal -> source_risk={liked_post_result.risk_score}, "
        f"weight={engagement_weight}, repeated_count={repeated_engagement_count}, "
        f"contribution={weighted_contribution}"
    )

    return EngagementSignal(
        source_post_risk_score=liked_post_result.risk_score,
        weighted_risk_contribution=weighted_contribution,
        engagement_weight=engagement_weight,
    )


def record_engagement_signal(
    liked_post_result: PipelineResult,
    liker_risk_history: list[float],
    repeated_engagement_count: int = 1,
) -> tuple[list[float], EngagementSignal]:
    """
    Call from your like-endpoint handler. Appends the weighted contribution
    to the liker's risk_history -- does not touch the post's own author or
    its stored PipelineResult.
    """
    signal = compute_engagement_signal(liked_post_result, repeated_engagement_count=repeated_engagement_count)
    updated_history = (liker_risk_history or []) + [signal.weighted_risk_contribution]

    logger.info(
        f"Engagement signal recorded -- liked post risk={signal.source_post_risk_score}, "
        f"weighted contribution={signal.weighted_risk_contribution}"
    )

    return updated_history, signal


# =============================================================================
# 4. CORE TEXT PIPELINE
# =============================================================================
SENTIMENT_MAP = {
    "negative": ("negative", -1.0),
    "neutral":  ("neutral",   0.0),
    "positive": ("positive",  1.0),
    "label_0": ("negative", -1.0),
    "label_1": ("neutral",   0.0),
    "label_2": ("positive",  1.0),
}

EMOTION_LABELS = {
    "anger": "anger",
    "disgust": "anger",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "fear",
}

NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sadness"}
SARCASM_OVERRIDE_THRESHOLD = 0.7

MEDIA_SENTIMENT_WEIGHT = 0.4
TEXT_SENTIMENT_WEIGHT = 1.0 - MEDIA_SENTIMENT_WEIGHT

# Emotion blending -- same 60/40 text-leads split as sentiment, for the same
# reason (captions are the user's own words). This fixes the gap where a
# post with a thin/empty caption but an emotionally clear image (e.g.
# crying eyes) previously kept whatever the text classifier defaulted to
# ("neutral") in the emotion badge, since blend_media_signal used to only
# touch sentiment_score/risk_score.
MEDIA_EMOTION_WEIGHT = 0.4
TEXT_EMOTION_WEIGHT = 1.0 - MEDIA_EMOTION_WEIGHT


def _top_result(raw) -> dict:
    result = raw[0]
    if isinstance(result, list):
        result = result[0]
    return result


def run_sentiment(text: str) -> tuple[str, float]:
    model = get_model("sentiment")
    result = _top_result(model(text))
    label_raw = result["label"].lower()
    conf = result["score"]
    label, direction = SENTIMENT_MAP.get(label_raw, ("neutral", 0.0))
    score = direction * conf
    return label, round(score, 4)


# Below this confidence, the top-1 label isn't a real signal -- it's the
# model picking the least-bad option among several it's unsure about.
# Falling back to neutral here is honest about that uncertainty instead
# of quietly presenting a coin-flip guess as a confident classification.
EMOTION_CONFIDENCE_FLOOR = 0.5


def run_emotion(text: str) -> tuple[str, float]:
    model = get_model("emotion")
    result = _top_result(model(text))
    label = result["label"].lower()
    label = EMOTION_LABELS.get(label, "neutral")
    score = round(result["score"], 4)
    if score < EMOTION_CONFIDENCE_FLOOR:
        return "neutral", score
    return label, score


def run_sarcasm(text: str) -> tuple[bool, float]:
    model = get_model("sarcasm")
    result = _top_result(model(text))
    is_irony = result["label"].lower() == "irony"
    return is_irony, round(result["score"], 4)


def resolve_effective_sentiment(
    sentiment_score: float,
    is_sarcastic: bool,
    sarcasm_score: float,
) -> tuple[float, str]:
    if is_sarcastic and sarcasm_score >= SARCASM_OVERRIDE_THRESHOLD and sentiment_score > 0:
        corrected_score = round(-sentiment_score * 0.8, 4)
        return corrected_score, "negative"
    label = "negative" if sentiment_score < 0 else ("positive" if sentiment_score > 0 else "neutral")
    return sentiment_score, label


def resolve_effective_emotion(
    text_emotion: str,
    text_emotion_score: float,
    media_emotion: str,
    media_emotion_score: float,
    text_is_empty: bool = False,
) -> tuple[str, float]:
    """
    Blends text-derived and media-derived emotion using a weighted-confidence
    vote, mirroring resolve_effective_sentiment's philosophy but adapted for
    a categorical label instead of a continuous score.

    text_is_empty bypass: an empty/near-empty caption still produces a
    confident-looking output from the text classifier (e.g. "neutral" at
    79%) -- that's the model's default read of nothing, not genuine signal.
    Letting it vote at that confidence would let "neutral" win almost every
    weighted comparison against a real image emotion read, which defeats
    the point of blending in the first place. So: no caption, no text vote
    -- go with whatever the media says.

    IMPORTANT: text_is_empty must be computed from the user's actual caption
    (see analyze_text's `original_content` parameter), never from whatever
    placeholder text a caller substituted in for the classifier to have
    something to run on ("shared a photo" etc. is not empty by length, but
    it is empty of real signal).

    Otherwise: two candidate labels (text's pick, media's pick -- may be
    the same). Each side's weighted confidence is its own confidence times
    its blend weight (text leads at TEXT_EMOTION_WEIGHT=0.6, since a real
    caption is the user's own words and should carry more trust than an
    automated visual read); higher weighted confidence wins.
    """
    if text_is_empty:
        return media_emotion, media_emotion_score

    if text_emotion == media_emotion:
        combined_score = round(
            text_emotion_score * TEXT_EMOTION_WEIGHT + media_emotion_score * MEDIA_EMOTION_WEIGHT, 4
        )
        return text_emotion, combined_score

    text_weighted = text_emotion_score * TEXT_EMOTION_WEIGHT
    media_weighted = media_emotion_score * MEDIA_EMOTION_WEIGHT

    if media_weighted > text_weighted:
        return media_emotion, round(media_weighted, 4)
    return text_emotion, round(text_weighted, 4)


def blend_media_signal(
    sentiment_score: float,
    sentiment: str,
    risk_score: float,
    emotion: str,
    emotion_score: float,
    media_source: str | None,
    text_is_empty: bool = False,
) -> tuple[float, str, float, str, float, dict | None]:
    """
    Blends CLIP-based media analysis (image or video, auto-detected via
    analyze_media above) into the text-derived sentiment/risk/emotion.

    - sentiment_score: weighted blend (text 60%, media 40%).
    - risk_score: max(text_risk, media_risk) -- never diluted.
    - emotion: weighted-confidence vote between text and media reads, with
      an empty-caption bypass -- see resolve_effective_emotion.
    """
    if not media_source:
        return sentiment_score, sentiment, risk_score, emotion, emotion_score, None

    media_result = analyze_media(media_source)
    print(f"DEBUG media_result={media_result}", flush=True)

    blended_sentiment_score = round(
        sentiment_score * TEXT_SENTIMENT_WEIGHT + media_result["sentiment_score"] * MEDIA_SENTIMENT_WEIGHT,
        4,
    )
    if blended_sentiment_score < 0:
        blended_sentiment = "negative"
    elif blended_sentiment_score > 0:
        blended_sentiment = "positive"
    else:
        blended_sentiment = "neutral"

    blended_risk_score = round(max(risk_score, media_result["risk_score"]), 4)

    blended_emotion, blended_emotion_score = resolve_effective_emotion(
        emotion, emotion_score, media_result["emotion"], media_result["emotion_score"], text_is_empty
    )
    if blended_emotion != emotion:
        logger.info(
            f"Emotion override from media -- text emotion={emotion}({emotion_score}), "
            f"media emotion={media_result['emotion']}({media_result['emotion_score']}), "
            f"text_is_empty={text_is_empty} -> blended={blended_emotion}({blended_emotion_score})"
        )

    logger.debug(
        f"Media blend -> text_sentiment={sentiment_score}, media_sentiment={media_result['sentiment_score']}, "
        f"blended={blended_sentiment_score}; text_risk={risk_score}, media_risk={media_result['risk_score']}, "
        f"blended_risk={blended_risk_score}; emotion={blended_emotion}({blended_emotion_score})"
    )

    return (
        blended_sentiment_score,
        blended_sentiment,
        blended_risk_score,
        blended_emotion,
        blended_emotion_score,
        media_result,
    )


def compute_instant_risk(
    sentiment_score: float,
    emotion: str,
    emotion_score: float,
    is_sarcastic: bool,
    numbness_flagged: bool = False,
    numbness_strength: float = 0.0,
) -> float:
    base = max(0.0, -sentiment_score)

    emotion_weight = 0.0
    if emotion in NEGATIVE_EMOTIONS:
        emotion_weight = emotion_score * 0.4
    elif emotion in ("neutral", "surprise") and sentiment_score < -0.4:
        emotion_weight = 0.15

    sarcasm_boost = 0.1 if is_sarcastic and sentiment_score >= 0 else 0.0
    numbness_weight = numbness_strength * 0.35 if numbness_flagged else 0.0

    risk = min(1.0, base * 0.5 + emotion_weight + sarcasm_boost + numbness_weight)
    return round(risk, 4)


def compute_feed_score(
    sentiment_score: float,
    risk_score: float,
    likes_count: int = 0,
    comments_count: int = 0,
) -> float:
    positivity = (sentiment_score + 1.0) / 2.0
    engagement = min(1.0, (likes_count + comments_count * 2) / 100.0)
    penalty = risk_score * 0.6
    score = positivity * 0.5 + engagement * 0.3 - penalty
    return round(max(0.0, min(1.0, score + 0.2)), 4)


def run_lstm_risk(user_risk_history: list[float]) -> float:
    if not user_risk_history:
        return 0.0
    lstm = get_model("lstm")
    seq = user_risk_history[-20:]
    padded = [0.0] * (20 - len(seq)) + seq
    arr = np.array(padded, dtype=np.float32).reshape(1, 20, 1)
    prediction = lstm.predict(arr, verbose=0)[0][0]
    return round(float(np.clip(prediction, 0.0, 1.0)), 4)


def analyze_text(
    text: str,
    user_risk_history: list[float] | None = None,
    likes_count: int = 0,
    comments_count: int = 0,
    media_source: str | None = None,
    original_content: str | None = None,
) -> PipelineResult:
    """
    Full pipeline: text (+ optional image or video) -> PipelineResult.
    See section headers above for hard-floor bypass, sarcasm correction,
    and media blending rationale.

    `text` is what gets fed to the sentiment/emotion/sarcasm/numbness/risk
    classifiers. For posts with no real caption, callers may pass a
    placeholder ("shared a photo") here so those classifiers have something
    to run on.

    `original_content` is the user's ACTUAL caption, used only to decide
    whether the caption is empty for the purposes of the text/media emotion
    blend (see resolve_effective_emotion's text_is_empty bypass). If omitted,
    falls back to treating `text` itself as the real caption (preserves
    prior behavior for callers that don't pass a placeholder).
    """
    raw_sentiment, raw_sentiment_score = run_sentiment(text)
    emotion, emotion_score     = run_emotion(text)
    is_sarcastic, sarcasm_score = run_sarcasm(text)

    import ai.pipeline.numbness_detector as numbness_detector_module
    numbness_detector_module.get_model = get_model
    numbness_detector_module._detector = None

    try:
        numbness_flagged, numbness_strength = detect_numbness_signal(text)
    except Exception:
        numbness_flagged, numbness_strength = False, 0.0

    import ai.pipeline.risk_detector as risk_detector_module
    risk_detector_module.get_model = get_model
    risk_detector_module._detector = None

    try:
        risk_detail = detect_depression_suicide_risk(text)
    except Exception:
        risk_detail = risk_detector_module.RiskDetectionResult(
            tier=RiskTier.NONE,
            score=0.0,
            hard_floor_triggered=False,
            matched_phrase=None,
            closest_reference=None,
            similarity=None,
        )

    # Below this length, the REAL caption reflects no genuine content -- used
    # to bypass the text side of emotion blending (see resolve_effective_emotion).
    # Deliberately computed from original_content (the user's actual caption),
    # not from `text`, since `text` may be a non-empty placeholder substituted
    # in when there was no caption at all.
    caption_for_emptiness_check = (
        original_content if original_content is not None else text
    )
    text_is_empty = len((caption_for_emptiness_check or "").strip()) < 3
    print(f"DEBUG media_source={media_source!r}, text_is_empty={text_is_empty}, text_emotion={emotion}({emotion_score})", flush=True)

    sentiment_score, sentiment = resolve_effective_sentiment(
        raw_sentiment_score, is_sarcastic, sarcasm_score
    )
    if sentiment_score != raw_sentiment_score:
        logger.info(
            f"Sarcasm override -- raw sentiment={raw_sentiment}({raw_sentiment_score}), "
            f"sarcasm_score={sarcasm_score} -> corrected sentiment={sentiment}({sentiment_score})"
        )

    if risk_detail.hard_floor_triggered:
        logger.warning(
            f"HARD FLOOR triggered -- matched phrase: {risk_detail.matched_phrase!r}. "
            f"Routing directly to CRITICAL tier, bypassing ML averaging."
        )
        risk_score = 1.0
        sentiment_score, sentiment, risk_score, emotion, emotion_score, _ = blend_media_signal(
            sentiment_score, sentiment, risk_score, emotion, emotion_score, media_source, text_is_empty
        )
        feed_score = compute_feed_score(sentiment_score, risk_score, likes_count, comments_count)
        return PipelineResult(
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            emotion=emotion,
            emotion_score=emotion_score,
            sarcasm=is_sarcastic,
            sarcasm_score=sarcasm_score,
            numbness=numbness_flagged,
            numbness_score=numbness_strength,
            risk_score=risk_score,
            feed_score=feed_score,
        )

    instant_risk = compute_instant_risk(
        sentiment_score, emotion, emotion_score, is_sarcastic,
        numbness_flagged, numbness_strength,
    )
    instant_risk = max(instant_risk, risk_detail.score)

    history = (user_risk_history or []) + [instant_risk]
    lstm_risk = run_lstm_risk(history)

    risk_score = round(instant_risk * 0.3 + lstm_risk * 0.7, 4) if lstm_risk > 0 else instant_risk

    sentiment_score, sentiment, risk_score, emotion, emotion_score, _ = blend_media_signal(
        sentiment_score, sentiment, risk_score, emotion, emotion_score, media_source, text_is_empty
    )

    feed_score = compute_feed_score(sentiment_score, risk_score, likes_count, comments_count)

    logger.debug(
        f"Pipeline -> sentiment={sentiment}({sentiment_score}), "
        f"emotion={emotion}({emotion_score}), sarcasm={is_sarcastic}, "
        f"numbness={numbness_flagged}({numbness_strength}), "
        f"risk_tier={risk_detail.tier}({risk_detail.score}), "
        f"risk={risk_score}, feed={feed_score}, media_analyzed={bool(media_source)}"
    )

    return PipelineResult(
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        emotion=emotion,
        emotion_score=emotion_score,
        sarcasm=is_sarcastic,
        sarcasm_score=sarcasm_score,
        numbness=numbness_flagged,
        numbness_score=numbness_strength,
        risk_score=risk_score,
        feed_score=feed_score,
    )


def analyze_liked_post(
    liked_post_result: PipelineResult,
    liker_risk_history: list[float] | None = None,
    repeated_engagement_count: int = 1,
) -> tuple[list[float], EngagementSignal]:
    """
    Call when a user LIKES a post/story that already has a computed
    PipelineResult -- folds a reduced-weight signal into the LIKER's own
    risk_history without re-running the full text/media pipeline.
    """
    return record_engagement_signal(
        liked_post_result=liked_post_result,
        liker_risk_history=liker_risk_history or [],
        repeated_engagement_count=repeated_engagement_count,
    )