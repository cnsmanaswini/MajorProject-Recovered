"""
Emotion Arbiter -- Groq-based second opinion for borderline emotion calls.

The local emotion classifier (j-hartmann/emotion-english-distilroberta-base)
is a single-pass model that often anchors on the presence of an emotionally
loaded WORD rather than the sentence's actual meaning -- e.g. "helps me with
my anger" gets read as anger, "gloomy but beautiful" gets read as sadness,
even though both sentences are net-positive to a human reader. This shows up
specifically in the confidence band just above the neutral floor (~0.5-0.65)
-- the model is unsure enough to be wrong, but not unsure enough to fall back
to neutral.

This module is only called for that narrow borderline band (see
EMOTION_BORDERLINE_CEILING in analyzer.py). It asks an LLM to actually read
the sentence and pick from the SAME 5-label space the rest of the pipeline
uses, so its output can be blended in the exact same way as the local
classifier's.

Fails soft: any error (missing key, network issue, malformed response)
returns None, and the caller falls back to the local classifier's original
result. This must never take down a real request over a nice-to-have
second opinion.
"""

import json
import logging
import os
from functools import lru_cache

logger = logging.getLogger("mindgram.emotion_arbiter")

try:
    from groq import Groq
except ImportError:  # pragma: no cover - optional dependency
    Groq = None

GROQ_MODEL = "openai/gpt-oss-20b" # fast + cheap, plenty for a 5-way classification
VALID_LABELS = {"anger", "fear", "joy", "neutral", "sadness"}  # matches EMOTION_LABELS space

import re

CONTRAST_PATTERN = re.compile(r"\b(but|however|though|although|except|despite|yet|not)\b", re.IGNORECASE)
# Words strongly associated with each label -- used to detect when the local
# classifier likely anchored on a literal keyword match rather than the
# sentence's actual meaning (e.g. "anger" appearing in "helps me with my
# anger" and the model just picking anger regardless of context).
EMOTION_KEYWORDS = {
    "anger": ["anger", "angry", "mad", "furious", "rage", "pissed"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "anxiety"],
    "joy": ["joy", "happy", "happiness", "excited", "glad", "delighted"],
    "sadness": ["sad", "sadness", "depressed", "unhappy", "gloomy", "down", "crying"],
    "neutral": [],
}


def should_arbitrate(label: str, score: float, text: str, borderline_ceiling: float) -> bool:
    """
    Decides whether a local emotion call needs a Groq second opinion.
    Triggers on: genuinely low-but-above-floor confidence, OR a contrast
    clause (often signals mixed/reversed sentiment a single-pass classifier
    misses), OR the predicted label's own keyword appearing literally in the
    text (a strong sign the model just keyword-matched instead of reading
    the sentence).
    """
    if score <= borderline_ceiling:
        return True
    if CONTRAST_PATTERN.search(text):
        return True
    lowered = text.lower()
    if any(kw in lowered for kw in EMOTION_KEYWORDS.get(label, [])):
        return True
    return False

SYSTEM_PROMPT = (
    "You classify the overall emotional tone of a short social media caption. "
    "Read the FULL sentence's meaning, not just individual emotionally-loaded "
    "words -- e.g. \"this helps with my anger\" is calming/positive, not angry; "
    "\"gloomy but beautiful\" is a positive aesthetic take despite the word "
    "\"gloomy\". Watch for sarcasm, slang, fandom in-jokes, and contrast clauses "
    "(\"but\", \"not\", \"sorry not sorry\").\n\n"
    "Respond with ONLY a JSON object, no other text: "
    '{"emotion": "<one of: anger, fear, joy, neutral, sadness>", "confidence": <0.0-1.0>}'
)


@lru_cache(maxsize=1)
def _get_client():
    if Groq is None:
        raise RuntimeError("groq package not installed (pip install groq)")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment/.env")
    return Groq(api_key=api_key)


def arbitrate_emotion(text: str, local_label: str, local_score: float) -> tuple[str, float] | None:
    """
    Asks Groq for a second opinion on a borderline emotion call.
    Returns (label, confidence) on success, None on any failure -- caller
    should fall back to (local_label, local_score) when this returns None.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_completion_tokens=200,   # room for reasoning tokens + the JSON output
            reasoning_effort="low",      # this is a simple classification, no need for deep reasoning
            timeout=5,
        )
        raw = response.choices[0].message.content.strip()
        # Models occasionally wrap JSON in markdown fences despite instructions
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        label = str(parsed.get("emotion", "")).lower().strip()
        confidence = float(parsed.get("confidence", 0.0))

        if label not in VALID_LABELS:
            logger.warning(f"Groq returned invalid label {label!r}, ignoring")
            return None

        confidence = max(0.0, min(1.0, confidence))

        logger.info(
            f"Emotion arbitration -- text={text[:80]!r}, "
            f"local={local_label}({local_score}) -> groq={label}({confidence})"
        )
        return label, round(confidence, 4)

    except Exception as exc:
        # Missing key, package not installed, network timeout, bad JSON --
        # all treated the same way: log it, fall back to the local result.
        logger.warning(f"Emotion arbitration failed, falling back to local result: {exc}")
        return None