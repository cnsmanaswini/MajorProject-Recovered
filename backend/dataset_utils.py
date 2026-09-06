"""
dataset_utils.py
Single shared source of real short-text captions for ALL seed scripts.

Every seed script that needs caption/comment text should import
`fetch_captions` from here instead of hardcoding its own filler lines.
This is the one and only place scene/caption text gets pulled from a
real dataset — TweetEval (cardiffnlp, GitHub), an academic Twitter
emotion benchmark.
"""

import re
import httpx

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

# Targets the specific failure mode observed in seeded runs: short
# distilled emotion classifiers don't parse negation well, so a sentence
# that NEGATES a bad thing ("worry is a problem you may never have",
# "you are NOT your struggle") gets classified AS the bad thing. This
# doesn't fix the classifier -- it just avoids feeding it the shape of
# text it's known to get backwards.
NEGATED_REASSURANCE_PATTERN = re.compile(
    r"\b(is not|isn't|are not|aren't|you are not|you're not)\b.{0,25}"
    r"\b(worry|problem|struggle|weak|broken|hopeless|afraid|fear|alone)\b"
    r"|\bnever\b.{0,20}\bhave\b"
    r"|\bdown payment\b"
    r"|\bspec of hope\b",
    re.IGNORECASE,
)

# Generic third-person "inspirational quote" shape (no first-person
# pronoun, no personal disclosure) -- these are exactly the sentences
# most likely to be reposted quotes rather than a real feeling, and the
# classifier's negation-blindness hits this shape hardest.
FIRST_PERSON_PATTERN = re.compile(r"\b(i|i'm|i've|i'll|my|me|mine)\b", re.IGNORECASE)


def clean_caption(raw: str, min_len: int = 15, max_len: int = 220) -> str | None:
    text = raw.strip()
    if not text:
        return None

    text = re.sub(r"@user\b", "", text)          # TweetEval anonymizes handles as literal "@user"
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < min_len or len(text) > max_len:
        return None
    if BLOCKLIST_PATTERN.search(text):
        return None
    hashtag_chars = sum(len(w) for w in text.split() if w.startswith("#"))
    if hashtag_chars > len(text) * 0.4:
        return None
    if NEGATED_REASSURANCE_PATTERN.search(text):
        return None
    # Third-person aphorism with no personal pronoun AND no emoji/exclamation
    # -- reads like a quote, not a real post. Skip it rather than feed the
    # classifier a shape it tends to get backwards.
    has_personal_marker = bool(FIRST_PERSON_PATTERN.search(text))
    has_emotive_marker = "!" in text or any(ord(c) > 0x2600 for c in text)  # rough emoji check
    if not has_personal_marker and not has_emotive_marker:
        return None

    return text


async def fetch_captions(count: int, min_len: int = 15, max_len: int = 220,
                          shuffle: bool = True) -> list[str]:
    """Pulls, cleans, and dedupes real TweetEval lines. Returns up to
    `count` of them (fewer if the dataset doesn't have that many clean
    lines after filtering)."""
    import random

    seen, cleaned = set(), []
    async with httpx.AsyncClient(timeout=30) as client:
        for url in TWEETEVAL_URLS:
            resp = await client.get(url)
            resp.raise_for_status()
            for line in resp.text.splitlines():
                c = clean_caption(line, min_len, max_len)
                if c and c not in seen:
                    seen.add(c)
                    cleaned.append(c)

    if shuffle:
        random.shuffle(cleaned)

    if len(cleaned) < count:
        print(f"⚠️  Only found {len(cleaned)} clean captions after filtering "
              f"(asked for {count}). Returning all of them.")
        return cleaned
    return cleaned[:count]