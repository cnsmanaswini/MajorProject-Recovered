"""
eval_sentiment_emotion.py
Computes precision / recall / F1 for your sentiment and emotion classifiers
against TweetEval's REAL labeled test sets — not synthetic app data, since
your app has no ground truth of its own to check predictions against.

IMPORTANT CAVEATS (state these in your report, don't hide them):
  1. Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest was
     trained/fine-tuned specifically on this exact TweetEval sentiment
     benchmark. This reproduces its published state-of-the-art numbers,
     it does not independently validate the model on unseen data.
  2. Emotion: TweetEval's emotion task uses 4 labels (anger, joy,
     optimism, sadness). Your app's emotion space uses 5 (anger, fear,
     joy, neutral, sadness) — no "optimism", no matching label for
     "fear"/"neutral" in TweetEval. Rows labeled "optimism" are DROPPED
     from this eval rather than force-mapped to something incorrect.
     This means the emotion metrics only cover 3 of your app's 5 classes
     (anger, joy, sadness) — fear and neutral are simply untested here,
     because no public labeled data exists for them in this benchmark.
  3. This evaluates the LOCAL model only (run_sentiment/run_emotion
     without Groq arbitration) — running Groq on ~3000 test rows would
     mean thousands of live API calls. If you want arbitration's effect
     shown too, run --sample 200 with --with-groq to test a subset.

Usage (from backend/, venv active):
    python eval_sentiment_emotion.py
    python eval_sentiment_emotion.py --sample 300              # faster, subset
    python eval_sentiment_emotion.py --sample 200 --with-groq  # include Groq arbitration
"""

import argparse
import asyncio
import random
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from ai.pipeline.loader import preload_models          # noqa: E402
from ai.pipeline.analyzer import run_sentiment, run_emotion, run_sarcasm  # noqa: E402

try:
    from ai.pipeline.emotion_arbiter import arbitrate_emotion
except ImportError:
    arbitrate_emotion = None

from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

random.seed(42)

BASE = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets"

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
EMOTION_MAP = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}


async def fetch_lines(url: str) -> list[str]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        return [line for line in resp.text.strip().split("\n")]


async def run_eval(sample: int | None, with_groq: bool):
    preload_models()

    # ---------- SENTIMENT ----------
    print("=" * 60)
    print("SENTIMENT — vs TweetEval sentiment test set")
    print("=" * 60)

    sent_texts = await fetch_lines(f"{BASE}/sentiment/test_text.txt")
    sent_labels_raw = await fetch_lines(f"{BASE}/sentiment/test_labels.txt")
    sent_labels = [SENTIMENT_MAP[int(x)] for x in sent_labels_raw]

    pairs = list(zip(sent_texts, sent_labels))
    if sample:
        pairs = random.sample(pairs, min(sample, len(pairs)))

    y_true_sent, y_pred_sent = [], []
    for text, gold in pairs:
        pred_label, _ = run_sentiment(text)
        y_true_sent.append(gold)
        y_pred_sent.append(pred_label)

    print(f"\nEvaluated on {len(pairs)} examples\n")
    print(classification_report(y_true_sent, y_pred_sent, digits=3))
    print("Confusion matrix (rows=true, cols=pred), labels=negative/neutral/positive:")
    print(confusion_matrix(y_true_sent, y_pred_sent, labels=["negative", "neutral", "positive"]))

    # ---------- EMOTION ----------
    print("\n" + "=" * 60)
    print("EMOTION — vs TweetEval emotion test set (optimism rows dropped)")
    print("=" * 60)

    emo_texts = await fetch_lines(f"{BASE}/emotion/test_text.txt")
    emo_labels_raw = await fetch_lines(f"{BASE}/emotion/test_labels.txt")
    emo_labels = [EMOTION_MAP[int(x)] for x in emo_labels_raw]

    emo_pairs = [(t, l) for t, l in zip(emo_texts, emo_labels) if l != "optimism"]
    dropped = len(emo_texts) - len(emo_pairs)
    print(f"\nDropped {dropped} 'optimism'-labeled rows (no equivalent in app's emotion space)")

    if sample:
        emo_pairs = random.sample(emo_pairs, min(sample, len(emo_pairs)))

    y_true_emo, y_pred_emo_local = [], []
    y_pred_emo_final = []  # local, or Groq-arbitrated if --with-groq

    for i, (text, gold) in enumerate(emo_pairs):
        local_label, local_score = run_emotion(text)
        y_true_emo.append(gold)
        y_pred_emo_local.append(local_label)

        final_label = local_label
        if with_groq and arbitrate_emotion is not None:
            arbitrated = arbitrate_emotion(text, local_label, local_score)
            if arbitrated is not None:
                groq_label, _ = arbitrated
                final_label = groq_label
        y_pred_emo_final.append(final_label)

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(emo_pairs)}")

    print(f"\nEvaluated on {len(emo_pairs)} examples (labels: anger, joy, sadness)\n")
    print("--- LOCAL model only ---")
    print(classification_report(y_true_emo, y_pred_emo_local, labels=["anger", "joy", "sadness"], digits=3, zero_division=0))

    if with_groq:
        print("--- WITH Groq arbitration ---")
        print(classification_report(y_true_emo, y_pred_emo_final, labels=["anger", "joy", "sadness"], digits=3, zero_division=0))

    # ---------- SARCASM ----------
    print("\n" + "=" * 60)
    print("SARCASM — vs TweetEval irony test set")
    print("=" * 60)

    IRONY_MAP = {0: "not_sarcastic", 1: "sarcastic"}
    sarc_texts = await fetch_lines(f"{BASE}/irony/test_text.txt")
    sarc_labels_raw = await fetch_lines(f"{BASE}/irony/test_labels.txt")
    sarc_labels = [IRONY_MAP[int(x)] for x in sarc_labels_raw]

    sarc_pairs = list(zip(sarc_texts, sarc_labels))
    if sample:
        sarc_pairs = random.sample(sarc_pairs, min(sample, len(sarc_pairs)))

    y_true_sarc, y_pred_sarc = [], []
    for text, gold in sarc_pairs:
        is_sarcastic, _ = run_sarcasm(text)
        y_true_sarc.append(gold)
        y_pred_sarc.append("sarcastic" if is_sarcastic else "not_sarcastic")

    print(f"\nEvaluated on {len(sarc_pairs)} examples\n")
    print(classification_report(y_true_sarc, y_pred_sarc, labels=["sarcastic", "not_sarcastic"], digits=3, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                         help="Evaluate on a random subset instead of the full test set (faster)")
    parser.add_argument("--with-groq", action="store_true",
                         help="Also run Groq arbitration on the emotion set (costs API calls)")
    args = parser.parse_args()
    asyncio.run(run_eval(args.sample, args.with_groq))