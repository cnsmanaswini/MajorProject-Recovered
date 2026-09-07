"""
eval_risk.py
Computes precision/recall/F1 for the risk-scoring component by comparing
model risk_score (thresholded) against your manually-labeled ground truth
from label_risk_sample.py.

Run from backend/, venv active, AFTER running label_risk_sample.py:
    python eval_risk.py
    python eval_risk.py --threshold 0.6   # try a different cutoff
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import select                    # noqa: E402
from models.database import AsyncSessionLocal    # noqa: E402
from models.models import Post                    # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

LABELS_FILE = Path(__file__).resolve().parent / "risk_labels.json"


async def evaluate(threshold: float):
    if not LABELS_FILE.exists():
        print("❌ No risk_labels.json found — run label_risk_sample.py first.")
        return

    labels = json.loads(LABELS_FILE.read_text())
    post_ids = [int(pid) for pid in labels.keys()]

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Post.id, Post.risk_score).where(Post.id.in_(post_ids)))
        risk_by_id = {pid: score for pid, score in result.all()}

    y_true, y_pred = [], []
    for pid_str, human_label in labels.items():
        pid = int(pid_str)
        if pid not in risk_by_id:
            continue
        y_true.append("concerning" if human_label == 1 else "not_concerning")
        y_pred.append("concerning" if risk_by_id[pid] >= threshold else "not_concerning")

    print(f"Evaluated on {len(y_true)} manually-labeled posts (threshold={threshold})\n")
    print(classification_report(y_true, y_pred, labels=["concerning", "not_concerning"], digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), labels=concerning/not_concerning:")
    print(confusion_matrix(y_true, y_pred, labels=["concerning", "not_concerning"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="risk_score >= this value counts as a 'concerning' prediction")
    args = parser.parse_args()
    asyncio.run(evaluate(args.threshold))