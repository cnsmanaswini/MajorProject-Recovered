"""
label_risk_sample.py
Interactive, resumable manual labeling for a small ground-truth sample to
evaluate the LSTM risk-scoring component. You judge each caption blind
(risk_score is hidden) and type 1 (concerning) or 0 (not concerning).

Run from backend/, venv active:
    python label_risk_sample.py --count 30

Safe to interrupt (Ctrl+C) and rerun — already-labeled posts are skipped.
Labels are saved to risk_labels.json as you go.
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import select                    # noqa: E402
from models.database import AsyncSessionLocal    # noqa: E402
from models.models import Post                    # noqa: E402

LABELS_FILE = Path(__file__).resolve().parent / "risk_labels.json"


def load_labels() -> dict:
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text())
    return {}


def save_labels(labels: dict):
    LABELS_FILE.write_text(json.dumps(labels, indent=2))


async def label(count: int):
    labels = load_labels()
    print(f"({len(labels)} posts already labeled from a previous run)\n")

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Post.id, Post.content).where(Post.content != ""))
        all_posts = result.all()

    random.shuffle(all_posts)
    todo = [(pid, content) for pid, content in all_posts if str(pid) not in labels][:count]

    if not todo:
        print("Nothing left to label at this count — all sampled posts already done.")
        return

    print(f"Labeling {len(todo)} posts. For each caption, ask yourself:")
    print('  "If a friend posted this, would I be genuinely worried about them?"')
    print("Type 1 = concerning, 0 = not concerning, s = skip, q = save & quit\n")

    for pid, content in todo:
        print("-" * 60)
        print(f'"{content}"')
        while True:
            choice = input("Concerning? [1/0/s/q]: ").strip().lower()
            if choice in ("1", "0"):
                labels[str(pid)] = int(choice)
                save_labels(labels)
                break
            elif choice == "s":
                break
            elif choice == "q":
                print(f"\nSaved. {len(labels)} total labeled so far.")
                return
            else:
                print("  please type 1, 0, s, or q")

    print(f"\n✅ Done. {len(labels)} total labeled posts saved to {LABELS_FILE.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(label(args.count))