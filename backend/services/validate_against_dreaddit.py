"""
Validate ai/pipeline/analyzer.py's risk scoring against Dreaddit's human-
labeled stress annotations.

This is NOT a training script — Dreaddit's binary "stress / not stress"
label is a different (looser) construct than your risk_score, which is
meant to capture depression/suicide-risk severity via risk_detector.py's
tiers. Treat this as a sanity check: does risk_score trend meaningfully
higher for texts humans labeled "stress" than for texts labeled "not
stress"? It should NOT be read as "risk_score should equal the label."

Usage:
    python validate_against_dreaddit.py path/to/dreaddit_test.csv

Expected columns (per the Dreaddit repo/paper): at minimum a text column
and a binary label column. Common variants: "text"/"label" or
"post_text"/"label". Adjust TEXT_COL / LABEL_COL below if your CSV headers
differ once you inspect it.
"""
import sys
import csv
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # adjust if run from elsewhere

from ai.pipeline.analyzer import analyze_text
from ai.pipeline.loader import preload_models

TEXT_COL = "text"
LABEL_COL = "label"   # 1 = stress, 0 = not stress (per Dreaddit paper)


def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV appears empty")
    if TEXT_COL not in rows[0] or LABEL_COL not in rows[0]:
        raise ValueError(
            f"Expected columns '{TEXT_COL}' and '{LABEL_COL}' — found: {list(rows[0].keys())}. "
            f"Update TEXT_COL/LABEL_COL at the top of this script to match."
        )
    return rows


def main():
    preload_models()
    if len(sys.argv) < 2:
        print("Usage: python validate_against_dreaddit.py path/to/dreaddit.csv")
        sys.exit(1)

    rows = load_rows(sys.argv[1])
    print(f"Loaded {len(rows)} rows\n")

    stress_scores = []
    not_stress_scores = []
    mismatches = []  # cases worth eyeballing: label=stress but risk_score very low, or vice versa

    for i, row in enumerate(rows):
        text = row[TEXT_COL]
        label = int(row[LABEL_COL])

        result = analyze_text(text)
        score = result.risk_score

        if label == 1:
            stress_scores.append(score)
        else:
            not_stress_scores.append(score)

        # Flag notable mismatches for manual review
        if label == 1 and score < 0.2:
            mismatches.append(("stress_but_low_risk", text[:120], score))
        elif label == 0 and score > 0.6:
            mismatches.append(("not_stress_but_high_risk", text[:120], score))

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(rows)}")

    print("\n── Summary ──────────────────────────────")
    print(f"Stress-labeled     (n={len(stress_scores)}): "
          f"mean risk_score={statistics.mean(stress_scores):.3f}, "
          f"median={statistics.median(stress_scores):.3f}")
    print(f"Not-stress-labeled (n={len(not_stress_scores)}): "
          f"mean risk_score={statistics.mean(not_stress_scores):.3f}, "
          f"median={statistics.median(not_stress_scores):.3f}")

    gap = statistics.mean(stress_scores) - statistics.mean(not_stress_scores)
    print(f"\nMean gap (stress - not_stress): {gap:.3f}")
    print("Positive gap = pipeline is directionally correct. "
          "Gap near 0 or negative = risk scoring isn't tracking human-perceived stress at all.")

    print(f"\n── Notable mismatches ({len(mismatches)}) ──")
    for kind, snippet, score in mismatches[:15]:
        print(f"[{kind}] score={score:.3f} | {snippet}...")

    if len(mismatches) > 15:
        print(f"...({len(mismatches) - 15} more, not shown)")


if __name__ == "__main__":
    main()