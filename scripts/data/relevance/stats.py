"""Print summary stats for a relevance-labeled JSONL file.

Reports label (positive/negative) and relevance.decision distributions, the
top-N most frequent risk_factors, and how positive/negative labels break down
against the relevance decision (and vice versa).
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def get_decision(rec: dict) -> str:
    rel = rec.get("relevance") or {}
    return rel.get("decision") or "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print stats for a relevance-labeled JSONL file.")
    parser.add_argument("--input", required=True, type=Path, help="Path to relevance-labeled JSONL")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top risk_factors to show")
    args = parser.parse_args()

    labels: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    risk_factors: Counter[str] = Counter()
    label_by_decision: Counter[tuple[str, str]] = Counter()
    total = 0

    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1

            label = rec.get("label") or "none"
            decision = get_decision(rec)
            labels[label] += 1
            decisions[decision] += 1
            label_by_decision[(label, decision)] += 1

            for rf in rec.get("risk_factors") or []:
                risk_factors[rf] += 1

    print(f"Total records: {total} ({args.input})")
    print()

    print("Label counts (positive/negative):")
    for label, count in labels.most_common():
        pct = count / total * 100 if total else 0.0
        print(f"  {label}: {count} ({pct:.1f}%)")
    print()

    print("Relevance decision counts:")
    for decision, count in decisions.most_common():
        pct = count / total * 100 if total else 0.0
        print(f"  {decision}: {count} ({pct:.1f}%)")
    print()

    print(f"Top {args.top_n} risk_factors:")
    for rf, count in risk_factors.most_common(args.top_n):
        print(f"  {rf}: {count}")
    print()

    print("Label vs. relevance decision:")
    for label in sorted(labels):
        label_total = labels[label]
        print(f"  {label} ({label_total}):")
        for decision in sorted(decisions):
            count = label_by_decision[(label, decision)]
            if count == 0:
                continue
            pct = count / label_total * 100 if label_total else 0.0
            print(f"    {decision}: {count} ({pct:.1f}%)")
    print()

    print("Relevance decision vs. label:")
    for decision in sorted(decisions):
        decision_total = decisions[decision]
        print(f"  {decision} ({decision_total}):")
        for label in sorted(labels):
            count = label_by_decision[(label, decision)]
            if count == 0:
                continue
            pct = count / decision_total * 100 if decision_total else 0.0
            print(f"    {label}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
