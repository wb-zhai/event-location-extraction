"""
Compute Recall@K for a candidates file.

Each record must have:
  annotation.events[].event_type  — gold event types
  candidates[]                    — ranked list of candidate strings

Recall@K for a document = fraction of gold event types found in top-K candidates.
Macro Recall@K = mean over all documents that have at least one gold event.

Usage:
  python scripts/eval_recall_at_k.py dataset/zhai/v3/science/dev.candidates.jsonl
  python scripts/eval_recall_at_k.py path/to/file.jsonl --k 1 5 10 20 50
"""

import argparse
import json
import sys
from pathlib import Path


def recall_at_k(gold: set[str], candidates: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top_k = set(candidates[:k])
    return len(gold & top_k) / len(gold)


def evaluate(path: Path, ks: list[int]) -> None:
    totals = {k: 0.0 for k in ks}
    n_docs = 0

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            events = rec.get("annotation", {}).get("events", [])
            gold = {e["event_type"] for e in events if e.get("event_type")}
            if not gold:
                continue

            candidates = rec.get("candidates", [])
            n_docs += 1
            for k in ks:
                totals[k] += recall_at_k(gold, candidates, k)

    if n_docs == 0:
        print("No documents with gold annotations found.", file=sys.stderr)
        sys.exit(1)

    print(f"Documents evaluated: {n_docs}")
    print(f"{'K':>6}  {'Recall@K':>10}")
    print("-" * 20)
    for k in ks:
        print(f"{k:>6}  {totals[k] / n_docs:>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="Path to .candidates.jsonl file")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20, 50, 70, 100],
        metavar="K",
        help="Values of K to evaluate (default: 1 3 5 10 20 50 70 100)",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    evaluate(args.file, sorted(args.k))


if __name__ == "__main__":
    main()
