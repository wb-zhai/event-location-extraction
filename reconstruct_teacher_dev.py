#!/usr/bin/env python3
"""Reconstruct the exact article-level Teacher dev split used by V3 SFT.

This imports the split implementation from the current repo's to_sft.py so the
evaluation set cannot silently drift from the training split logic.

Default expected split for the frozen 21,979-article dataset:
  2,000 positive + 198 no-event = 2,198 dev articles
with dev_ratio=0.10 and seed=32.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scripts.event_extraction.generation.to_sft import _load_rows, _split_rows


def has_events(row: dict) -> bool:
    return bool((row.get("annotation") or {}).get("events"))


def rid(row: dict) -> str:
    value = row.get("id")
    return "" if value is None else str(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("data/extraction/eval"), type=Path)
    ap.add_argument("--dev-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--expect-total", type=int, default=21979)
    ap.add_argument("--expect-dev", type=int, default=2198)
    args = ap.parse_args()

    rows = _load_rows([args.input])
    if args.expect_total and len(rows) != args.expect_total:
        raise SystemExit(
            f"ERROR: expected {args.expect_total:,} input rows, found {len(rows):,}"
        )

    rng = random.Random(args.seed)
    train_rows, dev_rows = _split_rows(rows, args.dev_ratio, rng)

    train_ids = [rid(row) for _, row in train_rows]
    dev_ids = [rid(row) for _, row in dev_rows]

    if any(not x for x in train_ids + dev_ids):
        raise SystemExit("ERROR: one or more rows have no id")
    if len(set(train_ids)) != len(train_ids):
        raise SystemExit("ERROR: duplicate article ids in reconstructed train split")
    if len(set(dev_ids)) != len(dev_ids):
        raise SystemExit("ERROR: duplicate article ids in reconstructed dev split")

    overlap = set(train_ids) & set(dev_ids)
    if overlap:
        raise SystemExit(f"ERROR: train/dev article leakage: {len(overlap)} ids")

    if args.expect_dev and len(dev_rows) != args.expect_dev:
        raise SystemExit(
            f"ERROR: expected {args.expect_dev:,} dev articles, got {len(dev_rows):,}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev_jsonl = args.out_dir / "v3_teacher_dev_articles.jsonl"
    dev_id_path = args.out_dir / "v3_teacher_dev_ids.txt"
    train_id_path = args.out_dir / "v3_teacher_train_ids.txt"
    summary_path = args.out_dir / "v3_teacher_split_summary.json"

    with dev_jsonl.open("w", encoding="utf-8") as f:
        for _, row in dev_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dev_id_path.write_text("\n".join(dev_ids) + "\n", encoding="utf-8")
    train_id_path.write_text("\n".join(train_ids) + "\n", encoding="utf-8")

    dev_pos = sum(has_events(row) for _, row in dev_rows)
    train_pos = sum(has_events(row) for _, row in train_rows)

    groups = Counter()
    for source_name, row in dev_rows:
        groups[f"{source_name}|{'positive' if has_events(row) else 'empty'}"] += 1

    summary = {
        "input": str(args.input),
        "split_implementation": "scripts.event_extraction.generation.to_sft._split_rows",
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "input_articles": len(rows),
        "train_articles": len(train_rows),
        "dev_articles": len(dev_rows),
        "train_positive": train_pos,
        "train_empty": len(train_rows) - train_pos,
        "dev_positive": dev_pos,
        "dev_empty": len(dev_rows) - dev_pos,
        "train_dev_id_overlap": 0,
        "dev_groups": dict(sorted(groups.items())),
        "outputs": {
            "dev_articles": str(dev_jsonl),
            "dev_ids": str(dev_id_path),
            "train_ids": str(train_id_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
