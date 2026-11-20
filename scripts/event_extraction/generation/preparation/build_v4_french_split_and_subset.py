#!/usr/bin/env python3
"""
Build the French V4 held-out Teacher evaluation set and the French source
subset to add to the frozen V3 English article dataset.

Policy:
- keep only successful Gemini Teacher rows
- keep only canonical bona.v4 events in annotation.events
- preserve top-level language
- reserve a 10% article-level French eval split using the exact to_sft.py
  _split_rows implementation and seed 32
- from the remaining French pool, keep ALL naturally occurring no-event rows
- downsample only positive rows using the same V3 inverse-sqrt article-level
  weighting, bottom-10% rare-label protection, and sampling seed 42
- default final French SFT-source contribution: 2,173 articles
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from scripts.event_extraction.generation.to_sft import _split_rows


def load_ontology_labels(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events")
    if isinstance(events, dict):
        labels = list(events.keys())
    elif isinstance(events, list):
        labels = []
        for item in events:
            if isinstance(item, str):
                labels.append(item)
            elif isinstance(item, dict):
                label = item.get("name") or item.get("label") or item.get("event_type")
                if label:
                    labels.append(str(label))
    else:
        raise RuntimeError(f"Unsupported ontology schema in {path}")
    if not labels:
        raise RuntimeError(f"No ontology labels found in {path}")
    return labels


def rid(row: dict) -> str:
    value = row.get("id")
    if value is None:
        value = row.get("article_id")
    return "" if value is None else str(value)


def clean_row(row: dict, canonical: set[str], off: Counter) -> dict:
    out = dict(row)
    ann = dict(out.get("annotation") or {})
    clean_events = []
    for event in ann.get("events") or []:
        label = event.get("event_type")
        if label in canonical:
            clean_events.append(event)
        elif label:
            off[label] += 1
    ann["events"] = clean_events
    out["annotation"] = ann
    return out


def has_events(row: dict) -> bool:
    return bool((row.get("annotation") or {}).get("events"))


def labels_in_row(row: dict) -> set[str]:
    return {
        e["event_type"]
        for e in (row.get("annotation") or {}).get("events", [])
        if e.get("event_type")
    }


def article_label_counts(rows: list[dict]) -> Counter:
    c = Counter()
    for row in rows:
        for label in labels_in_row(row):
            c[label] += 1
    return c


def quantile_nearest(values: list[int], q: float) -> int:
    values = sorted(values)
    if not values:
        return 0
    idx = round((len(values) - 1) * q)
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]


def weighted_sample_without_replacement(items, weights, k, rng):
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    scored = []
    for item, weight in zip(items, weights):
        if weight <= 0:
            continue
        u = max(rng.random(), 1e-15)
        priority = -math.log(u) / weight
        scored.append((priority, item))
    scored.sort(key=lambda x: x[0])
    return [item for _, item in scored[:k]]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument(
        "--ontology",
        type=Path,
        default=Path("ontologies/zhai/bona.v4.json"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/extraction/v4_french"),
    )
    ap.add_argument("--target-total", type=int, default=2173)
    ap.add_argument("--dev-ratio", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=32)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--rare-quantile", type=float, default=0.10)
    ap.add_argument("--expect-success", type=int, default=4373)
    args = ap.parse_args()

    canonical_labels = load_ontology_labels(args.ontology)
    canonical = set(canonical_labels)

    raw_rows = []
    error_rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                raw_rows.append(row)
            else:
                error_rows.append(row)

    if args.expect_success and len(raw_rows) != args.expect_success:
        raise SystemExit(
            f"ERROR: expected {args.expect_success:,} successful rows, found {len(raw_rows):,}"
        )

    if any(not rid(r) for r in raw_rows):
        raise SystemExit("ERROR: one or more successful rows have no id")
    if len({rid(r) for r in raw_rows}) != len(raw_rows):
        raise SystemExit("ERROR: duplicate successful article ids")

    languages = Counter(str(r.get("language")) for r in raw_rows)
    if set(languages) != {"fra"}:
        raise SystemExit(f"ERROR: expected only language='fra', got {dict(languages)}")

    off = Counter()
    cleaned = [clean_row(r, canonical, off) for r in raw_rows]

    source_pos = sum(has_events(r) for r in cleaned)
    source_neg = len(cleaned) - source_pos

    # Exact article-level split implementation used by to_sft.py.
    source_name = args.input.name
    wrapped = [(source_name, row) for row in cleaned]
    rng_split = random.Random(args.split_seed)
    train_wrapped, dev_wrapped = _split_rows(wrapped, args.dev_ratio, rng_split)
    train_pool = [row for _, row in train_wrapped]
    dev_rows = [row for _, row in dev_wrapped]

    train_ids = {rid(r) for r in train_pool}
    dev_ids = {rid(r) for r in dev_rows}
    if train_ids & dev_ids:
        raise SystemExit("ERROR: train/eval article leakage")

    train_pos = [r for r in train_pool if has_events(r)]
    train_neg = [r for r in train_pool if not has_events(r)]
    dev_pos = [r for r in dev_rows if has_events(r)]
    dev_neg = [r for r in dev_rows if not has_events(r)]

    target_pos = args.target_total - len(train_neg)
    if target_pos <= 0:
        raise SystemExit(
            f"ERROR: {len(train_neg):,} no-event rows already exceed target total "
            f"{args.target_total:,}"
        )
    if target_pos > len(train_pos):
        raise SystemExit(
            f"ERROR: need {target_pos:,} positives but only {len(train_pos):,} available"
        )

    freq = article_label_counts(train_pos)
    observed_freqs = [freq[label] for label in canonical_labels if freq[label] > 0]
    rare_cutoff = quantile_nearest(observed_freqs, args.rare_quantile)
    protected_labels = {
        label
        for label in canonical_labels
        if 0 < freq[label] <= rare_cutoff
    }

    protected = []
    sampleable = []
    for row in train_pos:
        if labels_in_row(row) & protected_labels:
            protected.append(row)
        else:
            sampleable.append(row)

    if len(protected) > target_pos:
        raise SystemExit(
            f"ERROR: protected positives ({len(protected):,}) exceed target positives "
            f"({target_pos:,})"
        )

    remaining = target_pos - len(protected)
    max_freq = max(freq.values())
    weights = []
    for row in sampleable:
        labels = labels_in_row(row)
        weights.append(
            max(math.sqrt(max_freq / freq[label]) for label in labels)
        )

    rng_sample = random.Random(args.sample_seed)
    sampled = weighted_sample_without_replacement(
        sampleable, weights, remaining, rng_sample
    )
    selected_pos = protected + sampled

    selected_ids = {rid(r) for r in selected_pos + train_neg}

    # Restore original successful-source order.
    final_french = [r for r in cleaned if rid(r) in selected_ids]
    dev_ordered = [r for r in cleaned if rid(r) in dev_ids]

    if len(final_french) != args.target_total:
        raise SystemExit(
            f"ERROR: expected final French total {args.target_total:,}, "
            f"got {len(final_french):,}"
        )
    if {rid(r) for r in final_french} & {rid(r) for r in dev_ordered}:
        raise SystemExit("ERROR: final French source overlaps held-out French eval")

    final_pos = sum(has_events(r) for r in final_french)
    final_neg = len(final_french) - final_pos

    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_path = args.output_dir / "v4_french_teacher_eval_articles.jsonl"
    eval_ids_path = args.output_dir / "v4_french_teacher_eval_ids.txt"
    train_path = args.output_dir / f"v4_french_sft_source_{args.target_total}.jsonl"
    train_ids_path = args.output_dir / f"v4_french_sft_source_{args.target_total}_ids.txt"
    pool_path = args.output_dir / "v4_french_train_pool_before_downsample.jsonl"
    summary_path = args.output_dir / "v4_french_split_sampling_summary.json"

    write_jsonl(eval_path, dev_ordered)
    write_jsonl(train_path, final_french)
    write_jsonl(pool_path, [r for r in cleaned if rid(r) in train_ids])

    eval_ids_path.write_text(
        "\n".join(rid(r) for r in dev_ordered) + "\n", encoding="utf-8"
    )
    train_ids_path.write_text(
        "\n".join(rid(r) for r in final_french) + "\n", encoding="utf-8"
    )

    summary = {
        "input_rows": len(raw_rows) + len(error_rows),
        "successful_rows": len(raw_rows),
        "error_rows": len(error_rows),
        "source_positive": source_pos,
        "source_no_event": source_neg,
        "off_ontology_mentions_removed": sum(off.values()),
        "off_ontology_by_label": dict(off),
        "split": {
            "dev_ratio": args.dev_ratio,
            "seed": args.split_seed,
            "train_pool_total": len(train_pool),
            "train_pool_positive": len(train_pos),
            "train_pool_no_event": len(train_neg),
            "heldout_eval_total": len(dev_ordered),
            "heldout_eval_positive": len(dev_pos),
            "heldout_eval_no_event": len(dev_neg),
        },
        "downsample": {
            "method": "article-level inverse-sqrt weighted sampling",
            "seed": args.sample_seed,
            "rare_protection_quantile": args.rare_quantile,
            "rare_cutoff_articles": rare_cutoff,
            "protected_labels": sorted(protected_labels),
            "protected_positive_articles": len(protected),
            "target_total": args.target_total,
            "target_positive": target_pos,
            "kept_all_train_pool_no_event": True,
            "selected_positive": final_pos,
            "selected_no_event": final_neg,
            "selected_total": len(final_french),
        },
        "paths": {
            "heldout_eval": str(eval_path),
            "heldout_eval_ids": str(eval_ids_path),
            "french_sft_source": str(train_path),
            "french_sft_source_ids": str(train_ids_path),
            "train_pool_before_downsample": str(pool_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("DONE: V4 French split + source subset")
    print(f"successful Teacher rows:          {len(raw_rows):,}")
    print(f"failed Teacher rows excluded:     {len(error_rows):,}")
    print(f"source positives:                 {source_pos:,}")
    print(f"source no-event:                  {source_neg:,}")
    print(f"off-ontology mentions removed:    {sum(off.values()):,}")
    print()
    print(f"held-out French eval:             {len(dev_ordered):,}")
    print(f"  positives:                      {len(dev_pos):,}")
    print(f"  no-event:                       {len(dev_neg):,}")
    print()
    print(f"French train pool before sampling:{len(train_pool):,}")
    print(f"  positives:                      {len(train_pos):,}")
    print(f"  no-event:                       {len(train_neg):,}")
    print()
    print(f"final French SFT source:          {len(final_french):,}")
    print(f"  positives:                      {final_pos:,}")
    print(f"  no-event:                       {final_neg:,}")
    print(f"  no-event share:                 {100*final_neg/len(final_french):.2f}%")
    print(f"rare cutoff:                      <= {rare_cutoff:,} train-pool articles")
    print(f"protected labels:                 {len(protected_labels):,}")
    print()
    print(f"eval:                             {eval_path}")
    print(f"French SFT source:                {train_path}")
    print(f"summary:                          {summary_path}")


if __name__ == "__main__":
    main()
