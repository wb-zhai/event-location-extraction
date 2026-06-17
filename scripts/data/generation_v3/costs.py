#!/usr/bin/env python3
"""Report token usage and estimated cost from pipeline JSONL files.

Token counts are stored per-record in llm.metadata by generate.py and fix_events.py.
For fix_events.py output in --mode per-article, multiple rows share the same LLM call.
Pass --dedup-key doc_id to count each article's tokens only once in that case.
"""
import argparse
import json
from pathlib import Path
from typing import Any

# USD per 1M tokens. Update as pricing changes.
# Each model has "standard" and "batch" tiers (batch = Gemini Batch API, ~50% off input/output).
PRICING: dict[str, dict[str, dict[str, float]]] = {
    "gemini-2.5-flash": {
        "standard": {"input": 0.075,  "output": 0.30,  "cached": 0.01875, "thinking": 0.10},
        "batch":    {"input": 0.0375, "output": 0.15,  "cached": 0.01875, "thinking": 0.05},
    },
    "gemini-2.5-flash-preview-05-20": {
        "standard": {"input": 0.075,  "output": 0.30,  "cached": 0.01875, "thinking": 0.10},
        "batch":    {"input": 0.0375, "output": 0.15,  "cached": 0.01875, "thinking": 0.05},
    },
    "gemini-2.5-pro": {
        "standard": {"input": 1.25,   "output": 10.0,  "cached": 0.31,    "thinking": 3.50},
        "batch":    {"input": 0.625,  "output": 5.0,   "cached": 0.31,    "thinking": 1.75},
    },
    "gemini-2.5-pro-preview-05-06": {
        "standard": {"input": 1.25,   "output": 10.0,  "cached": 0.31,    "thinking": 3.50},
        "batch":    {"input": 0.625,  "output": 5.0,   "cached": 0.31,    "thinking": 1.75},
    },
    "gemini-3.1-pro-preview": {
        "standard": {"input": 2.50,   "output": 15.0,  "cached": 0.625,   "thinking": 3.50},
        "batch":    {"input": 1.25,   "output": 7.50,  "cached": 0.625,   "thinking": 1.75},
    },
}


def _extract_tokens(record: dict[str, Any]) -> tuple[str, int, int, int, int] | None:
    llm = record.get("llm")
    if not isinstance(llm, dict):
        return None
    model = str(llm.get("model") or "")
    if not model:
        return None
    meta = llm.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    return (
        model,
        int(meta.get("prompt_tokens") or 0),
        int(meta.get("completion_tokens") or 0),
        int(meta.get("cached_tokens") or 0),
        int(meta.get("thoughts_token_count") or 0),
    )


def aggregate(
    records: list[dict[str, Any]],
    dedup_key: str | None = None,
) -> dict[str, dict[str, int]]:
    """Aggregate token counts by model.

    dedup_key: if set, only count the first record for each (model, record[dedup_key]) pair.
    Use dedup_key='doc_id' for fix_events.py per-article output to avoid counting the same
    LLM call multiple times (once per invalid event in the article).
    """
    totals: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str]] = set()

    for record in records:
        t = _extract_tokens(record)
        if t is None:
            continue
        model, prompt, completion, cached, thinking = t

        if dedup_key is not None:
            key = (model, str(record.get(dedup_key) or ""))
            if key in seen:
                continue
            seen.add(key)

        if model not in totals:
            totals[model] = {"records": 0, "prompt": 0, "completion": 0, "cached": 0, "thinking": 0}
        totals[model]["records"] += 1
        totals[model]["prompt"] += prompt
        totals[model]["completion"] += completion
        totals[model]["cached"] += cached
        totals[model]["thinking"] += thinking

    return totals


def _fmt(n: int) -> str:
    if n == 0:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def report(label: str, totals: dict[str, dict[str, int]], *, batch: bool = False) -> None:
    tier = "batch" if batch else "standard"
    tier_label = " [batch pricing]" if batch else ""

    if not totals:
        print(f"\n=== Token report: {label}{tier_label} ===")
        print("  (no LLM records found)")
        return

    W = 36
    print(f"\n=== Token report: {label}{tier_label} ===")
    print(f"{'Model':<{W}} {'Recs':>6}  {'Input':>10}  {'Output':>10}  {'Cached':>10}  {'Thinking':>10}")
    print("-" * (W + 52))

    grand: dict[str, int] = {"records": 0, "prompt": 0, "completion": 0, "cached": 0, "thinking": 0}
    total_cost = 0.0
    has_unknown = False

    for model, d in sorted(totals.items()):
        print(
            f"{model:<{W}} {d['records']:>6}  {_fmt(d['prompt']):>10}  "
            f"{_fmt(d['completion']):>10}  {_fmt(d['cached']):>10}  {_fmt(d['thinking']):>10}"
        )
        for k in grand:
            grand[k] += d[k]

    if len(totals) > 1:
        print("-" * (W + 52))
        print(
            f"{'Total':<{W}} {grand['records']:>6}  {_fmt(grand['prompt']):>10}  "
            f"{_fmt(grand['completion']):>10}  {_fmt(grand['cached']):>10}  {_fmt(grand['thinking']):>10}"
        )

    print()
    for model, d in sorted(totals.items()):
        tiers = PRICING.get(model)
        rates = tiers.get(tier) if tiers else None
        if rates is None:
            print(f"  {model}: pricing unknown — update PRICING dict in costs.py")
            has_unknown = True
            continue
        cost = (
            d["prompt"] / 1e6 * rates["input"]
            + d["completion"] / 1e6 * rates["output"]
            + d["cached"] / 1e6 * rates["cached"]
            + d["thinking"] / 1e6 * rates["thinking"]
        )
        total_cost += cost
        parts = [
            f"{lbl}=${d[tok] / 1e6 * rates[rate_key]:.4f}"
            for lbl, tok, rate_key in [
                ("input", "prompt", "input"),
                ("output", "completion", "output"),
                ("cached", "cached", "cached"),
                ("thinking", "thinking", "thinking"),
            ]
            if d[tok]
        ]
        print(f"  {model}: {' + '.join(parts)} → ${cost:.4f}")

    suffix = " (known models only)" if has_unknown else ""
    print(f"  Total{suffix}: ${total_cost:.4f}")
    if grand["records"]:
        print(f"  Avg/row: ${total_cost / grand['records']:.6f}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, metavar="FILE")
    parser.add_argument(
        "--dedup-key",
        default=None,
        metavar="FIELD",
        help=(
            "Deduplicate by record field before summing (e.g. 'doc_id' for "
            "fix_events.py per-article output to avoid counting one LLM call multiple times)"
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Batch API pricing (~50%% off input/output) instead of standard pricing",
    )
    args = parser.parse_args()

    for path in args.files:
        records = read_jsonl(path)
        totals = aggregate(records, dedup_key=args.dedup_key)
        report(path.name, totals, batch=args.batch)


if __name__ == "__main__":
    main()
