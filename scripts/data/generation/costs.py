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
# Each tier has "low"/"high" rates for prompts <=TIER_THRESHOLD / >TIER_THRESHOLD tokens
# (models without a context-size price break just repeat the same rates in both).
# Thinking tokens are billed at the output rate (no separate thinking price).
TIER_THRESHOLD = 200_000

PRICING: dict[str, dict[str, dict[str, dict[str, float]]]] = {
    "gemini-2.5-flash": {
        "standard": {
            "low":  {"input": 0.30,  "output": 2.50, "cached": 0.03,  "thinking": 2.50},
            "high": {"input": 0.30,  "output": 2.50, "cached": 0.03,  "thinking": 2.50},
        },
        "batch": {
            "low":  {"input": 0.15,  "output": 1.25, "cached": 0.03,  "thinking": 1.25},
            "high": {"input": 0.15,  "output": 1.25, "cached": 0.03,  "thinking": 1.25},
        },
    },
    "gemini-2.5-pro": {
        "standard": {
            "low":  {"input": 1.25,  "output": 10.0, "cached": 0.125, "thinking": 10.0},
            "high": {"input": 2.50,  "output": 15.0, "cached": 0.25,  "thinking": 15.0},
        },
        "batch": {
            "low":  {"input": 0.625, "output": 5.0,  "cached": 0.125, "thinking": 5.0},
            "high": {"input": 1.25,  "output": 7.50, "cached": 0.25,  "thinking": 7.50},
        },
    },
    "gemini-3.1-pro-preview": {
        "standard": {
            "low":  {"input": 2.00, "output": 12.0, "cached": 0.20, "thinking": 12.0},
            "high": {"input": 4.00, "output": 18.0, "cached": 0.40, "thinking": 18.0},
        },
        "batch": {
            "low":  {"input": 1.00, "output": 6.0,  "cached": 0.20, "thinking": 6.0},
            "high": {"input": 2.00, "output": 9.0,  "cached": 0.40, "thinking": 9.0},
        },
    },
    "gemini-3-flash-preview": {
        "standard": {
            "low":  {"input": 0.50, "output": 3.00, "cached": 0.05, "thinking": 3.00},
            "high": {"input": 0.50, "output": 3.00, "cached": 0.05, "thinking": 3.00},
        },
        "batch": {
            "low":  {"input": 0.25, "output": 1.50, "cached": 0.05, "thinking": 1.50},
            "high": {"input": 0.25, "output": 1.50, "cached": 0.05, "thinking": 1.50},
        },
    },
}


def _extract_tokens(record: dict[str, Any]) -> tuple[str, int, int, int, int] | None:
    # generate.py/fix_events.py write "generation_llm" now (to avoid clobbering an
    # upstream pipeline's own "llm" field on preserved rows); fall back to "llm"
    # for older output files.
    llm = record.get("generation_llm", record.get("llm"))
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


def _new_stats() -> dict[str, int]:
    return {"records": 0, "prompt": 0, "completion": 0, "cached": 0, "thinking": 0}


def aggregate(
    records: list[dict[str, Any]],
    dedup_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate token counts by model, split into "low"/"high" price-tier buckets.

    The >200k pricing tier applies per LLM call based on that call's own prompt size, so
    each record is bucketed by its own prompt token count before summing, in addition to
    the overall per-model totals used for the report table.

    dedup_key: if set, only count the first record for each (model, record[dedup_key]) pair.
    Use dedup_key='doc_id' for fix_events.py per-article output to avoid counting the same
    LLM call multiple times (once per invalid event in the article).
    """
    totals: dict[str, dict[str, Any]] = {}
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
            totals[model] = {**_new_stats(), "low": _new_stats(), "high": _new_stats()}
        bucket = totals[model]["high" if prompt > TIER_THRESHOLD else "low"]
        for target in (totals[model], bucket):
            target["records"] += 1
            target["prompt"] += prompt
            target["completion"] += completion
            target["cached"] += cached
            target["thinking"] += thinking

    return totals


def _fmt(n: int) -> str:
    if n == 0:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def report(label: str, totals: dict[str, dict[str, Any]], *, batch: bool = False) -> None:
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
        rates_by_bucket = tiers.get(tier) if tiers else None
        if rates_by_bucket is None:
            print(f"  {model}: pricing unknown — update PRICING dict in costs.py")
            has_unknown = True
            continue

        cost = 0.0
        parts = []
        for lbl, tok, rate_key in [
            ("input", "prompt", "input"),
            ("output", "completion", "output"),
            ("cached", "cached", "cached"),
            ("thinking", "thinking", "thinking"),
        ]:
            type_cost = sum(
                d[bucket][tok] / 1e6 * rates_by_bucket[bucket][rate_key] for bucket in ("low", "high")
            )
            cost += type_cost
            if d[tok]:
                parts.append(f"{lbl}=${type_cost:.4f}")
        total_cost += cost

        tier_note = f" ({_fmt(d['high']['prompt'])} tokens at >200k rate)" if d["high"]["records"] else ""
        print(f"  {model}: {' + '.join(parts)} → ${cost:.4f}{tier_note}")

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
