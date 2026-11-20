#!/usr/bin/env python3
"""Merge validated clean records with successfully fixed events into a final output JSONL.

Pipeline:
  generate.py  →  validate.py        →  fix_events.py  →  merge.py    →  to_sft.py
                  .clean.jsonl          .fixed.jsonl       .final.jsonl
                  .invalid.jsonl  ← inspect              ← inspect
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from costs import aggregate, report  # noqa: E402


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_fixed_events(
    path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Parse fix_events.py output.

    Returns (fixed_by_doc_id, all_raw_records).
    Only events where status=ok, decision=fixed, revalidation.valid=True are merged.
    """
    fixed: dict[str, list[dict[str, Any]]] = {}
    raw: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        raw.append(record)
        if record.get("status") != "ok":
            continue
        if record.get("decision") != "fixed":
            continue
        if not (record.get("revalidation") or {}).get("valid"):
            continue
        doc_id = str(record.get("doc_id") or "")
        fixed_event = record.get("fixed_event")
        if fixed_event:
            fixed.setdefault(doc_id, []).append(fixed_event)
    return fixed, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True, help="clean.jsonl from validate.py")
    parser.add_argument(
        "--fixed", type=Path, default=None,
        help="fixed.jsonl from fix_events.py (optional; omit if no invalid events to merge)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.clean.resolve() == args.output.resolve():
        raise SystemExit("--output must differ from --clean")

    fixed_by_doc: dict[str, list[dict[str, Any]]] = {}
    fixed_raw: list[dict[str, Any]] = []
    if args.fixed is not None:
        if not args.fixed.exists():
            raise SystemExit(f"--fixed file not found: {args.fixed}")
        fixed_by_doc, fixed_raw = load_fixed_events(args.fixed)

    clean_records = iter_jsonl(args.clean)

    docs_processed = 0
    events_merged = 0
    docs_with_fixes = 0
    total_events = 0
    event_counts: list[int] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for record in clean_records:
            if record.get("status") == "ok":
                doc_id = str(record.get("id") or "")
                fixed_events = fixed_by_doc.get(doc_id, [])
                if fixed_events:
                    annotation = record.get("annotation") or {}
                    merged = (annotation.get("events") or []) + fixed_events
                    record = {**record, "annotation": {**annotation, "events": merged}}
                    events_merged += len(fixed_events)
                    docs_with_fixes += 1
                n = len((record.get("annotation") or {}).get("events") or [])
                event_counts.append(n)
                total_events += n
                docs_processed += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    avg = total_events / docs_processed if docs_processed else 0.0
    docs_zero = sum(1 for c in event_counts if c == 0)

    print("\n=== Merge summary ===")
    print(f"Documents processed  : {docs_processed}")
    print(f"Total events         : {total_events}")
    print(f"Avg events/article   : {avg:.2f}")
    print(f"Docs with 0 events   : {docs_zero}")
    print(f"Fixed events merged  : {events_merged}  (from {docs_with_fixes} documents)")
    print(f"Output               : {args.output}")

    gen_totals = aggregate(clean_records)
    if gen_totals:
        report(f"{args.clean.name} (generation)", gen_totals)

    if fixed_raw:
        # dedup by doc_id: per-article mode creates one record per invalid event,
        # but they all share the same LLM call per article
        fix_totals = aggregate(fixed_raw, dedup_key="doc_id")
        if fix_totals:
            report(f"{args.fixed.name} (fix, deduped by doc_id)", fix_totals)


if __name__ == "__main__":
    main()
