"""Report completion and correction stats for an exported event-annotation JSONL
file (the output of `events_argilla.py export`).

Usage:
    python -m scripts.annotations.events_report --input dataset/manual/event-extraction/matrix_5M.sample_1000.3.1pro.2label_prompt.extracted.bona.v4.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def canonicalize_events(events: list[dict[str, Any]] | None) -> list[tuple]:
    """Order-insensitive representation of an event list, so reordering the
    same events isn't counted as a correction."""
    if not events:
        return []
    return sorted(tuple(sorted(ev.items())) for ev in events if isinstance(ev, dict))


def is_changed(annotation: dict[str, Any]) -> bool:
    events = annotation.get("events")
    original_events = (annotation.get("original_annotation") or {}).get("events")
    return canonicalize_events(events) != canonicalize_events(original_events)


def report(records: list[dict[str, Any]]) -> None:
    total = len(records)

    completed_by_user: Counter[str] = Counter()
    changed_by_user: Counter[str] = Counter()
    completed = 0
    changed = 0

    for rec in records:
        annotation = rec.get("annotation") or {}
        annotated_by = annotation.get("annotated_by")
        if not annotated_by:
            continue
        completed += 1
        completed_by_user[annotated_by] += 1
        if is_changed(annotation):
            changed += 1
            changed_by_user[annotated_by] += 1

    print(f"Total records:     {total}")
    print(f"Completed:         {completed} ({completed / total:.0%})")
    print(f"Not completed:     {total - completed}")
    print()
    print("Completed by user:")
    for user, count in completed_by_user.most_common():
        print(f"  {user:<20} {count}")
    print()
    print(f"Corrected/changed vs original annotation: {changed} ({changed / completed:.0%} of completed)" if completed else "Corrected/changed vs original annotation: 0")
    print()
    print("Corrected/changed by user:")
    for user, count in changed_by_user.most_common():
        pct = count / completed_by_user[user]
        print(f"  {user:<20} {count} / {completed_by_user[user]} ({pct:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=str, help="Exported event-annotation JSONL file")
    args = parser.parse_args()

    records = iter_jsonl(Path(args.input))
    report(records)


if __name__ == "__main__":
    main()
