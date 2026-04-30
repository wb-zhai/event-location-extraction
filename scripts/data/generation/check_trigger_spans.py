from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = Path("dataset/manual/manual_fixes.jsonl")


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def iter_records(path: Path) -> list[tuple[str, dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Record at {path}:{line_no} is not an object.")
                records.append((str(line_no), record))
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("records", "samples", "articles", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Input JSON must contain an object or list of objects: {path}")

    records = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Record at {path}[{index}] is not an object.")
        records.append((str(index + 1), record))
    return records


def record_text(record: dict[str, Any]) -> str:
    source = record.get("source")
    if isinstance(source, dict):
        return str(source.get("text") or record.get("text") or record.get("body") or "")
    return str(record.get("text") or record.get("body") or "")


def event_offsets(event: dict[str, Any]) -> tuple[int | None, int | None]:
    start = event.get("start_char", event.get("start"))
    end = event.get("end_char", event.get("end"))
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return None, None


def check_file(path: Path, max_examples: int) -> int:
    mismatches = 0
    checked = 0
    records = iter_records(path)

    for record_ref, record in records:
        text = record_text(record)
        record_id = record.get("id") or record.get("doc_id") or record_ref
        events = record.get("events") or []
        if not isinstance(events, list):
            print(f"{path}:{record_ref} id={record_id}: events is not a list")
            mismatches += 1
            continue

        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                print(
                    f"{path}:{record_ref} id={record_id} event[{event_index}]: "
                    "event is not an object"
                )
                mismatches += 1
                continue

            trigger_text = event.get("trigger_text")
            start, end = event_offsets(event)
            checked += 1

            if not isinstance(trigger_text, str) or start is None or end is None:
                actual = None
                ok = False
            elif 0 <= start <= end <= len(text):
                actual = text[start:end]
                ok = actual == trigger_text
            else:
                actual = None
                ok = False

            if ok:
                continue

            mismatches += 1
            if mismatches > max_examples:
                continue

            print(
                f"{path}:{record_ref} id={record_id} event[{event_index}] "
                f"{event.get('event_type', '')}"
            )
            print(f"  offsets: start={start!r} end={end!r} text_len={len(text)}")
            print(f"  expected: {trigger_text!r}")
            print(f"  actual:   {actual!r}")

    print(f"{path}: checked {checked} trigger spans in {len(records)} records")
    if mismatches:
        hidden = max(0, mismatches - max_examples)
        suffix = f" ({hidden} more not shown)" if hidden else ""
        print(f"{path}: {mismatches} mismatch(es){suffix}")
    else:
        print(f"{path}: all trigger spans match")
    return mismatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that event trigger offsets slice back to trigger_text."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[DEFAULT_INPUT],
        help="JSON or JSONL files to check. Defaults to dataset/manual/manual_fixes.jsonl.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum mismatch examples to print across each file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_mismatches = 0
    for input_path in args.inputs:
        total_mismatches += check_file(resolve_path(input_path), args.max_examples)
    return 1 if total_mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
