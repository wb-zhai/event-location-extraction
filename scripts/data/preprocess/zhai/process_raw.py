"""Convert Zhai raw tag rows into manual_fixes-style JSONL records."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = REPO_ROOT / "dataset/zhai/raw/sample_100_with_tags_raw.json"
DEFAULT_OUTPUT = REPO_ROOT / "dataset/zhai/raw/sample_100_with_tags_raw.jsonl"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Zhai raw tag rows into grouped manual_fixes-style JSONL."
    )
    parser.add_argument("input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return [row for row in data if isinstance(row, dict)]


def normalize_title_text(title: str, text: str) -> tuple[str, str]:
    title = title.strip()
    text = text.strip()
    if not title or not text:
        return title, text

    title_folded = " ".join(title.casefold().split())
    paragraphs = [part.strip() for part in text.split("\n\n")]
    if paragraphs:
        first_paragraph_folded = " ".join(paragraphs[0].casefold().split())
        if first_paragraph_folded == title_folded:
            text = "\n\n".join(paragraphs[1:]).strip()
            return title, text

    if text.casefold().startswith(title.casefold()):
        suffix = text[len(title) :].lstrip(" \n\r\t-:|")
        if suffix:
            text = suffix
    return title, text


def parse_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def find_all_casefold(text: str, needle: str) -> list[int]:
    text_folded = text.casefold()
    needle_folded = needle.casefold().strip()
    if not needle_folded:
        return []
    hits: list[int] = []
    start = 0
    while True:
        index = text_folded.find(needle_folded, start)
        if index == -1:
            return hits
        hits.append(index)
        start = index + 1


def resolve_span(text: str, label: str, hint_start: int | None) -> tuple[int, int, str] | None:
    label = label.strip()
    if not label:
        return None

    candidates = find_all_casefold(text, label)
    if candidates:
        if hint_start is None:
            start = candidates[0]
        else:
            start = min(candidates, key=lambda index: abs(index - hint_start))
        end = start + len(label)
        return start, end, text[start:end]

    return None


def append_argument(event: dict[str, Any], argument: dict[str, Any]) -> None:
    key = (argument["role"], argument["start_char"], argument["end_char"])
    seen = event.setdefault("_argument_keys", set())
    if key in seen:
        return
    seen.add(key)
    event["arguments"].append(argument)


def add_event(
    grouped_events: OrderedDict[tuple[str, int, int], dict[str, Any]],
    event_type: str,
    trigger_span: tuple[int, int, str],
    location_span: tuple[int, int, str] | None,
) -> None:
    start_char, end_char, trigger_text = trigger_span
    event_key = (event_type, start_char, end_char)
    event = grouped_events.get(event_key)
    if event is None:
        event = {
            "event_type": event_type,
            "trigger_text": trigger_text,
            "start_char": start_char,
            "end_char": end_char,
            "arguments": [],
            "_argument_keys": set(),
        }
        grouped_events[event_key] = event

    if location_span is not None:
        location_start, location_end, location_text = location_span
        append_argument(
            event,
            {
                "role": "location",
                "text": location_text,
                "start_char": location_start,
                "end_char": location_end,
            },
        )


def build_record(
    uri: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_row = rows[0]
    title, text = normalize_title_text(
        str(first_row.get("title") or ""),
        str(first_row.get("body") or ""),
    )

    grouped_events: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()
    for row in rows:
        event_type = str(row.get("risk_factor_tag") or "").strip()
        if not event_type:
            continue

        trigger_span = resolve_span(
            text,
            event_type,
            parse_int(row.get("risk_factor_position_start")),
        )
        if trigger_span is None:
            continue

        location_tag = str(row.get("location_tag") or "").strip()
        location_span = resolve_span(
            text,
            location_tag,
            parse_int(row.get("location_position_start")),
        )
        add_event(grouped_events, event_type, trigger_span, location_span)

    events: list[dict[str, Any]] = []
    for event in grouped_events.values():
        event["arguments"].sort(key=lambda arg: (arg["start_char"], arg["end_char"], arg["role"]))
        event.pop("_argument_keys", None)
        events.append(event)

    events.sort(key=lambda event: (event["start_char"], event["end_char"], event["event_type"]))
    return {
        "id": uri,
        "status": "ok",
        "source": {
            "title": title,
            "text": text,
            "source_url": "",
            "publish_date": "",
        },
        "events": events,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)

    grouped_rows: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        uri = str(row.get("uri") or "").strip()
        if not uri:
            continue
        grouped_rows.setdefault(uri, []).append(row)

    records = [
        build_record(uri=uri, rows=article_rows)
        for uri, article_rows in grouped_rows.items()
    ]
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
