"""Recover and visualize source-grounded annotations from AnnotationRecord JSONL."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v2.io_utils import (
    append_jsonl_row,
    iter_jsonl,
    resolve_path,
    write_jsonl,
)
from scripts.data.generation_v2.usage import add_usage


@dataclass(frozen=True)
class RecoveredSpan:
    article_id: str
    span_kind: str
    start: int
    end: int
    text: str
    label: str | None = None
    role: str | None = None
    location_type: str | None = None
    linked_event_id: str | None = None
    recovery_status: str = "exact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "span_kind": self.span_kind,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "label": self.label,
            "role": self.role,
            "location_type": self.location_type,
            "linked_event_id": self.linked_event_id,
            "recovery_status": self.recovery_status,
        }


def candidate_offsets(span_text: str, text: str) -> list[tuple[int, int]]:
    if not span_text:
        return []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    while True:
        found = text.find(span_text, cursor)
        if found < 0:
            break
        offsets.append((found, found + len(span_text)))
        cursor = found + 1
    return offsets


def recover_offsets(
    span_text: str,
    text: str,
    start: int,
    end: int,
    *,
    repair_window_chars: int = 200,
) -> tuple[int, int, str] | None:
    if 0 <= start < end <= len(text) and text[start:end] == span_text:
        return start, end, "exact"

    offsets = candidate_offsets(span_text, text)
    nearby = [
        offset for offset in offsets if abs(offset[0] - max(start, 0)) <= repair_window_chars
    ]
    if len(nearby) == 1:
        return nearby[0][0], nearby[0][1], "repaired_nearby"
    if len(offsets) == 1:
        return offsets[0][0], offsets[0][1], "repaired_unique"
    return None


def recover_record(record: dict[str, Any], *, repair_window_chars: int = 200) -> list[dict[str, Any]]:
    article_id = str(record.get("id") or record.get("article_id") or "")
    text = str(record.get("text") or "")
    recovered: list[RecoveredSpan] = []

    for event_index, event in enumerate(record.get("events") or []):
        event_id = f"{article_id}::event::{event_index}"
        event_offsets = recover_offsets(
            str(event.get("text") or event.get("trigger_text") or ""),
            text,
            int(event.get("start", event.get("start_char", -1))),
            int(event.get("end", event.get("end_char", -1))),
            repair_window_chars=repair_window_chars,
        )
        if event_offsets is not None:
            start, end, status = event_offsets
            recovered.append(
                RecoveredSpan(
                    article_id=article_id,
                    span_kind="event",
                    start=start,
                    end=end,
                    text=text[start:end],
                    label=str(event.get("event_type") or ""),
                    linked_event_id=event_id,
                    recovery_status=status,
                )
            )

        for argument in event.get("arguments") or []:
            argument_offsets = recover_offsets(
                str(argument.get("text") or ""),
                text,
                int(argument.get("start", argument.get("start_char", -1))),
                int(argument.get("end", argument.get("end_char", -1))),
                repair_window_chars=repair_window_chars,
            )
            if argument_offsets is None:
                continue
            start, end, status = argument_offsets
            recovered.append(
                RecoveredSpan(
                    article_id=article_id,
                    span_kind="argument",
                    start=start,
                    end=end,
                    text=text[start:end],
                    role=str(argument.get("role") or ""),
                    location_type=str(argument.get("location_type") or "") or None,
                    linked_event_id=event_id,
                    recovery_status=status,
                )
            )

    for location in record.get("locations") or []:
        location_offsets = recover_offsets(
            str(location.get("text") or ""),
            text,
            int(location.get("start", location.get("start_char", -1))),
            int(location.get("end", location.get("end_char", -1))),
            repair_window_chars=repair_window_chars,
        )
        if location_offsets is None:
            continue
        start, end, status = location_offsets
        recovered.append(
            RecoveredSpan(
                article_id=article_id,
                span_kind="location",
                start=start,
                end=end,
                text=text[start:end],
                location_type=str(location.get("location_type") or "") or None,
                recovery_status=status,
            )
        )

    return [span.to_dict() for span in sorted(recovered, key=lambda item: (item.start, item.end, item.span_kind))]


RECOVERY_RANK = {"exact": 2, "repaired_nearby": 1, "repaired_unique": 0}


def article_id(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("article_id") or "")


def overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["start"]) < int(right["end"]) and int(right["start"]) < int(left["end"])


def span_score(span: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        RECOVERY_RANK.get(str(span.get("_recovery_status")), -1),
        int(span["end"]) - int(span["start"]),
        int(span.get("_support", 1)),
        -int(span.get("_source_order", 0)),
    )


def select_best_non_overlapping(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for span in sorted(
        spans,
        key=lambda item: (
            -span_score(item)[0],
            -span_score(item)[1],
            -span_score(item)[2],
            int(item.get("_source_order", 0)),
        ),
    ):
        if any(overlaps(span, existing) for existing in selected):
            continue
        selected.append(span)
    return sorted(selected, key=lambda item: (int(item["start"]), int(item["end"])))


def strip_internal_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def legacy_location(location: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": location["text"],
        "location_type": location.get("location_type") or "other",
        "start_char": int(location["start"]),
        "end_char": int(location["end"]),
        **({"support": int(location["_support"])} if "_support" in location else {}),
    }


def legacy_argument(argument: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": argument.get("role") or "",
        **legacy_location(argument),
    }


def legacy_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": event.get("event_type") or "",
        "trigger_text": event["text"],
        "start_char": int(event["start"]),
        "end_char": int(event["end"]),
        "arguments": [legacy_argument(argument) for argument in event.get("arguments") or []],
        "rationale": "",
        **({"support": int(event["_support"])} if "_support" in event else {}),
        "window_indices": sorted(set(event.get("_window_indices", []))),
        "core_window_indices": sorted(set(event.get("_core_window_indices", []))),
    }


def normalize_argument(
    argument: dict[str, Any],
    text: str,
    *,
    record_index: int,
    repair_window_chars: int,
) -> dict[str, Any] | None:
    offsets = recover_offsets(
        str(argument.get("text") or ""),
        text,
        int(argument.get("start", argument.get("start_char", -1))),
        int(argument.get("end", argument.get("end_char", -1))),
        repair_window_chars=repair_window_chars,
    )
    if offsets is None:
        return None
    start, end, status = offsets
    return {
        "role": str(argument.get("role") or ""),
        "start": start,
        "end": end,
        "text": text[start:end],
        "location_type": str(argument.get("location_type") or "other"),
        "_recovery_status": status,
        "_support": 1,
        "_source_order": record_index,
        "_window_indices": list(argument.get("window_indices") or []),
    }


def normalize_event(
    event: dict[str, Any],
    text: str,
    *,
    record_index: int,
    repair_window_chars: int,
) -> dict[str, Any] | None:
    offsets = recover_offsets(
        str(event.get("text") or event.get("trigger_text") or ""),
        text,
        int(event.get("start", event.get("start_char", -1))),
        int(event.get("end", event.get("end_char", -1))),
        repair_window_chars=repair_window_chars,
    )
    if offsets is None:
        return None
    start, end, status = offsets
    arguments = [
        normalized
        for argument in event.get("arguments") or []
        if (
            normalized := normalize_argument(
                argument,
                text,
                record_index=record_index,
                repair_window_chars=repair_window_chars,
            )
        )
        is not None
    ]
    return {
        "event_type": str(event.get("event_type") or ""),
        "start": start,
        "end": end,
        "text": text[start:end],
        "arguments": arguments,
        "_recovery_status": status,
        "_support": 1,
        "_source_order": record_index,
        "_window_indices": list(event.get("window_indices") or []),
        "_core_window_indices": list(event.get("core_window_indices") or []),
    }


def normalize_location(
    location: dict[str, Any],
    text: str,
    *,
    record_index: int,
    repair_window_chars: int,
) -> dict[str, Any] | None:
    offsets = recover_offsets(
        str(location.get("text") or ""),
        text,
        int(location.get("start", location.get("start_char", -1))),
        int(location.get("end", location.get("end_char", -1))),
        repair_window_chars=repair_window_chars,
    )
    if offsets is None:
        return None
    start, end, status = offsets
    return {
        "start": start,
        "end": end,
        "text": text[start:end],
        "location_type": str(location.get("location_type") or "other"),
        "_recovery_status": status,
        "_support": 1,
        "_source_order": record_index,
        "_window_indices": list(location.get("window_indices") or []),
    }


def merge_arguments(arguments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for argument in arguments:
        key = (
            int(argument["start"]),
            int(argument["end"]),
            str(argument.get("role") or ""),
            str(argument.get("location_type") or ""),
        )
        if key not in grouped or span_score(argument) > span_score(grouped[key]):
            grouped[key] = {**argument, "_support": grouped.get(key, {}).get("_support", 0) + 1}
        else:
            grouped[key]["_support"] = int(grouped[key].get("_support", 1)) + 1
        grouped[key]["_window_indices"] = sorted(
            set(grouped[key].get("_window_indices", [])) | set(argument.get("_window_indices", []))
        )
    return select_best_non_overlapping(list(grouped.values()))


def merge_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], dict[str, Any]] = {}
    grouped_arguments: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = (int(event["start"]), int(event["end"]), str(event.get("event_type") or ""))
        grouped_arguments[key].extend(event.get("arguments") or [])
        if key not in grouped or span_score(event) > span_score(grouped[key]):
            grouped[key] = {**event, "_support": grouped.get(key, {}).get("_support", 0) + 1}
        else:
            grouped[key]["_support"] = int(grouped[key].get("_support", 1)) + 1
        grouped[key]["_window_indices"] = sorted(
            set(grouped[key].get("_window_indices", [])) | set(event.get("_window_indices", []))
        )
        grouped[key]["_core_window_indices"] = sorted(
            set(grouped[key].get("_core_window_indices", [])) | set(event.get("_core_window_indices", []))
        )

    candidates: list[dict[str, Any]] = []
    for key, event in grouped.items():
        candidates.append({**event, "arguments": merge_arguments(grouped_arguments[key])})
    return select_best_non_overlapping(candidates)


def merge_locations(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], dict[str, Any]] = {}
    for location in locations:
        key = (
            int(location["start"]),
            int(location["end"]),
            str(location.get("location_type") or ""),
        )
        if key not in grouped or span_score(location) > span_score(grouped[key]):
            grouped[key] = {**location, "_support": grouped.get(key, {}).get("_support", 0) + 1}
        else:
            grouped[key]["_support"] = int(grouped[key].get("_support", 1)) + 1
        grouped[key]["_window_indices"] = sorted(
            set(grouped[key].get("_window_indices", [])) | set(location.get("_window_indices", []))
        )
    return select_best_non_overlapping(list(grouped.values()))


def merge_usage(metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metadata in metadata_rows:
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
        for component, values in usage.items():
            merged[component] = add_usage(merged.get(component), values)
    return merged


def usage_totals(usage: dict[str, Any]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "thoughts_tokens": 0}
    if "total" in usage:
        return add_usage(None, usage.get("total"))
    for component, values in usage.items():
        if component == "total":
            continue
        totals = add_usage(totals, values)
    return totals


def source_payload(record: dict[str, Any], text: str) -> dict[str, Any]:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    return {
        "title": source.get("title") or record.get("title") or "",
        "text": source.get("text") or text,
        "source_url": source.get("source_url") or record.get("source_url") or "",
        "publish_date": source.get("publish_date") or record.get("publish_date") or "",
    }


def llm_payload(records: list[dict[str, Any]], recovered_count: int, source_error_count: int) -> dict[str, Any]:
    metadata_rows = [record.get("metadata") or {} for record in records]
    base = dict(metadata_rows[0]) if metadata_rows else {}
    task_keys = [
        str(metadata.get("task_key"))
        for metadata in metadata_rows
        if metadata.get("task_key") is not None
    ]
    window_indices = sorted(
        {
            int(index)
            for metadata in metadata_rows
            for index in (metadata.get("window_indices") or [])
        }
    )
    settings = base.get("prompt_settings") if isinstance(base.get("prompt_settings"), dict) else {}
    usage = merge_usage(metadata_rows)
    totals = usage_totals(usage)
    return {
        "model": base.get("annotation_model") or "",
        "metadata": {
            "cached_tokens": totals["cached_input_tokens"],
            "completion_tokens": totals["output_tokens"],
            "prompt_tokens": totals["input_tokens"],
            "thoughts_token_count": totals["thoughts_tokens"],
            "generation_v2": {
                **base,
                "task_key": None,
                "merged_task_keys": task_keys,
                "window_indices": window_indices,
                "usage": usage,
                "recovery_status": "ok",
                "recovered_count": recovered_count,
                "source_row_count": len(records),
                "source_error_count": source_error_count,
            },
        },
        "output_mode": "events-with-args-and-locations",
        "pipeline": {
            "mode": "generation_v2",
            "enable_verifier": any(metadata.get("verifier_model") for metadata in metadata_rows),
            "batch_api": base.get("api_mode") == "batch",
            "window_strategy": "sentence_adaptive_overlap",
            "prompt_settings": settings,
        },
    }


def merge_article_records(
    records: list[dict[str, Any]],
    *,
    repair_window_chars: int = 200,
    source_error_count: int = 0,
) -> dict[str, Any]:
    text = str(records[0].get("text") or "")
    merged_events = merge_events(
        [
            normalized
            for record_index, record in enumerate(records)
            for event in record.get("events") or []
            if (
                normalized := normalize_event(
                    event,
                    text,
                    record_index=record_index,
                    repair_window_chars=repair_window_chars,
                )
            )
            is not None
        ]
    )
    merged_locations = merge_locations(
        [
            normalized
            for record_index, record in enumerate(records)
            for location in record.get("locations") or []
            if (
                normalized := normalize_location(
                    location,
                    text,
                    record_index=record_index,
                    repair_window_chars=repair_window_chars,
                )
            )
            is not None
        ]
    )
    legacy_events = [legacy_event(event) for event in merged_events]
    legacy_locations = [legacy_location(location) for location in merged_locations]
    row = {
        "id": article_id(records[0]),
        "status": "ok",
        "source": source_payload(records[0], text),
        "events": legacy_events,
        "locations": legacy_locations,
    }
    row["llm"] = llm_payload(
        records,
        recovered_count=len(legacy_events) + len(legacy_locations),
        source_error_count=source_error_count,
    )
    return row


def recover_file(input_path: Path, output_path: Path, *, repair_window_chars: int, overwrite: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    write_jsonl(output_path, [], overwrite=overwrite)
    records_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    article_ids: list[str] = []
    for record in iter_jsonl(input_path):
        current_article_id = article_id(record)
        if current_article_id not in records_by_article and current_article_id not in errors_by_article:
            article_ids.append(current_article_id)
        is_error = (
            record.get("status") == "error"
            or (record.get("metadata") or {}).get("verifier_decision") == "error"
        )
        target = errors_by_article if is_error else records_by_article
        target[current_article_id].append(record)

    for current_article_id in tqdm(article_ids, desc="Recovering"):
        successful_records = records_by_article.get(current_article_id, [])
        if successful_records:
            row = merge_article_records(
                successful_records,
                repair_window_chars=repair_window_chars,
                source_error_count=len(errors_by_article.get(current_article_id, [])),
            )
            rows.append(row)
            append_jsonl_row(output_path, row)
            continue

        record = errors_by_article[current_article_id][0]
        metadata = record.get("metadata") or {}
        text = str(record.get("text") or "")
        row = {
            "id": record.get("id"),
            "status": "error",
            "error": record.get("error") or "source record failed before recovery",
            "source": source_payload(record, text),
            "events": [],
            "locations": [],
            "llm": {
                "model": metadata.get("annotation_model") or "",
                "metadata": {
                    "generation_v2": {
                        **metadata,
                        "recovery_status": "source_error",
                        "recovered_count": 0,
                        "source_error_count": len(errors_by_article[current_article_id]),
                    }
                },
                "output_mode": "events-with-args-and-locations",
                "pipeline": {"mode": "generation_v2"},
            },
        }
        rows.append(row)
        append_jsonl_row(output_path, row)
    return rows


def render_record_html(record: dict[str, Any]) -> str:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    text = str(record.get("text") or source.get("text") or "")
    spans = list(record.get("recovered_annotations", []) or [])
    if not spans:
        for event_index, event in enumerate(record.get("events") or []):
            spans.append(
                {
                    "span_kind": "event",
                    "start": event.get("start", event.get("start_char", -1)),
                    "end": event.get("end", event.get("end_char", -1)),
                    "label": event.get("event_type"),
                    "recovery_status": "exact",
                    "linked_event_id": f"{record.get('id')}::event::{event_index}",
                }
            )
            for argument in event.get("arguments") or []:
                spans.append(
                    {
                        "span_kind": "argument",
                        "start": argument.get("start", argument.get("start_char", -1)),
                        "end": argument.get("end", argument.get("end_char", -1)),
                        "role": argument.get("role"),
                        "location_type": argument.get("location_type"),
                        "recovery_status": "exact",
                        "linked_event_id": f"{record.get('id')}::event::{event_index}",
                    }
                )
        for location in record.get("locations") or []:
            spans.append(
                {
                    "span_kind": "location",
                    "start": location.get("start", location.get("start_char", -1)),
                    "end": location.get("end", location.get("end_char", -1)),
                    "location_type": location.get("location_type"),
                    "recovery_status": "exact",
                }
            )
    spans = [
        span
        for span in spans
        if 0 <= int(span.get("start", -1)) < int(span.get("end", -1)) <= len(text)
    ]
    spans.sort(key=lambda span: (int(span["start"]), -int(span["end"])))

    chunks: list[str] = []
    cursor = 0
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        if start < cursor:
            continue
        chunks.append(html.escape(text[cursor:start]))
        title = " | ".join(
            str(value)
            for value in (
                span.get("span_kind"),
                span.get("label") or span.get("role") or span.get("location_type"),
                span.get("recovery_status"),
            )
            if value
        )
        chunks.append(
            f'<mark class="{html.escape(str(span.get("span_kind")))}" '
            f'title="{html.escape(title)}">{html.escape(text[start:end])}</mark>'
        )
        cursor = end
    chunks.append(html.escape(text[cursor:]))
    return f"<section><h2>{html.escape(str(record.get('id')))}</h2><p>{''.join(chunks)}</p></section>"


def write_html(records: list[dict[str, Any]], html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(render_record_html(record) for record in records)
    html_path.write_text(
        """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body { font-family: Georgia, serif; max-width: 1100px; margin: 40px auto; line-height: 1.55; }
section { border-bottom: 1px solid #ddd; padding: 24px 0; }
mark { padding: 0.1em 0.25em; border-radius: 0.25em; }
mark.event { background: #ffd166; }
mark.argument { background: #9ad0f5; }
mark.location { background: #b8e6b1; }
p { white-space: pre-wrap; }
</style>
</head>
<body>
"""
        + body
        + "\n</body>\n</html>\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover source-grounded annotations.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--repair-window-chars", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = resolve_path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"Output already exists, skipping: {output_path}")
        if args.html is not None:
            html_path = resolve_path(args.html)
            if not html_path.exists():
                write_html(list(iter_jsonl(output_path)), html_path)
        return 0
    records = recover_file(
        resolve_path(args.input),
        output_path,
        repair_window_chars=args.repair_window_chars,
        overwrite=True,
    )
    if args.html is not None:
        write_html(records, resolve_path(args.html))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
