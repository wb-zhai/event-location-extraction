#!/usr/bin/env python3
"""Validate v3 generation output JSONL.

Checks per event:
  - event_type is in the ontology (ontologies/zhai/ontology.events.json)
  - grounding_quote is a non-empty exact substring of source.text
  - event_location_text (when not "not_stated") is an exact substring of source.text
  - event_time_text (when not "not_stated") is an exact substring of source.text
  - time_status / severity / modality are valid enum values
  - no duplicate events per document (same event_type + grounding_quote)

Emits:
  <output_stem>.clean.jsonl   — all status=ok records, with only valid events retained
  <output_stem>.invalid.jsonl — one line per invalid event, with doc context + errors

Usage:
    python validate_zhai_v3.py --input silver.jsonl --output-stem silver.validated
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_PATH = REPO_ROOT / "ontologies" / "zhai" / "risk.label.description.training.json"
SYSTEM_PROMPT_PATH = REPO_ROOT / "scripts" / "data" / "generation_v3" / "prompts" / "teacher" / "system_prompt.txt"

VALID_TIME_STATUS = {"past", "ongoing", "forecast", "not_stated"}
VALID_SEVERITY = {"low", "medium", "high", "extreme", "not_stated"}
VALID_MODALITY = {"asserted", "reported", "projected", "hedged"}
VALID_DOC_RELEVANCE = {"relevant", "not_relevant"}


def load_ontology_labels(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data["events"].keys())


def load_prompt_labels(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"<allowed_event_types>(.*?)</allowed_event_types>", text, re.DOTALL)
    if not m:
        raise ValueError("Could not find <allowed_event_types> in system_prompt.txt")
    return {line.strip() for line in m.group(1).splitlines() if line.strip()}


def check_event(
    event: dict[str, Any],
    source_text: str,
    ontology: set[str],
    seen_keys: set[tuple[str, str]],
) -> tuple[list[str], set[str]]:
    """Return (error_messages, error_categories). Categories: off_ontology, grounding, enum, dedup."""
    errors: list[str] = []
    categories: set[str] = set()

    event_type = event.get("event_type", "")
    if event_type not in ontology:
        errors.append(f"event_type {event_type!r} not in ontology")
        categories.add("off_ontology")

    gq = event.get("grounding_quote", "")
    if not gq:
        errors.append("grounding_quote is empty")
        categories.add("grounding")
    elif gq not in source_text:
        errors.append("grounding_quote not found in source.text")
        categories.add("grounding")

    loc_text = event.get("event_location_text", "")
    if loc_text and loc_text != "not_stated" and loc_text not in source_text:
        errors.append("event_location_text not found in source.text")
        categories.add("grounding")

    time_text = event.get("event_time_text", "")
    if time_text and time_text != "not_stated" and time_text not in source_text:
        errors.append("event_time_text not found in source.text")
        categories.add("grounding")

    if event.get("time_status") not in VALID_TIME_STATUS:
        errors.append(f"invalid time_status: {event.get('time_status')!r}")
        categories.add("enum")
    if event.get("severity") not in VALID_SEVERITY:
        errors.append(f"invalid severity: {event.get('severity')!r}")
        categories.add("enum")
    if event.get("modality") not in VALID_MODALITY:
        errors.append(f"invalid modality: {event.get('modality')!r}")
        categories.add("enum")

    key = (event_type, gq)
    if key in seen_keys:
        errors.append("duplicate event (same event_type + grounding_quote)")
        categories.add("dedup")
    else:
        seen_keys.add(key)

    return errors, categories


def process(
    records: list[dict[str, Any]],
    ontology: set[str],
    clean_handle,
    invalid_handle,
) -> dict[str, int]:
    stats: dict[str, int] = {
        "total": 0,
        "skipped_error_status": 0,
        "total_events": 0,
        "events_clean": 0,
        "events_invalid": 0,
        "by_off_ontology": 0,
        "by_grounding": 0,
        "by_enum": 0,
        "by_dedup": 0,
    }

    for record in records:
        if record.get("status") != "ok":
            stats["skipped_error_status"] += 1
            continue

        stats["total"] += 1
        annotation = record.get("annotation") or {}
        source = record.get("source") or {}
        source_text = source.get("text", "")
        doc_id = str(record.get("id", ""))

        events: list[dict[str, Any]] = annotation.get("events") or []
        stats["total_events"] += len(events)

        seen_keys: set[tuple[str, str]] = set()
        clean_events: list[dict[str, Any]] = []

        for event in events:
            errors, categories = check_event(event, source_text, ontology, seen_keys)

            if errors:
                stats["events_invalid"] += 1
                for cat in categories:
                    stats[f"by_{cat}"] += 1
                invalid_handle.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "source_text": source_text,
                            "publish_date": source.get("publish_date", ""),
                            "event": event,
                            "errors": errors,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            else:
                clean_events.append(event)
                stats["events_clean"] += 1

        clean_record = {**record, "annotation": {**annotation, "events": clean_events}}
        clean_handle.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v3 generation output JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output-stem",
        required=True,
        type=Path,
        help="Path stem; emits <stem>.clean.jsonl and <stem>.invalid.jsonl",
    )
    parser.add_argument("--ontology", type=Path, default=ONTOLOGY_PATH)
    parser.add_argument("--system-prompt", type=Path, default=SYSTEM_PROMPT_PATH)
    args = parser.parse_args()

    ontology = load_ontology_labels(args.ontology)
    prompt_labels = load_prompt_labels(args.system_prompt)
    if any("{{" in label for label in prompt_labels):
        print("Ontology cross-check skipped — system_prompt contains unfilled placeholders", file=sys.stderr)
    elif ontology != prompt_labels:
        only_ont = sorted(ontology - prompt_labels)
        only_pmt = sorted(prompt_labels - ontology)
        print("WARNING: ontology / system_prompt mismatch!", file=sys.stderr)
        if only_ont:
            print(f"  in ontology only : {only_ont}", file=sys.stderr)
        if only_pmt:
            print(f"  in prompt only   : {only_pmt}", file=sys.stderr)
    else:
        print(f"Ontology cross-check OK — {len(ontology)} labels match prompt", file=sys.stderr)

    records: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    stem = str(args.output_stem)
    clean_path = Path(stem + ".clean.jsonl")
    invalid_path = Path(stem + ".invalid.jsonl")
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    with clean_path.open("w", encoding="utf-8") as ch, \
         invalid_path.open("w", encoding="utf-8") as ih:
        stats = process(records, ontology, ch, ih)

    total = stats["total_events"]
    invalid = stats["events_invalid"]
    clean = stats["events_clean"]

    print(f"\n=== Validation summary ===")
    print(f"Records processed   : {stats['total']}  ({stats['skipped_error_status']} error-status skipped)")
    print(f"Events total        : {total}")
    print(f"Events clean        : {clean}  ({100*clean/max(total,1):.1f}%)")
    print(f"Events invalid      : {invalid}  ({100*invalid/max(total,1):.1f}%)")
    print(f"  off-ontology      : {stats['by_off_ontology']}")
    print(f"  grounding fail    : {stats['by_grounding']}")
    print(f"  enum invalid      : {stats['by_enum']}")
    print(f"  duplicates        : {stats['by_dedup']}")
    print(f"\nOutput:")
    print(f"  {clean_path}")
    print(f"  {invalid_path}")


if __name__ == "__main__":
    main()
