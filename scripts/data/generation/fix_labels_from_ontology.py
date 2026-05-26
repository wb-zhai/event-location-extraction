from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def canonicalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("-", " ").replace("_", " ").split())


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".jsonl":
        return input_path.with_name(f"{input_path.stem}.labels-fixed.jsonl")
    return input_path.with_name(f"{input_path.name}.labels-fixed")


def _build_canonical_mapping(
    labels: Any,
    *,
    label_kind: str,
    aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"Ontology must contain a non-empty object at {label_kind!r}.")

    aliases = aliases or {}
    labels_by_key: dict[str, str] = {}
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            continue
        canonical_key = canonicalize_label(label)
        existing = labels_by_key.get(canonical_key)
        if existing is not None and existing != label:
            raise ValueError(
                f"Ontology has ambiguous {label_kind} labels after normalization: "
                f"{existing!r} and {label!r}"
            )
        labels_by_key[canonical_key] = label

    for source_key, target_key in aliases.items():
        source_label = labels_by_key.get(canonicalize_label(source_key))
        target_label = labels_by_key.get(canonicalize_label(target_key))
        if source_label is not None and target_label is None:
            labels_by_key[canonicalize_label(target_key)] = source_label
        labels_by_key.pop(canonicalize_label(source_key), None)

    if not labels_by_key:
        raise ValueError(f"Ontology did not yield any usable {label_kind} labels.")
    return labels_by_key


def load_ontology_labels(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    event_labels = _build_canonical_mapping(payload.get("events"), label_kind="events")
    location_type_labels = _build_canonical_mapping(
        payload.get("location_types"),
        label_kind="location_types",
        aliases={"state": "province", "county": "district"},
    )
    return event_labels, location_type_labels


def iter_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
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
            records.append((line_no, record))
    return records


def fix_record_labels(
    record: dict[str, Any],
    ontology_event_labels: dict[str, str],
    ontology_location_type_labels: dict[str, str],
    event_replacements: Counter[tuple[str, str]],
    unresolved_event_labels: Counter[str],
    location_type_replacements: Counter[tuple[str, str]],
    unresolved_location_types: Counter[str],
) -> None:
    events = record.get("events")
    if not isinstance(events, list):
        return

    for event in events:
        if not isinstance(event, dict):
            continue
        label = event.get("event_type")
        if not isinstance(label, str) or not label.strip():
            continue

        fixed_label = ontology_event_labels.get(canonicalize_label(label))
        if fixed_label is None:
            unresolved_event_labels[label] += 1
        elif fixed_label != label:
            event["event_type"] = fixed_label
            event_replacements[(label, fixed_label)] += 1

        arguments = event.get("arguments")
        if not isinstance(arguments, list):
            continue

        for argument in arguments:
            if not isinstance(argument, dict):
                continue
            location_type = argument.get("location_type")
            if not isinstance(location_type, str) or not location_type.strip():
                continue

            fixed_location_type = ontology_location_type_labels.get(
                canonicalize_label(location_type)
            )
            if fixed_location_type is None:
                unresolved_location_types[location_type] += 1
                continue
            if fixed_location_type == location_type:
                continue

            argument["location_type"] = fixed_location_type
            location_type_replacements[(location_type, fixed_location_type)] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite JSONL event labels and location type labels to match ontology names."
        )
    )
    parser.add_argument("input", type=Path, help="Input JSONL file to repair.")
    parser.add_argument(
        "ontology",
        type=Path,
        help="Ontology JSON file containing canonical labels under 'events' and 'location_types'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSONL path. Defaults to INPUT.labels-fixed.jsonl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    ontology_path = resolve_path(args.ontology)
    output_path = resolve_path(args.output) if args.output else default_output_path(input_path)

    ontology_event_labels, ontology_location_type_labels = load_ontology_labels(ontology_path)
    records = iter_jsonl(input_path)

    event_replacements: Counter[tuple[str, str]] = Counter()
    unresolved_event_labels: Counter[str] = Counter()
    location_type_replacements: Counter[tuple[str, str]] = Counter()
    unresolved_location_types: Counter[str] = Counter()

    for _, record in records:
        fix_record_labels(
            record,
            ontology_event_labels,
            ontology_location_type_labels,
            event_replacements,
            unresolved_event_labels,
            location_type_replacements,
            unresolved_location_types,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for _, record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    print(f"input: {input_path}")
    print(f"ontology: {ontology_path}")
    print(f"output: {output_path}")
    print(f"records: {len(records)}")
    print(f"event_labels_fixed: {sum(event_replacements.values())}")
    print(f"location_types_fixed: {sum(location_type_replacements.values())}")
    print(f"unresolved_event_labels: {sum(unresolved_event_labels.values())}")
    print(f"unresolved_location_types: {sum(unresolved_location_types.values())}")

    if event_replacements:
        print("event_replacements:")
        for (before, after), count in sorted(event_replacements.items()):
            print(f"  {before!r} -> {after!r}: {count}")

    if location_type_replacements:
        print("location_type_replacements:")
        for (before, after), count in sorted(location_type_replacements.items()):
            print(f"  {before!r} -> {after!r}: {count}")

    if unresolved_event_labels:
        print("unresolved_event_labels:")
        for label, count in unresolved_event_labels.most_common():
            print(f"  {label!r}: {count}")

    if unresolved_location_types:
        print("unresolved_location_types:")
        for label, count in unresolved_location_types.most_common():
            print(f"  {label!r}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
