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


def load_ontology_labels(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events")
    if not isinstance(events, dict) or not events:
        raise ValueError(f"Ontology must contain a non-empty object at 'events': {path}")

    labels_by_key: dict[str, str] = {}
    for label in events:
        if not isinstance(label, str) or not label.strip():
            continue
        canonical_key = canonicalize_label(label)
        existing = labels_by_key.get(canonical_key)
        if existing is not None and existing != label:
            raise ValueError(
                "Ontology has ambiguous labels after normalization: "
                f"{existing!r} and {label!r}"
            )
        labels_by_key[canonical_key] = label

    if not labels_by_key:
        raise ValueError(f"Ontology did not yield any usable event labels: {path}")
    return labels_by_key


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
    ontology_labels: dict[str, str],
    replacements: Counter[tuple[str, str]],
    unresolved: Counter[str],
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

        fixed_label = ontology_labels.get(canonicalize_label(label))
        if fixed_label is None:
            unresolved[label] += 1
            continue
        if fixed_label == label:
            continue

        event["event_type"] = fixed_label
        replacements[(label, fixed_label)] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite JSONL event labels to match ontology event names."
    )
    parser.add_argument("input", type=Path, help="Input JSONL file to repair.")
    parser.add_argument(
        "ontology",
        type=Path,
        help="Ontology JSON file containing canonical labels under 'events'.",
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

    ontology_labels = load_ontology_labels(ontology_path)
    records = iter_jsonl(input_path)

    replacements: Counter[tuple[str, str]] = Counter()
    unresolved: Counter[str] = Counter()

    for _, record in records:
        fix_record_labels(record, ontology_labels, replacements, unresolved)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for _, record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    print(f"input: {input_path}")
    print(f"ontology: {ontology_path}")
    print(f"output: {output_path}")
    print(f"records: {len(records)}")
    print(f"labels_fixed: {sum(replacements.values())}")
    print(f"unresolved_labels: {sum(unresolved.values())}")

    if replacements:
        print("replacements:")
        for (before, after), count in sorted(replacements.items()):
            print(f"  {before!r} -> {after!r}: {count}")

    if unresolved:
        print("unresolved:")
        for label, count in unresolved.most_common():
            print(f"  {label!r}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
