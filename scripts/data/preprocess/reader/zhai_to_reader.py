from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tokenization_utils import build_fast_tokenizer, char_span_to_piece_span, tokenize_text


def normalize_label(label: str) -> str:
    return label.strip().lower()


def load_samples(input_path: Path) -> list[tuple[Path, dict[str, Any]]]:
    samples: list[tuple[Path, dict[str, Any]]] = []
    if input_path.is_dir():
        for file_path in sorted(input_path.glob("*.jsonl")):
            with open(file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    samples.append((file_path, json.loads(line)))
        return samples

    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append((input_path, json.loads(line)))
    return samples


def canonicalize_argument_spans(
    arguments: list[dict[str, int]],
    relations: list[dict[str, Any]],
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    canonical_arguments: list[dict[str, int]] = []
    argument_index_map: dict[int, int] = {}
    argument_index_by_span: dict[tuple[int, int], int] = {}

    for old_index, argument in enumerate(arguments):
        span_key = (argument["start"], argument["end"])
        canonical_index = argument_index_by_span.get(span_key)
        if canonical_index is None:
            canonical_index = len(canonical_arguments)
            argument_index_by_span[span_key] = canonical_index
            canonical_arguments.append(argument)
        argument_index_map[old_index] = canonical_index

    canonical_relations: list[dict[str, Any]] = []
    seen_relations: set[tuple[int, int, str]] = set()
    for relation in relations:
        relation_key = (
            relation["event_idx"],
            argument_index_map[relation["argument_idx"]],
            relation["label"],
        )
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)
        canonical_relations.append(
            {
                "event_idx": relation_key[0],
                "argument_idx": relation_key[1],
                "label": relation_key[2],
            }
        )

    return canonical_arguments, canonical_relations


def build_document_annotations(
    data: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any] | None:
    source = data.get("source")
    if not isinstance(source, dict):
        return None

    document = source.get("text")
    if not isinstance(document, str) or not document.strip():
        return None

    tokenized = tokenize_text(tokenizer, document)

    events: list[dict[str, Any]] = []
    arguments: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    event_labels: set[str] = set()
    argument_labels: set[str] = set()
    argument_index_by_span: dict[tuple[int, int], int] = {}
    seen_relations: set[tuple[int, int, str]] = set()

    for event in data.get("events", []):
        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type")
        if not isinstance(event_type, str) or not event_type.strip():
            continue

        event_span = char_span_to_piece_span(
            event.get("start_char"),
            event.get("end_char"),
            tokenized.offsets,
        )
        if event_span is None:
            continue

        normalized_event_type = normalize_label(event_type)
        event_idx = len(events)
        events.append(
            {
                "start": event_span[0],
                "end": event_span[1],
                "label": normalized_event_type,
            }
        )
        event_labels.add(normalized_event_type)

        for argument in event.get("arguments", []):
            if not isinstance(argument, dict):
                continue

            role = argument.get("role")
            if not isinstance(role, str) or not role.strip():
                continue

            argument_span = char_span_to_piece_span(
                argument.get("start_char"),
                argument.get("end_char"),
                tokenized.offsets,
            )
            if argument_span is None:
                continue

            argument_idx = argument_index_by_span.get(argument_span)
            if argument_idx is None:
                argument_idx = len(arguments)
                argument_index_by_span[argument_span] = argument_idx
                arguments.append(
                    {
                        "start": argument_span[0],
                        "end": argument_span[1],
                    }
                )

            relation_label = normalize_label(role)
            relation_key = (event_idx, argument_idx, relation_label)
            if relation_key in seen_relations:
                continue
            seen_relations.add(relation_key)
            relations.append(
                {
                    "event_idx": event_idx,
                    "argument_idx": argument_idx,
                    "label": relation_label,
                }
            )
            argument_labels.add(relation_label)

    if not events:
        return None

    arguments, relations = canonicalize_argument_spans(arguments, relations)

    return {
        "id": str(data.get("id", "")),
        "input_ids": list(tokenized.input_ids),
        "tokenizer_tokens": list(tokenized.tokenizer_tokens),
        "event_labels": sorted(event_labels),
        "argument_labels": sorted(argument_labels),
        "events": events,
        "arguments": arguments,
        "relations": relations,
        "metadata": {
            "dataset": "risk-factor-zhai",
            "doc_id": data.get("id", ""),
            "title": source.get("title", ""),
            "event_count": len(events),
            "argument_count": len(arguments),
            "relation_count": len(relations),
            "source_status": data.get("status", ""),
            "tokenizer_name": tokenizer.name_or_path,
            "tokenizer_tokens_are_model_pieces": True,
        },
    }


def build_windows(
    token_count: int,
    window_size: int | None,
    window_stride: int | None,
) -> list[tuple[int, int]]:
    if window_size is None or window_size >= token_count:
        return [(0, token_count)]

    stride = window_stride if window_stride is not None else window_size
    windows: list[tuple[int, int]] = []
    start = 0
    while start < token_count:
        end = min(start + window_size, token_count)
        windows.append((start, end))
        if end >= token_count:
            break
        start += stride
    return windows


def slice_window(
    sample: dict[str, Any],
    window_start: int,
    window_end: int,
    window_index: int,
    window_count: int,
    window_size: int,
    window_stride: int,
) -> dict[str, Any] | None:
    windowed_events: list[dict[str, Any]] = []
    event_index_map: dict[int, int] = {}
    for old_idx, event in enumerate(sample["events"]):
        if window_start <= event["start"] and event["end"] < window_end:
            event_index_map[old_idx] = len(windowed_events)
            windowed_events.append(
                {
                    "start": event["start"] - window_start,
                    "end": event["end"] - window_start,
                    "label": event["label"],
                }
            )

    if not windowed_events:
        return None

    windowed_arguments: list[dict[str, Any]] = []
    argument_index_map: dict[int, int] = {}
    for old_idx, argument in enumerate(sample["arguments"]):
        if window_start <= argument["start"] and argument["end"] < window_end:
            argument_index_map[old_idx] = len(windowed_arguments)
            windowed_arguments.append(
                {
                    "start": argument["start"] - window_start,
                    "end": argument["end"] - window_start,
                }
            )

    windowed_relations: list[dict[str, Any]] = []
    seen_relations: set[tuple[int, int, str]] = set()
    for relation in sample["relations"]:
        old_event_idx = relation["event_idx"]
        old_argument_idx = relation["argument_idx"]
        if (
            old_event_idx not in event_index_map
            or old_argument_idx not in argument_index_map
        ):
            continue
        relation_key = (
            event_index_map[old_event_idx],
            argument_index_map[old_argument_idx],
            relation["label"],
        )
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)
        windowed_relations.append(
            {
                "event_idx": relation_key[0],
                "argument_idx": relation_key[1],
                "label": relation_key[2],
            }
        )

    windowed_arguments, windowed_relations = canonicalize_argument_spans(
        windowed_arguments,
        windowed_relations,
    )

    metadata = dict(sample["metadata"])
    metadata.update(
        {
            "windowed": True,
            "window_index": window_index,
            "window_count": window_count,
            "window_start": window_start,
            "window_end": window_end,
            "window_size": window_size,
            "window_stride": window_stride,
        }
    )

    return {
        "id": f"{sample['id']}__w{window_index}",
        "input_ids": sample["input_ids"][window_start:window_end],
        "tokenizer_tokens": sample["tokenizer_tokens"][window_start:window_end],
        "event_labels": list(sample["event_labels"]),
        "argument_labels": list(sample["argument_labels"]),
        "events": windowed_events,
        "arguments": windowed_arguments,
        "relations": windowed_relations,
        "metadata": metadata,
    }


def convert_document(
    data: dict[str, Any],
    tokenizer: Any,
    window_size: int | None = None,
    window_stride: int | None = None,
) -> list[dict[str, Any]]:
    sample = build_document_annotations(data, tokenizer)
    if sample is None:
        return []

    if window_size is None:
        return [sample]

    stride = window_stride if window_stride is not None else window_size
    windows = build_windows(len(sample["input_ids"]), window_size, stride)
    return [
        windowed
        for window_index, (window_start, window_end) in enumerate(windows)
        if (
            windowed := slice_window(
                sample=sample,
                window_start=window_start,
                window_end=window_end,
                window_index=window_index,
                window_count=len(windows),
                window_size=window_size,
                window_stride=stride,
            )
        )
        is not None
    ]


def output_path_for(source_path: Path, input_path: Path, output_path: Path) -> Path:
    if input_path.is_dir():
        return output_path / source_path.name
    return output_path


def ontology_source_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl"))

    sibling_files = sorted(input_path.parent.glob("*.jsonl"))
    if sibling_files:
        return sibling_files
    return [input_path]


def build_ontology(source_files: list[Path], tokenizer: Any) -> dict[str, Any]:
    event_label_counts: dict[str, int] = {}
    argument_label_counts: dict[str, int] = {}

    for source_file in source_files:
        for _, data in load_samples(source_file):
            sample = build_document_annotations(data, tokenizer)
            if sample is None:
                continue

            for event in sample["events"]:
                label = event["label"]
                event_label_counts[label] = event_label_counts.get(label, 0) + 1

            for relation in sample["relations"]:
                label = relation["label"]
                argument_label_counts[label] = argument_label_counts.get(label, 0) + 1

    return {
        "dataset": "risk-factor-zhai",
        "event_labels": sorted(event_label_counts),
        "argument_labels": sorted(argument_label_counts),
        "event_label_counts": dict(sorted(event_label_counts.items())),
        "argument_label_counts": dict(sorted(argument_label_counts.items())),
        "source_files": [str(path) for path in source_files],
    }


def default_ontology_output_path(input_path: Path, output_path: Path) -> Path:
    if input_path.is_dir() or output_path.is_dir():
        return output_path / "ontology.json"
    return output_path.parent / f"{output_path.name}.ontology.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert risk-factor raw JSONL to reader-normalized JSONL format"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a risk-factor raw JSONL file or directory of JSONL split files",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Output JSONL path, or output directory when input_path is a directory",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Optional document window size in tokenizer pieces",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Optional document window stride in tokenizer pieces; defaults to --window-size",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Fast Hugging Face tokenizer used to build token-piece reader data.",
    )
    parser.add_argument(
        "--ontology-output",
        type=str,
        default=None,
        help="Optional path for the dataset ontology JSON",
    )
    args = parser.parse_args()

    if args.window_size is not None and args.window_size <= 0:
        raise ValueError("--window-size must be a positive integer")
    if args.window_stride is not None and args.window_stride <= 0:
        raise ValueError("--window-stride must be a positive integer")
    if args.window_size is None and args.window_stride is not None:
        raise ValueError("--window-stride requires --window-size")

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    tokenizer = build_fast_tokenizer(args.tokenizer_name)

    samples_by_source: dict[Path, list[dict[str, Any]]] = {}
    processed = 0
    written = 0

    for source_path, data in load_samples(input_path):
        processed += 1
        converted_samples = convert_document(
            data,
            tokenizer=tokenizer,
            window_size=args.window_size,
            window_stride=args.window_stride,
        )
        if not converted_samples:
            continue
        samples_by_source.setdefault(source_path, []).extend(converted_samples)
        written += len(converted_samples)

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        for source_path, samples in samples_by_source.items():
            target_path = output_path_for(source_path, input_path, output_path)
            with open(target_path, "w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for sample in samples_by_source.get(input_path, []):
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    source_files = ontology_source_files(input_path)
    ontology = build_ontology(source_files, tokenizer)
    ontology_output = (
        Path(args.ontology_output)
        if args.ontology_output is not None
        else default_ontology_output_path(input_path, output_path)
    )
    ontology_output.parent.mkdir(parents=True, exist_ok=True)
    with open(ontology_output, "w", encoding="utf-8") as handle:
        json.dump(ontology, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Processed documents: {processed}")
    print(f"Written samples: {written}")
    print(
        f"Ontology labels: {len(ontology['event_labels'])} events, "
        f"{len(ontology['argument_labels'])} argument roles"
    )


if __name__ == "__main__":
    main()
