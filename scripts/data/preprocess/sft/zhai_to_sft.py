from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"\S+")


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


def safe_text_span(document: str, start: Any, end: Any) -> str:
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    if start < 0 or end <= start or end > len(document):
        return ""
    return document[start:end]


def tokenize_document(document: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    token_char_spans: list[tuple[int, int]] = []
    for match in TOKEN_PATTERN.finditer(document):
        start, end = match.span()
        tokens.append(match.group(0))
        token_char_spans.append((start, end))
    return tokens, token_char_spans


def load_tokenizer(tokenizer_name_or_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required when --tokenizer is provided"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("--tokenizer requires a fast tokenizer with offset mappings")
    return tokenizer


def tokenize_document_with_tokenizer(
    document: str,
    tokenizer: Any,
) -> tuple[list[str], list[tuple[int, int]]]:
    encoded = tokenizer(
        document,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = encoded.get("input_ids", [])
    offsets = encoded.get("offset_mapping", [])
    if len(input_ids) != len(offsets):
        raise ValueError("Tokenizer returned mismatched input_ids and offset_mapping")

    tokens: list[str] = []
    token_char_spans: list[tuple[int, int]] = []
    for token_id, offset in zip(input_ids, offsets):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            continue
        start, end = offset
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if end <= start:
            continue

        tokens.append(tokenizer.convert_ids_to_tokens(token_id))
        token_char_spans.append((start, end))

    return tokens, token_char_spans


def convert_argument(document: str, argument: dict[str, Any]) -> dict[str, Any] | None:
    start = argument.get("start_char")
    end = argument.get("end_char")
    text = safe_text_span(document, start, end)
    if not text:
        return None

    role = argument.get("role")
    if not isinstance(role, str) or not role.strip():
        return None

    converted_argument = {
        "role": role.strip(),
        "start": start,
        "end": end,
        "text": text,
    }
    location_type = argument.get("location_type")
    if isinstance(location_type, str) and location_type.strip():
        converted_argument["location_type"] = location_type.strip()
    return converted_argument


def convert_event(
    document: str,
    event: dict[str, Any],
    include_arguments: bool = True,
) -> dict[str, Any] | None:
    event_type = event.get("event_type")
    if not isinstance(event_type, str) or not event_type.strip():
        return None

    start = event.get("start_char")
    end = event.get("end_char")
    text = safe_text_span(document, start, end)
    if not text:
        return None

    converted = {
        "event_type": event_type.strip(),
        "start": start,
        "end": end,
        "text": text,
    }
    if not include_arguments:
        return converted

    arguments: list[dict[str, Any]] = []
    seen_arguments: set[tuple[str, int, int]] = set()
    for argument in event.get("arguments", []):
        if not isinstance(argument, dict):
            continue
        converted_argument = convert_argument(document, argument)
        if converted_argument is None:
            continue

        argument_key = (
            converted_argument["role"],
            converted_argument["start"],
            converted_argument["end"],
        )
        if argument_key in seen_arguments:
            continue
        seen_arguments.add(argument_key)
        arguments.append(converted_argument)

    converted["arguments"] = arguments
    return converted


def convert_document(
    data: dict[str, Any],
    include_arguments: bool = True,
) -> dict[str, Any] | None:
    source = data.get("source")
    if not isinstance(source, dict):
        return None

    document = source.get("text")
    if not isinstance(document, str) or not document.strip():
        return None

    events: list[dict[str, Any]] = []
    for event in data.get("events", []):
        if not isinstance(event, dict):
            continue
        converted_event = convert_event(
            document,
            event,
            include_arguments=include_arguments,
        )
        if converted_event is None:
            continue
        events.append(converted_event)

    events.sort(key=lambda item: (item["start"], item["end"], item["event_type"]))
    return {"question": document, "answer": {"events": events}}


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
    token_char_spans: list[tuple[int, int]],
    window_start: int,
    window_end: int,
    include_arguments: bool = True,
) -> dict[str, Any] | None:
    if not token_char_spans or window_start >= window_end:
        return None

    window_char_start = token_char_spans[window_start][0]
    window_char_end = token_char_spans[window_end - 1][1]
    window_text = sample["question"][window_char_start:window_char_end]

    events: list[dict[str, Any]] = []
    for event in sample["answer"]["events"]:
        if not (window_char_start <= event["start"] and event["end"] <= window_char_end):
            continue

        windowed_event = {
            "event_type": event["event_type"],
            "start": event["start"] - window_char_start,
            "end": event["end"] - window_char_start,
            "text": event["text"],
        }
        if include_arguments:
            arguments: list[dict[str, Any]] = []
            for argument in event.get("arguments", []):
                if not (
                    window_char_start <= argument["start"]
                    and argument["end"] <= window_char_end
                ):
                    continue
                arguments.append(
                    {
                        "role": argument["role"],
                        "start": argument["start"] - window_char_start,
                        "end": argument["end"] - window_char_start,
                        "text": argument["text"],
                    }
                )
                if "location_type" in argument:
                    arguments[-1]["location_type"] = argument["location_type"]
            windowed_event["arguments"] = arguments

        events.append(windowed_event)

    return {"question": window_text, "answer": {"events": events}}


def convert_to_sft_records(
    data: dict[str, Any],
    window_size: int | None = None,
    window_stride: int | None = None,
    tokenizer: Any | None = None,
    include_arguments: bool = True,
) -> list[dict[str, Any]]:
    sample = convert_document(data, include_arguments=include_arguments)
    if sample is None:
        return []

    if window_size is None:
        return [sample]

    if tokenizer is None:
        tokens, token_char_spans = tokenize_document(sample["question"])
    else:
        tokens, token_char_spans = tokenize_document_with_tokenizer(
            sample["question"],
            tokenizer,
        )
    if not tokens:
        return [sample]

    stride = window_stride if window_stride is not None else window_size
    windows = build_windows(len(tokens), window_size, stride)
    records: list[dict[str, Any]] = []
    for window_start, window_end in windows:
        windowed = slice_window(
            sample,
            token_char_spans,
            window_start,
            window_end,
            include_arguments=include_arguments,
        )
        if windowed is not None:
            records.append(windowed)
    return records


def build_ontology(
    source_files: list[Path],
    include_arguments: bool = True,
) -> dict[str, Any]:
    event_label_counts: dict[str, int] = {}
    argument_label_counts: dict[str, int] = {}

    for source_file in source_files:
        for _, data in load_samples(source_file):
            converted = convert_document(data, include_arguments=include_arguments)
            if converted is None:
                continue

            for event in converted["answer"]["events"]:
                event_label = event["event_type"]
                event_label_counts[event_label] = (
                    event_label_counts.get(event_label, 0) + 1
                )

                if not include_arguments:
                    continue

                for argument in event.get("arguments", []):
                    argument_label = argument["role"]
                    argument_label_counts[argument_label] = (
                        argument_label_counts.get(argument_label, 0) + 1
                    )

    ontology = {
        "dataset": "ZHAI",
        "event_labels": sorted(event_label_counts),
        "event_label_counts": dict(sorted(event_label_counts.items())),
        "source_files": [str(path) for path in source_files],
    }
    if include_arguments:
        ontology["argument_labels"] = sorted(argument_label_counts)
        ontology["argument_label_counts"] = dict(sorted(argument_label_counts.items()))
    return ontology


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ZHAI risk-factor dataset to SFT JSONL format"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a ZHAI JSONL file or directory of JSONL split files",
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
        help="Optional document window size in tokens",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Optional document window stride in tokens; defaults to --window-size",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional Hugging Face tokenizer name or path for tokenizer-based window sizes",
    )
    parser.add_argument(
        "--ontology-output",
        type=str,
        default=None,
        help="Optional path for the dataset ontology JSON",
    )
    parser.add_argument(
        "--only-events",
        action="store_true",
        help="Only output event triggers and omit event arguments",
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
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else None

    samples_by_source: dict[Path, list[dict[str, Any]]] = {}
    processed = 0
    written = 0
    total_events = 0
    total_arguments = 0

    for source_path, data in load_samples(input_path):
        processed += 1
        converted_records = convert_to_sft_records(
            data,
            window_size=args.window_size,
            window_stride=args.window_stride,
            tokenizer=tokenizer,
            include_arguments=not args.only_events,
        )
        if not converted_records:
            continue

        for converted in converted_records:
            events = converted["answer"]["events"]
            total_events += len(events)
            total_arguments += sum(len(event.get("arguments", [])) for event in events)

        samples_by_source.setdefault(source_path, []).extend(converted_records)
        written += len(converted_records)

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    for source_path, records in samples_by_source.items():
        destination = output_path_for(source_path, input_path, output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.ontology_output:
        ontology_output = Path(args.ontology_output)
        ontology_output.parent.mkdir(parents=True, exist_ok=True)
        ontology = build_ontology(
            ontology_source_files(input_path),
            include_arguments=not args.only_events,
        )
        with open(ontology_output, "w", encoding="utf-8") as handle:
            json.dump(ontology, handle, indent=2, ensure_ascii=False)
            print(f"Wrote ontology to {ontology_output}")

    print(f"Processed {processed} samples")
    print(f"Wrote {written} normalized samples")
    average_events_per_sample = total_events / written if written else 0.0
    average_arguments_per_sample = total_arguments / written if written else 0.0
    average_arguments_per_event = (
        total_arguments / total_events if total_events else 0.0
    )

    print(f"Average events per sample: {average_events_per_sample:.4f}")
    print(f"Average arguments per sample: {average_arguments_per_sample:.4f}")
    print(f"Average arguments per event: {average_arguments_per_event:.4f}")


if __name__ == "__main__":
    main()
