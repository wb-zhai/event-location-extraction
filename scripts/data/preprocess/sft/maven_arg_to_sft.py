from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.inference.text_anchor import TextAnchorResolver

TOKEN_PATTERN = re.compile(r"\S+")
ANCHOR_RESOLVER = TextAnchorResolver()


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


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


def tokenize_document(
    document: str,
) -> tuple[list[str], list[tuple[int, int]], dict[int, int], dict[int, int]]:
    tokens: list[str] = []
    token_char_spans: list[tuple[int, int]] = []
    char_start_to_token: dict[int, int] = {}
    char_end_to_token: dict[int, int] = {}

    for token_index, match in enumerate(TOKEN_PATTERN.finditer(document)):
        start, end = match.span()
        tokens.append(match.group(0))
        token_char_spans.append((start, end))
        char_start_to_token[start] = token_index
        char_end_to_token[end] = token_index

    return tokens, token_char_spans, char_start_to_token, char_end_to_token


def char_offset_to_token_span(
    offset: Any,
    char_start_to_token: dict[int, int],
    char_end_to_token: dict[int, int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(offset, list) or len(offset) != 2:
        return None

    start, end = offset
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        return None
    if start not in char_start_to_token or end not in char_end_to_token:
        return None

    return char_start_to_token[start], char_end_to_token[end], start, end


def offset_or_text_to_token_span(
    document: str,
    offset: Any,
    text_value: Any,
    char_start_to_token: dict[int, int],
    char_end_to_token: dict[int, int],
) -> tuple[int, int, int, int] | None:
    span = char_offset_to_token_span(offset, char_start_to_token, char_end_to_token)
    if span is not None:
        return span

    if not isinstance(text_value, str) or not text_value.strip():
        return None

    match = ANCHOR_RESOLVER.resolve(document, text_value)
    if match.start is None or match.end is None:
        return None
    if match.start not in char_start_to_token or match.end not in char_end_to_token:
        return None

    return (
        char_start_to_token[match.start],
        char_end_to_token[match.end],
        match.start,
        match.end,
    )


def dedupe_mentions(
    mentions: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    deduped: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for mention in mentions:
        if mention in seen:
            continue
        seen.add(mention)
        deduped.append(mention)
    return deduped


def safe_text_span(document: str, start: int, end: int) -> str:
    if start < 0 or end <= start or end > len(document):
        return ""
    return document[start:end]


def project_events_to_window(
    document: str,
    events: list[dict[str, Any]],
    window_start: int,
    window_end: int,
    char_offset: int,
    include_arguments: bool = True,
) -> list[dict[str, Any]]:
    projected_events: list[dict[str, Any]] = []
    for event in events:
        if not (
            window_start <= event["token_start"] and event["token_end"] < window_end
        ):
            continue

        projected_event = {
            "event_type": event["event_type"],
            "start": event["start"] - char_offset,
            "end": event["end"] - char_offset,
            "text": safe_text_span(document, event["start"], event["end"]),
        }
        if include_arguments:
            projected_arguments: list[dict[str, Any]] = []
            seen_arguments: set[tuple[str, int, int]] = set()
            for argument in event["arguments"]:
                if not (
                    window_start <= argument["token_start"]
                    and argument["token_end"] < window_end
                ):
                    continue

                argument_key = (
                    argument["role"],
                    argument["token_start"],
                    argument["token_end"],
                )
                if argument_key in seen_arguments:
                    continue
                seen_arguments.add(argument_key)

                projected_arguments.append(
                    {
                        "role": argument["role"],
                        "start": argument["start"] - char_offset,
                        "end": argument["end"] - char_offset,
                        "text": safe_text_span(
                            document,
                            argument["start"],
                            argument["end"],
                        ),
                    }
                )

            projected_event["arguments"] = projected_arguments

        projected_events.append(projected_event)

    return projected_events


def format_sft_record(question: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    return {"question": question, "answer": {"events": events}}


def collect_argument_events(
    document: str,
    argument_values: Any,
    entity_mentions_by_id: dict[str, list[tuple[int, int, int, int]]],
    char_start_to_token: dict[int, int],
    char_end_to_token: dict[int, int],
) -> list[tuple[int, int, int, int]]:
    if not isinstance(argument_values, list):
        return []

    mentions: list[tuple[int, int, int, int]] = []
    for argument_value in argument_values:
        if not isinstance(argument_value, dict):
            continue

        entity_id = argument_value.get("entity_id")
        if isinstance(entity_id, str) and entity_id:
            mentions.extend(entity_mentions_by_id.get(entity_id, []))
            continue

        span = offset_or_text_to_token_span(
            document,
            argument_value.get("offset"),
            argument_value.get("text") or argument_value.get("mention"),
            char_start_to_token,
            char_end_to_token,
        )
        if span is None:
            continue
        mentions.append(span)

    return dedupe_mentions(mentions)


def build_entity_mentions(
    document: str,
    data: dict[str, Any],
    char_start_to_token: dict[int, int],
    char_end_to_token: dict[int, int],
) -> dict[str, list[tuple[int, int, int, int]]]:
    entity_mentions_by_id: dict[str, list[tuple[int, int, int, int]]] = {}

    for entity in data.get("entities", []):
        entity_id = entity.get("id")
        if not entity_id:
            continue

        mentions: list[tuple[int, int, int, int]] = []
        for mention in entity.get("mention", []):
            span = offset_or_text_to_token_span(
                document,
                mention.get("offset"),
                mention.get("text") or mention.get("mention"),
                char_start_to_token,
                char_end_to_token,
            )
            if span is None:
                continue
            mentions.append(span)

        if mentions:
            entity_mentions_by_id[entity_id] = dedupe_mentions(mentions)

    return entity_mentions_by_id


def build_document_annotations(data: dict[str, Any]) -> dict[str, Any] | None:
    document = data.get("document", "")
    if not isinstance(document, str):
        return None

    tokens, token_char_spans, char_start_to_token, char_end_to_token = (
        tokenize_document(document)
    )
    if not tokens:
        return None

    entity_mentions_by_id = build_entity_mentions(
        document,
        data,
        char_start_to_token,
        char_end_to_token,
    )

    events: list[dict[str, Any]] = []
    event_labels: set[str] = set()
    argument_labels: set[str] = set()

    for event in data.get("events", []):
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            continue

        normalized_event_type = normalize_label(event_type)
        event_arguments_map = event.get("argument", {})
        if not isinstance(event_arguments_map, dict):
            event_arguments_map = {}

        for mention in event.get("mention", []):
            mention_span = offset_or_text_to_token_span(
                document,
                mention.get("offset"),
                mention.get("text") or mention.get("mention"),
                char_start_to_token,
                char_end_to_token,
            )
            if mention_span is None:
                continue

            (
                mention_token_start,
                mention_token_end,
                mention_char_start,
                mention_char_end,
            ) = mention_span
            event_arguments: list[dict[str, Any]] = []
            seen_arguments: set[tuple[str, int, int]] = set()

            for role, argument_values in event_arguments_map.items():
                if not isinstance(role, str) or not role:
                    continue

                normalized_role = normalize_label(role)
                argument_mentions = collect_argument_events(
                    document,
                    argument_values,
                    entity_mentions_by_id,
                    char_start_to_token,
                    char_end_to_token,
                )
                if not argument_mentions:
                    continue

                for argument_mention in argument_mentions:
                    (
                        argument_token_start,
                        argument_token_end,
                        argument_char_start,
                        argument_char_end,
                    ) = argument_mention
                    argument_key = (
                        normalized_role,
                        argument_token_start,
                        argument_token_end,
                    )
                    if argument_key in seen_arguments:
                        continue
                    seen_arguments.add(argument_key)

                    event_arguments.append(
                        {
                            "role": normalized_role,
                            "start": argument_char_start,
                            "end": argument_char_end,
                            "token_start": argument_token_start,
                            "token_end": argument_token_end,
                        }
                    )
                    argument_labels.add(normalized_role)

            events.append(
                {
                    "event_type": normalized_event_type,
                    "start": mention_char_start,
                    "end": mention_char_end,
                    "token_start": mention_token_start,
                    "token_end": mention_token_end,
                    "arguments": event_arguments,
                }
            )
            event_labels.add(normalized_event_type)

    if not events:
        return None

    doc_id = data.get("id")
    if not isinstance(doc_id, str) or not doc_id:
        doc_id = ""

    argument_count = sum(len(event["arguments"]) for event in events)
    return {
        "id": doc_id,
        "document": document,
        "tokens": tokens,
        "token_char_spans": token_char_spans,
        "event_labels": sorted(event_labels),
        "argument_labels": sorted(argument_labels),
        "events": events,
        "metadata": {
            "dataset": "MAVEN-Arg",
            "doc_id": doc_id,
            "title": data.get("title", ""),
            "event_count": len(events),
            "argument_count": argument_count,
            "relation_count": argument_count,
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
    include_arguments: bool = True,
) -> dict[str, Any] | None:
    token_char_spans = sample["token_char_spans"]
    if not token_char_spans or window_start >= window_end:
        return None

    window_char_start = token_char_spans[window_start][0]
    window_char_end = token_char_spans[window_end - 1][1]
    windowed_events = project_events_to_window(
        sample["document"],
        sample["events"],
        window_start,
        window_end,
        window_char_start,
        include_arguments=include_arguments,
    )
    if not windowed_events:
        return None

    question = sample["document"][window_char_start:window_char_end]
    return format_sft_record(question, windowed_events)


def convert_document(
    data: dict[str, Any],
    window_size: int | None = None,
    window_stride: int | None = None,
    include_arguments: bool = True,
) -> list[dict[str, Any]]:
    sample = build_document_annotations(data)
    if sample is None:
        return []

    if window_size is None:
        full_events = project_events_to_window(
            sample["document"],
            sample["events"],
            0,
            len(sample["tokens"]),
            0,
            include_arguments=include_arguments,
        )
        if not full_events:
            return []
        return [format_sft_record(sample["document"], full_events)]

    stride = window_stride if window_stride is not None else window_size
    windows = build_windows(len(sample["tokens"]), window_size, stride)
    return [
        windowed
        for window_start, window_end in windows
        if (
            windowed := slice_window(
                sample,
                window_start,
                window_end,
                include_arguments=include_arguments,
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


def build_ontology(
    source_files: list[Path],
    include_arguments: bool = True,
) -> dict[str, Any]:
    event_label_counts: dict[str, int] = {}
    argument_label_counts: dict[str, int] = {}

    for source_file in source_files:
        for _, data in load_samples(source_file):
            sample = build_document_annotations(data)
            if sample is None:
                continue

            for event in sample["events"]:
                label = event["event_type"]
                event_label_counts[label] = event_label_counts.get(label, 0) + 1

                if not include_arguments:
                    continue

                for argument in event["arguments"]:
                    label = argument["role"]
                    argument_label_counts[label] = (
                        argument_label_counts.get(label, 0) + 1
                    )

    ontology = {
        "dataset": "MAVEN-Arg",
        "event_labels": sorted(event_label_counts),
        "event_label_counts": dict(sorted(event_label_counts.items())),
        "source_files": [str(path) for path in source_files],
    }
    if include_arguments:
        ontology["argument_labels"] = sorted(argument_label_counts)
        ontology["argument_label_counts"] = dict(sorted(argument_label_counts.items()))
    return ontology


def default_ontology_output_path(input_path: Path, output_path: Path) -> Path:
    if input_path.is_dir() or output_path.is_dir():
        return output_path / "ontology.json"
    return output_path.parent / f"{output_path.name}.ontology.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MAVEN-Arg dataset to reader-normalized JSONL format"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a MAVEN-Arg JSONL file or directory of JSONL split files",
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

    samples_by_source: dict[Path, list[dict[str, Any]]] = {}
    processed = 0
    written = 0
    total_events = 0
    total_arguments = 0

    for source_path, data in load_samples(input_path):
        processed += 1
        converted_samples = convert_document(
            data,
            window_size=args.window_size,
            window_stride=args.window_stride,
            include_arguments=not args.only_events,
        )
        if not converted_samples:
            continue

        for converted_sample in converted_samples:
            events = converted_sample.get("answer", {}).get("events", [])
            total_events += len(events)
            total_arguments += sum(len(event.get("arguments", [])) for event in events)

        samples_by_source.setdefault(source_path, []).extend(converted_samples)
        written += len(converted_samples)

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
