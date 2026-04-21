from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROLE_PATTERN = re.compile(r"arg\d+(?P<role>.+)$")


def normalize_label(label: str) -> str:
    return (
        label.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def parse_role_label(raw_role: str) -> str:
    role = raw_role.strip().lower()
    match = ROLE_PATTERN.search(role)
    if match and match.group("role"):
        return normalize_label(match.group("role"))
    return normalize_label(role)


def flatten_sentences(sentences: list[list[str]]) -> list[str]:
    return [token for sentence in sentences for token in sentence]


def load_samples(input_path: Path) -> list[tuple[Path, dict[str, Any]]]:
    samples: list[tuple[Path, dict[str, Any]]] = []

    def iter_json_lines(path: Path) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                samples.append((path, json.loads(line)))

    if input_path.is_dir():
        for pattern in ("*.jsonlines", "*.jsonl"):
            for file_path in sorted(input_path.glob(pattern)):
                iter_json_lines(file_path)
        return samples

    iter_json_lines(input_path)
    return samples


def is_valid_span(start: Any, end: Any, token_count: int) -> bool:
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and start >= 0
        and end >= start
        and end < token_count
    )


def build_document_annotations(data: dict[str, Any]) -> dict[str, Any] | None:
    doc_id = data.get("doc_key")
    if not isinstance(doc_id, str) or not doc_id:
        return None

    sentences = data.get("sentences", [])
    if not isinstance(sentences, list) or not sentences:
        return None

    flat_tokens = flatten_sentences(sentences)
    token_count = len(flat_tokens)
    if token_count == 0:
        return None

    events: list[dict[str, Any]] = []
    arguments: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    event_span_to_indices: dict[tuple[int, int], list[int]] = {}
    argument_span_to_indices: dict[tuple[int, int], list[int]] = {}

    for trigger in data.get("evt_triggers", []):
        if not isinstance(trigger, list) or len(trigger) < 3:
            continue
        start = trigger[0]
        end = trigger[1]
        trigger_info = trigger[2]
        if not is_valid_span(start, end, token_count):
            continue
        if not isinstance(trigger_info, list) or not trigger_info:
            continue
        if not isinstance(trigger_info[0], list) or not trigger_info[0]:
            continue

        event_type = trigger_info[0][0]
        if not isinstance(event_type, str) or not event_type:
            continue
        event_label = normalize_label(event_type.replace(".n/a", ""))

        event_idx = len(events)
        events.append({"start": start, "end": end, "label": event_label})
        event_span_to_indices.setdefault((start, end), []).append(event_idx)

    if not events:
        return None

    for ent_span in data.get("ent_spans", []):
        if not isinstance(ent_span, list) or len(ent_span) < 3:
            continue
        start = ent_span[0]
        end = ent_span[1]
        if not is_valid_span(start, end, token_count):
            continue

        argument_idx = len(arguments)
        arguments.append({"start": start, "end": end})
        argument_span_to_indices.setdefault((start, end), []).append(argument_idx)

    argument_labels: set[str] = set()
    seen_relations: set[tuple[int, int, str]] = set()

    for link in data.get("gold_evt_links", []):
        if not isinstance(link, list) or len(link) != 3:
            continue

        event_span = link[0]
        argument_span = link[1]
        raw_role = link[2]
        if (
            not isinstance(event_span, list)
            or len(event_span) != 2
            or not isinstance(argument_span, list)
            or len(argument_span) != 2
            or not isinstance(raw_role, str)
        ):
            continue

        event_start, event_end = event_span
        argument_start, argument_end = argument_span
        if not is_valid_span(event_start, event_end, token_count):
            continue
        if not is_valid_span(argument_start, argument_end, token_count):
            continue

        relation_label = parse_role_label(raw_role)
        argument_labels.add(relation_label)

        event_indices = event_span_to_indices.get((event_start, event_end), [])
        if not event_indices:
            event_idx = len(events)
            event_indices = [event_idx]
            events.append(
                {
                    "start": event_start,
                    "end": event_end,
                    "label": "unknown_event",
                }
            )
            event_span_to_indices[(event_start, event_end)] = event_indices

        argument_indices = argument_span_to_indices.get(
            (argument_start, argument_end), []
        )
        if not argument_indices:
            argument_idx = len(arguments)
            argument_indices = [argument_idx]
            arguments.append({"start": argument_start, "end": argument_end})
            argument_span_to_indices[(argument_start, argument_end)] = argument_indices

        for event_idx in event_indices:
            for argument_idx in argument_indices:
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

    event_labels = sorted({event["label"] for event in events})

    return {
        "id": doc_id,
        "tokens": flat_tokens,
        "event_labels": event_labels,
        "argument_labels": sorted(argument_labels),
        "events": events,
        "arguments": arguments,
        "relations": relations,
        "metadata": {
            "dataset": "RAMS",
            "doc_id": doc_id,
            "split": data.get("split", ""),
            "source_url": data.get("source_url", ""),
            "event_count": len(events),
            "argument_count": len(arguments),
            "relation_count": len(relations),
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
    events = sample["events"]
    arguments = sample["arguments"]
    relations = sample["relations"]

    windowed_events: list[dict[str, Any]] = []
    event_index_map: dict[int, int] = {}
    for old_idx, event in enumerate(events):
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
    for old_idx, argument in enumerate(arguments):
        if window_start <= argument["start"] and argument["end"] < window_end:
            argument_index_map[old_idx] = len(windowed_arguments)
            windowed_arguments.append(
                {
                    "start": argument["start"] - window_start,
                    "end": argument["end"] - window_start,
                }
            )

    windowed_relations: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for relation in relations:
        old_event_idx = relation["event_idx"]
        old_argument_idx = relation["argument_idx"]
        if (
            old_event_idx not in event_index_map
            or old_argument_idx not in argument_index_map
        ):
            continue

        pair = (event_index_map[old_event_idx], argument_index_map[old_argument_idx])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        windowed_relations.append(
            {
                "event_idx": pair[0],
                "argument_idx": pair[1],
                "label": relation["label"],
            }
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
        "tokens": sample["tokens"][window_start:window_end],
        "event_labels": list(sample["event_labels"]),
        "argument_labels": list(sample["argument_labels"]),
        "events": windowed_events,
        "arguments": windowed_arguments,
        "relations": windowed_relations,
        "metadata": metadata,
    }


def convert_document(
    data: dict[str, Any],
    window_size: int | None = None,
    window_stride: int | None = None,
) -> list[dict[str, Any]]:
    sample = build_document_annotations(data)
    if sample is None:
        return []

    if window_size is None:
        return [sample]

    stride = window_stride if window_stride is not None else window_size
    windows = build_windows(len(sample["tokens"]), window_size, stride)
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
        stem = source_path.stem
        if source_path.suffix == ".jsonlines":
            return output_path / f"{stem}.jsonl"
        return output_path / source_path.name
    return output_path


def count_self_related_events(sample: dict[str, Any]) -> int:
    events = sample.get("events", [])
    arguments = sample.get("arguments", [])
    relations = sample.get("relations", [])

    self_related_event_indices: set[int] = set()
    for relation in relations:
        event_idx = relation.get("event_idx")
        argument_idx = relation.get("argument_idx")
        if (
            not isinstance(event_idx, int)
            or not isinstance(argument_idx, int)
            or event_idx < 0
            or argument_idx < 0
            or event_idx >= len(events)
            or argument_idx >= len(arguments)
        ):
            continue

        event = events[event_idx]
        argument = arguments[argument_idx]
        if event.get("start") == argument.get("start") and event.get(
            "end"
        ) == argument.get("end"):
            self_related_event_indices.add(event_idx)

    return len(self_related_event_indices)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RAMS dataset to reader-normalized JSONL format"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a RAMS JSONL/JSONLINES file or directory of split files",
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
    samples_without_relations = 0
    total_events = 0
    total_arguments = 0
    total_relations = 0
    total_self_related_events = 0

    for source_path, data in load_samples(input_path):
        processed += 1
        converted_samples = convert_document(
            data,
            window_size=args.window_size,
            window_stride=args.window_stride,
        )
        if not converted_samples:
            continue

        samples_by_source.setdefault(source_path, []).extend(converted_samples)
        written += len(converted_samples)

        for sample in converted_samples:
            event_count = len(sample["events"])
            argument_count = len(sample["arguments"])
            relation_count = len(sample["relations"])

            total_events += event_count
            total_arguments += argument_count
            total_relations += relation_count
            total_self_related_events += count_self_related_events(sample)
            if relation_count == 0:
                samples_without_relations += 1

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

    print(f"Processed {processed} samples")
    print(f"Wrote {written} normalized samples")
    if written > 0:
        avg_samples_without_relations = samples_without_relations / written
        avg_events_per_sample = total_events / written
        avg_arguments_per_sample = total_arguments / written
        avg_relations_per_sample = total_relations / written
        avg_self_related_events_per_sample = total_self_related_events / written

        print(
            "Average number of samples without relations per sample "
            f"(indicator mean): {avg_samples_without_relations:.4f}"
        )
        print(f"Average number of events per sample: {avg_events_per_sample:.4f}")
        print(f"Average number of arguments per sample: {avg_arguments_per_sample:.4f}")
        print(f"Average number of relations per sample: {avg_relations_per_sample:.4f}")
        print(
            "Average number of events with self-relations per sample: "
            f"{avg_self_related_events_per_sample:.4f}"
        )
    else:
        print(
            "Average number of samples without relations per sample (indicator mean): 0.0000"
        )
        print("Average number of events per sample: 0.0000")
        print("Average number of arguments per sample: 0.0000")
        print("Average number of relations per sample: 0.0000")
        print("Average number of events with self-relations per sample: 0.0000")


if __name__ == "__main__":
    main()
