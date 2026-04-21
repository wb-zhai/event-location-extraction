from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def build_sentence_offsets(token_sentences: list[list[str]]) -> list[int]:
    offsets: list[int] = []
    running_total = 0
    for sentence in token_sentences:
        offsets.append(running_total)
        running_total += len(sentence)
    return offsets


def globalize_span(
    sentence_offsets: list[int], sent_id: int, offset: list[int]
) -> tuple[int, int]:
    sentence_offset = sentence_offsets[sent_id]
    start = sentence_offset + offset[0]
    end = sentence_offset + offset[1] - 1
    return start, end


def relation_label_sets(data: dict[str, Any]) -> set[str]:
    labels = {
        f"temporal_{normalize_label(label)}"
        for label, pairs in data.get("temporal_relations", {}).items()
        if pairs
    }
    labels.update(
        f"causal_{normalize_label(label)}"
        for label, pairs in data.get("causal_relations", {}).items()
        if pairs
    )
    if data.get("subevent_relations"):
        labels.add("subevent")
    return labels


def build_document_annotations(data: dict[str, Any]) -> dict[str, Any] | None:
    token_sentences = data.get("tokens", [])
    flat_tokens = [token for sentence in token_sentences for token in sentence]
    sentence_offsets = build_sentence_offsets(token_sentences)

    events: list[dict[str, Any]] = []
    arguments: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    event_labels = sorted(
        {
            normalize_label(event["type"])
            for event in data.get("events", [])
            if event.get("type") and event.get("mention")
        }
    )
    argument_labels = sorted(relation_label_sets(data))

    event_mentions_by_id: dict[str, list[int]] = {}
    argument_mentions_by_id: dict[str, list[int]] = {}

    for event in data.get("events", []):
        event_type = event.get("type")
        if not event_type:
            continue
        normalized_type = normalize_label(event_type)
        mention_indices: list[int] = []
        mirrored_argument_indices: list[int] = []
        for mention in event.get("mention", []):
            sent_id = mention.get("sent_id")
            offset = mention.get("offset")
            if (
                not isinstance(sent_id, int)
                or sent_id < 0
                or sent_id >= len(sentence_offsets)
                or not isinstance(offset, list)
                or len(offset) != 2
            ):
                continue
            start, end = globalize_span(sentence_offsets, sent_id, offset)
            mention_indices.append(len(events))
            events.append({"start": start, "end": end, "label": normalized_type})
            mirrored_argument_indices.append(len(arguments))
            arguments.append({"start": start, "end": end})
        if mention_indices:
            event_mentions_by_id[event["id"]] = mention_indices
            argument_mentions_by_id[event["id"]] = mirrored_argument_indices

    if not events:
        return None

    for timex in data.get("TIMEX", []):
        sent_id = timex.get("sent_id")
        offset = timex.get("offset")
        timex_id = timex.get("id")
        if (
            not timex_id
            or not isinstance(sent_id, int)
            or sent_id < 0
            or sent_id >= len(sentence_offsets)
            or not isinstance(offset, list)
            or len(offset) != 2
        ):
            continue
        start, end = globalize_span(sentence_offsets, sent_id, offset)
        argument_mentions_by_id[timex_id] = [len(arguments)]
        arguments.append({"start": start, "end": end})

    seen_relations: set[tuple[int, int, str]] = set()

    def add_relations(
        pairs: list[list[str]],
        relation_label: str,
    ) -> None:
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            head_ids = event_mentions_by_id.get(pair[0], [])
            tail_ids = argument_mentions_by_id.get(pair[1], [])
            for event_idx in head_ids:
                for argument_idx in tail_ids:
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

    for label, pairs in data.get("temporal_relations", {}).items():
        add_relations(pairs, f"temporal_{normalize_label(label)}")
    for label, pairs in data.get("causal_relations", {}).items():
        add_relations(pairs, f"causal_{normalize_label(label)}")
    add_relations(data.get("subevent_relations", []), "subevent")

    return {
        "id": data["id"],
        "tokens": flat_tokens,
        "event_labels": event_labels,
        "argument_labels": argument_labels,
        "events": events,
        "arguments": arguments,
        "relations": relations,
        "metadata": {
            "dataset": "MAVEN-ERE",
            "doc_id": data["id"],
            "title": data.get("title", ""),
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
        pair = (
            event_index_map[old_event_idx],
            argument_index_map[old_argument_idx],
        )
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
        return output_path / source_path.name
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MAVEN-ERE dataset to reader-normalized JSONL format"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a MAVEN-ERE JSONL file or directory of JSONL split files",
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


if __name__ == "__main__":
    main()
