"""Dataset utilities for event/argument reader.

Assumption: normalized span indices use inclusive word boundaries, so a span with
``start=2`` and ``end=4`` covers ``tokens[2:5]``.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, get_worker_info

EVENT_MARKER = "[E]"
ARGUMENT_MARKER = "[A]"
DEFAULT_CANDIDATE_SAMPLING_SEED = 13

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpanAnnotation:
    start: int
    end: int
    label: str | None = None


@dataclass(frozen=True)
class RelationAnnotation:
    event_idx: int
    argument_idx: int
    label: str


@dataclass(frozen=True)
class NormalizedSample:
    sample_id: str
    tokens: list[str]
    event_labels: list[str]
    argument_labels: list[str]
    events: list[SpanAnnotation]
    arguments: list[SpanAnnotation]
    relations: list[RelationAnnotation]
    metadata: dict[str, Any]

    def to_reference(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "tokens": list(self.tokens),
            "events": [
                {"start": span.start, "end": span.end, "label": span.label}
                for span in self.events
            ],
            "arguments": [
                {"start": span.start, "end": span.end} for span in self.arguments
            ],
            "relations": [
                {
                    "event_idx": relation.event_idx,
                    "argument_idx": relation.argument_idx,
                    "label": relation.label,
                }
                for relation in self.relations
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CandidateOntology:
    event_labels: list[str]
    argument_labels: list[str]


@dataclass(frozen=True)
class CandidateSample:
    sample: NormalizedSample
    event_labels: list[str]
    argument_labels: list[str]
    required_event_labels: list[str]
    required_argument_labels: list[str]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from exc
    return records


def _validate_string_list(
    name: str,
    value: Any,
    sample_id: str,
    *,
    require_unique: bool = True,
) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{sample_id}: '{name}' must be a list[str]")
    if require_unique and len(set(value)) != len(value):
        raise ValueError(f"{sample_id}: '{name}' must not contain duplicates")
    return value


def _validate_span(
    span: Any,
    sample_id: str,
    field_name: str,
    token_count: int,
    require_label: bool,
) -> SpanAnnotation:
    if not isinstance(span, dict):
        raise ValueError(f"{sample_id}: every '{field_name}' item must be an object")
    if "start" not in span or "end" not in span:
        raise ValueError(
            f"{sample_id}: every '{field_name}' item needs 'start' and 'end'"
        )
    start = span["start"]
    end = span["end"]
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"{sample_id}: '{field_name}' span bounds must be integers")
    if start < 0 or end < start or end >= token_count:
        raise ValueError(
            f"{sample_id}: invalid {field_name} span [{start}, {end}] for {token_count} tokens"
        )
    label = span.get("label")
    if require_label:
        if not isinstance(label, str):
            raise ValueError(
                f"{sample_id}: every '{field_name}' item needs a string 'label'"
            )
    elif label is not None:
        raise ValueError(f"{sample_id}: '{field_name}' items must not define 'label'")
    return SpanAnnotation(start=start, end=end, label=label)


def _canonicalize_arguments(
    arguments: list[SpanAnnotation],
) -> tuple[list[SpanAnnotation], dict[int, int]]:
    canonical_arguments: list[SpanAnnotation] = []
    argument_index_map: dict[int, int] = {}
    argument_index_by_span: dict[tuple[int, int], int] = {}

    for old_index, argument in enumerate(arguments):
        span_key = (argument.start, argument.end)
        canonical_index = argument_index_by_span.get(span_key)
        if canonical_index is None:
            canonical_index = len(canonical_arguments)
            argument_index_by_span[span_key] = canonical_index
            canonical_arguments.append(argument)
        argument_index_map[old_index] = canonical_index

    return canonical_arguments, argument_index_map


def normalize_record(record: dict[str, Any]) -> NormalizedSample:
    required_fields = {
        "id",
        "tokens",
        "event_labels",
        "argument_labels",
        "events",
        "arguments",
        "relations",
        "metadata",
    }
    missing_fields = required_fields.difference(record)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Normalized record is missing required fields: {missing}")

    sample_id = record["id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Normalized record 'id' must be a non-empty string")

    tokens = _validate_string_list(
        "tokens", record["tokens"], sample_id, require_unique=False
    )
    if not tokens:
        raise ValueError(f"{sample_id}: 'tokens' must not be empty")

    event_labels = _validate_string_list(
        "event_labels", record["event_labels"], sample_id
    )
    argument_labels = _validate_string_list(
        "argument_labels", record["argument_labels"], sample_id
    )
    if not event_labels:
        raise ValueError(f"{sample_id}: 'event_labels' must not be empty")

    if not isinstance(record["events"], list):
        raise ValueError(f"{sample_id}: 'events' must be a list")
    events = [
        _validate_span(span, sample_id, "events", len(tokens), require_label=True)
        for span in record["events"]
    ]
    for event in events:
        if event.label not in event_labels:
            raise ValueError(
                f"{sample_id}: event label '{event.label}' is not present in event_labels"
            )

    if not isinstance(record["arguments"], list):
        raise ValueError(f"{sample_id}: 'arguments' must be a list")
    arguments = [
        _validate_span(span, sample_id, "arguments", len(tokens), require_label=False)
        for span in record["arguments"]
    ]
    arguments, argument_index_map = _canonicalize_arguments(arguments)

    if not isinstance(record["relations"], list):
        raise ValueError(f"{sample_id}: 'relations' must be a list")

    relations: list[RelationAnnotation] = []
    seen_relations: set[tuple[int, int, str]] = set()

    for relation in record["relations"]:
        if not isinstance(relation, dict):
            raise ValueError(f"{sample_id}: every 'relations' item must be an object")
        event_idx = relation.get("event_idx")
        argument_idx = relation.get("argument_idx")
        label = relation.get("label")
        if not isinstance(event_idx, int) or not isinstance(argument_idx, int):
            raise ValueError(
                f"{sample_id}: relation indices must be integers for events and arguments"
            )
        if not isinstance(label, str):
            raise ValueError(f"{sample_id}: relation labels must be strings")
        if not argument_labels:
            raise ValueError(
                f"{sample_id}: 'relations' must be empty when 'argument_labels' is empty"
            )
        if event_idx < 0 or event_idx >= len(events):
            raise ValueError(
                f"{sample_id}: relation event_idx {event_idx} is out of range"
            )
        if argument_idx < 0 or argument_idx >= len(record["arguments"]):
            raise ValueError(
                f"{sample_id}: relation argument_idx {argument_idx} is out of range"
            )
        if label not in argument_labels:
            raise ValueError(
                f"{sample_id}: relation label '{label}' is not present in argument_labels"
            )
        argument_idx = argument_index_map[argument_idx]
        relation_key = (event_idx, argument_idx, label)
        if relation_key in seen_relations:
            raise ValueError(
                f"{sample_id}: duplicate relation {(event_idx, argument_idx, label)} is not allowed"
            )
        seen_relations.add(relation_key)
        relations.append(
            RelationAnnotation(
                event_idx=event_idx, argument_idx=argument_idx, label=label
            )
        )

    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"{sample_id}: 'metadata' must be an object")

    return NormalizedSample(
        sample_id=sample_id,
        tokens=tokens,
        event_labels=event_labels,
        argument_labels=argument_labels,
        events=events,
        arguments=arguments,
        relations=relations,
        metadata=metadata,
    )


def load_normalized_jsonl(path: str | Path) -> list[NormalizedSample]:
    return [normalize_record(record) for record in read_jsonl(path)]


def load_candidate_ontology(path: str | Path) -> CandidateOntology:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Candidate ontology file '{path}' must contain a JSON object")
    if "event_labels" not in payload or "argument_labels" not in payload:
        raise ValueError(
            f"Candidate ontology file '{path}' must define both 'event_labels' and 'argument_labels'"
        )

    event_labels = _validate_string_list(
        "event_labels",
        payload.get("event_labels", []),
        str(path),
    )
    argument_labels = _validate_string_list(
        "argument_labels",
        payload.get("argument_labels", []),
        str(path),
    )
    return CandidateOntology(
        event_labels=event_labels,
        argument_labels=argument_labels,
    )


def _labels_in_document_order(
    document_labels: list[str], required_labels: list[str]
) -> list[str]:
    required_set = set(required_labels)
    ordered = [label for label in document_labels if label in required_set]
    missing = [label for label in required_labels if label not in set(ordered)]
    return ordered + missing


def _extract_required_event_labels(sample: NormalizedSample) -> list[str]:
    return _labels_in_document_order(
        sample.event_labels,
        [event.label for event in sample.events if event.label is not None],
    )


def _extract_required_argument_labels(sample: NormalizedSample) -> list[str]:
    return _labels_in_document_order(
        sample.argument_labels,
        [relation.label for relation in sample.relations],
    )


def _build_candidate_labels(
    *,
    sample_id: str,
    label_kind: str,
    required_labels: list[str],
    document_labels: list[str],
    ontology_labels: list[str] | None,
    requested_total: int | None,
    rng: random.Random,
) -> list[str]:
    if requested_total is None:
        return list(document_labels)

    required_set = set(required_labels)
    missing_required_labels = [
        label for label in required_labels if label not in document_labels
    ]
    if missing_required_labels:
        raise ValueError(
            f"{sample_id}: required {label_kind} labels {missing_required_labels} are missing from document-level labels"
        )

    if requested_total < len(required_labels):
        LOGGER.warning(
            "%s: requested %s %s candidates but sample has %s required labels from annotations; overriding the limit for this sample",
            sample_id,
            requested_total,
            label_kind,
            len(required_labels),
        )
        return list(required_labels)

    if ontology_labels is not None:
        missing_required_in_ontology = [
            label for label in required_labels if label not in ontology_labels
        ]
        if missing_required_in_ontology:
            raise ValueError(
                f"{sample_id}: required {label_kind} labels {missing_required_in_ontology} are missing from the ontology"
            )

    selected_labels = list(required_labels)
    remaining_capacity = requested_total - len(selected_labels)
    if remaining_capacity <= 0:
        return selected_labels

    document_extra_labels = [
        label for label in document_labels if label not in required_set
    ]
    if document_extra_labels:
        selected_labels.extend(document_extra_labels[:remaining_capacity])
        remaining_capacity = requested_total - len(selected_labels)

    if remaining_capacity <= 0 or ontology_labels is None:
        return selected_labels

    selected_set = set(selected_labels)
    ontology_extra_pool = [
        label for label in ontology_labels if label not in selected_set
    ]
    if not ontology_extra_pool:
        return selected_labels

    sample_size = min(remaining_capacity, len(ontology_extra_pool))
    if sample_size <= 0:
        return selected_labels
    selected_labels.extend(rng.sample(ontology_extra_pool, sample_size))
    return selected_labels


def _apply_training_candidate_transform(
    *,
    labels: list[str],
    required_labels: list[str],
    ontology_labels: list[str] | None,
    shuffle_probability: float,
    gold_dropout_probability: float,
    rng: random.Random,
) -> list[str]:
    transformed_labels = list(labels)
    if not transformed_labels:
        return transformed_labels

    if required_labels and ontology_labels and gold_dropout_probability > 0.0:
        required_set = set(required_labels)
        refillable_negatives = [
            label
            for label in ontology_labels
            if label not in required_set and label not in transformed_labels
        ]
        dropped_required_labels = [
            label for label in required_labels if rng.random() < gold_dropout_probability
        ]
        max_required_drops = min(
            len(refillable_negatives),
            max(len(required_labels) - 1, 0),
        )
        if len(dropped_required_labels) > max_required_drops:
            rng.shuffle(dropped_required_labels)
            dropped_required_labels = dropped_required_labels[:max_required_drops]

        if dropped_required_labels:
            dropped_required_set = set(dropped_required_labels)
            transformed_labels = [
                label for label in transformed_labels if label not in dropped_required_set
            ]
            transformed_labels.extend(
                rng.sample(refillable_negatives, len(dropped_required_labels))
            )

    if len(transformed_labels) > 1 and rng.random() < shuffle_probability:
        rng.shuffle(transformed_labels)

    return transformed_labels


def _make_candidate_sample(
    sample: NormalizedSample,
    event_labels: list[str],
    argument_labels: list[str],
    required_event_labels: list[str],
    required_argument_labels: list[str],
) -> CandidateSample:
    return CandidateSample(
        sample=sample,
        event_labels=event_labels,
        argument_labels=argument_labels,
        required_event_labels=required_event_labels,
        required_argument_labels=required_argument_labels,
    )


def _materialize_candidate_sample(
    candidate_sample: CandidateSample,
) -> NormalizedSample:
    return replace(
        candidate_sample.sample,
        event_labels=list(candidate_sample.event_labels),
        argument_labels=list(candidate_sample.argument_labels),
    )


def _tokenize_words_with_alignment(
    tokenizer: Any,
    words: list[str],
    sample_id: str,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            f"{sample_id}: reader encoding requires a fast tokenizer with word_ids() support"
        )

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=False,
    )
    if not hasattr(encoding, "word_ids"):
        raise ValueError(
            f"{sample_id}: tokenizer output does not expose word_ids() alignment"
        )

    piece_ids = encoding["input_ids"]
    word_ids = encoding.word_ids()
    if not isinstance(piece_ids, list) or not isinstance(word_ids, list):
        raise ValueError(f"{sample_id}: tokenizer alignment returned an invalid shape")
    if len(piece_ids) != len(word_ids):
        raise ValueError(
            f"{sample_id}: tokenizer alignment length mismatch between pieces and word ids"
        )
    if not piece_ids:
        raise ValueError(f"{sample_id}: tokenizer produced no document pieces")

    word_to_token_start = [-1] * len(words)
    word_to_token_end = [-1] * len(words)
    token_to_word: list[int] = []
    word_start_mask: list[int] = []
    word_end_mask: list[int] = []

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or not isinstance(word_idx, int):
            raise ValueError(
                f"{sample_id}: tokenizer alignment produced an unexpected non-word token"
            )
        if word_idx < 0 or word_idx >= len(words):
            raise ValueError(
                f"{sample_id}: tokenizer alignment produced an out-of-range word index"
            )
        token_to_word.append(word_idx)
        if word_to_token_start[word_idx] == -1:
            word_to_token_start[word_idx] = token_idx
            word_start_mask.append(1)
        else:
            word_start_mask.append(0)
        next_word_idx = word_ids[token_idx + 1] if token_idx + 1 < len(word_ids) else None
        is_word_end = next_word_idx != word_idx
        word_end_mask.append(1 if is_word_end else 0)
        if is_word_end:
            word_to_token_end[word_idx] = token_idx

    missing_words = [
        str(word_idx)
        for word_idx, (start, end) in enumerate(
            zip(word_to_token_start, word_to_token_end, strict=True)
        )
        if start == -1 or end == -1
    ]
    if missing_words:
        raise ValueError(
            f"{sample_id}: tokenizer failed to align words at indices {', '.join(missing_words)}"
        )

    return (
        piece_ids,
        word_to_token_start,
        word_to_token_end,
        token_to_word,
        word_start_mask,
        word_end_mask,
    )


def _tokenize_label(tokenizer: Any, label: str) -> list[int]:
    pieces = tokenizer(label, add_special_tokens=False)["input_ids"]
    if pieces:
        return pieces
    if tokenizer.unk_token_id is None:
        raise ValueError(f"Tokenizer produced no pieces for label '{label}'")
    return [tokenizer.unk_token_id]


def _trim_sample_to_word_count(
    sample: NormalizedSample, keep_words: int
) -> NormalizedSample:
    events = [
        SpanAnnotation(start=span.start, end=span.end, label=span.label)
        for span in sample.events
        if span.end < keep_words
    ]
    arguments = [
        SpanAnnotation(start=span.start, end=span.end, label=None)
        for span in sample.arguments
        if span.end < keep_words
    ]
    valid_event_indices = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(
            idx for idx, span in enumerate(sample.events) if span.end < keep_words
        )
    }
    valid_argument_indices = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(
            idx for idx, span in enumerate(sample.arguments) if span.end < keep_words
        )
    }
    relations = [
        RelationAnnotation(
            event_idx=valid_event_indices[relation.event_idx],
            argument_idx=valid_argument_indices[relation.argument_idx],
            label=relation.label,
        )
        for relation in sample.relations
        if relation.event_idx in valid_event_indices
        and relation.argument_idx in valid_argument_indices
    ]
    metadata = dict(sample.metadata)
    metadata["truncated"] = keep_words < len(sample.tokens)
    metadata["original_token_count"] = len(sample.tokens)
    return NormalizedSample(
        sample_id=sample.sample_id,
        tokens=sample.tokens[:keep_words],
        event_labels=list(sample.event_labels),
        argument_labels=list(sample.argument_labels),
        events=events,
        arguments=arguments,
        relations=relations,
        metadata=metadata,
    )


def _fit_sample_to_max_length(
    sample: NormalizedSample,
    tokenizer: Any,
    max_length: int,
) -> NormalizedSample:
    event_bank_length = sum(
        1 + len(_tokenize_label(tokenizer, label)) for label in sample.event_labels
    )
    argument_bank_length = sum(
        1 + len(_tokenize_label(tokenizer, label)) for label in sample.argument_labels
    )
    reserved_length = 4 + event_bank_length + argument_bank_length
    if reserved_length >= max_length:
        raise ValueError(
            f"{sample.sample_id}: label banks consume {reserved_length} tokens, which exceeds max_length={max_length}"
        )

    available_doc_pieces = max_length - reserved_length
    _, _, word_to_token_end, _, _, _ = _tokenize_words_with_alignment(
        tokenizer, sample.tokens, sample.sample_id
    )
    kept_word_count = 0
    for word_end in word_to_token_end:
        if word_end >= available_doc_pieces:
            break
        kept_word_count += 1

    if kept_word_count == 0:
        raise ValueError(
            f"{sample.sample_id}: no document tokens fit within max_length={max_length}"
        )

    return _trim_sample_to_word_count(sample, kept_word_count)


def encode_sample(
    sample: NormalizedSample,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    sample = _fit_sample_to_max_length(sample, tokenizer, max_length)
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    if cls_token_id is None or sep_token_id is None or pad_token_id is None:
        raise ValueError(
            "Tokenizer must define cls_token_id, sep_token_id, and pad_token_id"
        )

    event_marker_id = tokenizer.convert_tokens_to_ids(EVENT_MARKER)
    argument_marker_id = tokenizer.convert_tokens_to_ids(ARGUMENT_MARKER)
    if (
        event_marker_id == tokenizer.unk_token_id
        or argument_marker_id == tokenizer.unk_token_id
    ):
        raise ValueError("Tokenizer is missing reader marker tokens")

    (
        document_piece_ids,
        word_to_token_start,
        word_to_token_end,
        document_token_to_word,
        document_word_start_mask,
        document_word_end_mask,
    ) = _tokenize_words_with_alignment(tokenizer, sample.tokens, sample.sample_id)

    input_ids = [cls_token_id]
    attention_mask = [1]
    token_to_word = [-1]
    word_start_mask = [0]
    word_end_mask = [0]
    input_ids.extend(document_piece_ids)
    attention_mask.extend([1] * len(document_piece_ids))
    token_to_word.extend(document_token_to_word)
    word_start_mask.extend(document_word_start_mask)
    word_end_mask.extend(document_word_end_mask)
    input_ids.append(sep_token_id)
    attention_mask.append(1)
    token_to_word.append(-1)
    word_start_mask.append(0)
    word_end_mask.append(0)

    document_offset = 1
    token_word_starts = [document_offset + index for index in word_to_token_start]
    token_word_ends = [document_offset + index for index in word_to_token_end]

    event_marker_positions: list[int] = []
    event_label_token_starts: list[int] = []
    event_label_token_ends: list[int] = []
    for label in sample.event_labels:
        label_start = len(input_ids)
        event_marker_positions.append(label_start)
        event_label_token_starts.append(label_start)
        input_ids.append(event_marker_id)
        attention_mask.append(1)
        token_to_word.append(-1)
        word_start_mask.append(0)
        word_end_mask.append(0)
        label_piece_ids = _tokenize_label(tokenizer, label)
        input_ids.extend(label_piece_ids)
        attention_mask.extend([1] * len(label_piece_ids))
        token_to_word.extend([-1] * len(label_piece_ids))
        word_start_mask.extend([0] * len(label_piece_ids))
        word_end_mask.extend([0] * len(label_piece_ids))
        event_label_token_ends.append(len(input_ids) - 1)

    input_ids.append(sep_token_id)
    attention_mask.append(1)
    token_to_word.append(-1)
    word_start_mask.append(0)
    word_end_mask.append(0)

    argument_marker_positions: list[int] = []
    argument_label_token_starts: list[int] = []
    argument_label_token_ends: list[int] = []
    for label in sample.argument_labels:
        label_start = len(input_ids)
        argument_marker_positions.append(label_start)
        argument_label_token_starts.append(label_start)
        input_ids.append(argument_marker_id)
        attention_mask.append(1)
        token_to_word.append(-1)
        word_start_mask.append(0)
        word_end_mask.append(0)
        label_piece_ids = _tokenize_label(tokenizer, label)
        input_ids.extend(label_piece_ids)
        attention_mask.extend([1] * len(label_piece_ids))
        token_to_word.extend([-1] * len(label_piece_ids))
        word_start_mask.extend([0] * len(label_piece_ids))
        word_end_mask.extend([0] * len(label_piece_ids))
        argument_label_token_ends.append(len(input_ids) - 1)

    input_ids.append(sep_token_id)
    attention_mask.append(1)
    token_to_word.append(-1)
    word_start_mask.append(0)
    word_end_mask.append(0)

    event_start_labels = [-100] * len(input_ids)
    event_end_labels = [-100] * len(input_ids)
    argument_start_labels = [-100] * len(input_ids)
    argument_end_labels = [-100] * len(input_ids)
    for token_index, is_word_start in enumerate(word_start_mask):
        if is_word_start:
            event_start_labels[token_index] = 0
            argument_start_labels[token_index] = 0
    for token_index, is_word_end in enumerate(word_end_mask):
        if is_word_end:
            event_end_labels[token_index] = 0
            argument_end_labels[token_index] = 0

    event_label_to_index = {label: idx for idx, label in enumerate(sample.event_labels)}
    argument_label_to_index = {
        label: idx for idx, label in enumerate(sample.argument_labels)
    }

    gold_event_token_starts: list[int] = []
    gold_event_token_ends: list[int] = []
    gold_event_type_labels: list[int] = []
    for span in sample.events:
        start_token = token_word_starts[span.start]
        end_token = token_word_ends[span.end]
        event_start_labels[start_token] = 1
        event_end_labels[end_token] = 1
        gold_event_token_starts.append(start_token)
        gold_event_token_ends.append(end_token)
        gold_event_type_labels.append(event_label_to_index.get(span.label or "", -100))

    gold_argument_token_starts: list[int] = []
    gold_argument_token_ends: list[int] = []
    for span in sample.arguments:
        start_token = token_word_starts[span.start]
        end_token = token_word_ends[span.end]
        argument_start_labels[start_token] = 1
        argument_end_labels[end_token] = 1
        gold_argument_token_starts.append(start_token)
        gold_argument_token_ends.append(end_token)

    relation_targets = [
        (
            relation.event_idx,
            relation.argument_idx,
            argument_label_to_index[relation.label],
        )
        for relation in sample.relations
        if relation.label in argument_label_to_index
    ]

    return {
        "sample_id": sample.sample_id,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_to_word": token_to_word,
        "word_start_mask": word_start_mask,
        "word_end_mask": word_end_mask,
        "event_marker_positions": event_marker_positions,
        "argument_marker_positions": argument_marker_positions,
        "event_label_token_starts": event_label_token_starts,
        "event_label_token_ends": event_label_token_ends,
        "argument_label_token_starts": argument_label_token_starts,
        "argument_label_token_ends": argument_label_token_ends,
        "event_start_labels": event_start_labels,
        "event_end_labels": event_end_labels,
        "argument_start_labels": argument_start_labels,
        "argument_end_labels": argument_end_labels,
        "gold_event_token_starts": gold_event_token_starts,
        "gold_event_token_ends": gold_event_token_ends,
        "gold_event_type_labels": gold_event_type_labels,
        "gold_argument_token_starts": gold_argument_token_starts,
        "gold_argument_token_ends": gold_argument_token_ends,
        "relation_targets": relation_targets,
        "event_label_texts": list(sample.event_labels),
        "argument_label_texts": list(sample.argument_labels),
        "reference": sample.to_reference(),
    }


class EventReaderDataset(Dataset):
    def __init__(
        self,
        samples: list[NormalizedSample],
        tokenizer: Any,
        max_length: int,
        *,
        ontology: CandidateOntology | None = None,
        num_event_candidates: int | None = None,
        num_relation_candidates: int | None = None,
        is_training: bool = False,
        candidate_shuffle_probability: float = 0.0,
        gold_candidate_dropout_probability: float = 0.0,
        random_seed: int = DEFAULT_CANDIDATE_SAMPLING_SEED,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.ontology = ontology
        self.num_event_candidates = num_event_candidates
        self.num_relation_candidates = num_relation_candidates
        self.candidate_shuffle_probability = candidate_shuffle_probability
        self.gold_candidate_dropout_probability = gold_candidate_dropout_probability
        self.random_seed = random_seed

        candidate_rng = random.Random(random_seed)
        ontology_event_labels = ontology.event_labels if ontology is not None else None
        ontology_argument_labels = (
            ontology.argument_labels if ontology is not None else None
        )
        self.samples = [
            _make_candidate_sample(
                sample,
                _build_candidate_labels(
                    sample_id=sample.sample_id,
                    label_kind="event",
                    required_labels=_extract_required_event_labels(sample),
                    document_labels=sample.event_labels,
                    ontology_labels=ontology_event_labels,
                    requested_total=num_event_candidates,
                    rng=candidate_rng,
                ),
                _build_candidate_labels(
                    sample_id=sample.sample_id,
                    label_kind="relation",
                    required_labels=_extract_required_argument_labels(sample),
                    document_labels=sample.argument_labels,
                    ontology_labels=ontology_argument_labels,
                    requested_total=num_relation_candidates,
                    rng=candidate_rng,
                ),
                _extract_required_event_labels(sample),
                _extract_required_argument_labels(sample),
            )
            for sample in samples
        ]

    def _item_rng(self, index: int) -> random.Random:
        worker_info = get_worker_info()
        worker_seed = worker_info.seed if worker_info is not None else torch.initial_seed()
        return random.Random(
            f"{self.random_seed}:{worker_seed}:{index}:{self.samples[index].sample.sample_id}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        candidate_sample = self.samples[index]
        event_labels = list(candidate_sample.event_labels)
        argument_labels = list(candidate_sample.argument_labels)
        if self.is_training and self.ontology is not None:
            rng = self._item_rng(index)
            event_labels = _apply_training_candidate_transform(
                labels=event_labels,
                required_labels=candidate_sample.required_event_labels,
                ontology_labels=(
                    self.ontology.event_labels
                    if self.num_event_candidates is not None
                    else None
                ),
                shuffle_probability=(
                    self.candidate_shuffle_probability
                    if self.num_event_candidates is not None
                    else 0.0
                ),
                gold_dropout_probability=(
                    self.gold_candidate_dropout_probability
                    if self.num_event_candidates is not None
                    else 0.0
                ),
                rng=rng,
            )
            argument_labels = _apply_training_candidate_transform(
                labels=argument_labels,
                required_labels=candidate_sample.required_argument_labels,
                ontology_labels=(
                    self.ontology.argument_labels
                    if self.num_relation_candidates is not None
                    else None
                ),
                shuffle_probability=(
                    self.candidate_shuffle_probability
                    if self.num_relation_candidates is not None
                    else 0.0
                ),
                gold_dropout_probability=(
                    self.gold_candidate_dropout_probability
                    if self.num_relation_candidates is not None
                    else 0.0
                ),
                rng=rng,
            )

        return encode_sample(
            _materialize_candidate_sample(
                _make_candidate_sample(
                    candidate_sample.sample,
                    event_labels,
                    argument_labels,
                    candidate_sample.required_event_labels,
                    candidate_sample.required_argument_labels,
                )
            ),
            self.tokenizer,
            self.max_length,
        )


class EventReaderCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    @staticmethod
    def _pad_1d(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        max_length = max((len(sequence) for sequence in sequences), default=0)
        tensor = torch.full((len(sequences), max_length), pad_value, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            if sequence:
                tensor[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        return tensor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = {
            "input_ids": self._pad_1d(
                [item["input_ids"] for item in features], self.pad_token_id
            ),
            "attention_mask": self._pad_1d(
                [item["attention_mask"] for item in features], 0
            ),
            "token_to_word": self._pad_1d(
                [item["token_to_word"] for item in features], -1
            ),
            "word_start_mask": self._pad_1d(
                [item["word_start_mask"] for item in features], 0
            ),
            "word_end_mask": self._pad_1d(
                [item["word_end_mask"] for item in features], 0
            ),
            "event_marker_positions": self._pad_1d(
                [item["event_marker_positions"] for item in features], -1
            ),
            "argument_marker_positions": self._pad_1d(
                [item["argument_marker_positions"] for item in features], -1
            ),
            "event_label_token_starts": self._pad_1d(
                [item["event_label_token_starts"] for item in features], -1
            ),
            "event_label_token_ends": self._pad_1d(
                [item["event_label_token_ends"] for item in features], -1
            ),
            "argument_label_token_starts": self._pad_1d(
                [item["argument_label_token_starts"] for item in features], -1
            ),
            "argument_label_token_ends": self._pad_1d(
                [item["argument_label_token_ends"] for item in features], -1
            ),
            "event_start_labels": self._pad_1d(
                [item["event_start_labels"] for item in features], -100
            ),
            "event_end_labels": self._pad_1d(
                [item["event_end_labels"] for item in features], -100
            ),
            "argument_start_labels": self._pad_1d(
                [item["argument_start_labels"] for item in features], -100
            ),
            "argument_end_labels": self._pad_1d(
                [item["argument_end_labels"] for item in features], -100
            ),
            "gold_event_token_starts": self._pad_1d(
                [item["gold_event_token_starts"] for item in features], -1
            ),
            "gold_event_token_ends": self._pad_1d(
                [item["gold_event_token_ends"] for item in features], -1
            ),
            "gold_event_type_labels": self._pad_1d(
                [item["gold_event_type_labels"] for item in features], -100
            ),
            "gold_argument_token_starts": self._pad_1d(
                [item["gold_argument_token_starts"] for item in features], -1
            ),
            "gold_argument_token_ends": self._pad_1d(
                [item["gold_argument_token_ends"] for item in features], -1
            ),
            "sample_ids": [item["sample_id"] for item in features],
            "event_label_texts": [item["event_label_texts"] for item in features],
            "argument_label_texts": [item["argument_label_texts"] for item in features],
            "references": [item["reference"] for item in features],
        }

        batch_size = len(features)
        max_events = batch["gold_event_token_starts"].shape[1]
        max_arguments = batch["gold_argument_token_starts"].shape[1]
        max_roles = batch["argument_marker_positions"].shape[1]
        relation_labels = torch.full(
            (batch_size, max_events, max_arguments, max_roles),
            -100,
            dtype=torch.long,
        )

        for batch_index, item in enumerate(features):
            event_count = len(item["gold_event_token_starts"])
            argument_count = len(item["gold_argument_token_starts"])
            role_count = len(item["argument_label_texts"])
            if event_count == 0 or argument_count == 0 or role_count == 0:
                continue
            relation_labels[batch_index, :event_count, :argument_count, :role_count] = 0
            for event_idx, argument_idx, role_idx in item["relation_targets"]:
                relation_labels[batch_index, event_idx, argument_idx, role_idx] = 1

        batch["relation_labels"] = relation_labels
        return batch
