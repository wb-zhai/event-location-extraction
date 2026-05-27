from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from src.data.dataset import (
    DEFAULT_CANDIDATE_SAMPLING_SEED,
    _apply_training_candidate_transform,
    _build_candidate_labels,
)
from src.sft_prompt import render_chat, build_messages


@dataclass(frozen=True)
class SftCandidateOntology:
    event_descriptions: dict[str, str]
    argument_role_descriptions: dict[str, str]
    location_type_descriptions: dict[str, str]

    @property
    def event_labels(self) -> list[str]:
        return list(self.event_descriptions)

    @property
    def argument_role_labels(self) -> list[str]:
        return list(self.argument_role_descriptions)

    @property
    def location_type_labels(self) -> list[str]:
        return list(self.location_type_descriptions)


def _safe_substring(text: str, start: Any, end: Any) -> str:
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    if start < 0 or end <= start or end > len(text):
        return ""
    return text[start:end]


def _coerce_span(
    document: str,
    span_like: dict[str, Any] | None,
    *,
    start: Any = None,
    end: Any = None,
    text: Any = None,
) -> dict[str, Any]:
    if isinstance(span_like, dict):
        start = span_like.get("start", start)
        end = span_like.get("end", end)
        text = span_like.get("text", text)

    span_text = _safe_substring(document, start, end)
    normalized = {
        "text": span_text if span_text else (text if isinstance(text, str) else ""),
    }
    if start is not None:
        normalized["start"] = start
    if end is not None:
        normalized["end"] = end
    if isinstance(span_like, dict):
        for field_name in ("left_context", "right_context"):
            value = span_like.get(field_name)
            if isinstance(value, str) and value:
                normalized[field_name] = value
    return normalized


def _load_sft_candidate_ontology(path: str | Path) -> SftCandidateOntology:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Ontology file '{path}' must contain a JSON object")

    def _require_description_map(field_name: str) -> dict[str, str]:
        value = payload.get(field_name)
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(description, str)
            for key, description in value.items()
        ):
            raise ValueError(
                f"Ontology file '{path}' must define '{field_name}' as an object[str, str]"
            )
        return dict(value)

    return SftCandidateOntology(
        event_descriptions=_require_description_map("events"),
        argument_role_descriptions=_require_description_map("argument_roles"),
        location_type_descriptions=_require_description_map("location_types"),
    )


def _normalize_candidate_count(name: str, value: int | None) -> int | None:
    if value in (None, -1):
        return None
    if value < 0:
        raise ValueError(f"{name} must be -1, 0, or a positive integer, got {value}")
    return value


def _normalize_max_empty_event_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError(
            f"--max_empty_event_ratio must be >= 0 when provided, got {value}"
        )
    return value


def _select_label_descriptions(
    labels: list[str],
    descriptions: dict[str, str] | None,
    *,
    include_descriptions: bool,
) -> list[str] | dict[str, str]:
    if not include_descriptions or descriptions is None:
        return labels
    return {
        label: descriptions.get(label, "") for label in labels
    }


def _enrich_events(document: str, events: list[dict[str, Any]], *, events_only: bool = False) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for event in events:
        trigger = _coerce_span(
            document,
            event.get("trigger"),
            start=event.get("start"),
            end=event.get("end"),
            text=event.get("text"),
        )
        out_event = {
            "event_type": event.get("event_type", ""),
            "trigger": trigger,
        }
        
        if not events_only:
            out_event["arguments"] = []
            for arg in event.get("arguments", []):
                span = _coerce_span(
                    document,
                    arg.get("span"),
                    start=arg.get("start"),
                    end=arg.get("end"),
                    text=arg.get("text"),
                )
                out_event["arguments"].append(
                    {
                        "role": arg.get("role", ""),
                        "span": span,
                        "location_type": arg.get("location_type", ""),
                    }
                )
            out_event["arguments"].sort(
                key=lambda x: (
                    x["span"].get("start", 10**9),
                    x["span"].get("end", 10**9),
                    x.get("role", ""),
                    x.get("location_type", ""),
                )
            )

        enriched.append(out_event)

    enriched.sort(
        key=lambda x: (
            x["trigger"].get("start", 10**9),
            x["trigger"].get("end", 10**9),
            x.get("event_type", ""),
        )
    )
    return enriched


def _extract_required_labels(
    events: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    event_labels: list[str] = []
    argument_role_labels: list[str] = []
    location_type_labels: list[str] = []
    seen_event_labels: set[str] = set()
    seen_argument_role_labels: set[str] = set()
    seen_location_type_labels: set[str] = set()

    for event in events:
        event_type = event.get("event_type")
        if (
            isinstance(event_type, str)
            and event_type
            and event_type not in seen_event_labels
        ):
            seen_event_labels.add(event_type)
            event_labels.append(event_type)

        for argument in event.get("arguments", []):
            role = argument.get("role")
            if (
                isinstance(role, str)
                and role
                and role not in seen_argument_role_labels
            ):
                seen_argument_role_labels.add(role)
                argument_role_labels.append(role)

            location_type = argument.get("location_type")
            if (
                isinstance(location_type, str)
                and location_type
                and location_type not in seen_location_type_labels
            ):
                seen_location_type_labels.add(location_type)
                location_type_labels.append(location_type)

    return event_labels, argument_role_labels, location_type_labels


def _require_label_list(
    row: dict[str, Any],
    field_name: str,
    *,
    fallback_labels: list[str] | None = None,
) -> list[str]:
    labels = row.get(field_name)
    if labels is None:
        if fallback_labels is not None:
            return list(fallback_labels)
        raise ValueError(
            f"SFT row is missing a valid '{field_name}' list required for prompt candidate labels"
        )
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError(
            f"SFT row is missing a valid '{field_name}' list required for prompt candidate labels"
        )
    if len(labels) != len(set(labels)):
        raise ValueError(f"SFT row '{field_name}' must not contain duplicates")
    return labels


def _item_rng(random_seed: int, index: int, sample_id: str) -> random.Random:
    return random.Random(f"{random_seed}:{index}:{sample_id}")


def _resolve_candidate_labels(
    row: dict[str, Any],
    *,
    ontology: SftCandidateOntology | None,
    num_event_candidates: int | None,
    num_relation_candidates: int | None,
    is_training: bool,
    candidate_shuffle_probability: float,
    gold_candidate_dropout_probability: float,
    random_seed: int,
    candidate_rng: random.Random,
    index: int,
) -> tuple[list[str], list[str], list[str]]:
    raw_events = row["answer"]["events"]
    (
        required_event_labels,
        required_argument_role_labels,
        required_location_type_labels,
    ) = _extract_required_labels(raw_events)
    document = row.get("question", "")
    sample_id = row.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        sample_id = document if isinstance(document, str) and document else str(index)

    document_event_labels = _require_label_list(
        row,
        "events",
        fallback_labels=required_event_labels,
    )
    document_argument_role_labels = _require_label_list(
        row,
        "argument_roles",
        fallback_labels=required_argument_role_labels,
    )
    document_location_type_labels = _require_label_list(
        row,
        "location_types",
        fallback_labels=required_location_type_labels,
    )
    requested_event_total = num_event_candidates
    if requested_event_total is None and ontology is not None:
        requested_event_total = len(ontology.event_labels)
    requested_relation_total = num_relation_candidates
    if requested_relation_total is None and ontology is not None:
        requested_relation_total = len(ontology.argument_role_labels)
    requested_location_type_total = num_relation_candidates
    if requested_location_type_total is None and ontology is not None:
        requested_location_type_total = len(ontology.location_type_labels)

    event_labels = _build_candidate_labels(
        sample_id=sample_id,
        label_kind="event",
        required_labels=required_event_labels,
        document_labels=document_event_labels,
        ontology_labels=ontology.event_labels if ontology is not None else None,
        requested_total=requested_event_total,
        rng=candidate_rng,
    )
    argument_role_labels = _build_candidate_labels(
        sample_id=sample_id,
        label_kind="argument role",
        required_labels=required_argument_role_labels,
        document_labels=document_argument_role_labels,
        ontology_labels=(
            ontology.argument_role_labels if ontology is not None else None
        ),
        requested_total=requested_relation_total,
        rng=candidate_rng,
    )
    location_type_labels = _build_candidate_labels(
        sample_id=sample_id,
        label_kind="location type",
        required_labels=required_location_type_labels,
        document_labels=document_location_type_labels,
        ontology_labels=(
            ontology.location_type_labels if ontology is not None else None
        ),
        requested_total=requested_location_type_total,
        rng=candidate_rng,
    )

    if is_training and ontology is not None:
        rng = _item_rng(random_seed, index, sample_id)
        event_labels = _apply_training_candidate_transform(
            labels=event_labels,
            required_labels=required_event_labels,
            ontology_labels=(
                ontology.event_labels if num_event_candidates is not None else None
            ),
            shuffle_probability=(
                candidate_shuffle_probability
                if num_event_candidates is not None
                else 0.0
            ),
            gold_dropout_probability=(
                gold_candidate_dropout_probability
                if num_event_candidates is not None
                else 0.0
            ),
            rng=rng,
        )
        argument_role_labels = _apply_training_candidate_transform(
            labels=argument_role_labels,
            required_labels=required_argument_role_labels,
            ontology_labels=(
                ontology.argument_role_labels
                if num_relation_candidates is not None
                else None
            ),
            shuffle_probability=(
                candidate_shuffle_probability
                if num_relation_candidates is not None
                else 0.0
            ),
            gold_dropout_probability=(
                gold_candidate_dropout_probability
                if num_relation_candidates is not None
                else 0.0
            ),
            rng=rng,
        )
        location_type_labels = _apply_training_candidate_transform(
            labels=location_type_labels,
            required_labels=required_location_type_labels,
            ontology_labels=(
                ontology.location_type_labels
                if num_relation_candidates is not None
                else None
            ),
            shuffle_probability=(
                candidate_shuffle_probability
                if num_relation_candidates is not None
                else 0.0
            ),
            gold_dropout_probability=(
                gold_candidate_dropout_probability
                if num_relation_candidates is not None
                else 0.0
            ),
            rng=rng,
        )

    return event_labels, argument_role_labels, location_type_labels


def _chat_text(
    tokenizer,
    document: str,
    event_labels: list[str],
    argument_role_labels: list[str],
    location_type_labels: list[str],
    *,
    ontology: SftCandidateOntology | None,
    include_descriptions: bool,
    answer_obj: dict[str, Any],
    events_only: bool = False,
    omit_offsets: bool = False,
) -> str:
    return render_chat(
        tokenizer,
        document,
        _select_label_descriptions(
            event_labels,
            ontology.event_descriptions if ontology is not None else None,
            include_descriptions=include_descriptions,
        ),
        _select_label_descriptions(
            argument_role_labels,
            (
                ontology.argument_role_descriptions
                if ontology is not None
                else None
            ),
            include_descriptions=include_descriptions,
        ),
        _select_label_descriptions(
            location_type_labels,
            (
                ontology.location_type_descriptions
                if ontology is not None
                else None
            ),
            include_descriptions=include_descriptions,
        ),
        answer_obj=answer_obj,
        add_generation_prompt=False,
        events_only=events_only,
        omit_offsets=omit_offsets,
    )


def _chat_parts(
    tokenizer,
    document: str,
    event_labels: list[str],
    argument_role_labels: list[str],
    location_type_labels: list[str],
    *,
    ontology: SftCandidateOntology | None,
    include_descriptions: bool,
    answer_obj: dict[str, Any],
    events_only: bool = False,
    omit_offsets: bool = False,
) -> tuple[list[dict[str, str]], str]:
    messages = build_messages(
        document,
        _select_label_descriptions(
            event_labels,
            ontology.event_descriptions if ontology is not None else None,
            include_descriptions=include_descriptions,
        ),
        _select_label_descriptions(
            argument_role_labels,
            (
                ontology.argument_role_descriptions
                if ontology is not None
                else None
            ),
            include_descriptions=include_descriptions,
        ),
        _select_label_descriptions(
            location_type_labels,
            (
                ontology.location_type_descriptions
                if ontology is not None
                else None
            ),
            include_descriptions=include_descriptions,
        ),
        events_only=events_only,
        omit_offsets=omit_offsets,
    )
    label_text = json.dumps(answer_obj, ensure_ascii=False)
    return messages, label_text


def _format_row(
    row: dict[str, Any],
    tokenizer,
    *,
    ontology: SftCandidateOntology | None = None,
    num_event_candidates: int | None = None,
    num_relation_candidates: int | None = None,
    is_training: bool = False,
    candidate_shuffle_probability: float = 0.0,
    gold_candidate_dropout_probability: float = 0.0,
    random_seed: int = DEFAULT_CANDIDATE_SAMPLING_SEED,
    include_descriptions: bool = False,
    candidate_rng: random.Random | None = None,
    index: int = 0,
    events_only: bool = False,
    omit_offsets: bool = False,
) -> dict[str, str]:
    document = row["question"]
    raw_events = row["answer"]["events"]
    answer_obj = {"events": _enrich_events(document, raw_events, events_only=events_only)}
    event_labels, argument_role_labels, location_type_labels = _resolve_candidate_labels(
        row,
        ontology=ontology,
        num_event_candidates=num_event_candidates,
        num_relation_candidates=num_relation_candidates,
        is_training=is_training,
        candidate_shuffle_probability=candidate_shuffle_probability,
        gold_candidate_dropout_probability=gold_candidate_dropout_probability,
        random_seed=random_seed,
        candidate_rng=(
            candidate_rng if candidate_rng is not None else random.Random(random_seed)
        ),
        index=index,
    )
    text = _chat_text(
        tokenizer,
        document,
        event_labels,
        argument_role_labels,
        location_type_labels,
        ontology=ontology,
        include_descriptions=include_descriptions,
        answer_obj=answer_obj,
        events_only=events_only,
        omit_offsets=omit_offsets,
    )
    return {
        "text": text,
        "row_index": index,
        "seq_length": _get_sequence_length(tokenizer, text),
    }


def _build_map_fn(
    tokenizer,
    *,
    ontology: SftCandidateOntology | None,
    num_event_candidates: int | None,
    num_relation_candidates: int | None,
    is_training: bool,
    candidate_shuffle_probability: float,
    gold_candidate_dropout_probability: float,
    random_seed: int,
    include_descriptions: bool,
    events_only: bool = False,
    omit_offsets: bool = False,
):
    candidate_rng = random.Random(random_seed)

    def _map_fn(row: dict[str, Any], index: int) -> dict[str, str]:
        return _format_row(
            row,
            tokenizer,
            ontology=ontology,
            num_event_candidates=num_event_candidates,
            num_relation_candidates=num_relation_candidates,
            is_training=is_training,
            candidate_shuffle_probability=candidate_shuffle_probability,
            gold_candidate_dropout_probability=gold_candidate_dropout_probability,
            random_seed=random_seed,
            include_descriptions=include_descriptions,
            candidate_rng=candidate_rng,
            index=index,
            events_only=events_only,
            omit_offsets=omit_offsets,
        )

    return _map_fn


def _build_sample_preview(
    row: dict[str, Any],
    tokenizer,
    *,
    ontology: SftCandidateOntology | None,
    num_event_candidates: int | None,
    num_relation_candidates: int | None,
    is_training: bool,
    candidate_shuffle_probability: float,
    gold_candidate_dropout_probability: float,
    random_seed: int,
    include_descriptions: bool,
    index: int,
    events_only: bool = False,
    omit_offsets: bool = False,
) -> tuple[list[dict[str, str]], str]:
    document = row["question"]
    raw_events = row["answer"]["events"]
    answer_obj = {"events": _enrich_events(document, raw_events, events_only=events_only)}
    event_labels, argument_role_labels, location_type_labels = _resolve_candidate_labels(
        row,
        ontology=ontology,
        num_event_candidates=num_event_candidates,
        num_relation_candidates=num_relation_candidates,
        is_training=is_training,
        candidate_shuffle_probability=candidate_shuffle_probability,
        gold_candidate_dropout_probability=gold_candidate_dropout_probability,
        random_seed=random_seed,
        candidate_rng=random.Random(random_seed),
        index=index,
    )
    return _chat_parts(
        tokenizer,
        document,
        event_labels,
        argument_role_labels,
        location_type_labels,
        ontology=ontology,
        include_descriptions=include_descriptions,
        answer_obj=answer_obj,
        events_only=events_only,
        omit_offsets=omit_offsets,
    )


def _get_sequence_length(tokenizer, text: str) -> int:
    tokenized = tokenizer(
        text=text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = tokenized["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _filter_overlong_samples(dataset, max_seq_length: int, split_name: str):
    original_size = len(dataset)
    filtered_dataset = dataset.filter(
        lambda row: row["seq_length"] <= max_seq_length,
        num_proc=4,
    )
    removed_count = original_size - len(filtered_dataset)
    print(
        f"{split_name} samples within max_seq_length={max_seq_length}: "
        f"{len(filtered_dataset)}/{original_size} kept, {removed_count} removed."
    )
    return filtered_dataset


def _subsample_dataset(dataset, max_samples: int, seed: int, split_name: str):
    if max_samples <= 0:
        raise ValueError(f"--max_train_samples must be > 0, got {max_samples}")

    dataset_size = len(dataset)
    if max_samples >= dataset_size:
        print(
            f"{split_name} subsampling skipped: requested {max_samples} samples, "
            f"dataset has {dataset_size}."
        )
        return dataset

    subsampled_dataset = dataset.shuffle(seed=seed).select(range(max_samples))
    print(
        f"{split_name} subsampled to {len(subsampled_dataset)}/{dataset_size} rows "
        f"using seed={seed}."
    )
    return subsampled_dataset


def _row_has_events(row: dict[str, Any]) -> bool:
    answer = row.get("answer")
    if not isinstance(answer, dict):
        return False
    events = answer.get("events")
    return isinstance(events, list) and len(events) > 0


def _limit_empty_event_rows(dataset, max_ratio: float, seed: int):
    empty_indices: list[int] = []
    non_empty_indices: list[int] = []
    for index, row in enumerate(dataset):
        if _row_has_events(row):
            non_empty_indices.append(index)
        else:
            empty_indices.append(index)

    non_empty_count = len(non_empty_indices)
    empty_count = len(empty_indices)
    max_empty_count = int(max_ratio * non_empty_count)

    if empty_count <= max_empty_count:
        print(
            "Train empty-event ratio already within limit: "
            f"{empty_count} empty, {non_empty_count} non-empty, ratio<={max_ratio}."
        )
        return dataset

    kept_empty_indices = (
        random.Random(seed).sample(empty_indices, k=max_empty_count)
        if max_empty_count > 0
        else []
    )
    kept_indices = sorted(non_empty_indices + kept_empty_indices)
    limited_dataset = dataset.select(kept_indices)
    print(
        "Train empty-event rows limited: "
        f"{len(kept_empty_indices)}/{empty_count} empty kept with "
        f"{non_empty_count} non-empty rows using max ratio={max_ratio} and seed={seed}."
    )
    return limited_dataset


def _print_train_dataset_preview(
    train_ds,
    raw_train_ds,
    tokenizer,
    *,
    ontology: SftCandidateOntology | None,
    num_event_candidates: int | None,
    num_relation_candidates: int | None,
    candidate_shuffle_probability: float,
    gold_candidate_dropout_probability: float,
    random_seed: int,
    include_descriptions: bool,
    events_only: bool = False,
    omit_offsets: bool = False,
) -> None:
    if len(train_ds) == 0:
        print("Training dataset is empty; no preview available.")
        return

    sequence_lengths = train_ds["seq_length"]

    sample_index = random.Random(random_seed).randrange(len(train_ds))
    raw_sample_index = train_ds[sample_index]["row_index"]
    messages, label_text = _build_sample_preview(
        raw_train_ds[raw_sample_index],
        tokenizer,
        ontology=ontology,
        num_event_candidates=num_event_candidates,
        num_relation_candidates=num_relation_candidates,
        is_training=True,
        candidate_shuffle_probability=candidate_shuffle_probability,
        gold_candidate_dropout_probability=gold_candidate_dropout_probability,
        random_seed=random_seed,
        include_descriptions=include_descriptions,
        index=raw_sample_index,
        events_only=events_only,
        omit_offsets=omit_offsets,
    )

    avg_length = sum(sequence_lengths) / len(sequence_lengths)
    print(
        "Training sequence length stats "
        f"(tokens): avg={avg_length:.2f}, min={min(sequence_lengths)}, "
        f"max={max(sequence_lengths)}"
    )
    print(f"Random training sample index: {sample_index}")
    for msg in messages:
        if msg["role"] == "system":
            print("System prompt:")
            print(msg["content"])
        elif msg["role"] == "user":
            print("User prompt:")
            print(msg["content"])
    print("Sample answer:")
    print(label_text)


def _print_dataset_max_sequence_length(dataset, split_name: str) -> None:
    if len(dataset) == 0:
        print(f"{split_name} max sequence length (tokens): dataset is empty")
        return
    max_length = max(dataset["seq_length"])
    print(f"{split_name} max sequence length (tokens): {max_length}")


def _print_dataset_max_sequence_length_before_filtering(
    dataset,
    split_name: str,
) -> None:
    if len(dataset) == 0:
        print(f"{split_name} max sequence length before filtering (tokens): dataset is empty")
        return
    max_length = max(dataset["seq_length"])
    print(f"{split_name} max sequence length before filtering (tokens): {max_length}")


def _response_only(trainer: SFTTrainer, model_name: str) -> SFTTrainer:
    if "lfm" in model_name.lower():
        print("Applying response-only training template for LFM model")
        return train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )
    if "qwen3.5" in model_name.lower():
        print("Applying response-only training template for Qwen3.5 model in no-thinking mode")
        return train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )
    
    raise ValueError(
        "Response-only training template is not defined for the specified model. "
        "Supported models for response-only training: LFM, Qwen3.5."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Randomly subsample this many training rows before formatting.",
    )
    parser.add_argument(
        "--max_empty_event_ratio",
        type=float,
        default=None,
        help=(
            "Maximum allowed ratio of empty-event rows to non-empty rows in the "
            "training split. Empty rows are deterministically downsampled before formatting."
        ),
    )
    parser.add_argument(
        "--filter_overlong_samples",
        action="store_true",
        help="Drop formatted samples whose tokenized length exceeds --max_seq_length.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--full_finetuning", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_16bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--train_on_responses_only", action="store_true")
    parser.add_argument(
        "--events_only",
        action="store_true",
        help="Train only on event extraction (no arguments).",
    )
    parser.add_argument(
        "--omit_offsets",
        action="store_true",
        help="Omit character offsets in prompts and responses.",
    )
    parser.add_argument(
        "--ontology_file",
        type=str,
        default=None,
        help="Ontology JSON with 'events', 'argument_roles', and 'location_types' used for prompt context and candidate sampling.",
    )
    parser.add_argument(
        "--description",
        action="store_true",
        help="Include ontology label descriptions in prompts in addition to label keys.",
    )
    parser.add_argument(
        "--num_event_candidates",
        type=int,
        default=None,
        help="Total number of event candidates per sample, including gold labels. Use -1 for all candidates.",
    )
    parser.add_argument(
        "--num_relation_candidates",
        type=int,
        default=None,
        help="Total number of argument-role and location-type candidates per sample, including gold labels. Use -1 for all candidates.",
    )
    parser.add_argument(
        "--train_candidate_shuffle_prob",
        type=float,
        default=0.5,
        help="Probability of shuffling candidate labels for each training sample.",
    )
    parser.add_argument(
        "--train_gold_candidate_dropout_prob",
        type=float,
        default=0.05,
        help="Probability of dropping each gold candidate label during training.",
    )
    parser.add_argument(
        "--candidate_sampling_seed",
        type=int,
        default=DEFAULT_CANDIDATE_SAMPLING_SEED,
        help="Random seed used for deterministic candidate sampling.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.num_event_candidates = _normalize_candidate_count(
        "--num_event_candidates",
        args.num_event_candidates,
    )
    args.max_empty_event_ratio = _normalize_max_empty_event_ratio(
        args.max_empty_event_ratio
    )
    args.num_relation_candidates = _normalize_candidate_count(
        "--num_relation_candidates",
        args.num_relation_candidates,
    )

    candidate_sampling_enabled = (
        args.num_event_candidates is not None
        or args.num_relation_candidates is not None
    )
    if candidate_sampling_enabled and args.ontology_file is None:
        raise ValueError(
            "--ontology_file is required when candidate sampling is enabled"
        )
    ontology = (
        _load_sft_candidate_ontology(args.ontology_file)
        if args.ontology_file is not None
        else None
    )

    # check that only one of load_in_4bit, load_in_8bit, load_in_16bit is set
    precision_flags = [
        args.load_in_4bit,
        args.load_in_8bit,
        args.load_in_16bit,
    ]
    if sum(precision_flags) > 1:
        raise ValueError(
            "Only one of --load_in_4bit, --load_in_8bit, --load_in_16bit can be set"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        load_in_16bit=args.load_in_16bit,
        full_finetuning=args.full_finetuning,
    )

    if not args.full_finetuning:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_r,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=False, #"unsloth",
            random_state=3407,
            max_seq_length=args.max_seq_length,
            finetune_vision_layers=False,
        )

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    if args.max_train_samples is not None:
        train_ds = _subsample_dataset(
            train_ds,
            max_samples=args.max_train_samples,
            seed=args.candidate_sampling_seed,
            split_name="Train",
        )
    if args.max_empty_event_ratio is not None:
        train_ds = _limit_empty_event_rows(
            train_ds,
            max_ratio=args.max_empty_event_ratio,
            seed=args.candidate_sampling_seed,
        )
    raw_train_ds = train_ds
    train_ds = train_ds.map(
        _build_map_fn(
            tokenizer,
            ontology=ontology,
            num_event_candidates=args.num_event_candidates,
            num_relation_candidates=args.num_relation_candidates,
            is_training=True,
            candidate_shuffle_probability=args.train_candidate_shuffle_prob,
            gold_candidate_dropout_probability=args.train_gold_candidate_dropout_prob,
            random_seed=args.candidate_sampling_seed,
            include_descriptions=args.description,
            events_only=args.events_only,
            omit_offsets=args.omit_offsets,
        ),
        with_indices=True,
        num_proc=4,
    )
    _print_dataset_max_sequence_length_before_filtering(train_ds, "Train")
    if args.filter_overlong_samples:
        train_ds = _filter_overlong_samples(train_ds, args.max_seq_length, "Train")
    _print_dataset_max_sequence_length(train_ds, "Train")

    eval_ds = None
    if args.eval_file:
        eval_ds = load_dataset("json", data_files=args.eval_file, split="train")
        eval_ds = eval_ds.map(
            _build_map_fn(
                tokenizer,
                ontology=ontology,
                num_event_candidates=args.num_event_candidates,
                num_relation_candidates=args.num_relation_candidates,
                is_training=False,
                candidate_shuffle_probability=args.train_candidate_shuffle_prob,
                gold_candidate_dropout_probability=args.train_gold_candidate_dropout_prob,
                random_seed=args.candidate_sampling_seed,
                include_descriptions=args.description,
            ),
            with_indices=True,
            num_proc=4,
        )
        _print_dataset_max_sequence_length_before_filtering(eval_ds, "Eval")
        if args.filter_overlong_samples:
            eval_ds = _filter_overlong_samples(eval_ds, args.max_seq_length, "Eval")
        _print_dataset_max_sequence_length(eval_ds, "Eval")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        args=SFTConfig(
            packing=False,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            num_train_epochs=args.epochs,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=eval_ds is not None,
            metric_for_best_model="eval_loss" if eval_ds is not None else None,
            greater_is_better=False if eval_ds is not None else None,
            # save_total_limit=2 if eval_ds is not None else None,
            dataset_num_proc=4,
            ddp_find_unused_parameters=False,
            optim="adamw_8bit",
            bf16=use_bf16,
            fp16=not use_bf16,
            seed=3407,
            report_to="none",
        ),
    )

    if args.train_on_responses_only:
        trainer = _response_only(trainer, args.model_name)

    _print_train_dataset_preview(
        train_ds,
        raw_train_ds,
        tokenizer,
        ontology=ontology,
        num_event_candidates=args.num_event_candidates,
        num_relation_candidates=args.num_relation_candidates,
        candidate_shuffle_probability=args.train_candidate_shuffle_prob,
        gold_candidate_dropout_probability=args.train_gold_candidate_dropout_prob,
        random_seed=args.candidate_sampling_seed,
        include_descriptions=args.description,
        events_only=args.events_only,
        omit_offsets=args.omit_offsets,
    )

    trainer.train()
    trainer.save_model(str(Path(args.output_dir) / "final_adapter"))
    tokenizer.save_pretrained(str(Path(args.output_dir) / "final_adapter"))


if __name__ == "__main__":
    main()
