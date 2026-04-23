from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from src.data.dataset import (
    DEFAULT_CANDIDATE_SAMPLING_SEED,
    CandidateOntology,
    _apply_training_candidate_transform,
    _build_candidate_labels,
    load_candidate_ontology,
)
from src.sft_prompt import render_chat


def _safe_substring(text: str, start: Any, end: Any) -> str:
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    if start < 0 or end <= start or end > len(text):
        return ""
    return text[start:end]


def _enrich_events(document: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for event in events:
        ev_start = event.get("start")
        ev_end = event.get("end")
        out_event = {
            "event_type": event.get("event_type", ""),
            "start": ev_start,
            "end": ev_end,
            "text": _safe_substring(document, ev_start, ev_end),
            "arguments": [],
        }
        for arg in event.get("arguments", []):
            a_start = arg.get("start")
            a_end = arg.get("end")
            out_event["arguments"].append(
                {
                    "role": arg.get("role", ""),
                    "start": a_start,
                    "end": a_end,
                    "text": _safe_substring(document, a_start, a_end),
                }
            )
        out_event["arguments"].sort(
            key=lambda x: (
                x.get("start", 10**9),
                x.get("end", 10**9),
                x.get("role", ""),
            )
        )

        out_event.pop("arguments")

        enriched.append(out_event)

    enriched.sort(
        key=lambda x: (
            x.get("start", 10**9),
            x.get("end", 10**9),
            x.get("event_type", ""),
        )
    )
    return enriched


def _extract_required_labels(
    events: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    event_labels: list[str] = []
    argument_labels: list[str] = []
    seen_event_labels: set[str] = set()
    seen_argument_labels: set[str] = set()

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
            if isinstance(role, str) and role and role not in seen_argument_labels:
                seen_argument_labels.add(role)
                argument_labels.append(role)

    return event_labels, argument_labels


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
    ontology: CandidateOntology | None,
    num_event_candidates: int | None,
    num_relation_candidates: int | None,
    is_training: bool,
    candidate_shuffle_probability: float,
    gold_candidate_dropout_probability: float,
    random_seed: int,
    candidate_rng: random.Random,
    index: int,
) -> tuple[list[str], list[str]]:
    raw_events = row["answer"]["events"]
    required_event_labels, required_argument_labels = _extract_required_labels(
        raw_events
    )
    document = row.get("question", "")
    sample_id = row.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        sample_id = document if isinstance(document, str) and document else str(index)

    document_event_labels = _require_label_list(
        row,
        "event_labels",
        fallback_labels=required_event_labels if ontology is not None else None,
    )
    document_argument_labels = _require_label_list(
        row,
        "argument_labels",
        fallback_labels=required_argument_labels if ontology is not None else None,
    )

    event_labels = _build_candidate_labels(
        sample_id=sample_id,
        label_kind="event",
        required_labels=required_event_labels,
        document_labels=document_event_labels,
        ontology_labels=ontology.event_labels if ontology is not None else None,
        requested_total=num_event_candidates,
        rng=candidate_rng,
    )
    argument_labels = _build_candidate_labels(
        sample_id=sample_id,
        label_kind="relation",
        required_labels=required_argument_labels,
        document_labels=document_argument_labels,
        ontology_labels=ontology.argument_labels if ontology is not None else None,
        requested_total=num_relation_candidates,
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
        argument_labels = _apply_training_candidate_transform(
            labels=argument_labels,
            required_labels=required_argument_labels,
            ontology_labels=(
                ontology.argument_labels
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

    return event_labels, argument_labels


def _chat_text(
    tokenizer,
    document: str,
    event_labels: list[str],
    argument_labels: list[str],
    answer_obj: dict[str, Any],
) -> str:
    del argument_labels
    return render_chat(
        tokenizer,
        document,
        event_labels,
        answer_obj=answer_obj,
        add_generation_prompt=False,
    )


def _chat_parts(
    tokenizer,
    document: str,
    event_labels: list[str],
    argument_labels: list[str],
    answer_obj: dict[str, Any],
) -> tuple[str, str]:
    del argument_labels
    prompt_text = render_chat(
        tokenizer,
        document,
        event_labels,
        add_generation_prompt=True,
    )
    label_text = json.dumps(answer_obj, ensure_ascii=False)
    return prompt_text, label_text


def _format_row(
    row: dict[str, Any],
    tokenizer,
    *,
    ontology: CandidateOntology | None = None,
    num_event_candidates: int | None = None,
    num_relation_candidates: int | None = None,
    is_training: bool = False,
    candidate_shuffle_probability: float = 0.0,
    gold_candidate_dropout_probability: float = 0.0,
    random_seed: int = DEFAULT_CANDIDATE_SAMPLING_SEED,
    candidate_rng: random.Random | None = None,
    index: int = 0,
) -> dict[str, str]:
    document = row["question"]
    raw_events = row["answer"]["events"]
    answer_obj = {"events": _enrich_events(document, raw_events)}
    event_labels, argument_labels = _resolve_candidate_labels(
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
    text = _chat_text(tokenizer, document, event_labels, argument_labels, answer_obj)
    return {"text": text, "row_index": index}


def _build_map_fn(
    tokenizer,
    *,
    ontology: CandidateOntology | None,
    num_event_candidates: int | None,
    num_relation_candidates: int | None,
    is_training: bool,
    candidate_shuffle_probability: float,
    gold_candidate_dropout_probability: float,
    random_seed: int,
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
            candidate_rng=candidate_rng,
            index=index,
        )

    return _map_fn


def _build_sample_preview(
    row: dict[str, Any],
    tokenizer,
    *,
    ontology: CandidateOntology | None,
    num_event_candidates: int | None,
    num_relation_candidates: int | None,
    is_training: bool,
    candidate_shuffle_probability: float,
    gold_candidate_dropout_probability: float,
    random_seed: int,
    index: int,
) -> tuple[str, str]:
    document = row["question"]
    raw_events = row["answer"]["events"]
    answer_obj = {"events": _enrich_events(document, raw_events)}
    event_labels, argument_labels = _resolve_candidate_labels(
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
        argument_labels,
        answer_obj,
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


def _filter_overlong_samples(dataset, tokenizer, max_seq_length: int, split_name: str):
    original_size = len(dataset)
    filtered_dataset = dataset.filter(
        lambda row: _get_sequence_length(tokenizer, row["text"]) <= max_seq_length,
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


def _print_train_dataset_preview(
    train_ds,
    raw_train_ds,
    tokenizer,
    *,
    ontology: CandidateOntology | None,
    num_event_candidates: int | None,
    num_relation_candidates: int | None,
    candidate_shuffle_probability: float,
    gold_candidate_dropout_probability: float,
    random_seed: int,
) -> None:
    if len(train_ds) == 0:
        print("Training dataset is empty; no preview available.")
        return

    sequence_lengths: list[int] = []
    for text in train_ds["text"]:
        sequence_lengths.append(_get_sequence_length(tokenizer, text))

    sample_index = random.Random(random_seed).randrange(len(train_ds))
    raw_sample_index = train_ds[sample_index]["row_index"]
    prompt_text, label_text = _build_sample_preview(
        raw_train_ds[raw_sample_index],
        tokenizer,
        ontology=ontology,
        num_event_candidates=num_event_candidates,
        num_relation_candidates=num_relation_candidates,
        is_training=True,
        candidate_shuffle_probability=candidate_shuffle_probability,
        gold_candidate_dropout_probability=gold_candidate_dropout_probability,
        random_seed=random_seed,
        index=raw_sample_index,
    )

    avg_length = sum(sequence_lengths) / len(sequence_lengths)
    print(
        "Training sequence length stats "
        f"(tokens): avg={avg_length:.2f}, min={min(sequence_lengths)}, "
        f"max={max(sequence_lengths)}"
    )
    print(f"Random training sample index: {sample_index}")
    print("Sample input:")
    print(prompt_text)
    print("Sample label:")
    print(label_text)

def _response_only(trainer: SFTTrainer, model_name: str) -> SFTTrainer:
    if "lfm" in model_name.lower():
        print("Applying response-only training template for LFM model")
        return train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )
    if "qwen3.5" in model_name.lower():
        print("Applying response-only training template for Qwen3.5 model")
        return train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n<think>",
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
        "--ontology_file",
        type=str,
        default=None,
        help="Ontology JSON with 'event_labels' and 'argument_labels' used for candidate sampling.",
    )
    parser.add_argument(
        "--num_event_candidates",
        type=int,
        default=None,
        help="Total number of event candidates per sample, including gold labels.",
    )
    parser.add_argument(
        "--num_relation_candidates",
        type=int,
        default=None,
        help="Total number of relation candidates per sample, including gold labels.",
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

    candidate_sampling_enabled = (
        args.num_event_candidates is not None
        or args.num_relation_candidates is not None
    )
    if candidate_sampling_enabled and args.ontology_file is None:
        raise ValueError(
            "--ontology_file is required when candidate sampling is enabled"
        )
    ontology = (
        load_candidate_ontology(args.ontology_file)
        if candidate_sampling_enabled and args.ontology_file is not None
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
        ),
        with_indices=True,
        num_proc=4,
    )
    if args.filter_overlong_samples:
        train_ds = _filter_overlong_samples(
            train_ds, tokenizer, args.max_seq_length, "Train"
        )

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
            ),
            with_indices=True,
            num_proc=4,
        )
        if args.filter_overlong_samples:
            eval_ds = _filter_overlong_samples(
                eval_ds, tokenizer, args.max_seq_length, "Eval"
            )

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        packing=False,
        args=SFTConfig(
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
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
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
    )

    trainer.train()
    trainer.save_model(str(Path(args.output_dir) / "final_adapter"))
    tokenizer.save_pretrained(str(Path(args.output_dir) / "final_adapter"))


if __name__ == "__main__":
    main()
