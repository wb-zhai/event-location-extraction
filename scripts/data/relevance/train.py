"""Fine-tune an encoder text classifier on relevance-labeled JSONL data.

Trains a binary `relevant` / `irrelevant` sequence classifier (default backbone:
answerdotai/ModernBERT-base) on the `relevance.is_relevant` field written by
relevance_filter.py (or any file with the same shape). Self-contained: no imports
from other scripts in this repo.

    python scripts/data/relevance/train.py \\
        --input dataset/db/relevance/matrix_5M.sample_1000.3.1pro.2label_prompt.jsonl \\
        --output-dir /tmp/relevance-modernbert-smoke \\
        --num-epochs 1 --batch-size 8 --precision fp32
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"
DEFAULT_MAX_LENGTH = 512
DEFAULT_MAX_CHARS = 2000
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 3.0
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_EVAL_FRAC = 0.15
DEFAULT_SEED = 42

ID2LABEL = {0: "irrelevant", 1: "relevant"}
LABEL2ID = {"irrelevant": 0, "relevant": 1}


# ---------------------------------------------------------------------------
# Data loading (inline, self-contained — no imports from other repo scripts)
# ---------------------------------------------------------------------------


def record_key(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("url") or "")


def get_label(record: dict[str, Any]) -> int | None:
    rel = record.get("relevance") or {}
    is_relevant = rel.get("is_relevant")
    if is_relevant is None:
        return None
    return 1 if is_relevant else 0


def build_text(record: dict[str, Any], max_chars: int) -> str:
    source = record.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    title = str(record.get("title") or source.get("title") or "")
    text = str(record.get("text") or source.get("text") or "")
    text = text[:max_chars]
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def load_examples(path: Path, max_chars: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped_no_label = 0
    skipped_dupe = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            key = record_key(record)
            if key and key in seen:
                skipped_dupe += 1
                continue
            if key:
                seen.add(key)

            label = get_label(record)
            if label is None:
                skipped_no_label += 1
                continue

            examples.append(
                {
                    "id": key,
                    "text": build_text(record, max_chars),
                    "label": label,
                }
            )

    n = len(examples)
    n_relevant = sum(e["label"] for e in examples)
    LOGGER.info(
        "Loaded %d examples from %s (skipped %d missing-label, %d duplicate ids)",
        n,
        path,
        skipped_no_label,
        skipped_dupe,
    )
    if n:
        LOGGER.info(
            "Class balance: relevant=%d (%.1f%%) irrelevant=%d (%.1f%%)",
            n_relevant,
            n_relevant / n * 100,
            n - n_relevant,
            (n - n_relevant) / n * 100,
        )
    return examples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[1], average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


# ---------------------------------------------------------------------------
# Optional class-weighted loss
# ---------------------------------------------------------------------------


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def build_dataset(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    dataset = Dataset.from_list(examples)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text", "id"])
    return dataset


# ---------------------------------------------------------------------------
# Precision resolution (mps/cpu-aware)
# ---------------------------------------------------------------------------


def resolve_precision_flags(precision: str) -> tuple[str, bool, bool]:
    if precision == "fp32":
        return "fp32", False, False

    cuda_available = torch.cuda.is_available()

    if precision == "auto":
        if not cuda_available:
            return "fp32", False, False
        if torch.cuda.is_bf16_supported():
            return "bf16", False, True
        return "fp16", True, False

    if not cuda_available:
        raise ValueError(
            f"--precision {precision} requires CUDA. Use --precision fp32 on mps/cpu."
        )
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("--precision bf16 requires CUDA bf16 support on this device.")

    return precision, precision == "fp16", precision == "bf16"


def detect_device_label() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an encoder classifier on relevance-labeled JSONL data."
    )
    parser.add_argument("--input", required=True, type=Path, help="Training JSONL file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="Optional separate eval JSONL file. If omitted, a stratified split of "
        "--input is used (see --eval-frac).",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-epochs", type=float, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--eval-frac", type=float, default=DEFAULT_EVAL_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--precision", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"]
    )
    parser.add_argument(
        "--metric-for-best",
        type=str,
        default="f1",
        help="Metric to select the best checkpoint (accuracy/precision/recall/f1).",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Use inverse-frequency class weights in the loss.",
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    args = parse_args(argv)

    resolved_precision, fp16_enabled, bf16_enabled = resolve_precision_flags(args.precision)
    device_label = detect_device_label()
    LOGGER.info(
        "Training precision requested=%s resolved=%s (fp16=%s, bf16=%s) on device=%s",
        args.precision,
        resolved_precision,
        fp16_enabled,
        bf16_enabled,
        device_label,
    )

    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if args.eval_file is not None:
        train_examples = load_examples(args.input, args.max_chars)
        eval_examples = load_examples(args.eval_file, args.max_chars)
    else:
        all_examples = load_examples(args.input, args.max_chars)
        labels = [e["label"] for e in all_examples]
        train_examples, eval_examples = train_test_split(
            all_examples,
            test_size=args.eval_frac,
            random_state=args.seed,
            stratify=labels,
        )
        LOGGER.info(
            "Stratified split of %s: train=%d eval=%d (eval_frac=%.2f)",
            args.input,
            len(train_examples),
            len(eval_examples),
            args.eval_frac,
        )

    train_dataset = build_dataset(train_examples, tokenizer, args.max_length)
    eval_dataset = build_dataset(eval_examples, tokenizer, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    class_weights = None
    if args.balance_classes:
        train_labels = np.array([e["label"] for e in train_examples])
        counts = np.bincount(train_labels, minlength=2).astype(np.float32)
        weights = counts.sum() / (2.0 * np.clip(counts, 1.0, None))
        class_weights = torch.tensor(weights, dtype=torch.float32)
        LOGGER.info("Class weights (irrelevant, relevant): %s", weights.tolist())

    report_to = ["wandb"] if args.wandb_project else "none"
    training_args_kwargs: dict[str, Any] = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best,
        greater_is_better=True,
        report_to=report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        seed=args.seed,
    )
    if args.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb_project

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_cls = WeightedLossTrainer if args.balance_classes else Trainer
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if args.balance_classes:
        trainer_kwargs["class_weights"] = class_weights
    trainer = trainer_cls(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    metrics = trainer.evaluate()
    LOGGER.info("Final eval metrics: %s", json.dumps(metrics, indent=2))

    final_output_dir = args.output_dir / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(final_output_dir)

    with open(args.output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    LOGGER.info("Saved model to %s", final_output_dir)


if __name__ == "__main__":
    main()
