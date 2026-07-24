"""Fine-tune an encoder text classifier on relevance-labeled JSONL data.

Trains a binary `relevant` / `irrelevant` sequence classifier (default backbone:
answerdotai/ModernBERT-base) on the `relevance.is_relevant` field written by
relevance_filter.py (or any file with the same shape). Self-contained: no imports
from other scripts in this repo. Accepts one or more --input files (concatenated,
deduped by id/url). The pretrained backbone trains at --backbone-lr (default 2e-4)
while the classification head trains at --learning-rate.

    python scripts/data/relevance/train.py \\
        --input dataset/db/relevance/matrix_5M.sample_1000.3.1pro.2label_prompt.jsonl \\
        --output-dir /tmp/relevance-modernbert-smoke \\
        --num-epochs 1 --batch-size 8 --precision fp32
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
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
DEFAULT_MAX_LENGTH = 2048
DEFAULT_MAX_CHARS = 4000
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_BACKBONE_LR = 2e-5
DEFAULT_NUM_EPOCHS = 5.0
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_EVAL_FRAC = 0.10
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


def load_examples(paths: Path | list[Path], max_chars: int) -> list[dict[str, Any]]:
    if isinstance(paths, Path):
        paths = [paths]

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped_no_label = 0
    skipped_dupe = 0

    for path in paths:
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
                        "source": str(path),
                    }
                )

    n = len(examples)
    n_relevant = sum(e["label"] for e in examples)
    LOGGER.info(
        "Loaded %d examples from %s (skipped %d missing-label, %d duplicate ids)",
        n,
        ", ".join(str(p) for p in paths),
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


class RelevanceTrainer(Trainer):
    """Trainer with optional class-weighted loss and optional backbone/head learning rates."""

    def __init__(
        self,
        *args,
        class_weights: torch.Tensor | None = None,
        backbone_lr: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.backbone_lr = backbone_lr

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.class_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.backbone_lr is None or self.optimizer is not None:
            return super().create_optimizer()

        opt_model = self.model
        base_prefix = f"{opt_model.base_model_prefix}."
        decay_parameters = set(self.get_decay_parameter_names(opt_model))

        buckets: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            part = "backbone" if name.startswith(base_prefix) else "head"
            buckets.setdefault((part, name in decay_parameters), []).append(param)

        lrs = {"backbone": self.backbone_lr, "head": self.args.learning_rate}
        LOGGER.info(
            "Param groups: backbone=%d lr=%s | head=%d lr=%s",
            sum(len(p) for (part, _), p in buckets.items() if part == "backbone"),
            self.backbone_lr,
            sum(len(p) for (part, _), p in buckets.items() if part == "head"),
            self.args.learning_rate,
        )

        optimizer_grouped_parameters = [
            {"params": params, "lr": lrs[part], "weight_decay": self.args.weight_decay if decay else 0.0}
            for (part, decay), params in buckets.items()
            if params
        ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def log_split_stats(name: str, examples: list[dict[str, Any]]) -> None:
    n = len(examples)
    if not n:
        LOGGER.warning("%s split is empty!", name)
        return
    n_relevant = sum(e["label"] for e in examples)
    LOGGER.info(
        "%s split: %d examples | relevant=%d (%.1f%%) irrelevant=%d (%.1f%%)",
        name,
        n,
        n_relevant,
        n_relevant / n * 100,
        n - n_relevant,
        (n - n_relevant) / n * 100,
    )


def save_split_by_source(run_dir: Path, split_name: str, examples: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        grouped.setdefault(ex["source"], []).append(ex)

    used_names: set[str] = set()
    for source, group in grouped.items():
        stem = Path(source).stem
        name = stem
        suffix = 2
        while name in used_names:
            name = f"{stem}_{suffix}"
            suffix += 1
        used_names.add(name)

        split_path = run_dir / f"{split_name}_{name}.jsonl"
        with split_path.open("w", encoding="utf-8") as f:
            for ex in group:
                f.write(json.dumps({k: v for k, v in ex.items() if k != "source"}, ensure_ascii=False) + "\n")
        LOGGER.info("Saved %s split (%d examples) from %s to %s", split_name, len(group), source, split_path)


def build_dataset(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    dataset = Dataset.from_list(examples)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text", "id", "source"])
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
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        nargs="+",
        help="Training JSONL file(s). Multiple files are concatenated (deduped by id/url).",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--eval-file",
        type=Path,
        nargs="+",
        default=None,
        help="Optional separate eval JSONL file(s). If omitted, a stratified split of "
        "--input is used (see --eval-frac).",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate for the classification head (and the whole model, if --backbone-lr is not set)."
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=DEFAULT_BACKBONE_LR,
        help="Learning rate for the pretrained backbone; the classification head trains at "
        "--learning-rate. Set --backbone-lr equal to --learning-rate to train the whole "
        "model at a single rate.",
    )
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
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
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

    model_max = tokenizer.model_max_length
    if model_max < 1_000_000:
        LOGGER.info(
            "Tokenizer model_max_length=%d; requested --max-length=%d", model_max, args.max_length
        )
        if args.max_length > model_max:
            raise ValueError(
                f"--max-length {args.max_length} exceeds the model's maximum supported length "
                f"({model_max}). Lower --max-length or choose a model with a larger context window."
            )
    else:
        LOGGER.info(
            "Tokenizer does not declare a max length; using --max-length=%d", args.max_length
        )

    if args.eval_file is not None:
        train_examples = load_examples(args.input, args.max_chars)
        eval_examples = load_examples(args.eval_file, args.max_chars)
        log_split_stats("train", train_examples)
        log_split_stats("eval", eval_examples)
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
            ", ".join(str(p) for p in args.input),
            len(train_examples),
            len(eval_examples),
            args.eval_frac,
        )
        log_split_stats("train", train_examples)
        log_split_stats("eval", eval_examples)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.wandb_run_name or f"{args.output_dir.name}_{timestamp}"

    if args.resume_from_checkpoint:
        run_dir = args.output_dir
    else:
        run_dir = args.output_dir / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Run directory: %s", run_dir)

    save_split_by_source(run_dir, "train", train_examples)
    save_split_by_source(run_dir, "dev", eval_examples)

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
        output_dir=str(run_dir),
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
        os.environ["WANDB_RUN_NAME"] = run_name

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = RelevanceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        backbone_lr=args.backbone_lr,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    metrics = trainer.evaluate()
    LOGGER.info("Final eval metrics: %s", json.dumps(metrics, indent=2))

    final_output_dir = run_dir / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(final_output_dir)

    with open(run_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    LOGGER.info("Saved model to %s", final_output_dir)


if __name__ == "__main__":
    main()
