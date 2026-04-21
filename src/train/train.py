from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

from src.data.dataset import (
    ARGUMENT_MARKER,
    EVENT_MARKER,
    EventReaderCollator,
    EventReaderDataset,
    load_normalized_jsonl,
)
from src.modeling.model import (
    DEFAULT_MODEL_NAME,
    DEFAULT_RELATION_THRESHOLD,
    EventReaderConfig,
    EventReader,
    compute_reader_metrics,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 4096
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_NUM_EPOCHS = 3.0
DEFAULT_TASK_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_RELATION_LOSS_WEIGHT = 2.0
RELATION_THRESHOLD_GRID = [step / 20 for step in range(1, 20)]


def _prediction_with_relation_threshold(
    prediction: dict[str, Any], threshold: float
) -> dict[str, Any]:
    return {
        **prediction,
        "relations": [
            relation
            for relation in prediction.get("relation_candidates", [])
            if relation["score"] >= threshold
        ],
    }


def tune_relation_threshold(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    threshold_grid: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    best_threshold = DEFAULT_RELATION_THRESHOLD
    best_metrics = compute_reader_metrics(predictions, references)
    best_f1 = best_metrics["relation_classification_f1"]

    for threshold in threshold_grid or RELATION_THRESHOLD_GRID:
        thresholded_predictions = [
            _prediction_with_relation_threshold(prediction, threshold)
            for prediction in predictions
        ]
        metrics = compute_reader_metrics(thresholded_predictions, references)
        f1 = metrics["relation_classification_f1"]
        if f1 > best_f1:
            best_threshold = threshold
            best_metrics = metrics
            best_f1 = f1

    return best_threshold, best_metrics


class EventArgumentTrainer(Trainer):
    NON_MODEL_KEYS = {"sample_ids", "references"}

    def _model_inputs(
        self, inputs: dict[str, Any], decode_predictions: bool
    ) -> dict[str, Any]:
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in self.NON_MODEL_KEYS
        }
        model_inputs["decode_predictions"] = decode_predictions
        return model_inputs

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        LOGGER.info("Starting evaluation with metric prefix '%s'.", metric_key_prefix)
        if isinstance(eval_dataset, dict):
            raise ValueError(
                "Dictionary eval datasets are not supported by EventArgumentTrainer.evaluate"
            )
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        if model is None:
            raise ValueError("Trainer model is not initialized")
        model.eval()

        references: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        losses: list[float] = []
        loss_buckets: dict[str, list[float]] = {}

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            references.extend(batch["references"])
            model_inputs = self._prepare_inputs(
                self._model_inputs(batch, decode_predictions=True)
            )
            with torch.no_grad():
                outputs = model(**model_inputs)
            if "loss" in outputs:
                losses.append(float(outputs["loss"].detach().cpu().item()))
            for key, value in outputs.items():
                if not key.startswith("loss_"):
                    continue
                loss_buckets.setdefault(key, []).append(
                    float(value.detach().cpu().item())
                )
            predictions.extend(outputs["decoded_predictions"])

        if predictions and references:
            LOGGER.info(
                "First eval sample prediction: %s | gold labels: %s",
                json.dumps(predictions[0], ensure_ascii=False),
                json.dumps(references[0], ensure_ascii=False),
            )

        best_relation_threshold, metrics = tune_relation_threshold(
            predictions, references
        )
        metrics["best_relation_threshold"] = best_relation_threshold
        LOGGER.info(
            "Best relation threshold on %s: %.2f",
            metric_key_prefix,
            best_relation_threshold,
        )
        if losses:
            metrics["loss"] = sum(losses) / len(losses)
        for key, values in loss_buckets.items():
            if values:
                metrics[key] = sum(values) / len(values)
        metrics = {
            f"{metric_key_prefix}_{key}": value for key, value in metrics.items()
        }
        self.log(metrics)
        return metrics

    def create_optimizer(
        self, model: torch.nn.Module | None = None
    ) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer

        model = model if model is not None else self.model
        if model is None:
            raise ValueError("Trainer model is not initialized")

        encoder_lr = getattr(
            self.args, "encoder_learning_rate", self.args.learning_rate
        )
        task_lr = getattr(self.args, "task_learning_rate", self.args.learning_rate)
        weight_decay = self.args.weight_decay
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad
                    and n.startswith("encoder.")
                    and not any(term in n for term in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": encoder_lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad
                    and n.startswith("encoder.")
                    and any(term in n for term in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": encoder_lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad
                    and not n.startswith("encoder.")
                    and not any(term in n for term in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": task_lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad
                    and not n.startswith("encoder.")
                    and any(term in n for term in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": task_lr,
            },
        ]
        optimizer_grouped_parameters = [
            group for group in optimizer_grouped_parameters if group["params"]
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "test",
    ) -> Any:
        test_dataloader = self.get_test_dataloader(test_dataset)
        model = self.model
        if model is None:
            raise ValueError("Trainer model is not initialized")
        model.eval()

        references: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        for batch in test_dataloader:
            references.extend(batch["references"])
            model_inputs = self._prepare_inputs(
                self._model_inputs(batch, decode_predictions=True)
            )
            with torch.no_grad():
                outputs = model(**model_inputs)
            predictions.extend(outputs["decoded_predictions"])

        metrics = compute_reader_metrics(predictions, references)
        metrics = {
            f"{metric_key_prefix}_{key}": value for key, value in metrics.items()
        }
        return {
            "predictions": predictions,
            "metrics": metrics,
            "references": references,
        }


def build_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [EVENT_MARKER, ARGUMENT_MARKER]}
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must define a pad_token or eos_token")
    return tokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an event/argument reader with Hugging Face Trainer."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, type=str)
    parser.add_argument("--train_file", required=True, type=str)
    parser.add_argument("--eval_file", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        type=str,
        help="Resume training from a saved Trainer checkpoint directory.",
    )
    parser.add_argument("--max_length", default=DEFAULT_MAX_LENGTH, type=int)
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--learning_rate", default=DEFAULT_LEARNING_RATE, type=float)
    parser.add_argument(
        "--task_learning_rate",
        default=DEFAULT_TASK_LEARNING_RATE,
        type=float,
        help="Learning rate for non-encoder task heads.",
    )
    parser.add_argument(
        "--num_epochs",
        default=DEFAULT_NUM_EPOCHS,
        type=float,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        type=int,
        help="Accumulate gradients across this many steps before optimizer update.",
    )
    parser.add_argument(
        "--weight_decay",
        default=DEFAULT_WEIGHT_DECAY,
        type=float,
        help="Weight decay applied to non-bias and non-LayerNorm parameters.",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=DEFAULT_MAX_GRAD_NORM,
        type=float,
        help="Global gradient clipping norm.",
    )
    parser.add_argument(
        "--relation_threshold", default=DEFAULT_RELATION_THRESHOLD, type=float
    )
    parser.add_argument(
        "--dataloader_num_workers",
        default=4,
        type=int,
        help="Number of subprocesses to use for data loading.",
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Precision for training (default: auto)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for scheduler (default: 0.1)",
    )
    parser.add_argument(
        "--relation_loss_weight",
        type=float,
        default=DEFAULT_RELATION_LOSS_WEIGHT,
        help="Relative weight for relation loss in multi-task training.",
    )

    return parser.parse_args(argv)


def resolve_precision_flags(precision: str) -> tuple[str, bool, bool]:
    if precision == "auto":
        if not torch.cuda.is_available():
            return "fp32", False, False
        if torch.cuda.is_bf16_supported():
            return "bf16", False, True
        return "fp16", True, False

    if precision == "fp32":
        return "fp32", False, False

    if not torch.cuda.is_available():
        raise ValueError(
            f"--precision {precision} requires CUDA. "
            "Use --precision fp32 when training on CPU."
        )

    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError(
            "--precision bf16 requires CUDA bf16 support on this device/runtime."
        )

    return precision, precision == "fp16", precision == "bf16"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    args = parse_args(argv)

    resolved_precision, fp16_enabled, bf16_enabled = resolve_precision_flags(
        args.precision
    )
    device_label = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(
        "Training precision requested=%s resolved=%s (fp16=%s, bf16=%s) on device=%s",
        args.precision,
        resolved_precision,
        fp16_enabled,
        bf16_enabled,
        device_label,
    )
    if args.resume_from_checkpoint is not None:
        LOGGER.info("Resuming training from checkpoint: %s", args.resume_from_checkpoint)

    train_samples = load_normalized_jsonl(args.train_file)
    eval_samples = load_normalized_jsonl(args.eval_file)

    tokenizer = build_tokenizer(args.model_name)
    train_dataset = EventReaderDataset(train_samples, tokenizer, args.max_length)
    eval_dataset = EventReaderDataset(eval_samples, tokenizer, args.max_length)

    config = EventReaderConfig(
        model_name=args.model_name,
        relation_threshold=args.relation_threshold,
        relation_loss_weight=args.relation_loss_weight,
    )
    encoder = AutoModel.from_pretrained(args.model_name, dtype=torch.float32)
    model = EventReader(config, encoder)
    model.resize_token_embeddings(len(tokenizer))
    model = model.float()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        report_to=[],
        warmup_ratio=args.warmup_ratio,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
    )
    setattr(training_args, "encoder_learning_rate", args.learning_rate)
    setattr(training_args, "task_learning_rate", args.task_learning_rate)

    trainer = EventArgumentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=EventReaderCollator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    final_output_dir = Path(args.output_dir) / "final"
    best_relation_threshold = metrics.get("eval_best_relation_threshold")
    if best_relation_threshold is not None:
        model.config.relation_threshold = float(best_relation_threshold)
    model.config.encoder_vocab_size = len(tokenizer)
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(final_output_dir)
    with open(
        Path(args.output_dir) / "eval_metrics.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2)
    LOGGER.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
