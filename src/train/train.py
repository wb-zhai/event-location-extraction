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
    DEFAULT_ARGUMENT_THRESHOLD,
    DEFAULT_EVENT_THRESHOLD,
    DEFAULT_MODEL_NAME,
    DEFAULT_RELATION_THRESHOLD,
    EventReaderConfig,
    EventReader,
    compute_reader_metrics,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 4096
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 3.0


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
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        LOGGER.info("Starting evaluation with metric prefix '%s'.", metric_key_prefix)
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()

        references: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        losses: list[float] = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            references.extend(batch["references"])
            model_inputs = self._prepare_inputs(
                self._model_inputs(batch, decode_predictions=True)
            )
            with torch.no_grad():
                outputs = model(**model_inputs)
            if "loss" in outputs:
                losses.append(float(outputs["loss"].detach().cpu().item()))
            predictions.extend(outputs["decoded_predictions"])

        if predictions and references:
            LOGGER.info(
                "First eval sample prediction: %s | gold labels: %s",
                json.dumps(predictions[0], ensure_ascii=False),
                json.dumps(references[0], ensure_ascii=False),
            )

        metrics = compute_reader_metrics(predictions, references)
        if losses:
            metrics["loss"] = sum(losses) / len(losses)
        metrics = {
            f"{metric_key_prefix}_{key}": value for key, value in metrics.items()
        }
        self.log(metrics)
        return metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "test",
    ) -> Any:
        test_dataloader = self.get_test_dataloader(test_dataset)
        model = self.model
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an event/argument reader with Hugging Face Trainer."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, type=str)
    parser.add_argument("--train_file", required=True, type=str)
    parser.add_argument("--eval_file", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--max_length", default=DEFAULT_MAX_LENGTH, type=int)
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--learning_rate", default=DEFAULT_LEARNING_RATE, type=float)
    parser.add_argument(
        "--event_threshold", default=DEFAULT_EVENT_THRESHOLD, type=float
    )
    parser.add_argument(
        "--argument_threshold", default=DEFAULT_ARGUMENT_THRESHOLD, type=float
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
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Precision for training (default: fp16)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for scheduler (default: 0.1)",
    )

    return parser.parse_args()


def resolve_precision_flags(precision: str) -> tuple[bool, bool]:
    if precision == "fp32":
        return False, False

    if not torch.cuda.is_available():
        raise ValueError(
            f"--precision {precision} requires CUDA. "
            "Use --precision fp32 when training on CPU."
        )

    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError(
            "--precision bf16 requires CUDA bf16 support on this device/runtime."
        )

    return precision == "fp16", precision == "bf16"


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    args = parse_args()

    fp16_enabled, bf16_enabled = resolve_precision_flags(args.precision)
    device_label = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(
        "Training precision=%s (fp16=%s, bf16=%s) on device=%s",
        args.precision,
        fp16_enabled,
        bf16_enabled,
        device_label,
    )

    train_samples = load_normalized_jsonl(args.train_file)
    eval_samples = load_normalized_jsonl(args.eval_file)

    tokenizer = build_tokenizer(args.model_name)
    train_dataset = EventReaderDataset(train_samples, tokenizer, args.max_length)
    eval_dataset = EventReaderDataset(eval_samples, tokenizer, args.max_length)

    config = EventReaderConfig(
        model_name=args.model_name,
        event_threshold=args.event_threshold,
        argument_threshold=args.argument_threshold,
        relation_threshold=args.relation_threshold,
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
        num_train_epochs=DEFAULT_NUM_EPOCHS,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=1_000,
        save_steps=1_000,
        logging_steps=10,
        report_to=[],
        warmup_ratio=args.warmup_ratio,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
    )

    trainer = EventArgumentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=EventReaderCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(Path(args.output_dir) / "final")
    with open(
        Path(args.output_dir) / "eval_metrics.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2)
    LOGGER.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
