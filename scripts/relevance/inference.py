"""Run relevance inference with an encoder classifier trained by train.py.

Two interchangeable backends, selected with --backend:
  hf     transformers (AutoModelForSequenceClassification) -- cuda/mps/cpu
  vllm   vLLM pooling engine (LLM.classify) -- cuda only, higher throughput

Input is either free text on the command line (--text, one or more strings)
or a path (--input): a single JSONL file, or a folder of JSONL files (all
"*.jsonl" siblings are processed). Records may be raw articles (top-level or
source.title/source.text, as written by relevance_filter.py) or pre-built
train/dev splits (id/text/label, as saved by train.py) -- both shapes build
the same input text as at training time. Each record is enriched with a
"relevance" field (decision/is_relevant/confidence/probs) and written to
--output, one JSONL file in -> one JSONL file out.

    # Single string from the terminal
    python scripts/relevance/inference.py \\
        --checkpoint outputs/relevance/relevance-modernbert/20260715_123657/checkpoint-4618 \\
        --text "Flooding displaces thousands in southern province"

    # A single JSONL file, HF backend
    python scripts/relevance/inference.py \\
        --checkpoint outputs/relevance/relevance-modernbert/20260715_123657/checkpoint-4618 \\
        --input dataset/db/training_exp/matrix_5M.sample_20000.relevance.cascade.jsonl \\
        --output /tmp/matrix.relevance.jsonl --batch-size 64

    # A folder of JSONL files, vLLM backend
    python scripts/relevance/inference.py \\
        --checkpoint outputs/relevance/relevance-modernbert/20260715_123657/checkpoint-4618 \\
        --input dataset/db/training_exp --output /tmp/training_exp.relevance --backend vllm
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 2048
DEFAULT_MAX_CHARS = 4000
DEFAULT_BATCH_SIZE = 32
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9

FALLBACK_ID2LABEL = {0: "irrelevant", 1: "relevant"}


# ---------------------------------------------------------------------------
# Data loading (mirrors train.py's record_key/build_text -- self-contained,
# so tokenized input text matches what the model was trained on)
# ---------------------------------------------------------------------------


def record_key(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("url") or "")


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


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        if not files:
            raise SystemExit(f"No .jsonl files found in {input_path}")
        return files
    return [input_path]


def output_path_for(source_path: Path, input_path: Path, output_path: Path) -> Path:
    if input_path.is_dir():
        return output_path / source_path.name
    return output_path


# ---------------------------------------------------------------------------
# Precision / device helpers
# ---------------------------------------------------------------------------


def detect_device_label() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(precision: str, device: str):
    import torch

    if precision == "fp32":
        return torch.float32
    if precision == "auto":
        if device != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device != "cuda":
        raise ValueError(f"--precision {precision} requires CUDA. Use --precision fp32 on mps/cpu.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("--precision bf16 requires CUDA bf16 support on this device.")
    return torch.bfloat16 if precision == "bf16" else torch.float16


def resolve_vllm_dtype(precision: str) -> str:
    return {"auto": "auto", "fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[precision]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class HFPredictor:
    def __init__(self, checkpoint: str, device: str, precision: str, max_length: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device
        self.max_length = max_length

        dtype = resolve_torch_dtype(precision, device)
        LOGGER.info("Loading HF model from %s (device=%s, dtype=%s)", checkpoint, device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        torch = self._torch
        inputs = self.tokenizer(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        return [_to_prediction(p, self.id2label) for p in probs]


class VLLMPredictor:
    def __init__(
        self,
        checkpoint: str,
        precision: str,
        max_length: int,
        device: str | None,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
    ):
        from vllm import LLM

        kwargs: dict[str, Any] = dict(
            model=checkpoint,
            runner="pooling",
            dtype=resolve_vllm_dtype(precision),
            max_model_len=max_length,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if device:
            kwargs["device"] = device
        LOGGER.info("Loading vLLM model from %s (%s)", checkpoint, kwargs)
        self.llm = LLM(**kwargs)
        self.id2label = _load_id2label(checkpoint)

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        outputs = self.llm.classify(texts)
        return [_to_prediction(list(output.outputs.probs), self.id2label) for output in outputs]


def _load_id2label(checkpoint: str) -> dict[int, str]:
    config_path = Path(checkpoint) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("id2label"):
            return {int(k): v for k, v in config["id2label"].items()}
    return dict(FALLBACK_ID2LABEL)


def _to_prediction(probs: list[float], id2label: dict[int, str]) -> dict[str, Any]:
    pred_id = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "label": id2label.get(pred_id, str(pred_id)),
        "confidence": float(probs[pred_id]),
        "probs": {id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)},
    }


def build_predictor(args: argparse.Namespace, device: str):
    if args.backend == "hf":
        return HFPredictor(args.checkpoint, device, args.precision, args.max_length)
    return VLLMPredictor(
        args.checkpoint,
        args.precision,
        args.max_length,
        args.device,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def effective_batch_size(args: argparse.Namespace, num_records: int) -> int:
    """--batch-size chunks requests for the hf backend (bounded by GPU memory).

    vLLM does its own continuous batching/scheduling internally, so chunking
    ahead of time only throttles it -- hand it the whole set of records in one
    `classify()` call instead.
    """
    if args.backend == "vllm":
        return max(num_records, 1)
    return args.batch_size


def process_records(
    records: Iterable[dict[str, Any]],
    predictor: Any,
    batch_size: int,
    max_chars: int,
    checkpoint: str,
    backend: str,
) -> Iterator[dict[str, Any]]:
    for batch in chunked(list(records), batch_size):
        texts = [build_text(record, max_chars) for record in batch]
        predictions = predictor.predict(texts)
        for record, prediction, text in zip(batch, predictions, texts):
            record["relevance"] = {
                "decision": prediction["label"],
                "is_relevant": prediction["label"] == "relevant",
                "confidence": prediction["confidence"],
                "probs": prediction["probs"],
                "model": str(checkpoint),
                "backend": backend,
                "max_chars": max_chars,
                "text_chars_used": len(text),
            }
            yield record


def run_on_text(args: argparse.Namespace, predictor: Any) -> None:
    records = [{"id": f"text-{i}", "text": text} for i, text in enumerate(args.text)]
    batch_size = effective_batch_size(args, len(records))
    out_records = list(
        process_records(records, predictor, batch_size, args.max_chars, args.checkpoint, args.backend)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for record in out_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        LOGGER.info("Wrote %d predictions to %s", len(out_records), args.output)
    else:
        for record in out_records:
            print(json.dumps(record, ensure_ascii=False))


def run_on_input(args: argparse.Namespace, predictor: Any) -> None:
    input_files = iter_input_files(args.input)
    LOGGER.info("Found %d input file(s) under %s", len(input_files), args.input)

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)

    for source_path in input_files:
        out_path = output_path_for(source_path, args.input, args.output)
        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"Output {out_path} already exists. Use --overwrite to replace it.")

        records = list(read_jsonl(source_path))
        if args.limit:
            records = records[: args.limit]
        LOGGER.info("Processing %s (%d records) -> %s", source_path, len(records), out_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        counts = {"total": 0, "relevant": 0}
        batch_size = effective_batch_size(args, len(records))
        with out_path.open("w", encoding="utf-8") as f:
            predicted = process_records(
                records, predictor, batch_size, args.max_chars, args.checkpoint, args.backend
            )
            for record in tqdm(predicted, total=len(records), desc=source_path.name):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["total"] += 1
                counts["relevant"] += int(record["relevance"]["is_relevant"])

        total = counts["total"]
        if total:
            LOGGER.info(
                "%s: %d/%d relevant (%.1f%%)", out_path, counts["relevant"], total, counts["relevant"] / total * 100
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run relevance inference with an encoder classifier trained by train.py."
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="Path to a trained checkpoint dir (e.g. .../checkpoint-4618)"
    )
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="Inference backend")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, nargs="+", help="One or more raw text strings to classify")
    input_group.add_argument("--input", type=Path, help="A JSONL file, or a folder of JSONL files")

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Required with --input (a file, or a folder if --input is a folder); "
        "with --text, defaults to stdout if omitted.",
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Max article chars used for text")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Tokenizer max sequence length")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size (hf backend only; vllm submits all records per file in one call "
        "and batches internally)",
    )
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda/mps/cpu for --backend hf (auto-detected if omitted); a device string for --backend vllm",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM only: number of GPUs")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="vLLM only: fraction of GPU memory vLLM is allowed to use",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max records to process per input file")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    args = parse_args(argv)

    if args.input is not None and args.output is None:
        raise SystemExit("--output is required when using --input")

    if args.backend == "hf":
        # Only touch torch.cuda in the main process for the hf backend. For
        # vllm, doing so here initializes CUDA before the engine core forks
        # its worker subprocess, which then fails with "Cannot re-initialize
        # CUDA in forked subprocess".
        args.device = args.device or detect_device_label()
        device = args.device
    else:
        device = None

    predictor = build_predictor(args, device)

    if args.text is not None:
        run_on_text(args, predictor)
    else:
        run_on_input(args, predictor)


if __name__ == "__main__":
    main()
