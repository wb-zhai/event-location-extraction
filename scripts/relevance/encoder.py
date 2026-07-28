"""Run the relevance gate using a local encoder text classifier instead of an LLM.

Uses classla/multilingual-IPTC-news-topic-classifier (xlm-roberta-large, 94
languages) to assign each article one of 17 fixed IPTC top-level topics, then maps
that topic onto is_relevant via a fixed RELEVANT_LABELS set. No LLM client, no
prompt, no API cost -- runs fully locally via transformers.pipeline. Output has the
same relevance shape as relevance_filter.py (decision, is_relevant, confidence,
reason, filtered, ...), so it's diffable against a Gemini-labeled file with
agreement.py.

    python scripts/relevance/encoder.py \\
        --input dataset/zhai/v3/articles.jsonl \\
        --output /tmp/articles.encoder.jsonl --limit 50 --overwrite

    python scripts/relevance/agreement.py \\
        --a dataset/zhai/v3/articles.gemini.jsonl --b /tmp/articles.encoder.jsonl
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any


from scripts.relevance.relevance_filter import (
    _record_key,
    food_insecurity_regex,
    should_filter_by_relevance,
    truncate_text,
)

DEFAULT_MODEL = "classla/multilingual-IPTC-news-topic-classifier"

# IPTC top-level topics considered in-scope for the food-security event categories
# (agricultural production, conflicts/violence, economic issues, environmental
# issues, food crisis, forced displacement, humanitarian aid, land-related issues,
# pests/diseases, political instability, weather shocks). "society" covers
# poverty/human rights/family planning, the closest bucket for forced
# displacement, humanitarian aid, and land-related issues.
RELEVANT_LABELS = {
    "conflict, war and peace",
    "disaster, accident and emergency incident",
    "economy, business and finance",
    "environment",
    "politics",
    "weather",
    "health",
    "society",
}


def build_relevance_decision(
    label: str,
    score: float,
    model_name: str,
    max_chars: int,
    text_chars_used: int,
    confidence_threshold: float,
) -> dict[str, Any]:
    is_relevant = label in RELEVANT_LABELS
    reason = f"The news is classified as {label} therefore it is {'relevant' if is_relevant else 'not relevant'}."
    decision = {
        "is_relevant": is_relevant,
        "confidence": float(score),
        "reason": reason,
    }
    return {
        "decision": "relevant" if is_relevant else "irrelevant",
        "is_relevant": is_relevant,
        "confidence": float(score),
        "reason": reason,
        "filtered": should_filter_by_relevance(decision, confidence_threshold),
        "threshold": confidence_threshold,
        "model": model_name,
        "max_chars": max_chars,
        "text_chars_used": text_chars_used,
        "metadata": {"predicted_label": label, "score": float(score)},
    }


def record_preview_text(record: dict[str, Any], max_chars: int) -> tuple[str, str, str]:
    source = record.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    title = str(record.get("title") or source.get("title", ""))
    text = str(record.get("text") or source.get("text", ""))
    preview_text = truncate_text(text, max_chars)
    combined = f"{title}\n\n{preview_text}" if title else preview_text
    return title, preview_text, combined


def detect_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def process_file(args: argparse.Namespace) -> None:
    from transformers import pipeline

    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(
            f"Output file {output_path} already exists. Use --resume to continue "
            "or --overwrite to replace it."
        )

    done_keys: set[str] = set()
    if args.resume and output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_keys.add(_record_key(json.loads(line)))
        print(f"Resuming: {len(done_keys)} records already done, skipping.")

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if _record_key(rec) not in done_keys:
                    records.append(rec)
            if args.limit and len(records) >= args.limit:
                break

    device = detect_device(args.device)
    print(f"Loading {args.model} on device={device}...")
    classifier = pipeline(
        "text-classification",
        model=args.model,
        device=device,
        truncation=True,
        max_length=args.max_length,
    )

    write_mode = "a" if args.resume and done_keys else "w"
    out_f = open(output_path, write_mode, encoding="utf-8")
    counts = {"total": 0, "filtered": 0}

    try:
        from tqdm import tqdm

        chunks = _chunked(records, args.chunk_size)
        for chunk in tqdm(chunks, desc="Classifying"):
            to_classify: list[dict[str, Any]] = []
            texts: list[str] = []
            chunk_results: list[dict[str, Any]] = []

            for record in chunk:
                record_id = str(record.get("id", record.get("url", "")))
                try:
                    if args.use_regex:
                        title = str(
                            record.get("title")
                            or (record.get("source") or {}).get("title", "")
                        )
                        text = str(
                            record.get("text")
                            or (record.get("source") or {}).get("text", "")
                        )
                        if not food_insecurity_regex.search(title + " " + text):
                            record["relevance"] = {
                                "decision": "irrelevant",
                                "is_relevant": False,
                                "confidence": 1.0,
                                "reason": "Regex filter mismatch",
                                "filtered": True,
                                "threshold": 1.0,
                                "model": "regex",
                            }
                            chunk_results.append(record)
                            continue

                    _, preview_text, combined = record_preview_text(
                        record, args.max_chars
                    )
                    to_classify.append(record)
                    texts.append(combined)
                except Exception as e:
                    record["relevance"] = {"error": str(e), "filtered": False}
                    print(f"Error processing record {record_id}: {e}")
                    chunk_results.append(record)

            if texts:
                predictions = classifier(texts, batch_size=args.batch_size)
                for record, prediction, text in zip(to_classify, predictions, texts):
                    record["relevance"] = build_relevance_decision(
                        label=prediction["label"],
                        score=prediction["score"],
                        model_name=args.model,
                        max_chars=args.max_chars,
                        text_chars_used=len(text),
                        confidence_threshold=args.confidence_threshold,
                    )
                    chunk_results.append(record)

            for record in chunk_results:
                is_filtered = bool(record.get("relevance", {}).get("filtered", False))
                if not (args.filter_only and is_filtered):
                    out_f.write(json.dumps(record) + "\n")
                counts["total"] += 1
                counts["filtered"] += int(is_filtered)
            out_f.flush()
    finally:
        out_f.close()

    total = counts["total"]
    filtered = counts["filtered"]
    print(f"Total records: {total}")
    if total:
        print(f"Filtered out: {filtered} ({filtered/total:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the relevance gate using a local encoder text classifier."
    )
    parser.add_argument("--input", required=True, type=str, help="Input JSONL file")
    parser.add_argument("--output", required=True, type=str, help="Output JSONL file")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="HF text-classification model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cuda/mps/cpu). Default: auto-detect.",
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Tokenizer max sequence length"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Max characters to use for relevance",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Pipeline batch size"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Records per progress/flush chunk"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Confidence threshold to filter",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Do not write filtered records to output",
    )
    parser.add_argument(
        "--use-regex",
        action="store_true",
        help="Pre-filter with the food_insecurity_regex before calling the classifier.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of records to process."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records already present in the output file and append new results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file. Required if the output file "
        "already exists and --resume is not set.",
    )
    args = parser.parse_args()

    process_file(args)


if __name__ == "__main__":
    main()
