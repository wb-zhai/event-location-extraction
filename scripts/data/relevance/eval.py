"""Score a predictions JSONL file against a labeled JSONL file.

Compares two files matched by record `id`: one holding predictions (as written by
inference.py, relevance_filter.py, encoder.py, or local.py -- anything with a
`relevance.is_relevant` field) and one holding ground truth (either a train.py
train/dev split with a top-level `label`, or another `relevance.is_relevant`-shaped
file, e.g. a cascade-labeled file used as gold). Reports accuracy/precision/recall/F1
and a confusion matrix over the ids common to both files -- no model loading, no
inference, just the two files.

    # Score inference.py's predictions against the dev split it was held out from
    python scripts/data/relevance/eval.py \\
        --predictions /tmp/dev.predictions.jsonl \\
        --labels outputs/relevance/relevance-modernbert/20260715_123657/dev.jsonl

    # Score against a cascade-labeled file used as gold, save misclassified records
    python scripts/data/relevance/eval.py \\
        --predictions /tmp/matrix.relevance.jsonl \\
        --labels dataset/db/training_exp/matrix_5M.sample_20000.relevance.cascade.jsonl \\
        --errors /tmp/eval_errors.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ID2LABEL = {0: "irrelevant", 1: "relevant"}


def record_key(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("url") or "")


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record_key(record)
            if key:
                records[key] = record
    return records


def get_label(record: dict[str, Any]) -> int | None:
    """Ground truth or prediction label: `relevance.is_relevant` if present, else a
    top-level `label` (as saved by train.py's train/dev splits)."""
    rel = record.get("relevance") or {}
    is_relevant = rel.get("is_relevant")
    if is_relevant is not None:
        return 1 if is_relevant else 0
    label = record.get("label")
    if label is not None:
        return int(label)
    return None


def record_title(rec: dict[str, Any]) -> str:
    return str(rec.get("title") or (rec.get("source") or {}).get("title", ""))


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["relevant"], average="binary", pos_label="relevant", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["irrelevant", "relevant"]).ravel()
    return {
        "n": len(y_true),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score a predictions JSONL file against a labeled JSONL file."
    )
    parser.add_argument("--predictions", required=True, type=Path, help="JSONL file with predicted relevance")
    parser.add_argument("--labels", required=True, type=Path, help="JSONL file with ground-truth relevance/label")
    parser.add_argument(
        "--errors", type=Path, default=None, help="Optional path to write misclassified records as JSONL"
    )
    parser.add_argument(
        "--metrics-output", type=Path, default=None, help="Optional path to write the metrics JSON"
    )
    args = parser.parse_args(argv)

    predictions = load_records(args.predictions)
    labels = load_records(args.labels)

    common_ids = sorted(set(predictions) & set(labels))
    if not common_ids:
        raise SystemExit("No common ids between --predictions and --labels.")

    y_true: list[str] = []
    y_pred: list[str] = []
    errors: list[dict[str, Any]] = []
    skipped = 0

    for rid in common_ids:
        pred_record, label_record = predictions[rid], labels[rid]
        pred_label, true_label = get_label(pred_record), get_label(label_record)
        if pred_label is None or true_label is None:
            skipped += 1
            continue

        pred_str, true_str = ID2LABEL[pred_label], ID2LABEL[true_label]
        y_true.append(true_str)
        y_pred.append(pred_str)
        if pred_str != true_str:
            errors.append(
                {
                    "id": rid,
                    "title": record_title(label_record),
                    "true_label": true_str,
                    "predicted_label": pred_str,
                    "confidence": (pred_record.get("relevance") or {}).get("confidence"),
                }
            )

    print(f"predictions: {len(predictions)} records ({args.predictions})")
    print(f"labels: {len(labels)} records ({args.labels})")
    print(f"Common ids: {len(common_ids)} | scored: {len(y_true)} | skipped (missing label): {skipped}")
    print()

    metrics = compute_metrics(y_true, y_pred)
    print(json.dumps(metrics, indent=2))

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nWrote metrics to {args.metrics_output}")

    if args.errors:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        with args.errors.open("w", encoding="utf-8") as f:
            for row in errors:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(errors)} misclassified records to {args.errors}")


if __name__ == "__main__":
    main()
