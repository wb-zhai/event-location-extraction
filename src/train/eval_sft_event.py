import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _event_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [event for event in value if isinstance(event, dict)]


def _prediction_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        events = prediction.get("events")
        if events is not None:
            return _event_list(events)

    answer = row.get("answer")
    if isinstance(answer, dict):
        nested_prediction = answer.get("prediction")
        if isinstance(nested_prediction, dict):
            events = nested_prediction.get("events")
            if events is not None:
                return _event_list(events)

    return []


def _event_types(events: list[dict[str, Any]]) -> set[str]:
    return {
        event_type
        for event in events
        if isinstance(event, dict)
        and isinstance((event_type := event.get("event_type")), str)
        and event_type
    }


def _explicit_doc_key(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    candidates = [row]
    if isinstance(metadata, dict):
        candidates.append(metadata)

    for source in candidates:
        for key in ("doc_id", "document_id", "source_id", "article_id"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, int):
                return str(value)

    value = row.get("id")
    if isinstance(value, str) and value:
        base_id, marker, window_index = value.rpartition("__w")
        if marker and window_index.isdigit():
            return base_id
        return value
    if isinstance(value, int):
        return str(value)
    return None


def _document_group_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    doc_index = -1
    previous_start: int | None = None

    for row_index, row in enumerate(rows):
        explicit_key = _explicit_doc_key(row)
        if explicit_key is not None:
            keys.append(explicit_key)
            continue

        metadata = row.get("metadata")
        start = (
            metadata.get("document_char_start")
            if isinstance(metadata, dict)
            else None
        )
        if isinstance(start, int):
            if previous_start is None or start <= previous_start:
                doc_index += 1
            previous_start = start
            keys.append(f"offset-doc-{doc_index}")
            continue

        keys.append(f"row-doc-{row_index}")

    return keys


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _score_event_type_sets(
    gold_labels: list[set[str]], pred_labels: list[set[str]]
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    exact = 0

    for gold_set, pred_set in zip(gold_labels, pred_labels):
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        if gold_set == pred_set:
            exact += 1

    metrics = _prf(tp, fp, fn)
    metrics["exact_match"] = exact / len(gold_labels) if gold_labels else 0.0
    return metrics


def _macro_score_event_type_sets(
    gold_labels: list[set[str]], pred_labels: list[set[str]]
) -> dict[str, float]:
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    exact = 0

    for gold_set, pred_set in zip(gold_labels, pred_labels):
        doc_metrics = _prf(
            len(gold_set & pred_set),
            len(pred_set - gold_set),
            len(gold_set - pred_set),
        )
        precision += doc_metrics["precision"]
        recall += doc_metrics["recall"]
        f1 += doc_metrics["f1"]
        if gold_set == pred_set:
            exact += 1

    total = len(gold_labels)
    if total == 0:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "macro_exact_match": 0.0,
        }

    return {
        "macro_precision": precision / total,
        "macro_recall": recall / total,
        "macro_f1": f1 / total,
        "macro_exact_match": exact / total,
    }


def evaluate_window_events(
    gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> dict[str, float]:
    if len(gold_rows) != len(pred_rows):
        raise ValueError(
            f"Number of windows in gold and pred do not match: {len(gold_rows)} vs {len(pred_rows)}"
        )

    # we only care about whether the model can correctly identify which events are present in the document,
    # not whether it can correctly identify the spans or the number of events
    window_gold_labels = []
    window_pred_labels = []
    for gold_row, pred_row in zip(gold_rows, pred_rows):
        window_gold_labels.append(_event_types(gold_row["answer"]["events"]))
        window_pred_labels.append(_event_types(_prediction_events(pred_row)))

    return _score_event_type_sets(window_gold_labels, window_pred_labels)


def evaluate_document_events(
    gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(gold_rows) != len(pred_rows):
        raise ValueError(
            f"Number of windows in gold and pred do not match: {len(gold_rows)} vs {len(pred_rows)}"
        )

    gold_doc_labels: dict[str, set[str]] = {}
    pred_doc_labels: dict[str, set[str]] = {}
    for doc_key, gold_row, pred_row in zip(
        _document_group_keys(gold_rows), gold_rows, pred_rows
    ):
        gold_doc_labels.setdefault(doc_key, set()).update(
            _event_types(gold_row["answer"]["events"])
        )
        pred_doc_labels.setdefault(doc_key, set()).update(
            _event_types(_prediction_events(pred_row))
        )

    doc_keys = list(gold_doc_labels)
    gold_labels = [gold_doc_labels[doc_key] for doc_key in doc_keys]
    pred_labels = [pred_doc_labels.get(doc_key, set()) for doc_key in doc_keys]
    metrics = _score_event_type_sets(gold_labels, pred_labels)
    metrics.update(_macro_score_event_type_sets(gold_labels, pred_labels))
    metrics["documents"] = len(doc_keys)
    return metrics


def evaluate_events(
    gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        "window": evaluate_window_events(gold_rows, pred_rows),
        "document": evaluate_document_events(gold_rows, pred_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-jsonl", type=str, required=True)
    parser.add_argument("--pred-jsonl", type=str, required=True)
    # parser.add_argument("--report-json", type=str, default=None)
    args = parser.parse_args()

    gold_rows = _load_jsonl(Path(args.gold_jsonl))
    pred_rows = _load_jsonl(Path(args.pred_jsonl))
    metrics = evaluate_events(gold_rows, pred_rows)
    # if args.report_json is not None:
    #     write_doc_event_report(gold_rows, pred_rows, Path(args.report_json))

    print("Window-level metrics:")
    print(json.dumps(metrics["window"], indent=2, ensure_ascii=False))
    print("Document-level metrics:")
    print(json.dumps(metrics["document"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
