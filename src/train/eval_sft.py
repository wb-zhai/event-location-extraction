from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.inference.text_anchor import AnchorStatus, TextAnchorResolver


_ANCHOR_RESOLVER = TextAnchorResolver()
_WEAK_MATCH_THRESHOLD = 0.5
EventAnnotation = tuple[str, int, int]
ArgumentAnnotation = tuple[str, int, int, str, int, int]
EventSpanAnnotation = tuple[int, int]
ArgumentSpanAnnotation = tuple[int, int, int, int]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prediction_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        events = prediction.get("events")
        if isinstance(events, list):
            return events

    answer = row.get("answer")
    if isinstance(answer, dict):
        nested_prediction = answer.get("prediction")
        if isinstance(nested_prediction, dict):
            events = nested_prediction.get("events")
            if isinstance(events, list):
                return events

    return []


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def _span_valid(text: str, start: Any, end: Any) -> bool:
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
    )


def _resolve_span(text: str, obj: dict[str, Any]) -> tuple[tuple[int, int] | None, str]:
    start, end = obj.get("start"), obj.get("end")
    s = obj.get("text", "")
    if _span_valid(text, start, end):
        if not isinstance(start, int) or not isinstance(end, int):
            return None, AnchorStatus.NOT_FOUND
        span_start = start
        span_end = end
        if not s or text[span_start:span_end] == s:
            return (span_start, span_end), AnchorStatus.MATCH_EXACT
    if isinstance(s, str) and s:
        match = _ANCHOR_RESOLVER.resolve(text, s)
        if match.start is not None and match.end is not None:
            return (match.start, match.end), match.status
        return None, AnchorStatus.NOT_FOUND
    return None, AnchorStatus.NOT_FOUND


def _anchor_payload(
    obj: dict[str, Any], *, keys: tuple[str, ...] = ("trigger", "argument", "entity")
) -> dict[str, Any]:
    for key in keys:
        nested = obj.get(key)
        if isinstance(nested, dict):
            return nested
    return obj


def _collect_annotations(doc_text: str, events: list[dict[str, Any]]) -> tuple[
    set[EventAnnotation],
    set[ArgumentAnnotation],
    int,
    int,
    dict[str, int],
]:
    event_set: set[EventAnnotation] = set()
    arg_set: set[ArgumentAnnotation] = set()
    grounded = 0
    total = 0
    status_counts = {
        AnchorStatus.MATCH_EXACT: 0,
        AnchorStatus.MATCH_LESSER: 0,
        AnchorStatus.MATCH_FUZZY: 0,
        AnchorStatus.NOT_FOUND: 0,
    }

    for ev in events:
        total += 1
        ev_type = ev.get("event_type", "")
        ev_span, ev_status = _resolve_span(doc_text, _anchor_payload(ev))
        status_counts[ev_status] = status_counts.get(ev_status, 0) + 1
        if ev_span is None:
            continue
        grounded += 1
        es, ee = ev_span
        event_set.add((ev_type, es, ee))

        for arg in ev.get("arguments", []):
            total += 1
            role = arg.get("role", "")
            arg_span, arg_status = _resolve_span(doc_text, _anchor_payload(arg))
            status_counts[arg_status] = status_counts.get(arg_status, 0) + 1
            if arg_span is None:
                continue
            grounded += 1
            a_s, a_e = arg_span
            arg_set.add((ev_type, es, ee, role, a_s, a_e))

    return event_set, arg_set, grounded, total, status_counts


def _event_spans(events: set[EventAnnotation]) -> set[EventSpanAnnotation]:
    return {(start, end) for _, start, end in events}


def _argument_spans(args: set[ArgumentAnnotation]) -> set[ArgumentSpanAnnotation]:
    return {
        (event_start, event_end, arg_start, arg_end)
        for _, event_start, event_end, _, arg_start, arg_end in args
    }


def _merge_status_counts(
    totals: dict[str, int], counts: dict[str, int]
) -> dict[str, int]:
    for status, count in counts.items():
        totals[status] = totals.get(status, 0) + count
    return totals


def _score_annotation_sets(
    gold: set[tuple[Any, ...]], pred: set[tuple[Any, ...]]
) -> tuple[int, int, int]:
    return len(gold & pred), len(pred - gold), len(gold - pred)


def _span_overlap_ratio(
    first_span: tuple[int, int], second_span: tuple[int, int]
) -> float:
    first_start, first_end = first_span
    second_start, second_end = second_span
    overlap = max(0, min(first_end, second_end) - max(first_start, second_start))
    if overlap == 0:
        return 0.0
    first_length = max(0, first_end - first_start)
    second_length = max(0, second_end - second_start)
    denominator = max(first_length, second_length)
    return overlap / denominator if denominator else 0.0


def _match_weak_events(
    gold: set[EventAnnotation], pred: set[EventAnnotation]
) -> tuple[int, int, int]:
    available_gold = set(gold)
    matched = 0

    for pred_event in pred:
        pred_label, pred_start, pred_end = pred_event
        best_gold = None
        best_overlap = 0.0

        for gold_event in available_gold:
            gold_label, gold_start, gold_end = gold_event
            if pred_label != gold_label:
                continue
            overlap_ratio = _span_overlap_ratio(
                (pred_start, pred_end), (gold_start, gold_end)
            )
            if (
                overlap_ratio >= _WEAK_MATCH_THRESHOLD
                and overlap_ratio > best_overlap
            ):
                best_overlap = overlap_ratio
                best_gold = gold_event

        if best_gold is None:
            continue

        available_gold.remove(best_gold)
        matched += 1

    return matched, len(pred) - matched, len(gold) - matched


def _match_weak_event_spans(
    gold: set[EventSpanAnnotation], pred: set[EventSpanAnnotation]
) -> tuple[int, int, int]:
    available_gold = set(gold)
    matched = 0

    for pred_start, pred_end in pred:
        best_gold = None
        best_overlap = 0.0

        for gold_start, gold_end in available_gold:
            overlap_ratio = _span_overlap_ratio(
                (pred_start, pred_end), (gold_start, gold_end)
            )
            if (
                overlap_ratio >= _WEAK_MATCH_THRESHOLD
                and overlap_ratio > best_overlap
            ):
                best_overlap = overlap_ratio
                best_gold = (gold_start, gold_end)

        if best_gold is None:
            continue

        available_gold.remove(best_gold)
        matched += 1

    return matched, len(pred) - matched, len(gold) - matched


def _match_weak_arguments(
    gold: set[ArgumentAnnotation], pred: set[ArgumentAnnotation]
) -> tuple[int, int, int]:
    available_gold = set(gold)
    matched = 0

    for pred_arg in pred:
        (
            pred_event_type,
            pred_event_start,
            pred_event_end,
            pred_role,
            pred_arg_start,
            pred_arg_end,
        ) = pred_arg
        best_gold = None
        best_overlap = 0.0

        for gold_arg in available_gold:
            (
                gold_event_type,
                gold_event_start,
                gold_event_end,
                gold_role,
                gold_arg_start,
                gold_arg_end,
            ) = gold_arg
            if pred_event_type != gold_event_type:
                continue
            if pred_event_start != gold_event_start or pred_event_end != gold_event_end:
                continue
            if pred_role != gold_role:
                continue
            overlap_ratio = _span_overlap_ratio(
                (pred_arg_start, pred_arg_end), (gold_arg_start, gold_arg_end)
            )
            if (
                overlap_ratio >= _WEAK_MATCH_THRESHOLD
                and overlap_ratio > best_overlap
            ):
                best_overlap = overlap_ratio
                best_gold = gold_arg

        if best_gold is None:
            continue

        available_gold.remove(best_gold)
        matched += 1

    return matched, len(pred) - matched, len(gold) - matched


def _match_weak_argument_spans(
    gold: set[ArgumentSpanAnnotation], pred: set[ArgumentSpanAnnotation]
) -> tuple[int, int, int]:
    available_gold = set(gold)
    matched = 0

    for pred_event_start, pred_event_end, pred_arg_start, pred_arg_end in pred:
        best_gold = None
        best_overlap = 0.0

        for gold_event_start, gold_event_end, gold_arg_start, gold_arg_end in available_gold:
            if pred_event_start != gold_event_start or pred_event_end != gold_event_end:
                continue
            overlap_ratio = _span_overlap_ratio(
                (pred_arg_start, pred_arg_end), (gold_arg_start, gold_arg_end)
            )
            if (
                overlap_ratio >= _WEAK_MATCH_THRESHOLD
                and overlap_ratio > best_overlap
            ):
                best_overlap = overlap_ratio
                best_gold = (
                    gold_event_start,
                    gold_event_end,
                    gold_arg_start,
                    gold_arg_end,
                )

        if best_gold is None:
            continue

        available_gold.remove(best_gold)
        matched += 1

    return matched, len(pred) - matched, len(gold) - matched


def evaluate(
    gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> dict[str, float]:
    if len(gold_rows) != len(pred_rows):
        raise ValueError(
            f"Mismatched row counts: gold={len(gold_rows)} pred={len(pred_rows)}"
        )

    ev_tp = ev_fp = ev_fn = 0
    ev_span_tp = ev_span_fp = ev_span_fn = 0
    arg_tp = arg_fp = arg_fn = 0
    arg_span_tp = arg_span_fp = arg_span_fn = 0
    weak_ev_tp = weak_ev_fp = weak_ev_fn = 0
    weak_ev_span_tp = weak_ev_span_fp = weak_ev_span_fn = 0
    weak_arg_tp = weak_arg_fp = weak_arg_fn = 0
    weak_arg_span_tp = weak_arg_span_fp = weak_arg_span_fn = 0
    grounded_total = predicted_total = 0
    em_docs = 0
    status_totals = {
        AnchorStatus.MATCH_EXACT: 0,
        AnchorStatus.MATCH_LESSER: 0,
        AnchorStatus.MATCH_FUZZY: 0,
        AnchorStatus.NOT_FOUND: 0,
    }

    for g, p in zip(gold_rows, pred_rows):
        text = g["question"]
        g_events = g["answer"]["events"]
        p_events = _prediction_events(p)

        g_ev, g_arg, _, _, _ = _collect_annotations(text, g_events)
        p_ev, p_arg, grounded, total, status_counts = _collect_annotations(text, p_events)
        g_ev_spans = _event_spans(g_ev)
        p_ev_spans = _event_spans(p_ev)
        g_arg_spans = _argument_spans(g_arg)
        p_arg_spans = _argument_spans(p_arg)

        grounded_total += grounded
        predicted_total += total
        _merge_status_counts(status_totals, status_counts)

        doc_ev_tp, doc_ev_fp, doc_ev_fn = _score_annotation_sets(g_ev, p_ev)
        ev_tp += doc_ev_tp
        ev_fp += doc_ev_fp
        ev_fn += doc_ev_fn

        doc_ev_span_tp, doc_ev_span_fp, doc_ev_span_fn = _score_annotation_sets(
            g_ev_spans, p_ev_spans
        )
        ev_span_tp += doc_ev_span_tp
        ev_span_fp += doc_ev_span_fp
        ev_span_fn += doc_ev_span_fn

        doc_arg_tp, doc_arg_fp, doc_arg_fn = _score_annotation_sets(g_arg, p_arg)
        arg_tp += doc_arg_tp
        arg_fp += doc_arg_fp
        arg_fn += doc_arg_fn

        doc_arg_span_tp, doc_arg_span_fp, doc_arg_span_fn = _score_annotation_sets(
            g_arg_spans, p_arg_spans
        )
        arg_span_tp += doc_arg_span_tp
        arg_span_fp += doc_arg_span_fp
        arg_span_fn += doc_arg_span_fn

        doc_weak_ev_tp, doc_weak_ev_fp, doc_weak_ev_fn = _match_weak_events(g_ev, p_ev)
        weak_ev_tp += doc_weak_ev_tp
        weak_ev_fp += doc_weak_ev_fp
        weak_ev_fn += doc_weak_ev_fn

        (
            doc_weak_ev_span_tp,
            doc_weak_ev_span_fp,
            doc_weak_ev_span_fn,
        ) = _match_weak_event_spans(g_ev_spans, p_ev_spans)
        weak_ev_span_tp += doc_weak_ev_span_tp
        weak_ev_span_fp += doc_weak_ev_span_fp
        weak_ev_span_fn += doc_weak_ev_span_fn

        doc_weak_arg_tp, doc_weak_arg_fp, doc_weak_arg_fn = _match_weak_arguments(
            g_arg, p_arg
        )
        weak_arg_tp += doc_weak_arg_tp
        weak_arg_fp += doc_weak_arg_fp
        weak_arg_fn += doc_weak_arg_fn

        (
            doc_weak_arg_span_tp,
            doc_weak_arg_span_fp,
            doc_weak_arg_span_fn,
        ) = _match_weak_argument_spans(g_arg_spans, p_arg_spans)
        weak_arg_span_tp += doc_weak_arg_span_tp
        weak_arg_span_fp += doc_weak_arg_span_fp
        weak_arg_span_fn += doc_weak_arg_span_fn

        if g_ev == p_ev and g_arg == p_arg:
            em_docs += 1

    ev_p, ev_r, ev_f1 = _prf(ev_tp, ev_fp, ev_fn)
    ev_span_p, ev_span_r, ev_span_f1 = _prf(ev_span_tp, ev_span_fp, ev_span_fn)
    arg_p, arg_r, arg_f1 = _prf(arg_tp, arg_fp, arg_fn)
    arg_span_p, arg_span_r, arg_span_f1 = _prf(
        arg_span_tp, arg_span_fp, arg_span_fn
    )
    weak_ev_p, weak_ev_r, weak_ev_f1 = _prf(weak_ev_tp, weak_ev_fp, weak_ev_fn)
    weak_ev_span_p, weak_ev_span_r, weak_ev_span_f1 = _prf(
        weak_ev_span_tp, weak_ev_span_fp, weak_ev_span_fn
    )
    weak_arg_p, weak_arg_r, weak_arg_f1 = _prf(
        weak_arg_tp, weak_arg_fp, weak_arg_fn
    )
    weak_arg_span_p, weak_arg_span_r, weak_arg_span_f1 = _prf(
        weak_arg_span_tp, weak_arg_span_fp, weak_arg_span_fn
    )

    grounding_rate = grounded_total / predicted_total if predicted_total else 0.0
    hallucinated_span_rate = 1.0 - grounding_rate
    doc_em = em_docs / len(gold_rows) if gold_rows else 0.0
    exact_rate = (
        status_totals[AnchorStatus.MATCH_EXACT] / predicted_total
        if predicted_total
        else 0.0
    )
    lesser_rate = (
        status_totals[AnchorStatus.MATCH_LESSER] / predicted_total
        if predicted_total
        else 0.0
    )
    fuzzy_rate = (
        status_totals[AnchorStatus.MATCH_FUZZY] / predicted_total
        if predicted_total
        else 0.0
    )
    not_found_rate = (
        status_totals[AnchorStatus.NOT_FOUND] / predicted_total
        if predicted_total
        else 0.0
    )

    return {
        "event_precision": ev_p,
        "event_recall": ev_r,
        "event_f1": ev_f1,
        "event_span_precision": ev_span_p,
        "event_span_recall": ev_span_r,
        "event_span_f1": ev_span_f1,
        "event_weak_precision": weak_ev_p,
        "event_weak_recall": weak_ev_r,
        "event_weak_f1": weak_ev_f1,
        "event_span_weak_precision": weak_ev_span_p,
        "event_span_weak_recall": weak_ev_span_r,
        "event_span_weak_f1": weak_ev_span_f1,
        "argument_precision": arg_p,
        "argument_recall": arg_r,
        "argument_f1": arg_f1,
        "argument_span_precision": arg_span_p,
        "argument_span_recall": arg_span_r,
        "argument_span_f1": arg_span_f1,
        "argument_weak_precision": weak_arg_p,
        "argument_weak_recall": weak_arg_r,
        "argument_weak_f1": weak_arg_f1,
        "argument_span_weak_precision": weak_arg_span_p,
        "argument_span_weak_recall": weak_arg_span_r,
        "argument_span_weak_f1": weak_arg_span_f1,
        "grounding_rate": grounding_rate,
        "hallucinated_span_rate": hallucinated_span_rate,
        "doc_exact_match": doc_em,
        "anchor_exact_count": status_totals[AnchorStatus.MATCH_EXACT],
        "anchor_lesser_count": status_totals[AnchorStatus.MATCH_LESSER],
        "anchor_fuzzy_count": status_totals[AnchorStatus.MATCH_FUZZY],
        "anchor_not_found_count": status_totals[AnchorStatus.NOT_FOUND],
        "anchor_exact_rate": exact_rate,
        "anchor_lesser_rate": lesser_rate,
        "anchor_fuzzy_rate": fuzzy_rate,
        "anchor_not_found_rate": not_found_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-jsonl", type=str, required=True)
    parser.add_argument("--pred-jsonl", type=str, required=True)
    args = parser.parse_args()

    gold_rows = _load_jsonl(Path(args.gold_jsonl))
    pred_rows = _load_jsonl(Path(args.pred_jsonl))
    metrics = evaluate(gold_rows, pred_rows)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
