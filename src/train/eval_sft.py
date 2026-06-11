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
EventTextAnnotation = tuple[str, str]
ArgumentDetachedSpanAnnotation = tuple[str, str, int, int]
ArgumentTextAnnotation = tuple[str, str, str]


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


def _document_event_groups(
    gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    gold_doc_events: dict[str, list[dict[str, Any]]] = {}
    pred_doc_events: dict[str, list[dict[str, Any]]] = {}
    doc_keys = _document_group_keys(gold_rows)

    for doc_key, gold_row, pred_row in zip(doc_keys, gold_rows, pred_rows):
        gold_doc_events.setdefault(doc_key, []).extend(gold_row["answer"]["events"])
        pred_doc_events.setdefault(doc_key, []).extend(_prediction_events(pred_row))

    return gold_doc_events, pred_doc_events


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
    obj: dict[str, Any],
    *,
    keys: tuple[str, ...] = ("trigger", "argument", "entity", "span"),
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
        if not isinstance(ev, dict):
            continue
        total += 1
        ev_type = ev.get("event_type", "")
        ev_span, ev_status = _resolve_span(doc_text, _anchor_payload(ev))
        status_counts[ev_status] = status_counts.get(ev_status, 0) + 1
        if ev_span is None:
            continue
        grounded += 1
        es, ee = ev_span
        event_set.add((ev_type, es, ee))

        arguments = ev.get("arguments", [])
        if not isinstance(arguments, list):
            continue
        for arg in arguments:
            if not isinstance(arg, dict):
                continue
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


def _argument_detached_spans(
    args: set[ArgumentAnnotation],
) -> set[ArgumentDetachedSpanAnnotation]:
    return {
        (event_type, role, arg_start, arg_end)
        for event_type, _, _, role, arg_start, arg_end in args
    }


def _normalize_span_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _event_span_texts(
    doc_text: str, events: list[dict[str, Any]]
) -> set[EventTextAnnotation]:
    event_texts: set[EventTextAnnotation] = set()

    for ev in events:
        if not isinstance(ev, dict):
            continue
        ev_type = ev.get("event_type", "")
        if not isinstance(ev_type, str) or not ev_type:
            continue
        ev_span, _ = _resolve_span(doc_text, _anchor_payload(ev))
        if ev_span is None:
            continue
        start, end = ev_span
        span_text = _normalize_span_text(doc_text[start:end])
        if span_text:
            event_texts.add((ev_type, span_text))

    return event_texts


def _argument_texts(
    doc_text: str, events: list[dict[str, Any]]
) -> set[ArgumentTextAnnotation]:
    argument_texts: set[ArgumentTextAnnotation] = set()

    for ev in events:
        if not isinstance(ev, dict):
            continue
        ev_type = ev.get("event_type", "")
        if not isinstance(ev_type, str) or not ev_type:
            continue
        arguments = ev.get("arguments", [])
        if not isinstance(arguments, list):
            continue
        for arg in arguments:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role", "")
            if not isinstance(role, str) or not role:
                continue
            arg_span, _ = _resolve_span(doc_text, _anchor_payload(arg))
            if arg_span is None:
                continue
            start, end = arg_span
            span_text = _normalize_span_text(doc_text[start:end])
            if span_text:
                argument_texts.add((ev_type, role, span_text))

    return argument_texts


def _merge_status_counts(
    totals: dict[str, int], counts: dict[str, int]
) -> dict[str, int]:
    for status, count in counts.items():
        totals[status] = totals.get(status, 0) + count
    return totals


def _score_annotation_sets(
    gold: set[Any], pred: set[Any]
) -> tuple[int, int, int]:
    return len(gold & pred), len(pred - gold), len(gold - pred)


def _score_event_type_docs(
    gold_doc_events: dict[str, list[dict[str, Any]]],
    pred_doc_events: dict[str, list[dict[str, Any]]],
) -> tuple[int, int, int, int]:
    tp = fp = fn = exact_docs = 0

    for doc_key in gold_doc_events:
        gold_event_types = _event_types(gold_doc_events[doc_key])
        pred_event_types = _event_types(pred_doc_events.get(doc_key, []))
        doc_tp, doc_fp, doc_fn = _score_annotation_sets(
            gold_event_types, pred_event_types
        )
        tp += doc_tp
        fp += doc_fp
        fn += doc_fn
        if gold_event_types == pred_event_types:
            exact_docs += 1

    return tp, fp, fn, exact_docs


def _doc_event_comparisons(
    gold_doc_events: dict[str, list[dict[str, Any]]],
    pred_doc_events: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for doc_key in gold_doc_events:
        gold_event_types = _event_types(gold_doc_events[doc_key])
        pred_event_types = _event_types(pred_doc_events.get(doc_key, []))
        rows.append(
            {
                "doc_key": doc_key,
                "correct": sorted(gold_event_types & pred_event_types),
                "missed": sorted(gold_event_types - pred_event_types),
                "wrong": sorted(pred_event_types - gold_event_types),
            }
        )

    return rows


def write_doc_event_report(
    gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], path: Path
) -> None:
    if len(gold_rows) != len(pred_rows):
        raise ValueError(
            f"Mismatched row counts: gold={len(gold_rows)} pred={len(pred_rows)}"
        )

    gold_doc_events, pred_doc_events = _document_event_groups(gold_rows, pred_rows)
    report = {
        "documents": _doc_event_comparisons(gold_doc_events, pred_doc_events),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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


def _span_text_similarity(first_text: str, second_text: str) -> float:
    if first_text == second_text:
        return 1.0

    first_tokens = set(first_text.split())
    second_tokens = set(second_text.split())
    token_similarity = 0.0
    if first_tokens and second_tokens:
        token_similarity = len(first_tokens & second_tokens) / max(
            len(first_tokens), len(second_tokens)
        )

    substring_similarity = 0.0
    if first_text in second_text or second_text in first_text:
        substring_similarity = min(len(first_text), len(second_text)) / max(
            len(first_text), len(second_text)
        )

    return max(token_similarity, substring_similarity)


def _match_weak_event_texts(
    gold: set[EventTextAnnotation], pred: set[EventTextAnnotation]
) -> tuple[int, int, int]:
    available_gold = set(gold)
    matched = 0

    for pred_event_type, pred_text in pred:
        best_gold = None
        best_similarity = 0.0

        for gold_event_type, gold_text in available_gold:
            if pred_event_type != gold_event_type:
                continue
            similarity = _span_text_similarity(pred_text, gold_text)
            if (
                similarity >= _WEAK_MATCH_THRESHOLD
                and similarity > best_similarity
            ):
                best_similarity = similarity
                best_gold = (gold_event_type, gold_text)

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


def _match_weak_detached_argument_spans(
    gold: set[ArgumentDetachedSpanAnnotation], pred: set[ArgumentDetachedSpanAnnotation]
) -> tuple[int, int, int]:
    available_gold = set(gold)
    matched = 0

    for pred_event_type, pred_role, pred_arg_start, pred_arg_end in pred:
        best_gold = None
        best_overlap = 0.0

        for gold_event_type, gold_role, gold_arg_start, gold_arg_end in available_gold:
            if pred_event_type != gold_event_type or pred_role != gold_role:
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
                    gold_event_type,
                    gold_role,
                    gold_arg_start,
                    gold_arg_end,
                )

        if best_gold is None:
            continue

        available_gold.remove(best_gold)
        matched += 1

    return matched, len(pred) - matched, len(gold) - matched


def _match_weak_argument_texts(
    gold: set[ArgumentTextAnnotation], pred: set[ArgumentTextAnnotation]
) -> tuple[int, int, int]:
    available_gold = set(gold)
    matched = 0

    for pred_event_type, pred_role, pred_text in pred:
        best_gold = None
        best_similarity = 0.0

        for gold_event_type, gold_role, gold_text in available_gold:
            if pred_event_type != gold_event_type or pred_role != gold_role:
                continue
            similarity = _span_text_similarity(pred_text, gold_text)
            if (
                similarity >= _WEAK_MATCH_THRESHOLD
                and similarity > best_similarity
            ):
                best_similarity = similarity
                best_gold = (gold_event_type, gold_role, gold_text)

        if best_gold is None:
            continue

        available_gold.remove(best_gold)
        matched += 1

    return matched, len(pred) - matched, len(gold) - matched


def _metric_dict(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(gold_rows) != len(pred_rows):
        raise ValueError(
            f"Mismatched row counts: gold={len(gold_rows)} pred={len(pred_rows)}"
        )

    ev_tp = ev_fp = ev_fn = 0
    ev_span_tp = ev_span_fp = ev_span_fn = 0
    arg_tp = arg_fp = arg_fn = 0
    arg_span_tp = arg_span_fp = arg_span_fn = 0
    arg_detached_span_tp = arg_detached_span_fp = arg_detached_span_fn = 0
    weak_ev_tp = weak_ev_fp = weak_ev_fn = 0
    weak_ev_span_tp = weak_ev_span_fp = weak_ev_span_fn = 0
    doc_ev_text_tp = doc_ev_text_fp = doc_ev_text_fn = 0
    doc_weak_ev_text_tp = doc_weak_ev_text_fp = doc_weak_ev_text_fn = 0
    weak_arg_tp = weak_arg_fp = weak_arg_fn = 0
    weak_arg_span_tp = weak_arg_span_fp = weak_arg_span_fn = 0
    weak_arg_detached_span_tp = weak_arg_detached_span_fp = weak_arg_detached_span_fn = 0
    doc_arg_text_tp = doc_arg_text_fp = doc_arg_text_fn = 0
    doc_weak_arg_text_tp = doc_weak_arg_text_fp = doc_weak_arg_text_fn = 0
    event_level_tp = event_level_fp = event_level_fn = 0
    gold_doc_events, pred_doc_events = _document_event_groups(gold_rows, pred_rows)
    doc_keys = _document_group_keys(gold_rows)
    gold_doc_event_texts: dict[str, set[EventTextAnnotation]] = {}
    pred_doc_event_texts: dict[str, set[EventTextAnnotation]] = {}
    gold_doc_argument_texts: dict[str, set[ArgumentTextAnnotation]] = {}
    pred_doc_argument_texts: dict[str, set[ArgumentTextAnnotation]] = {}
    grounded_total = predicted_total = 0
    em_docs = 0
    event_level_em_docs = 0
    status_totals = {
        AnchorStatus.MATCH_EXACT: 0,
        AnchorStatus.MATCH_LESSER: 0,
        AnchorStatus.MATCH_FUZZY: 0,
        AnchorStatus.NOT_FOUND: 0,
    }

    for doc_key, g, p in zip(doc_keys, gold_rows, pred_rows):
        text = g["question"]
        g_events = g["answer"]["events"]
        p_events = _prediction_events(p)

        g_event_types = _event_types(g_events)
        p_event_types = _event_types(p_events)
        g_ev, g_arg, _, _, _ = _collect_annotations(text, g_events)
        p_ev, p_arg, grounded, total, status_counts = _collect_annotations(text, p_events)
        g_ev_spans = _event_spans(g_ev)
        p_ev_spans = _event_spans(p_ev)
        g_arg_spans = _argument_spans(g_arg)
        p_arg_spans = _argument_spans(p_arg)
        g_arg_detached_spans = _argument_detached_spans(g_arg)
        p_arg_detached_spans = _argument_detached_spans(p_arg)
        gold_doc_event_texts.setdefault(doc_key, set()).update(
            _event_span_texts(text, g_events)
        )
        pred_doc_event_texts.setdefault(doc_key, set()).update(
            _event_span_texts(text, p_events)
        )
        gold_doc_argument_texts.setdefault(doc_key, set()).update(
            _argument_texts(text, g_events)
        )
        pred_doc_argument_texts.setdefault(doc_key, set()).update(
            _argument_texts(text, p_events)
        )

        grounded_total += grounded
        predicted_total += total
        _merge_status_counts(status_totals, status_counts)

        doc_event_level_tp, doc_event_level_fp, doc_event_level_fn = (
            _score_annotation_sets(g_event_types, p_event_types)
        )
        event_level_tp += doc_event_level_tp
        event_level_fp += doc_event_level_fp
        event_level_fn += doc_event_level_fn

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

        (
            doc_arg_detached_span_tp,
            doc_arg_detached_span_fp,
            doc_arg_detached_span_fn,
        ) = _score_annotation_sets(g_arg_detached_spans, p_arg_detached_spans)
        arg_detached_span_tp += doc_arg_detached_span_tp
        arg_detached_span_fp += doc_arg_detached_span_fp
        arg_detached_span_fn += doc_arg_detached_span_fn

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

        (
            doc_weak_arg_detached_span_tp,
            doc_weak_arg_detached_span_fp,
            doc_weak_arg_detached_span_fn,
        ) = _match_weak_detached_argument_spans(
            g_arg_detached_spans, p_arg_detached_spans
        )
        weak_arg_detached_span_tp += doc_weak_arg_detached_span_tp
        weak_arg_detached_span_fp += doc_weak_arg_detached_span_fp
        weak_arg_detached_span_fn += doc_weak_arg_detached_span_fn

        if g_ev == p_ev and g_arg == p_arg:
            em_docs += 1
        if g_event_types == p_event_types:
            event_level_em_docs += 1

    for doc_key, g_event_texts in gold_doc_event_texts.items():
        p_event_texts = pred_doc_event_texts.get(doc_key, set())
        doc_text_tp, doc_text_fp, doc_text_fn = _score_annotation_sets(
            g_event_texts, p_event_texts
        )
        doc_ev_text_tp += doc_text_tp
        doc_ev_text_fp += doc_text_fp
        doc_ev_text_fn += doc_text_fn

        doc_weak_text_tp, doc_weak_text_fp, doc_weak_text_fn = (
            _match_weak_event_texts(g_event_texts, p_event_texts)
        )
        doc_weak_ev_text_tp += doc_weak_text_tp
        doc_weak_ev_text_fp += doc_weak_text_fp
        doc_weak_ev_text_fn += doc_weak_text_fn

    for doc_key, g_argument_texts in gold_doc_argument_texts.items():
        p_argument_texts = pred_doc_argument_texts.get(doc_key, set())
        doc_text_tp, doc_text_fp, doc_text_fn = _score_annotation_sets(
            g_argument_texts, p_argument_texts
        )
        doc_arg_text_tp += doc_text_tp
        doc_arg_text_fp += doc_text_fp
        doc_arg_text_fn += doc_text_fn

        doc_weak_text_tp, doc_weak_text_fp, doc_weak_text_fn = (
            _match_weak_argument_texts(g_argument_texts, p_argument_texts)
        )
        doc_weak_arg_text_tp += doc_weak_text_tp
        doc_weak_arg_text_fp += doc_weak_text_fp
        doc_weak_arg_text_fn += doc_weak_text_fn

    doc_event_level_tp, doc_event_level_fp, doc_event_level_fn, doc_event_level_em = (
        _score_event_type_docs(gold_doc_events, pred_doc_events)
    )

    grounding_rate = grounded_total / predicted_total if predicted_total else 0.0
    hallucinated_span_rate = 1.0 - grounding_rate
    doc_em = em_docs / len(gold_rows) if gold_rows else 0.0
    event_level_doc_em = event_level_em_docs / len(gold_rows) if gold_rows else 0.0
    doc_count = len(gold_doc_events)
    doc_event_level_doc_em = doc_event_level_em / doc_count if doc_count else 0.0
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
        "event_type": {
            "row": {
                **_metric_dict(event_level_tp, event_level_fp, event_level_fn),
                "exact_match": event_level_doc_em,
            },
            "document": {
                **_metric_dict(
                    doc_event_level_tp, doc_event_level_fp, doc_event_level_fn
                ),
                "exact_match": doc_event_level_doc_em,
            },
        },
        "event": {
            "strict": _metric_dict(ev_tp, ev_fp, ev_fn),
            "span_only": _metric_dict(ev_span_tp, ev_span_fp, ev_span_fn),
            "weak": _metric_dict(weak_ev_tp, weak_ev_fp, weak_ev_fn),
            "span_only_weak": _metric_dict(
                weak_ev_span_tp, weak_ev_span_fp, weak_ev_span_fn
            ),
            "document_text": _metric_dict(doc_ev_text_tp, doc_ev_text_fp, doc_ev_text_fn),
            "document_text_weak": _metric_dict(
                doc_weak_ev_text_tp, doc_weak_ev_text_fp, doc_weak_ev_text_fn
            ),
        },
        "argument": {
            "strict": _metric_dict(arg_tp, arg_fp, arg_fn),
            "trigger_span_only": _metric_dict(
                arg_span_tp, arg_span_fp, arg_span_fn
            ),
            "event_role_span": _metric_dict(
                arg_detached_span_tp,
                arg_detached_span_fp,
                arg_detached_span_fn,
            ),
            "weak": _metric_dict(weak_arg_tp, weak_arg_fp, weak_arg_fn),
            "trigger_span_only_weak": _metric_dict(
                weak_arg_span_tp, weak_arg_span_fp, weak_arg_span_fn
            ),
            "event_role_span_weak": _metric_dict(
                weak_arg_detached_span_tp,
                weak_arg_detached_span_fp,
                weak_arg_detached_span_fn,
            ),
            "document_text": _metric_dict(
                doc_arg_text_tp, doc_arg_text_fp, doc_arg_text_fn
            ),
            "document_text_weak": _metric_dict(
                doc_weak_arg_text_tp,
                doc_weak_arg_text_fp,
                doc_weak_arg_text_fn,
            ),
        },
        "exact_match": {
            "document": doc_em,
        },
        "grounding": {
            "rate": grounding_rate,
            "hallucinated_span_rate": hallucinated_span_rate,
            "anchors": {
                AnchorStatus.MATCH_EXACT: {
                    "count": status_totals[AnchorStatus.MATCH_EXACT],
                    "rate": exact_rate,
                },
                AnchorStatus.MATCH_LESSER: {
                    "count": status_totals[AnchorStatus.MATCH_LESSER],
                    "rate": lesser_rate,
                },
                AnchorStatus.MATCH_FUZZY: {
                    "count": status_totals[AnchorStatus.MATCH_FUZZY],
                    "rate": fuzzy_rate,
                },
                AnchorStatus.NOT_FOUND: {
                    "count": status_totals[AnchorStatus.NOT_FOUND],
                    "rate": not_found_rate,
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-jsonl", type=str, required=True)
    parser.add_argument("--pred-jsonl", type=str, required=True)
    parser.add_argument("--report-json", type=str, default=None)
    args = parser.parse_args()

    gold_rows = _load_jsonl(Path(args.gold_jsonl))
    pred_rows = _load_jsonl(Path(args.pred_jsonl))
    metrics = evaluate(gold_rows, pred_rows)
    if args.report_json is not None:
        write_doc_event_report(gold_rows, pred_rows, Path(args.report_json))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
