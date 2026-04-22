from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.inference.text_anchor import AnchorStatus, TextAnchorResolver


_ANCHOR_RESOLVER = TextAnchorResolver()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _normalize(doc_text: str, events: list[dict[str, Any]]) -> tuple[
    set[tuple[str, int, int]],
    set[tuple[str, int, int, str, int, int]],
    int,
    int,
    dict[str, int],
]:
    event_set: set[tuple[str, int, int]] = set()
    arg_set: set[tuple[str, int, int, str, int, int]] = set()
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
        ev_span, ev_status = _resolve_span(doc_text, ev)
        status_counts[ev_status] = status_counts.get(ev_status, 0) + 1
        if ev_span is None:
            continue
        grounded += 1
        es, ee = ev_span
        event_set.add((ev_type, es, ee))

        for arg in ev.get("arguments", []):
            total += 1
            role = arg.get("role", "")
            arg_span, arg_status = _resolve_span(doc_text, arg)
            status_counts[arg_status] = status_counts.get(arg_status, 0) + 1
            if arg_span is None:
                continue
            grounded += 1
            a_s, a_e = arg_span
            arg_set.add((ev_type, es, ee, role, a_s, a_e))

    return event_set, arg_set, grounded, total, status_counts


def evaluate(
    gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> dict[str, float]:
    if len(gold_rows) != len(pred_rows):
        raise ValueError(
            f"Mismatched row counts: gold={len(gold_rows)} pred={len(pred_rows)}"
        )

    ev_tp = ev_fp = ev_fn = 0
    arg_tp = arg_fp = arg_fn = 0
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
        p_events = p.get("answer", {}).get("events", [])

        g_ev, g_arg, _, _, _ = _normalize(text, g_events)
        p_ev, p_arg, grounded, total, status_counts = _normalize(text, p_events)

        grounded_total += grounded
        predicted_total += total
        for status, count in status_counts.items():
            status_totals[status] = status_totals.get(status, 0) + count

        ev_tp += len(g_ev & p_ev)
        ev_fp += len(p_ev - g_ev)
        ev_fn += len(g_ev - p_ev)

        arg_tp += len(g_arg & p_arg)
        arg_fp += len(p_arg - g_arg)
        arg_fn += len(g_arg - p_arg)

        if g_ev == p_ev and g_arg == p_arg:
            em_docs += 1

    ev_p, ev_r, ev_f1 = _prf(ev_tp, ev_fp, ev_fn)
    arg_p, arg_r, arg_f1 = _prf(arg_tp, arg_fp, arg_fn)

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
        "argument_precision": arg_p,
        "argument_recall": arg_r,
        "argument_f1": arg_f1,
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
