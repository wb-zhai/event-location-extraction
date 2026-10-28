#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def gold_events(row):
    ann = row.get("annotation")
    if isinstance(ann, dict):
        ev = ann.get("events") or []
        return ev if isinstance(ev, list) else []
    if isinstance(ann, list):
        return ann
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred-jsonl",
        default="data/extraction/eval/predictions/v1_teacher_predictions_v4.jsonl",
    )
    ap.add_argument(
        "--mapping-report",
        default="data/extraction/eval/predictions/v1_teacher_mapping_report.json",
    )
    ap.add_argument("--expected-rows", type=int, default=2198)
    args = ap.parse_args()

    p = Path(args.pred_jsonl)
    if not p.exists():
        raise SystemExit(f"Missing predictions file: {p}")

    n = 0
    empty_gold = 0
    positive_gold = 0
    predicted_empty = 0
    correct_empty = 0
    fp_empty_docs = 0
    fn_positive_docs = 0
    total_spurious_empty = 0

    pred_events = 0
    with_modality = 0
    with_time_status = 0
    with_admin = 0
    with_severity = 0

    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1

            gold = gold_events(row)
            pred = row.get("predictions") or []
            if not isinstance(pred, list):
                pred = []

            g_empty = len(gold) == 0
            p_empty = len(pred) == 0

            empty_gold += int(g_empty)
            positive_gold += int(not g_empty)
            predicted_empty += int(p_empty)

            if g_empty and p_empty:
                correct_empty += 1
            elif g_empty and not p_empty:
                fp_empty_docs += 1
                total_spurious_empty += len(pred)
            elif (not g_empty) and p_empty:
                fn_positive_docs += 1

            for e in pred:
                if not isinstance(e, dict):
                    continue
                pred_events += 1
                with_modality += int(e.get("modality") is not None)
                with_time_status += int(e.get("time_status") is not None)
                with_admin += int(e.get("event_location_admin_level") is not None)
                with_severity += int(e.get("severity") is not None)

    if n != args.expected_rows:
        raise SystemExit(f"Expected {args.expected_rows} rows, got {n}")

    noevent_precision = correct_empty / predicted_empty if predicted_empty else 0.0
    noevent_recall = correct_empty / empty_gold if empty_gold else 0.0
    noevent_f1 = (
        2 * noevent_precision * noevent_recall / (noevent_precision + noevent_recall)
        if (noevent_precision + noevent_recall) else 0.0
    )
    fp_rate_empty = fp_empty_docs / empty_gold if empty_gold else 0.0
    fn_rate_positive = fn_positive_docs / positive_gold if positive_gold else 0.0
    mean_spurious = (
        total_spurious_empty / fp_empty_docs if fp_empty_docs else 0.0
    )

    print("=" * 72)
    print("V1 TEACHER FINAL CHECKS")
    print("=" * 72)
    print(f"records:                         {n}")
    print(f"empty gold:                      {empty_gold}")
    print(f"positive gold:                   {positive_gold}")
    print(f"predicted-empty:                 {predicted_empty}")
    print(f"correct empty:                   {correct_empty}")
    print(f"FP empty-docs:                   {fp_empty_docs}")
    print(f"FP rate empty:                   {100*fp_rate_empty:.2f}%")
    print(f"false-negative positive docs:    {fn_positive_docs}")
    print(f"FN rate positive:                {100*fn_rate_positive:.2f}%")
    print(f"no-event precision:              {100*noevent_precision:.2f}%")
    print(f"no-event recall:                 {100*noevent_recall:.2f}%")
    print(f"no-event F1:                     {100*noevent_f1:.2f}%")
    print(f"mean spurious events / FP-empty: {mean_spurious:.3f}")
    print(f"total spurious events on empty:  {total_spurious_empty}")
    print()
    print("V1 mapped-prediction field coverage")
    print(f"prediction events:               {pred_events}")
    print(f"modality present:                {with_modality}/{pred_events}")
    print(f"time_status present:             {with_time_status}/{pred_events}")
    print(f"admin-level present:             {with_admin}/{pred_events}")
    print(f"severity present:                {with_severity}/{pred_events}")

    mr = Path(args.mapping_report)
    if mr.exists():
        r = json.loads(mr.read_text(encoding="utf-8"))
        print()
        print("Mapping audit")
        for key in (
            "raw_prediction_events",
            "mapped_prediction_events",
            "mapped_rate",
            "unmapped_prediction_events",
            "unmapped_rate",
            "intentionally_unmapped_prediction_events",
            "unknown_unmapped_prediction_events",
            "rows_with_unmapped_predictions",
        ):
            if key in r:
                val = r[key]
                if key.endswith("_rate"):
                    print(f"{key}: {100*float(val):.2f}%")
                else:
                    print(f"{key}: {val}")
        print("unmapped_labels:")
        for k, v in (r.get("unmapped_labels") or {}).items():
            print(f"  {k}: {v}")
    else:
        print()
        print(f"Mapping report not found: {mr}")

if __name__ == "__main__":
    main()
