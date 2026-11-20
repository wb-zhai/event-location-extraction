#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r.get("id"))
            rows[rid] = r
    return rows

def gold_events(r):
    ann = r.get("annotation") or {}
    return ann.get("events") or []

def pred_events(r):
    return r.get("predictions") or []

def safe_div(a, b):
    return a / b if b else 0.0

def summarize(name, rows):
    n = len(rows)
    gold_empty = 0
    gold_positive = 0
    correct_empty = 0
    fp_empty_docs = 0
    fn_positive_docs = 0
    pred_empty_total = 0
    spurious_counts = []

    for r in rows.values():
        g = gold_events(r)
        p = pred_events(r)
        ge = len(g) == 0
        pe = len(p) == 0

        gold_empty += ge
        gold_positive += not ge
        pred_empty_total += pe

        if ge and pe:
            correct_empty += 1
        elif ge and not pe:
            fp_empty_docs += 1
            spurious_counts.append(len(p))
        elif (not ge) and pe:
            fn_positive_docs += 1

    # Treat "no-event" as the positive class.
    tp = correct_empty
    fp = fn_positive_docs
    fn = fp_empty_docs
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)

    print(f"\n===== {name} =====")
    print(f"records:                         {n}")
    print(f"Teacher-empty articles:          {gold_empty}")
    print(f"Teacher-positive articles:       {gold_positive}")
    print(f"predicted-empty articles:        {pred_empty_total}")
    print(f"correct Teacher-empty:           {correct_empty}")
    print(f"false-positive empty-docs:       {fp_empty_docs}")
    print(f"FP rate on Teacher-empty:        {100*safe_div(fp_empty_docs, gold_empty):.2f}%")
    print(f"false-negative positive-docs:    {fn_positive_docs}")
    print(f"FN rate on Teacher-positive:     {100*safe_div(fn_positive_docs, gold_positive):.2f}%")
    print(f"no-event precision:              {100*prec:.2f}%")
    print(f"no-event recall:                 {100*rec:.2f}%")
    print(f"no-event F1:                     {100*f1:.2f}%")
    print(f"mean spurious events / FP empty: {safe_div(sum(spurious_counts), len(spurious_counts)):.3f}")
    print(f"total spurious events on empty:  {sum(spurious_counts)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v3", required=True)
    args = ap.parse_args()

    v2 = load(args.v2)
    v3 = load(args.v3)

    if set(v2) != set(v3):
        only2 = len(set(v2) - set(v3))
        only3 = len(set(v3) - set(v2))
        raise SystemExit(f"ID mismatch: only_v2={only2}, only_v3={only3}")

    # Gold annotations must be identical article-by-article.
    bad = []
    for rid in v2:
        if (v2[rid].get("annotation") or {}) != (v3[rid].get("annotation") or {}):
            bad.append(rid)
            if len(bad) >= 5:
                break
    if bad:
        raise SystemExit(f"Gold annotation mismatch for IDs: {bad}")

    summarize("V2", v2)
    summarize("V3", v3)

if __name__ == "__main__":
    main()
