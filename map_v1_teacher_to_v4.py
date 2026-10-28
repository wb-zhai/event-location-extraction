#!/usr/bin/env python3
"""
Map V1 legacy event_type labels into the v4 label space for Teacher evaluation.

This intentionally:
  - preserves each input row, including annotation/source/window_predictions;
  - rewrites only top-level predictions[*].event_type;
  - excludes unmapped legacy predictions from the common-v4 predictions list;
  - writes an audit report with mapped/unmapped counts and label frequencies;
  - does not fabricate V1-missing fields such as severity or
    event_location_admin_level.

The crosswalk is read directly from OLD_TO_V4 in baseline_clustering_v4.py using
AST, so importing the clustering module (and all its optional dependencies) is
not required.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_crosswalk(path: Path) -> dict[str, str | None]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target_names = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        target_names.append(t.id)
                value = node.value
            else:
                if isinstance(node.target, ast.Name):
                    target_names.append(node.target.id)
                value = node.value
            if "OLD_TO_V4" in target_names:
                obj = ast.literal_eval(value)
                if not isinstance(obj, dict):
                    raise TypeError("OLD_TO_V4 is not a dictionary")
                out = {}
                for k, v in obj.items():
                    kk = str(k).strip().lower()
                    vv = None if v is None else str(v).strip().lower()
                    out[kk] = vv
                return out
    raise RuntimeError(f"Could not find OLD_TO_V4 in {path}")


def load_active_labels(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    labels: set[str] = set()

    if isinstance(obj, dict):
        # bona.v4.clusters.json may be either name->cluster or a richer object.
        if all(isinstance(k, str) for k in obj):
            # If values look like cluster names, keys are labels.
            for k in obj:
                k = str(k).strip().lower()
                if k:
                    labels.add(k)
        # Also tolerate nested structures.
        for v in obj.values():
            if isinstance(v, dict) and v.get("name"):
                labels.add(str(v["name"]).strip().lower())
    elif isinstance(obj, list):
        for row in obj:
            if isinstance(row, dict) and row.get("name"):
                labels.add(str(row["name"]).strip().lower())
            elif isinstance(row, str):
                labels.add(row.strip().lower())

    if not labels:
        raise RuntimeError(f"No active v4 labels found in {path}")
    return labels


def norm_label(value: Any) -> str:
    return str(value or "").strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/extraction/eval/predictions/v1_teacher_predictions_raw.jsonl",
    )
    ap.add_argument(
        "--output",
        default="data/extraction/eval/predictions/v1_teacher_predictions_v4.jsonl",
    )
    ap.add_argument(
        "--crosswalk-source",
        default="baseline_clustering_v4.py",
    )
    ap.add_argument(
        "--taxonomy",
        default="ontologies/zhai/bona.v4.clusters.json",
    )
    ap.add_argument(
        "--report",
        default="data/extraction/eval/predictions/v1_teacher_mapping_report.json",
    )
    ap.add_argument("--expected-rows", type=int, default=2198)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    crosswalk_path = Path(args.crosswalk_source)
    taxonomy_path = Path(args.taxonomy)
    report_path = Path(args.report)

    for p in (in_path, crosswalk_path, taxonomy_path):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    crosswalk = load_crosswalk(crosswalk_path)
    active = load_active_labels(taxonomy_path)

    rows = 0
    raw_events = 0
    mapped_events = 0
    crosswalked_events = 0
    native_events = 0
    unmapped_events = 0
    intentionally_unmapped_events = 0
    unknown_unmapped_events = 0
    rows_with_unmapped = 0

    raw_label_counts = Counter()
    mapped_label_counts = Counter()
    unmapped_label_counts = Counter()
    intentional_unmapped_label_counts = Counter()
    unknown_unmapped_label_counts = Counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1

            preds = row.get("predictions") or []
            if not isinstance(preds, list):
                raise RuntimeError(f"Row {line_no}: predictions is not a list")

            mapped_preds = []
            row_had_unmapped = False

            for ev in preds:
                if not isinstance(ev, dict):
                    continue

                raw_events += 1
                raw = norm_label(ev.get("event_type"))
                raw_label_counts[raw or "<missing>"] += 1

                target = None
                status = None

                if raw in crosswalk:
                    target = crosswalk[raw]
                    if target is None:
                        status = "intentional_unmapped"
                    else:
                        status = "crosswalk"
                elif raw in active:
                    target = raw
                    status = "native_v4"
                else:
                    status = "unknown_unmapped"

                if target:
                    if target not in active:
                        raise RuntimeError(
                            f"Crosswalk target {target!r} for {raw!r} is not in active v4 taxonomy"
                        )
                    new_ev = dict(ev)
                    new_ev["event_type"] = target
                    mapped_preds.append(new_ev)

                    mapped_events += 1
                    mapped_label_counts[target] += 1
                    if status == "crosswalk":
                        crosswalked_events += 1
                    else:
                        native_events += 1
                else:
                    row_had_unmapped = True
                    unmapped_events += 1
                    unmapped_label_counts[raw or "<missing>"] += 1
                    if status == "intentional_unmapped":
                        intentionally_unmapped_events += 1
                        intentional_unmapped_label_counts[raw or "<missing>"] += 1
                    else:
                        unknown_unmapped_events += 1
                        unknown_unmapped_label_counts[raw or "<missing>"] += 1

            if row_had_unmapped:
                rows_with_unmapped += 1

            # Keep the original Teacher gold annotation untouched.
            row["predictions"] = mapped_preds
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows != args.expected_rows:
        raise RuntimeError(f"Expected {args.expected_rows} rows, got {rows}")

    mapped_rate = (mapped_events / raw_events) if raw_events else 0.0
    unmapped_rate = (unmapped_events / raw_events) if raw_events else 0.0

    report = {
        "input": str(in_path),
        "output": str(out_path),
        "crosswalk_source": str(crosswalk_path),
        "taxonomy": str(taxonomy_path),
        "rows": rows,
        "raw_prediction_events": raw_events,
        "mapped_prediction_events": mapped_events,
        "mapped_rate": mapped_rate,
        "crosswalked_prediction_events": crosswalked_events,
        "native_v4_prediction_events": native_events,
        "unmapped_prediction_events": unmapped_events,
        "unmapped_rate": unmapped_rate,
        "intentionally_unmapped_prediction_events": intentionally_unmapped_events,
        "unknown_unmapped_prediction_events": unknown_unmapped_events,
        "rows_with_unmapped_predictions": rows_with_unmapped,
        "unmapped_labels": dict(unmapped_label_counts.most_common()),
        "intentionally_unmapped_labels": dict(intentional_unmapped_label_counts.most_common()),
        "unknown_unmapped_labels": dict(unknown_unmapped_label_counts.most_common()),
        "raw_label_counts": dict(raw_label_counts.most_common()),
        "mapped_v4_label_counts": dict(mapped_label_counts.most_common()),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("V1 OLD -> V4 MAPPING")
    print("=" * 72)
    print(f"Rows:                     {rows:,}")
    print(f"Raw prediction events:    {raw_events:,}")
    print(f"Mapped prediction events: {mapped_events:,} ({mapped_rate:.2%})")
    print(f"  crosswalked:            {crosswalked_events:,}")
    print(f"  already/native v4:      {native_events:,}")
    print(f"Unmapped events:          {unmapped_events:,} ({unmapped_rate:.2%})")
    print(f"  intentionally unmapped: {intentionally_unmapped_events:,}")
    print(f"  unknown unmapped:       {unknown_unmapped_events:,}")
    print(f"Rows with >=1 unmapped:   {rows_with_unmapped:,}")
    print()
    print("Unmapped labels:")
    if unmapped_label_counts:
        for label, count in unmapped_label_counts.most_common():
            print(f"  {label:<40} {count:>7,}")
    else:
        print("  none")
    print()
    print(f"Mapped JSONL: {out_path}")
    print(f"Audit report: {report_path}")


if __name__ == "__main__":
    main()
