#!/usr/bin/env python3
"""
Map V1 legacy Teacher predictions to the V4 taxonomy.

Standalone: the historically established OLD_TO_V4 crosswalk is embedded here,
so this script does not depend on baseline_clustering_v4.py being present locally.

Unmapped legacy predictions are excluded from the common-V4 predictions list
and reported separately. Gold annotation and all other row fields are preserved.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

OLD_TO_V4 = {
    # conflict and security
    "conflict": "armed violence",
    "the offensive": "armed violence",
    "civil strife": "armed violence",
    "rebel insurgency": "militant activity",
    "terrorism": "militant activity",
    "terrorist": "militant activity",
    "air attack": "aerial bombardment",
    "bombing campaign": "aerial bombardment",
    "siege": "supply route blockade",
    "blockade": "supply route blockade",
    "looting": "looting",
    "human rights abuses": "human rights violation",
    "repression": "human rights violation",
    "violent suppression": "human rights violation",
    "infrastructure damage": "infrastructure destruction",

    # displacement and migration
    "displaced": "internal displacement",
    "flee": "internal displacement",
    "civilians uprooted": "internal displacement",
    "refugees": "refugee movement",
    "asylum seekers": "refugee movement",
    "migration": "migrant influx",
    "makeshift camps": "shelter loss",

    # food insecurity
    "food insecurity": "food scarcity",
    "food crisis": "food scarcity",
    "food shortage": "food scarcity",
    "acute hunger": "food scarcity",
    "massive starvation": "famine",
    "famine": "famine",
    "malnourished": "food scarcity",
    "harvest decline": "crop failure",
    "failed crops": "crop failure",

    # economic stress
    "economic crisis": "economic decline",
    "collapsing economy": "economic decline",
    "reduced national output": "economic decline",
    "economic impoverishment": "poverty",
    "rising inflation": "food price inflation",
    "price rise": "food price inflation",
    "rising food prices": "food price inflation",
    "international embargo": "trade disruption",

    # weather and natural hazards
    "floods": "flooding",
    "severe rains": "abnormal rainfall",
    "weather extremes": "abnormal rainfall",
    "drought": "drought",
    "water availability": "water shortage",
    "cyclone": "cyclone",
    "locusts": "pest infestation",

    # environment
    "environmental degradation": "environmental degradation",

    # public health
    "epidemics": "disease spread",
    "cholera outbreak": "disease spread",

    # political instability
    "coup": "governance breakdown",
    "overthrow": "governance breakdown",
    "power struggle": "governance breakdown",
    "authoritarian": "governance breakdown",
    "military dictatorship": "governance breakdown",
    "corruption": "governance breakdown",
    "secession": "geopolitical tension",

    # humanitarian disruption
    "restricted humanitarian access": "aid obstruction",
    "aid appeal": "funding shortfall",
    "food assistance": "aid reduction",
    "aid workers died": "civilian casualties",

    # deliberately unmapped in the established crosswalk
    "humanitarian situation": None,
    "humanitarian disaster": None,
    "natural disaster": None,
    "climate change": None,

    "burning houses": "fire",
}


def norm(x: Any) -> str:
    return str(x or "").strip().lower()


def load_active_labels(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    labels: set[str] = set()

    if isinstance(obj, dict):
        # Expected bona.v4.clusters.json form: label -> cluster.
        for k in obj:
            kk = norm(k)
            if kk:
                labels.add(kk)

        # Tolerate richer mappings too.
        for v in obj.values():
            if isinstance(v, dict) and v.get("name"):
                labels.add(norm(v["name"]))

    elif isinstance(obj, list):
        for row in obj:
            if isinstance(row, dict) and row.get("name"):
                labels.add(norm(row["name"]))
            elif isinstance(row, str):
                labels.add(norm(row))

    if not labels:
        raise RuntimeError(f"Could not read active V4 labels from {path}")
    return labels


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
    tax_path = Path(args.taxonomy)
    report_path = Path(args.report)

    for p in (in_path, tax_path):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    active = load_active_labels(tax_path)

    rows = 0
    raw_events = 0
    mapped_events = 0
    crosswalked_events = 0
    native_v4_events = 0
    unmapped_events = 0
    intentional_unmapped_events = 0
    unknown_unmapped_events = 0
    rows_with_unmapped = 0

    raw_counts = Counter()
    mapped_counts = Counter()
    unmapped_counts = Counter()
    intentional_counts = Counter()
    unknown_counts = Counter()

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
                raise RuntimeError(f"row {line_no}: predictions is not a list")

            mapped_preds = []
            had_unmapped = False

            for ev in preds:
                if not isinstance(ev, dict):
                    continue

                raw_events += 1
                label = norm(ev.get("event_type"))
                raw_counts[label or "<missing>"] += 1

                target = None
                status = None

                if label in OLD_TO_V4:
                    target = OLD_TO_V4[label]
                    status = "crosswalk" if target else "intentional_unmapped"
                elif label in active:
                    target = label
                    status = "native_v4"
                else:
                    status = "unknown_unmapped"

                if target:
                    if target not in active:
                        raise RuntimeError(
                            f"Crosswalk target {target!r} for legacy label {label!r} "
                            f"is not in {tax_path}"
                        )
                    new_ev = dict(ev)
                    new_ev["event_type"] = target
                    mapped_preds.append(new_ev)

                    mapped_events += 1
                    mapped_counts[target] += 1
                    if status == "crosswalk":
                        crosswalked_events += 1
                    else:
                        native_v4_events += 1
                else:
                    had_unmapped = True
                    unmapped_events += 1
                    unmapped_counts[label or "<missing>"] += 1
                    if status == "intentional_unmapped":
                        intentional_unmapped_events += 1
                        intentional_counts[label or "<missing>"] += 1
                    else:
                        unknown_unmapped_events += 1
                        unknown_counts[label or "<missing>"] += 1

            if had_unmapped:
                rows_with_unmapped += 1

            # Do not alter Teacher gold or invent V4-only V1 fields.
            row["predictions"] = mapped_preds
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows != args.expected_rows:
        raise RuntimeError(f"Expected {args.expected_rows} rows, got {rows}")

    report = {
        "rows": rows,
        "raw_prediction_events": raw_events,
        "mapped_prediction_events": mapped_events,
        "mapped_rate": mapped_events / raw_events if raw_events else 0.0,
        "crosswalked_prediction_events": crosswalked_events,
        "native_v4_prediction_events": native_v4_events,
        "unmapped_prediction_events": unmapped_events,
        "unmapped_rate": unmapped_events / raw_events if raw_events else 0.0,
        "intentionally_unmapped_prediction_events": intentional_unmapped_events,
        "unknown_unmapped_prediction_events": unknown_unmapped_events,
        "rows_with_unmapped_predictions": rows_with_unmapped,
        "unmapped_labels": dict(unmapped_counts.most_common()),
        "intentionally_unmapped_labels": dict(intentional_counts.most_common()),
        "unknown_unmapped_labels": dict(unknown_counts.most_common()),
        "raw_label_counts": dict(raw_counts.most_common()),
        "mapped_v4_label_counts": dict(mapped_counts.most_common()),
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("V1 OLD -> V4 MAPPING")
    print("=" * 72)
    print(f"Rows:                     {rows:,}")
    print(f"Raw prediction events:    {raw_events:,}")
    print(f"Mapped prediction events: {mapped_events:,} "
          f"({mapped_events/raw_events:.2%})" if raw_events else "Mapped prediction events: 0")
    print(f"  crosswalked:            {crosswalked_events:,}")
    print(f"  already/native v4:      {native_v4_events:,}")
    print(f"Unmapped events:          {unmapped_events:,} "
          f"({unmapped_events/raw_events:.2%})" if raw_events else "Unmapped events: 0")
    print(f"  intentionally unmapped: {intentional_unmapped_events:,}")
    print(f"  unknown unmapped:       {unknown_unmapped_events:,}")
    print(f"Rows with >=1 unmapped:   {rows_with_unmapped:,}")
    print()
    print("Unmapped labels:")
    if unmapped_counts:
        for label, count in unmapped_counts.most_common():
            print(f"  {label:<40} {count:>7,}")
    else:
        print("  none")
    print()
    print(f"Mapped JSONL: {out_path}")
    print(f"Audit report: {report_path}")


if __name__ == "__main__":
    main()
