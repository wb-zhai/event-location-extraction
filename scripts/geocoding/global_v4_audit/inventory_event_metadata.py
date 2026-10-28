#!/usr/bin/env python3

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("outputs/geography/global_v4_audit/postprocessing_v1")


def scan(i):
    p = (
        ROOT
        / f"shard-{i:03d}"
        / "event_metadata.jsonl"
    )

    c = {
        "rows": 0,
        "language": Counter(),
        "schema_status": Counter(),
        "grounding_status": Counter(),
        "clustering_time_status": Counter(),
        "temporal_syntax_kind": Counter(),
        "temporal_parse_status": Counter(),
        "start_precision": Counter(),
        "end_precision": Counter(),
        "changes_len": Counter(),
        "missing_publish_date": 0,
        "missing_source_gcs_path": 0,
        "missing_event_key": 0,
    }

    with p.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            c["rows"] += 1

            c["language"][str(x.get("language", ""))] += 1
            c["schema_status"][
                str(x.get("schema_status", ""))
            ] += 1
            c["grounding_status"][
                str(x.get("grounding_status", ""))
            ] += 1
            c["clustering_time_status"][
                str(x.get("clustering_time_status", ""))
            ] += 1

            t = x.get("temporal") or {}

            c["temporal_syntax_kind"][
                str(t.get("syntax_kind", ""))
            ] += 1
            c["temporal_parse_status"][
                str(t.get("parse_status", ""))
            ] += 1
            c["start_precision"][
                str(t.get("start_precision", ""))
            ] += 1
            c["end_precision"][
                str(t.get("end_precision", ""))
            ] += 1

            changes = x.get("changes")
            c["changes_len"][
                len(changes)
                if isinstance(changes, list)
                else -1
            ] += 1

            c["missing_publish_date"] += int(
                not x.get("publish_date")
            )
            c["missing_source_gcs_path"] += int(
                not x.get("source_gcs_path")
            )
            c["missing_event_key"] += int(
                not x.get("event_key")
            )

    return c


total = {
    "rows": 0,
    "language": Counter(),
    "schema_status": Counter(),
    "grounding_status": Counter(),
    "clustering_time_status": Counter(),
    "temporal_syntax_kind": Counter(),
    "temporal_parse_status": Counter(),
    "start_precision": Counter(),
    "end_precision": Counter(),
    "changes_len": Counter(),
    "missing_publish_date": 0,
    "missing_source_gcs_path": 0,
    "missing_event_key": 0,
}


with ProcessPoolExecutor(max_workers=4) as ex:
    for c in ex.map(scan, range(200)):
        total["rows"] += c["rows"]

        for k in (
            "language",
            "schema_status",
            "grounding_status",
            "clustering_time_status",
            "temporal_syntax_kind",
            "temporal_parse_status",
            "start_precision",
            "end_precision",
            "changes_len",
        ):
            total[k].update(c[k])

        for k in (
            "missing_publish_date",
            "missing_source_gcs_path",
            "missing_event_key",
        ):
            total[k] += c[k]


print("rows:", f"{total['rows']:,}")

for k in (
    "language",
    "schema_status",
    "grounding_status",
    "clustering_time_status",
    "temporal_syntax_kind",
    "temporal_parse_status",
    "start_precision",
    "end_precision",
    "changes_len",
):
    print("\n" + k)
    for value, n in total[k].most_common():
        print(f"  {value!r:<35} {n:>15,}")

print("\nmissing fields")
for k in (
    "missing_publish_date",
    "missing_source_gcs_path",
    "missing_event_key",
):
    print(f"  {k:<35} {total[k]:>15,}")
