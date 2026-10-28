#!/usr/bin/env python3

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


POST = Path("global_v4_audit/postprocessing_v1")
FINAL = Path("global_v4_audit/final_geocoded")


def audit_shard(i):
    shard = f"shard-{i:03d}"

    meta_path = POST / shard / "event_metadata.jsonl"
    geo_path = FINAL / f"{shard}.geo.jsonl"

    if not meta_path.is_file():
        return shard, "MISSING_METADATA", {}

    if not geo_path.is_file():
        return shard, "MISSING_GEO", {}

    meta = meta_path.open(encoding="utf-8")

    articles = 0
    events = 0
    metadata_rows = 0
    mismatches = []

    try:
        for line_no, line in enumerate(
            geo_path.open(encoding="utf-8"),
            1,
        ):
            obj = json.loads(line)
            article_id = str(obj["id"])

            articles += 1

            for event_index, event in enumerate(
                obj.get("predictions") or []
            ):
                events += 1

                mline = meta.readline()

                if not mline:
                    mismatches.append({
                        "kind": "metadata_ended_early",
                        "article_id": article_id,
                        "event_index": event_index,
                    })
                    return shard, "FAIL", {
                        "articles": articles,
                        "events": events,
                        "metadata_rows": metadata_rows,
                        "mismatches": mismatches[:20],
                    }

                m = json.loads(mline)
                metadata_rows += 1

                if str(m.get("id")) != article_id:
                    mismatches.append({
                        "kind": "id_mismatch",
                        "expected_id": article_id,
                        "metadata_id": m.get("id"),
                        "event_index": event_index,
                        "metadata_event_index":
                            m.get("event_index"),
                    })

                if int(m.get("event_index", -1)) != event_index:
                    mismatches.append({
                        "kind": "event_index_mismatch",
                        "id": article_id,
                        "expected_event_index": event_index,
                        "metadata_event_index":
                            m.get("event_index"),
                    })

                if len(mismatches) >= 20:
                    return shard, "FAIL", {
                        "articles": articles,
                        "events": events,
                        "metadata_rows": metadata_rows,
                        "mismatches": mismatches,
                    }

        # Metadata must also end here.
        extra = meta.readline()

        if extra:
            x = json.loads(extra)
            mismatches.append({
                "kind": "extra_metadata_rows",
                "first_extra_id": x.get("id"),
                "first_extra_event_index":
                    x.get("event_index"),
            })

    finally:
        meta.close()

    status = "PASS" if not mismatches else "FAIL"

    return shard, status, {
        "articles": articles,
        "events": events,
        "metadata_rows": metadata_rows,
        "mismatches": mismatches[:20],
    }


results = []

with ProcessPoolExecutor(max_workers=4) as ex:
    for r in ex.map(audit_shard, range(200)):
        results.append(r)

total_articles = 0
total_events = 0
total_metadata = 0
failures = []

for shard, status, d in results:
    total_articles += d.get("articles", 0)
    total_events += d.get("events", 0)
    total_metadata += d.get("metadata_rows", 0)

    if status != "PASS":
        failures.append((shard, status, d))

print("=== EVENT METADATA ALIGNMENT ===")
print("shards              :", len(results))
print("articles            :", f"{total_articles:,}")
print("final events        :", f"{total_events:,}")
print("metadata rows       :", f"{total_metadata:,}")
print("failed shards       :", len(failures))

for x in failures[:20]:
    print(json.dumps(x, ensure_ascii=False, indent=2))

if failures:
    raise RuntimeError(
        f"{len(failures)} shards failed metadata alignment"
    )

if total_events != 128_267_182:
    raise RuntimeError(
        f"Expected 128,267,182 events, got {total_events:,}"
    )

if total_metadata != total_events:
    raise RuntimeError(
        f"Metadata rows {total_metadata:,} != "
        f"events {total_events:,}"
    )

print()
print("EVENT METADATA ALIGNMENT: PASS")
