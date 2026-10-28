#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(
    "/home/bonadossou/global_v4_work/"
    "compact_event_metadata_v1"
)

ARTICLE_DIR = ROOT / "articles"
EVENT_DIR = ROOT / "events"
AUDIT_DIR = ROOT / "audits"

EXPECTED_ARTICLES = 29_236_324
EXPECTED_EVENTS = 128_267_182
EXPECTED_SHARDS = 200


articles = sorted(ARTICLE_DIR.glob("shard-*.parquet"))
events = sorted(EVENT_DIR.glob("shard-*.parquet"))
audits = sorted(AUDIT_DIR.glob("shard-*.audit.json"))


print("=== FILE COMPLETENESS ===")
print("article parquet files :", len(articles))
print("event parquet files   :", len(events))
print("audit files           :", len(audits))

if len(articles) != EXPECTED_SHARDS:
    raise RuntimeError(
        f"expected 200 article parquet files, got {len(articles)}"
    )

if len(events) != EXPECTED_SHARDS:
    raise RuntimeError(
        f"expected 200 event parquet files, got {len(events)}"
    )

if len(audits) != EXPECTED_SHARDS:
    raise RuntimeError(
        f"expected 200 audit files, got {len(audits)}"
    )


# ---------------------------------------------------------
# Validate Parquet footer row counts and schemas
# ---------------------------------------------------------

article_rows = 0
event_rows = 0

article_schema = None
event_schema = None

for p in articles:
    pf = pq.ParquetFile(p)

    article_rows += pf.metadata.num_rows

    s = pf.schema_arrow

    if article_schema is None:
        article_schema = s
    elif s != article_schema:
        raise RuntimeError(
            f"article schema mismatch: {p}"
        )


for p in events:
    pf = pq.ParquetFile(p)

    event_rows += pf.metadata.num_rows

    s = pf.schema_arrow

    if event_schema is None:
        event_schema = s
    elif s != event_schema:
        raise RuntimeError(
            f"event schema mismatch: {p}"
        )


# ---------------------------------------------------------
# Aggregate shard audits
# ---------------------------------------------------------

audit_articles = 0
audit_events = 0
audit_metadata = 0

invalid_event_keys = 0
raw_value_mismatches = 0

count_dist = Counter()
review_dist = Counter()
verified_dist = Counter()

bad_audits = []

article_bytes = 0
event_bytes = 0


for p in audits:
    x = json.loads(
        p.read_text(encoding="utf-8")
    )

    if x.get("status") != "PASS":
        bad_audits.append(str(p))

    audit_articles += int(
        x.get("articles", 0)
    )

    audit_events += int(
        x.get("events", 0)
    )

    audit_metadata += int(
        x.get("metadata_rows", 0)
    )

    invalid_event_keys += int(
        x.get("missing_or_invalid_event_key", 0)
    )

    raw_value_mismatches += int(
        x.get("raw_value_mismatch", 0)
    )

    count_dist.update(
        {
            str(k): int(v)
            for k, v in
            x.get("temporal_count", {}).items()
        }
    )

    review_dist.update(
        {
            str(k): int(v)
            for k, v in
            x.get("review_reason", {}).items()
        }
    )

    verified_dist.update(
        {
            str(k): int(v)
            for k, v in
            x.get("source_verified", {}).items()
        }
    )

    article_bytes += int(
        x.get("article_bytes", 0)
    )

    event_bytes += int(
        x.get("event_bytes", 0)
    )


print()
print("=== ROW ACCOUNTING ===")
print("article parquet rows :", f"{article_rows:,}")
print("event parquet rows   :", f"{event_rows:,}")
print("audit articles       :", f"{audit_articles:,}")
print("audit events         :", f"{audit_events:,}")
print("audit metadata rows  :", f"{audit_metadata:,}")

print()
print("=== DATA QUALITY ===")
print("bad audits            :", len(bad_audits))
print("invalid event keys    :", f"{invalid_event_keys:,}")
print("raw_value mismatches  :", f"{raw_value_mismatches:,}")


if article_rows != EXPECTED_ARTICLES:
    raise RuntimeError(
        f"article parquet rows: expected "
        f"{EXPECTED_ARTICLES:,}, got {article_rows:,}"
    )

if event_rows != EXPECTED_EVENTS:
    raise RuntimeError(
        f"event parquet rows: expected "
        f"{EXPECTED_EVENTS:,}, got {event_rows:,}"
    )

if audit_articles != EXPECTED_ARTICLES:
    raise RuntimeError(
        f"audit article count mismatch: {audit_articles:,}"
    )

if audit_events != EXPECTED_EVENTS:
    raise RuntimeError(
        f"audit event count mismatch: {audit_events:,}"
    )

if audit_metadata != EXPECTED_EVENTS:
    raise RuntimeError(
        f"audit metadata count mismatch: {audit_metadata:,}"
    )

if bad_audits:
    raise RuntimeError(
        f"{len(bad_audits)} shard audits are not PASS"
    )

if invalid_event_keys != 0:
    raise RuntimeError(
        f"{invalid_event_keys:,} invalid event keys"
    )

if raw_value_mismatches != 0:
    raise RuntimeError(
        f"{raw_value_mismatches:,} temporal.raw_value mismatches"
    )


print()
print("=== TEMPORAL.COUNT ===")
for k, n in count_dist.most_common():
    print(f"{k!r:<35} {n:>15,}")

print()
print("=== REVIEW_REASON ===")
for k, n in review_dist.most_common():
    print(f"{k!r:<35} {n:>15,}")

print()
print("=== SOURCE_VERIFIED ===")
for k, n in verified_dist.most_common():
    print(f"{k!r:<35} {n:>15,}")


manifest = {
    "status": "PASS",
    "version": "compact_event_metadata_v1",

    "shards": EXPECTED_SHARDS,
    "articles": article_rows,
    "events": event_rows,

    "source_alignment": {
        "event_bearing_articles": 29_236_324,
        "candidate_articles": 34_388_866,
        "genuine_no_event_articles": 5_152_542,
        "events": 128_267_182,
        "metadata_rows": 128_267_182,
        "id_mismatches": 0,
        "event_time_mismatches": 0,
        "ordering_errors": 0,
        "decreasing_shift_errors": 0,
        "event_index_semantics":
            "original pre-quarantine event index",
    },

    "removed_global_constants": {
        "schema_status":
            "structurally_valid",
        "grounding_status":
            "not_verified",
        "clustering_time_status":
            "source_review_required",
        "changes": [],
    },

    "normalizations": {
        "temporal.raw_value":
            "omitted after validation that it equals "
            "raw_event_time for every surviving event",
        "event_key":
            "SHA-256 hexadecimal string stored reversibly "
            "as 32-byte binary",
        "article_metadata":
            "language, publish_date and source_gcs_path "
            "stored once per article after validating "
            "within-article constancy",
    },

    "quality": {
        "invalid_event_keys": invalid_event_keys,
        "raw_value_mismatches": raw_value_mismatches,
    },

    "temporal_count_distribution":
        dict(count_dist),

    "review_reason_distribution":
        dict(review_dist),

    "source_verified_distribution":
        dict(verified_dist),

    "article_bytes": article_bytes,
    "event_bytes": event_bytes,
}


manifest_path = ROOT / "MANIFEST.json"

manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)


print()
print("=" * 60)
print("COMPACT EVENT METADATA VALIDATION: PASS")
print("=" * 60)
print("manifest:", manifest_path)
