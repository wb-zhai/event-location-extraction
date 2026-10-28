#!/usr/bin/env python3

import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SRC_ROOT = Path(
    "global_v4_audit/postprocessing_v1"
)

OUT_ROOT = Path(
    "/home/bonadossou/global_v4_work/"
    "compact_event_metadata_v1"
)

ARTICLE_DIR = OUT_ROOT / "articles"
EVENT_DIR = OUT_ROOT / "events"
AUDIT_DIR = OUT_ROOT / "audits"

ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
EVENT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

BATCH = 100_000

EXPECTED_CONSTANTS = {
    "schema_status": "structurally_valid",
    "grounding_status": "not_verified",
    "clustering_time_status": "source_review_required",
}

HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


article_schema = pa.schema([
    ("id", pa.string()),
    ("shard", pa.int16()),
    ("language", pa.string()),
    ("publish_date", pa.string()),
    ("source_gcs_path", pa.string()),
])


# event_key is stored reversibly as the raw 32 SHA-256 bytes,
# instead of 64 hexadecimal characters.
event_schema = pa.schema([
    ("event_key", pa.binary(32)),
    ("id", pa.string()),
    ("shard", pa.int16()),
    ("event_index", pa.int32()),

    ("raw_event_time", pa.string()),

    ("count", pa.string()),
    ("syntax_kind", pa.string()),
    ("parse_status", pa.string()),
    ("review_reason", pa.string()),

    ("start_token", pa.string()),
    ("end_token", pa.string()),
    ("start_precision", pa.string()),
    ("end_precision", pa.string()),

    ("lower_bound_date", pa.string()),
    ("upper_bound_date", pa.string()),

    ("source_verified", pa.string()),
])


ARTICLE_DICT_COLS = [
    "id",
    "language",
]

EVENT_DICT_COLS = [
    "id",
    "raw_event_time",
    "count",
    "syntax_kind",
    "parse_status",
    "review_reason",
    "start_token",
    "end_token",
    "start_precision",
    "end_precision",
    "lower_bound_date",
    "upper_bound_date",
    "source_verified",
]


def require_string_or_none(v, field, shard, row_num):
    if v is None:
        return None

    if not isinstance(v, str):
        raise RuntimeError(
            f"{shard} row {row_num}: "
            f"{field} expected str/None, got "
            f"{type(v).__name__}: {v!r}"
        )

    return v


def write_batch(writer, rows, schema):
    if not rows:
        return

    table = pa.Table.from_pylist(
        rows,
        schema=schema,
    )

    writer.write_table(
        table,
        row_group_size=len(rows),
    )

    rows.clear()


def process_shard(i):
    shard = f"shard-{i:03d}"

    src = (
        SRC_ROOT
        / shard
        / "event_metadata.jsonl"
    )

    if not src.is_file():
        raise RuntimeError(
            f"missing metadata file: {src}"
        )

    article_final = (
        ARTICLE_DIR / f"{shard}.parquet"
    )
    event_final = (
        EVENT_DIR / f"{shard}.parquet"
    )

    article_tmp = Path(
        str(article_final) + ".tmp"
    )
    event_tmp = Path(
        str(event_final) + ".tmp"
    )

    article_tmp.unlink(missing_ok=True)
    event_tmp.unlink(missing_ok=True)

    stats = Counter()

    temporal_count = Counter()
    review_reason = Counter()
    source_verified = Counter()

    article_rows = []
    event_rows = []

    # Verify article IDs are contiguous within a shard.
    closed_ids = set()

    current_id = None
    current_article = None

    article_writer = pq.ParquetWriter(
        article_tmp,
        article_schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=ARTICLE_DICT_COLS,
        write_statistics=True,
    )

    event_writer = pq.ParquetWriter(
        event_tmp,
        event_schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=EVENT_DICT_COLS,
        write_statistics=True,
    )

    try:
        with src.open(
            "r",
            encoding="utf-8",
        ) as f:

            for row_num, line in enumerate(f, 1):
                x = json.loads(line)

                stats["metadata_rows"] += 1

                # --------------------------------
                # Proven global constants
                # --------------------------------

                for field, expected in (
                    EXPECTED_CONSTANTS.items()
                ):
                    if x.get(field) != expected:
                        raise RuntimeError(
                            f"{shard} row {row_num}: "
                            f"{field}={x.get(field)!r}, "
                            f"expected {expected!r}"
                        )

                changes = x.get("changes")

                if changes != []:
                    raise RuntimeError(
                        f"{shard} row {row_num}: "
                        f"changes is not []: {changes!r}"
                    )

                # --------------------------------
                # Core identifiers
                # --------------------------------

                article_id = x.get("id")

                if not isinstance(article_id, str):
                    article_id = str(article_id)

                if int(x.get("shard", -1)) != i:
                    raise RuntimeError(
                        f"{shard} row {row_num}: "
                        f"metadata shard={x.get('shard')!r}"
                    )

                event_index = x.get("event_index")

                if (
                    not isinstance(event_index, int)
                    or event_index < 0
                ):
                    raise RuntimeError(
                        f"{shard} row {row_num}: "
                        f"bad event_index={event_index!r}"
                    )

                event_key = x.get("event_key")

                if (
                    not isinstance(event_key, str)
                    or not HEX64.fullmatch(event_key)
                ):
                    stats["missing_or_invalid_event_key"] += 1
                    raise RuntimeError(
                        f"{shard} row {row_num}: "
                        f"invalid event_key={event_key!r}"
                    )

                # --------------------------------
                # Article-level metadata
                # --------------------------------

                article_values = {
                    "id": article_id,
                    "shard": i,
                    "language":
                        require_string_or_none(
                            x.get("language"),
                            "language",
                            shard,
                            row_num,
                        ),
                    "publish_date":
                        require_string_or_none(
                            x.get("publish_date"),
                            "publish_date",
                            shard,
                            row_num,
                        ),
                    "source_gcs_path":
                        require_string_or_none(
                            x.get("source_gcs_path"),
                            "source_gcs_path",
                            shard,
                            row_num,
                        ),
                }

                if current_id != article_id:

                    if current_id is not None:
                        closed_ids.add(current_id)

                    if article_id in closed_ids:
                        raise RuntimeError(
                            f"{shard} row {row_num}: "
                            f"article {article_id!r} "
                            "reappeared non-contiguously"
                        )

                    current_id = article_id
                    current_article = article_values

                    article_rows.append(
                        article_values
                    )

                    stats["articles"] += 1

                    if len(article_rows) >= BATCH:
                        write_batch(
                            article_writer,
                            article_rows,
                            article_schema,
                        )

                else:
                    # All article-level fields must
                    # be identical across its events.
                    if article_values != current_article:
                        raise RuntimeError(
                            f"{shard} row {row_num}: "
                            f"article metadata conflict "
                            f"for id={article_id!r}\n"
                            f"first={current_article}\n"
                            f"now={article_values}"
                        )

                # --------------------------------
                # Temporal event metadata
                # --------------------------------

                raw_time = require_string_or_none(
                    x.get("raw_event_time"),
                    "raw_event_time",
                    shard,
                    row_num,
                )

                t = x.get("temporal")

                if not isinstance(t, dict):
                    raise RuntimeError(
                        f"{shard} row {row_num}: "
                        "temporal is not a dict"
                    )

                raw_value = require_string_or_none(
                    t.get("raw_value"),
                    "temporal.raw_value",
                    shard,
                    row_num,
                )

                if raw_value != raw_time:
                    stats["raw_value_mismatch"] += 1

                    raise RuntimeError(
                        f"{shard} row {row_num}: "
                        f"temporal.raw_value={raw_value!r} "
                        f"!= raw_event_time={raw_time!r}"
                    )

                def ts(name):
                    return require_string_or_none(
                        t.get(name),
                        f"temporal.{name}",
                        shard,
                        row_num,
                    )

                count = ts("count")
                syntax_kind = ts("syntax_kind")
                parse_status = ts("parse_status")
                rr = ts("review_reason")
                start_token = ts("start_token")
                end_token = ts("end_token")
                start_precision = ts("start_precision")
                end_precision = ts("end_precision")
                lower = ts("lower_bound_date")
                upper = ts("upper_bound_date")
                verified = ts("source_verified")

                temporal_count[count] += 1
                review_reason[rr] += 1
                source_verified[verified] += 1

                event_rows.append({
                    "event_key":
                        bytes.fromhex(event_key),
                    "id": article_id,
                    "shard": i,
                    "event_index": event_index,
                    "raw_event_time": raw_time,

                    "count": count,
                    "syntax_kind": syntax_kind,
                    "parse_status": parse_status,
                    "review_reason": rr,

                    "start_token": start_token,
                    "end_token": end_token,
                    "start_precision":
                        start_precision,
                    "end_precision":
                        end_precision,

                    "lower_bound_date": lower,
                    "upper_bound_date": upper,

                    "source_verified": verified,
                })

                stats["events"] += 1

                if len(event_rows) >= BATCH:
                    write_batch(
                        event_writer,
                        event_rows,
                        event_schema,
                    )

        write_batch(
            article_writer,
            article_rows,
            article_schema,
        )

        write_batch(
            event_writer,
            event_rows,
            event_schema,
        )

    except Exception:
        article_writer.close()
        event_writer.close()

        article_tmp.unlink(missing_ok=True)
        event_tmp.unlink(missing_ok=True)

        raise

    else:
        article_writer.close()
        event_writer.close()

    # --------------------------------
    # Verify Parquet footer row counts
    # --------------------------------

    a_rows = pq.ParquetFile(
        article_tmp
    ).metadata.num_rows

    e_rows = pq.ParquetFile(
        event_tmp
    ).metadata.num_rows

    if a_rows != stats["articles"]:
        raise RuntimeError(
            f"{shard}: article parquet rows "
            f"{a_rows:,} != {stats['articles']:,}"
        )

    if e_rows != stats["events"]:
        raise RuntimeError(
            f"{shard}: event parquet rows "
            f"{e_rows:,} != {stats['events']:,}"
        )

    os.replace(article_tmp, article_final)
    os.replace(event_tmp, event_final)

    audit = {
        "status": "PASS",
        "shard": i,
        "articles": stats["articles"],
        "events": stats["events"],
        "metadata_rows":
            stats["metadata_rows"],
        "missing_or_invalid_event_key":
            stats[
                "missing_or_invalid_event_key"
            ],
        "raw_value_mismatch":
            stats["raw_value_mismatch"],
        "temporal_count":
            dict(temporal_count),
        "review_reason":
            dict(review_reason),
        "source_verified":
            dict(source_verified),
        "article_parquet":
            str(article_final),
        "event_parquet":
            str(event_final),
        "article_bytes":
            article_final.stat().st_size,
        "event_bytes":
            event_final.stat().st_size,
    }

    audit_path = (
        AUDIT_DIR / f"{shard}.audit.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    return audit


def main():
    print("input :", SRC_ROOT)
    print("output:", OUT_ROOT)
    print("workers: 2")
    print()

    total = Counter()

    count_dist = Counter()
    review_dist = Counter()
    verified_dist = Counter()

    completed = 0

    with ProcessPoolExecutor(
        max_workers=2
    ) as ex:

        futures = {
            ex.submit(process_shard, i): i
            for i in range(200)
        }

        for fut in as_completed(futures):
            i = futures[fut]

            try:
                a = fut.result()
            except Exception as e:
                print(
                    f"FAILED shard-{i:03d}: {e}"
                )
                raise

            completed += 1

            total["articles"] += a["articles"]
            total["events"] += a["events"]
            total["metadata_rows"] += (
                a["metadata_rows"]
            )
            total["article_bytes"] += (
                a["article_bytes"]
            )
            total["event_bytes"] += (
                a["event_bytes"]
            )

            count_dist.update(
                a["temporal_count"]
            )
            review_dist.update(
                a["review_reason"]
            )
            verified_dist.update(
                a["source_verified"]
            )

            print(
                f"[{completed:03d}/200] "
                f"shard-{i:03d} "
                f"articles={a['articles']:,} "
                f"events={a['events']:,}"
            )

    if total["articles"] != 34_388_866:
        raise RuntimeError(
            f"expected 34,388,866 articles, "
            f"got {total['articles']:,}"
        )

    if total["events"] != 128_267_182:
        raise RuntimeError(
            f"expected 128,267,182 events, "
            f"got {total['events']:,}"
        )

    if total["metadata_rows"] != 128_267_182:
        raise RuntimeError(
            f"expected 128,267,182 metadata rows, "
            f"got {total['metadata_rows']:,}"
        )

    manifest = {
        "status": "PASS",
        "version":
            "compact_event_metadata_v1",
        "shards": 200,

        "articles":
            total["articles"],
        "events":
            total["events"],
        "metadata_rows":
            total["metadata_rows"],

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
                "omitted because validated "
                "equal to raw_event_time "
                "for every row",
            "event_key":
                "SHA-256 hex encoded as "
                "reversible 32-byte fixed binary; "
                "recover with event_key.hex()",
            "article_metadata":
                "language, publish_date and "
                "source_gcs_path stored once "
                "per article after validating "
                "within-article constancy",
            "event_index":
                "original pre-quarantine "
                "event index",
        },

        "temporal_count_distribution":
            dict(count_dist),

        "review_reason_distribution":
            dict(review_dist),

        "source_verified_distribution":
            dict(verified_dist),

        "article_bytes":
            total["article_bytes"],

        "event_bytes":
            total["event_bytes"],
    }

    manifest_path = (
        OUT_ROOT / "MANIFEST.json"
    )

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
    print("COMPACT EVENT METADATA: PASS")
    print("=" * 60)
    print(
        "articles      :",
        f"{total['articles']:,}",
    )
    print(
        "events        :",
        f"{total['events']:,}",
    )
    print(
        "metadata rows :",
        f"{total['metadata_rows']:,}",
    )
    print()
    print("temporal.count:")
    for k, n in count_dist.most_common():
        print(f"  {k!r:<30} {n:>15,}")

    print()
    print("review_reason:")
    for k, n in review_dist.most_common():
        print(f"  {k!r:<30} {n:>15,}")

    print()
    print("source_verified:")
    for k, n in verified_dist.most_common():
        print(f"  {k!r:<30} {n:>15,}")

    print()
    print("written:", OUT_ROOT)


if __name__ == "__main__":
    main()
