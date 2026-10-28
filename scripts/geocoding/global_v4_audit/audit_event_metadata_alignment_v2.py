#!/usr/bin/env python3

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


POST = Path("outputs/geography/global_v4_audit/postprocessing_v1")
FINAL = Path("outputs/geography/global_v4_audit/final_geocoded")

EXPECTED_EVENTS = 128_267_182
EXPECTED_ARTICLES = 34_388_866


def audit_shard(i):
    shard = f"shard-{i:03d}"

    meta_path = POST / shard / "event_metadata.jsonl"
    geo_path = FINAL / f"{shard}.geo.jsonl"

    stats = Counter()
    examples = []

    if not meta_path.is_file():
        return shard, "FAIL", {
            "error": f"missing metadata: {meta_path}"
        }

    if not geo_path.is_file():
        return shard, "FAIL", {
            "error": f"missing geo: {geo_path}"
        }

    with (
        meta_path.open(encoding="utf-8") as meta,
        geo_path.open(encoding="utf-8") as geo,
    ):
        for line_no, line in enumerate(geo, 1):
            obj = json.loads(line)

            article_id = str(obj["id"])
            preds = obj.get("predictions") or []

            stats["articles"] += 1

            previous_original_index = -1
            previous_shift = -1

            for compact_index, event in enumerate(preds):
                stats["events"] += 1

                mline = meta.readline()

                if not mline:
                    return shard, "FAIL", {
                        **stats,
                        "error": "metadata ended early",
                        "article_id": article_id,
                        "compact_index": compact_index,
                    }

                m = json.loads(mline)
                stats["metadata_rows"] += 1

                # 1. Same article.
                if str(m.get("id")) != article_id:
                    stats["id_mismatch"] += 1

                    if len(examples) < 20:
                        examples.append({
                            "kind": "id_mismatch",
                            "article_id": article_id,
                            "metadata_id": m.get("id"),
                            "compact_index": compact_index,
                            "metadata_event_index":
                                m.get("event_index"),
                        })

                # 2. Metadata must belong to this shard.
                if int(m.get("shard", -1)) != i:
                    stats["shard_mismatch"] += 1

                try:
                    original_index = int(m["event_index"])
                except Exception:
                    stats["invalid_original_index"] += 1
                    original_index = None

                if original_index is not None:

                    # Original indices must preserve order.
                    if original_index <= previous_original_index:
                        stats["nonincreasing_original_index"] += 1

                        if len(examples) < 20:
                            examples.append({
                                "kind":
                                    "nonincreasing_original_index",
                                "article_id": article_id,
                                "compact_index": compact_index,
                                "metadata_event_index":
                                    original_index,
                                "previous_original_index":
                                    previous_original_index,
                            })

                    # Removing events can only make compact index
                    # <= original index.
                    if original_index < compact_index:
                        stats["original_index_lt_compact"] += 1

                        if len(examples) < 20:
                            examples.append({
                                "kind":
                                    "original_index_lt_compact",
                                "article_id": article_id,
                                "compact_index": compact_index,
                                "metadata_event_index":
                                    original_index,
                            })

                    shift = original_index - compact_index

                    # As removed events accumulate, this shift can
                    # stay constant or increase, but cannot decrease.
                    if shift < previous_shift:
                        stats["decreasing_index_shift"] += 1

                        if len(examples) < 20:
                            examples.append({
                                "kind":
                                    "decreasing_index_shift",
                                "article_id": article_id,
                                "compact_index": compact_index,
                                "metadata_event_index":
                                    original_index,
                                "shift": shift,
                                "previous_shift":
                                    previous_shift,
                            })

                    if shift:
                        stats["shifted_events"] += 1
                        stats[f"shift_{shift}"] += 1

                    previous_original_index = original_index
                    previous_shift = shift

                # 3. Critical event-level identity check:
                # temporal raw value was copied from event_time.
                final_time = event.get("event_time")
                metadata_time = m.get("raw_event_time")

                if final_time != metadata_time:
                    stats["event_time_mismatch"] += 1

                    if len(examples) < 20:
                        examples.append({
                            "kind": "event_time_mismatch",
                            "article_id": article_id,
                            "compact_index": compact_index,
                            "metadata_event_index":
                                m.get("event_index"),
                            "final_event_time":
                                final_time,
                            "metadata_raw_event_time":
                                metadata_time,
                        })

            if previous_shift > 0:
                stats["articles_with_index_gaps"] += 1
                stats["inferred_missing_slots_before_last_valid"] += (
                    previous_shift
                )

        # Metadata must terminate exactly with final events.
        extra = meta.readline()

        if extra:
            x = json.loads(extra)

            return shard, "FAIL", {
                **stats,
                "error": "extra metadata rows remain",
                "first_extra_id": x.get("id"),
                "first_extra_event_index":
                    x.get("event_index"),
            }

    hard_errors = sum(
        stats[k]
        for k in (
            "id_mismatch",
            "shard_mismatch",
            "invalid_original_index",
            "nonincreasing_original_index",
            "original_index_lt_compact",
            "decreasing_index_shift",
            "event_time_mismatch",
        )
    )

    return shard, (
        "PASS" if hard_errors == 0 else "FAIL"
    ), {
        **stats,
        "hard_errors": hard_errors,
        "examples": examples,
    }


results = []

with ProcessPoolExecutor(max_workers=4) as ex:
    for result in ex.map(audit_shard, range(200)):
        results.append(result)


total = Counter()
failures = []

for shard, status, d in results:
    for k, v in d.items():
        if isinstance(v, int):
            total[k] += v

    if status != "PASS":
        failures.append((shard, status, d))


print("=== EVENT METADATA ALIGNMENT V2 ===")
print("shards                         :", len(results))
print("articles                       :", f"{total['articles']:,}")
print("final events                   :", f"{total['events']:,}")
print("metadata rows                  :", f"{total['metadata_rows']:,}")
print("articles with original gaps    :", f"{total['articles_with_index_gaps']:,}")
print("events with shifted index      :", f"{total['shifted_events']:,}")
print(
    "inferred missing slots*        :",
    f"{total['inferred_missing_slots_before_last_valid']:,}",
)
print("ID mismatches                  :", f"{total['id_mismatch']:,}")
print("event_time mismatches          :", f"{total['event_time_mismatch']:,}")
print("ordering errors                :", f"{total['nonincreasing_original_index']:,}")
print("decreasing shift errors        :", f"{total['decreasing_index_shift']:,}")
print("failed shards                  :", len(failures))

print()
print(
    "* only missing original indices occurring before the "
    "last surviving event of each article; not a global "
    "quarantine-event count"
)

if failures:
    print()
    for x in failures[:10]:
        print(json.dumps(x, ensure_ascii=False, indent=2))

    raise RuntimeError(
        f"{len(failures)} shards failed corrected alignment"
    )

if total["articles"] != EXPECTED_ARTICLES:
    raise RuntimeError(
        f"Expected {EXPECTED_ARTICLES:,} articles; "
        f"got {total['articles']:,}"
    )

if total["events"] != EXPECTED_EVENTS:
    raise RuntimeError(
        f"Expected {EXPECTED_EVENTS:,} events; "
        f"got {total['events']:,}"
    )

if total["metadata_rows"] != EXPECTED_EVENTS:
    raise RuntimeError(
        f"Expected {EXPECTED_EVENTS:,} metadata rows; "
        f"got {total['metadata_rows']:,}"
    )

print()
print("======================================")
print("EVENT METADATA ALIGNMENT V2: PASS")
print("======================================")
