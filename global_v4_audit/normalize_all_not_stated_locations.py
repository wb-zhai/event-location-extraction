#!/usr/bin/env python3

import json
import os
from collections import defaultdict
from pathlib import Path

GAP = Path(
    "global_v4_audit/stated_location_definition_gap.json"
)

FINAL_ROOT = Path(
    "global_v4_audit/final_geocoded"
)

AUDIT_OUT = Path(
    "global_v4_audit/final_geocoded/"
    "not_stated_normalization_audit.json"
)


def all_not_stated(value):
    if not isinstance(value, str):
        return False

    parts = [x.strip() for x in value.split(";")]

    if not parts:
        return False

    return all(
        (not x) or x.lower() == "not_stated"
        for x in parts
    )


gap = json.loads(GAP.read_text(encoding="utf-8"))
anomalies = gap["anomalies"]

if len(anomalies) != 13:
    raise RuntimeError(
        f"Expected 13 anomalies, found {len(anomalies)}"
    )

# Group expected corrections by shard.
targets = defaultdict(dict)

for a in anomalies:
    shard = Path(a["file"]).parent.name

    key = (
        str(a["id"]),
        int(a["prediction_index"]),
    )

    if key in targets[shard]:
        raise RuntimeError(
            f"Duplicate target: {shard} {key}"
        )

    targets[shard][key] = a


print("affected shards:", len(targets))
print("target events  :", len(anomalies))
print()

total_patched = 0
patched_records = []


for shard in sorted(targets):
    src = FINAL_ROOT / f"{shard}.geo.jsonl"
    tmp = Path(str(src) + ".normalize.tmp")

    if not src.is_file():
        raise RuntimeError(f"Missing final shard: {src}")

    if tmp.exists():
        tmp.unlink()

    expected = targets[shard]
    found = set()

    print(
        f"{shard}: targets={len(expected)}"
    )

    with (
        src.open("r", encoding="utf-8") as fin,
        tmp.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            obj = json.loads(line)

            article_id = str(obj.get("id"))
            preds = obj.get("predictions") or []

            for idx, event in enumerate(preds):
                key = (article_id, idx)

                if key not in expected:
                    continue

                a = expected[key]

                old_loc = event.get("event_location")
                old_level = event.get(
                    "event_location_admin_level"
                )

                # Strong safety checks before changing anything.
                if old_loc != a["event_location"]:
                    raise RuntimeError(
                        f"{shard} {key}: event_location mismatch\n"
                        f"expected={a['event_location']!r}\n"
                        f"actual={old_loc!r}"
                    )

                if old_level != a[
                    "event_location_admin_level"
                ]:
                    raise RuntimeError(
                        f"{shard} {key}: "
                        "admin-level mismatch"
                    )

                if not all_not_stated(old_loc):
                    raise RuntimeError(
                        f"{shard} {key}: location is not "
                        f"all-not_stated: {old_loc!r}"
                    )

                # These should have no geocodable components.
                geo = event.get("geotaxonomy")
                if geo not in (None, []):
                    raise RuntimeError(
                        f"{shard} {key}: unexpected "
                        f"geotaxonomy={geo!r}"
                    )

                # Preserve original extraction representation.
                event["event_location_raw"] = old_loc
                event[
                    "event_location_admin_level_raw"
                ] = old_level

                # Canonical downstream representation.
                event["event_location"] = "not_stated"
                event[
                    "event_location_admin_level"
                ] = "not_stated"

                found.add(key)
                total_patched += 1

                patched_records.append({
                    "shard": shard,
                    "id": article_id,
                    "prediction_index": idx,
                    "event_type": event.get("event_type"),
                    "event_location_raw": old_loc,
                    "event_location_admin_level_raw":
                        old_level,
                    "event_location": "not_stated",
                    "event_location_admin_level":
                        "not_stated",
                })

            fout.write(
                json.dumps(
                    obj,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    missing = set(expected) - found

    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{shard}: target events not found: "
            f"{sorted(missing)}"
        )

    os.replace(tmp, src)


if total_patched != 13:
    raise RuntimeError(
        f"Expected 13 patches, got {total_patched}"
    )


audit = {
    "status": "PASS",
    "rule":
        "all semicolon-separated location components "
        "are empty or not_stated -> canonical not_stated",
    "patched_events": total_patched,
    "affected_shards": len(targets),
    "provenance_fields": [
        "event_location_raw",
        "event_location_admin_level_raw",
    ],
    "patches": patched_records,
}

AUDIT_OUT.write_text(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print()
print("====================================")
print("NORMALIZATION PATCH: PASS")
print("====================================")
print("patched events :", total_patched)
print("affected shards:", len(targets))
print("audit          :", AUDIT_OUT)
