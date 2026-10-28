#!/usr/bin/env python3

import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path


BASE = Path(
    "/home/bonadossou/global_v4_work/"
    "global_v4_geotaxonomy_lookup.sqlite"
)

PATCHED = Path(
    "/home/bonadossou/global_v4_work/"
    "global_v4_geotaxonomy_lookup.PATCHED.sqlite"
)

EXCEPTIONS = Path(
    "global_v4_audit/"
    "missing_final_location_layer_enriched.json"
)

FAILED_SHARDS_FILE = Path(
    "global_v4_audit/"
    "failed_materialization_shards.txt"
)

POST_ROOT = Path(
    "global_v4_audit/postprocessing_v1"
)


PHOTON_LAYERS = {
    "country",
    "state",
    "county",
    "city",
    "district",
    "locality",
    "street",
    "house",
    "poi",
}


def event_components(event):
    loc = event.get("event_location")
    if not isinstance(loc, str):
        loc = "" if loc is None else str(loc)

    if not loc.strip() or loc.strip().lower() == "not_stated":
        return []

    admin = event.get("event_location_admin_level")
    if not isinstance(admin, str):
        admin = "" if admin is None else str(admin)

    raw_parts = loc.split(";")
    raw_levels = admin.split(";")

    if len(raw_levels) == 1:
        raw_levels = raw_levels * len(raw_parts)
    elif len(raw_levels) < len(raw_parts):
        raw_levels += [""] * (len(raw_parts) - len(raw_levels))

    out = []

    for i, raw_part in enumerate(raw_parts):
        query = raw_part.strip()

        if not query or query.lower() == "not_stated":
            continue

        raw_level = (
            raw_levels[i].strip()
            if i < len(raw_levels)
            else ""
        )

        layer = raw_level if raw_level in PHOTON_LAYERS else ""

        out.append((query, layer))

    return out


if not BASE.is_file():
    raise SystemExit(f"Missing base lookup: {BASE}")

if not EXCEPTIONS.is_file():
    raise SystemExit(f"Missing exception file: {EXCEPTIONS}")

exceptions = json.loads(
    EXCEPTIONS.read_text(encoding="utf-8")
)

if len(exceptions) != 13:
    raise RuntimeError(
        f"Expected 13 exceptions, found {len(exceptions)}"
    )

exception_keys = {
    (x["location"], x["layer"])
    for x in exceptions
}

if len(exception_keys) != 13:
    raise RuntimeError("Exception keys are not unique")


# ---------------------------------------------------------
# Count actual RAW materialization occurrences of each
# exceptional pair, using only failed shards.
#
# If a key occurred in a successful shard, that shard would
# also have failed during PASS 1.
# ---------------------------------------------------------

failed = [
    x.strip()
    for x in FAILED_SHARDS_FILE.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

counts = Counter()

for shard_name in failed:
    p = (
        POST_ROOT
        / shard_name
        / "candidate_geocoding_input.jsonl"
    )

    if not p.is_file():
        raise RuntimeError(f"Missing candidate file: {p}")

    with p.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            for event in obj.get("predictions") or []:
                if not isinstance(event, dict):
                    continue

                for key in event_components(event):
                    if key in exception_keys:
                        counts[key] += 1


print("=== EXCEPTION RAW OCCURRENCE COUNTS ===")

for key in sorted(exception_keys):
    print(f"{key!r:<55} {counts[key]:>8,}")

missing_counts = [
    key for key in exception_keys
    if counts[key] == 0
]

if missing_counts:
    raise RuntimeError(
        f"Exception keys not found in failed shards: "
        f"{missing_counts}"
    )


# ---------------------------------------------------------
# Create patched copy.
# ---------------------------------------------------------

if PATCHED.exists():
    PATCHED.unlink()

shutil.copy2(BASE, PATCHED)

con = sqlite3.connect(PATCHED)
con.row_factory = sqlite3.Row

before = con.execute(
    "SELECT COUNT(*) FROM geotaxonomy_lookup"
).fetchone()[0]

if before != 1_238_328:
    raise RuntimeError(
        f"Unexpected base row count: {before:,}"
    )


# Keep exact patch provenance separately rather than changing
# the production lookup table schema.
con.execute("""
CREATE TABLE IF NOT EXISTS materialization_exception_patch (
    location TEXT NOT NULL,
    layer TEXT NOT NULL,
    raw_occurrence_count INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (location, layer)
) WITHOUT ROWID
""")


insert_sql = """
INSERT INTO geotaxonomy_lookup (
    location,
    layer,
    mention_count,
    resolution_status,
    fallback_step,
    fallback_label,
    selected_query,
    selected_layer,
    resolved_name,
    zhai,
    photon_type,
    lat,
    lon,
    osm_id,
    osm_type,
    country,
    countrycode,
    province,
    district,
    city,
    wb_level,
    wb_matched,
    adm_code,
    adm_level,
    wb_match_status,
    wb_resolution_method
)
VALUES (
    ?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?
)
"""


for x in exceptions:
    key = (x["location"], x["layer"])

    exists = con.execute("""
        SELECT 1
        FROM geotaxonomy_lookup
        WHERE location=?
          AND layer=?
    """, key).fetchone()

    if exists:
        raise RuntimeError(
            f"Exception key already exists in base lookup: {key!r}"
        )

    con.execute(
        insert_sql,
        (
            x["location"],
            x["layer"],
            counts[key],

            x.get("resolution_status"),
            x.get("fallback_step"),
            x.get("fallback_label"),

            x.get("selected_query"),
            x.get("selected_layer"),

            x.get("resolved_name"),
            x.get("zhai"),
            x.get("photon_type"),

            x.get("lat"),
            x.get("lon"),

            x.get("osm_id"),
            x.get("osm_type"),

            x.get("country"),
            x.get("countrycode"),
            x.get("province"),
            x.get("district"),
            x.get("city"),

            x.get("wb_level"),
            x.get("wb_matched"),
            x.get("adm_code"),
            x.get("adm_level"),
            x.get("wb_match_status"),
            x.get("wb_resolution_method"),
        ),
    )

    con.execute("""
        INSERT INTO materialization_exception_patch (
            location,
            layer,
            raw_occurrence_count,
            payload_json
        )
        VALUES (?, ?, ?, ?)
    """, (
        x["location"],
        x["layer"],
        counts[key],
        json.dumps(
            x,
            ensure_ascii=False,
            sort_keys=True,
        ),
    ))


con.commit()

after = con.execute(
    "SELECT COUNT(*) FROM geotaxonomy_lookup"
).fetchone()[0]

patch_n = con.execute(
    "SELECT COUNT(*) FROM materialization_exception_patch"
).fetchone()[0]

quick = con.execute(
    "PRAGMA quick_check"
).fetchone()[0]

print()
print("=== PATCH AUDIT ===")
print(f"base rows       : {before:,}")
print(f"exception rows  : {patch_n:,}")
print(f"patched rows    : {after:,}")
print(f"quick_check     : {quick}")
print(f"output          : {PATCHED}")

if after != before + 13:
    raise RuntimeError(
        f"Expected {before + 13:,} patched rows, got {after:,}"
    )

if patch_n != 13:
    raise RuntimeError(
        f"Expected 13 provenance rows, got {patch_n:,}"
    )

if quick != "ok":
    raise RuntimeError(
        f"SQLite quick_check failed: {quick}"
    )


print()
print("=== VERIFY ALL 13 KEYS ===")

for key in sorted(exception_keys):
    row = con.execute("""
        SELECT
            resolution_status,
            wb_match_status,
            adm_code
        FROM geotaxonomy_lookup
        WHERE location=?
          AND layer=?
    """, key).fetchone()

    if row is None:
        raise RuntimeError(
            f"Patched key missing: {key!r}"
        )

    print(
        f"{key!r:<55} "
        f"photon={row['resolution_status']!s:<24} "
        f"wb={row['wb_match_status']!s:<40} "
        f"adm={row['adm_code']}"
    )

con.close()
