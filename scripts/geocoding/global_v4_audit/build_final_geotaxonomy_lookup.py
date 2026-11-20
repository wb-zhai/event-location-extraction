#!/usr/bin/env python3

import argparse
import sqlite3
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--photon-db", required=True)
ap.add_argument("--wb-db", required=True)
ap.add_argument("--out-db", required=True)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

photon = Path(args.photon_db).resolve()
wb = Path(args.wb_db).resolve()
out = Path(args.out_db).resolve()

for p in (photon, wb):
    if not p.is_file():
        raise SystemExit(f"Missing: {p}")

if out.exists():
    if not args.overwrite:
        raise SystemExit(f"{out} exists; use --overwrite if intentional")
    out.unlink()

con = sqlite3.connect(out)
con.execute("PRAGMA synchronous=NORMAL")

con.execute("ATTACH DATABASE ? AS photon", (str(photon),))
con.execute("ATTACH DATABASE ? AS wb", (str(wb),))

photon_n = con.execute(
    "SELECT COUNT(*) FROM photon.final_resolution"
).fetchone()[0]

wb_n = con.execute(
    "SELECT COUNT(*) FROM wb.wb_resolution"
).fetchone()[0]

print("Photon rows:", f"{photon_n:,}")
print("WB rows    :", f"{wb_n:,}")

if photon_n != 1_238_328:
    raise RuntimeError(f"Unexpected Photon count: {photon_n:,}")

if wb_n != 512_159:
    raise RuntimeError(f"Unexpected WB count: {wb_n:,}")

con.execute("""
CREATE TABLE geotaxonomy_lookup (
    location TEXT NOT NULL,
    layer TEXT NOT NULL,

    mention_count INTEGER,

    resolution_status TEXT,
    fallback_step INTEGER,
    fallback_label TEXT,

    selected_query TEXT,
    selected_layer TEXT,

    resolved_name TEXT,
    zhai TEXT,
    photon_type TEXT,

    lat REAL,
    lon REAL,

    osm_id,
    osm_type TEXT,

    country TEXT,
    countrycode TEXT,
    province TEXT,
    district TEXT,
    city TEXT,

    wb_level INTEGER,
    wb_matched INTEGER,
    adm_code TEXT,
    adm_level INTEGER,
    wb_match_status TEXT,
    wb_resolution_method TEXT,

    PRIMARY KEY (location, layer)
) WITHOUT ROWID
""")

con.execute("""
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
SELECT
    p.location,
    COALESCE(p.layer, ''),
    p.mention_count,
    p.resolution_status,
    p.fallback_step,
    p.fallback_label,
    p.selected_query,
    p.selected_layer,
    p.resolved_name,
    p.zhai,
    p.photon_type,
    p.lat,
    p.lon,
    p.osm_id,
    p.osm_type,
    p.country,
    p.countrycode,
    p.province,
    p.district,
    p.city,

    CASE
        WHEN p.lat IS NULL OR p.lon IS NULL THEN NULL
        WHEN p.zhai = 'country' THEN 0
        WHEN p.zhai = 'province' THEN 1
        ELSE 2
    END,

    w.wb_matched,
    w.adm_code,
    w.adm_level,
    w.wb_match_status,
    w.wb_resolution_method

FROM photon.final_resolution AS p

LEFT JOIN wb.wb_resolution AS w
    ON p.lat = w.lat
   AND p.lon = w.lon
   AND w.wb_level = CASE
        WHEN p.zhai = 'country' THEN 0
        WHEN p.zhai = 'province' THEN 1
        ELSE 2
   END
""")

con.commit()

total = con.execute("""
SELECT COUNT(*), SUM(mention_count)
FROM geotaxonomy_lookup
""").fetchone()

candidate = con.execute("""
SELECT COUNT(*), SUM(mention_count)
FROM geotaxonomy_lookup
WHERE lat IS NOT NULL
  AND lon IS NOT NULL
""").fetchone()

missing_wb = con.execute("""
SELECT COUNT(*), COALESCE(SUM(mention_count), 0)
FROM geotaxonomy_lookup
WHERE lat IS NOT NULL
  AND lon IS NOT NULL
  AND wb_match_status IS NULL
""").fetchone()

print()
print("=== FINAL LOOKUP AUDIT ===")
print(f"rows                         : {total[0]:,}")
print(f"weighted component mentions  : {total[1]:,}")
print(f"Photon candidate rows        : {candidate[0]:,}")
print(f"Photon candidate mentions    : {candidate[1]:,}")
print(f"candidate rows missing WB row: {missing_wb[0]:,}")
print(f"candidate mentions missing WB: {missing_wb[1]:,}")

print()
print("WB weighted status breakdown:")

for status, rows, mentions in con.execute("""
    SELECT
        COALESCE(wb_match_status, 'NO_PHOTON_CANDIDATE'),
        COUNT(*),
        SUM(mention_count)
    FROM geotaxonomy_lookup
    GROUP BY COALESCE(wb_match_status, 'NO_PHOTON_CANDIDATE')
    ORDER BY SUM(mention_count) DESC
"""):
    print(
        f"  {status:<28} "
        f"pairs={rows:>10,} "
        f"mentions={mentions:>12,}"
    )

if total[0] != 1_238_328:
    raise RuntimeError("lookup row accounting failed")

if total[1] != 137_210_352:
    raise RuntimeError("lookup mention accounting failed")

if candidate[1] != 135_568_783:
    raise RuntimeError("Photon candidate mention accounting failed")

if missing_wb[0] != 0:
    raise RuntimeError("Some Photon candidates have no WB-resolution row")

print()
print("quick_check:", con.execute("PRAGMA quick_check").fetchone()[0])
print("output:", out)

con.close()
