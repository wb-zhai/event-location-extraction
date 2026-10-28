#!/usr/bin/env python3
"""Rebuild location/layer counts from completed postprocessing shards.

Each run builds fresh temporary outputs, so reruns never add to prior counts.
Existing outputs remain intact if validation or processing fails.
"""
import argparse
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "outputs/geography/global_v4_audit/postprocessing_v1"
DB = REPO / "outputs/geography/global_v4_audit/location_layer_pairs.sqlite"
OUT = REPO / "outputs/geography/global_v4_audit/location_layer_pairs.jsonl"
VALID_LAYERS = {"country", "state", "county", "city", "district"}


def build_queue(root=ROOT, db=DB, out=OUT, expected_shards=200):
    root, db, out = Path(root), Path(db), Path(out)
    if db.resolve() == out.resolve():
        raise ValueError("Database and JSONL output paths must differ")
    # Validate all shard names before creating any output.
    expected = {f"shard-{i:03d}" for i in range(expected_shards)}
    actual = {p.parent.name for p in root.glob("shard-*/candidate_geocoding_input.jsonl")}
    if actual != expected:
        raise ValueError(f"Expected {expected_shards} candidate shards; missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    db.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Both temporary outputs live on the destination filesystem for atomic replace.
    with tempfile.TemporaryDirectory(prefix=".location-db-", dir=db.parent) as db_tmp,\
         tempfile.TemporaryDirectory(prefix=".location-jsonl-", dir=out.parent) as out_tmp:
        work_db = Path(db_tmp) / "pairs.sqlite"
        work_out = Path(out_tmp) / "pairs.jsonl"
        _populate(root, work_db, work_out, out, expected_shards)
        # Replacing an active SQLite file with sidecars could mix database versions.
        if any(Path(str(db) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
            raise RuntimeError("Destination SQLite has sidecar files; close its users before rebuilding")
        os.replace(work_db, db)
        os.replace(work_out, out)


def _populate(root, work_db, work_out, out, expected_shards):
    files = sorted(root.glob("shard-*/candidate_geocoding_input.jsonl"))

    if len(files) != expected_shards:
        raise SystemExit(f"Expected {expected_shards} candidate files, found {len(files)}")

    con = sqlite3.connect(work_db)
    try:

        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("""
        CREATE TABLE IF NOT EXISTS pairs (
            location TEXT NOT NULL,
            layer TEXT NOT NULL,
            mention_count INTEGER NOT NULL,
            PRIMARY KEY(location, layer)
        )
        """)
        con.commit()

        total_events = 0
        events_with_location = 0
        raw_component_mentions = 0
        clean_component_mentions = 0
        duplicates_removed = 0
        level_conflicts = 0
        conflict_examples = []

        for file_no, path in enumerate(files, 1):

            shard_counts = Counter()

            with path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):

                    obj = json.loads(line)

                    for event in obj.get("predictions", []):
                        total_events += 1

                        loc = event.get("event_location", "")
                        levels = event.get("event_location_admin_level", "")

                        if not isinstance(loc, str) or not loc.strip():
                            continue

                        if loc.strip().casefold() == "not_stated":
                            continue

                        parts = [x.strip() for x in loc.split(";")]
                        lvls = [x.strip() for x in levels.split(";")]

                        if len(parts) != len(lvls):
                            raise RuntimeError(
                                f"Unexpected location/admin mismatch in "
                                f"{path}:{line_no}: "
                                f"{len(parts)} locations vs {len(lvls)} levels"
                            )

                        events_with_location += 1

                        # Same cleanup policy used in the semicolon repair:
                        # remove not_stated and duplicate locations while
                        # preserving the first occurrence.
                        seen = {}

                        for part, lvl in zip(parts, lvls):

                            if not part or part.casefold() == "not_stated":
                                continue

                            raw_component_mentions += 1

                            layer = lvl if lvl in VALID_LAYERS else ""
                            dedupe_key = part.casefold()

                            if dedupe_key in seen:
                                duplicates_removed += 1

                                previous_layer = seen[dedupe_key]

                                if previous_layer != layer:
                                    level_conflicts += 1

                                    if len(conflict_examples) < 30:
                                        conflict_examples.append({
                                            "location": part,
                                            "first_layer": previous_layer or None,
                                            "duplicate_layer": layer or None,
                                            "raw_location": loc,
                                            "raw_admin_levels": levels,
                                        })

                                continue

                            seen[dedupe_key] = layer

                            shard_counts[(part, layer)] += 1
                            clean_component_mentions += 1

            con.executemany("""
                INSERT INTO pairs(location, layer, mention_count)
                VALUES (?, ?, ?)
                ON CONFLICT(location, layer)
                DO UPDATE SET mention_count =
                    mention_count + excluded.mention_count
            """, [
                (location, layer, count)
                for (location, layer), count in shard_counts.items()
            ])

            con.commit()

            print(
                f"[{file_no:03d}/{expected_shards:03d}] {path.parent.name} "
                f"pairs_in_shard={len(shard_counts):,}",
                flush=True,
            )


        unique_pairs = con.execute(
            "SELECT COUNT(*) FROM pairs"
        ).fetchone()[0]

        unique_locations = con.execute(
            "SELECT COUNT(DISTINCT location) FROM pairs"
        ).fetchone()[0]

        print()
        print("=== LOCATION + LAYER INVENTORY ===")
        print(f"events                     : {total_events:,}")
        print(f"events with stated location: {events_with_location:,}")
        print(f"raw component mentions     : {raw_component_mentions:,}")
        print(f"clean component mentions   : {clean_component_mentions:,}")
        print(f"duplicates removed         : {duplicates_removed:,}")
        print(f"level conflicts in duplicates: {level_conflicts:,}")
        print()
        print(f"unique locations           : {unique_locations:,}")
        print(f"unique location/layer pairs: {unique_pairs:,}")

        print("\nPairs by layer:")

        for layer, n, mentions in con.execute("""
            SELECT layer, COUNT(*), SUM(mention_count)
            FROM pairs
            GROUP BY layer
            ORDER BY SUM(mention_count) DESC
        """):
            label = layer if layer else "NO_LAYER"
            print(
                f"{label:12} unique={n:>10,} "
                f"mentions={mentions:>12,}"
            )

        print("\nDuplicate-level conflict examples:")

        for x in conflict_examples:
            print(json.dumps(x, ensure_ascii=False))


        with work_out.open("w", encoding="utf-8") as out:
            for location, layer, count in con.execute("""
                SELECT location, layer, mention_count
                FROM pairs
                ORDER BY mention_count DESC, location, layer
            """):
                out.write(json.dumps({
                    "location": location,
                    "layer": layer or None,
                    "count": count,
                }, ensure_ascii=False) + "\n")

        print()
        print(f"queue written: {out}")
        print(f"rows         : {unique_pairs:,}")


    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    build_queue(args.input_root, args.database, args.output)


if __name__ == "__main__":
    main()
