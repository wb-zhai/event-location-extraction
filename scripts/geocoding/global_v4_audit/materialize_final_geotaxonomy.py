#!/usr/bin/env python3

import argparse
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path


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
    """
    Preserve component order and raw admin-level alignment.

    A single admin level is broadcast across all semicolon-separated
    locations, matching the existing geocoder behaviour.
    """
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

    for component_index, raw_part in enumerate(raw_parts):
        query = raw_part.strip()

        if not query or query.lower() == "not_stated":
            continue

        raw_level = (
            raw_levels[component_index].strip()
            if component_index < len(raw_levels)
            else ""
        )

        layer = raw_level if raw_level in PHOTON_LAYERS else ""

        out.append({
            "component_index": component_index,
            "query": query,
            "raw_admin_level": raw_level,
            "layer": layer,
        })

    return out


def load_lookup_rows(con, keys):
    """
    Load only location/layer pairs required by this shard.
    """
    con.execute("DROP TABLE IF EXISTS temp.wanted")

    con.execute("""
        CREATE TEMP TABLE wanted (
            location TEXT NOT NULL,
            layer TEXT NOT NULL,
            PRIMARY KEY (location, layer)
        ) WITHOUT ROWID
    """)

    con.executemany(
        """
        INSERT OR IGNORE INTO wanted(location, layer)
        VALUES (?, ?)
        """,
        keys,
    )

    con.row_factory = sqlite3.Row

    result = {}

    for r in con.execute("""
        SELECT g.*
        FROM geotaxonomy_lookup AS g
        INNER JOIN wanted AS w
          ON g.location = w.location
         AND g.layer = w.layer
    """):
        row = dict(r)
        key = (row["location"], row["layer"])

        if key in result:
            raise RuntimeError(
                f"Duplicate geotaxonomy lookup key: {key!r}"
            )

        result[key] = row

    return result


def add_if_not_none(d, key, value):
    if value is not None:
        d[key] = value


def materialize_component(component, row):
    geo = {
        "component_index": component["component_index"],
        "query": component["query"],
        "raw_admin_level": component["raw_admin_level"],
    }

    # Keep the actual Photon layer visible. Empty string means that
    # no layer restriction was used.
    geo["photon_layer"] = component["layer"] or None

    geo["photon_resolution_status"] = row.get(
        "resolution_status"
    )

    selected_query = row.get("selected_query")
    selected_layer = row.get("selected_layer")

    if selected_query is not None:
        geo["photon_selected_query"] = selected_query

    if selected_layer is not None:
        geo["photon_selected_layer"] = selected_layer

    add_if_not_none(
        geo,
        "photon_fallback_step",
        row.get("fallback_step"),
    )

    add_if_not_none(
        geo,
        "photon_fallback_label",
        row.get("fallback_label"),
    )

    for key in (
        "resolved_name",
        "zhai",
        "photon_type",
        "lat",
        "lon",
        "osm_id",
        "osm_type",
        "country",
        "countrycode",
        "province",
        "district",
        "city",
    ):
        add_if_not_none(
            geo,
            key,
            row.get(key),
        )

    # Candidate exists.
    if row.get("lat") is not None and row.get("lon") is not None:

        add_if_not_none(
            geo,
            "wb_level",
            row.get("wb_level"),
        )

        wb_status = row.get("wb_match_status")

        if wb_status is None:
            raise RuntimeError(
                "Photon candidate unexpectedly has no WB status: "
                f"{component['query']!r}, "
                f"layer={component['layer']!r}"
            )

        geo["wb_match_status"] = wb_status

        add_if_not_none(
            geo,
            "wb_resolution_method",
            row.get("wb_resolution_method"),
        )

        add_if_not_none(
            geo,
            "adm_code",
            row.get("adm_code"),
        )

        add_if_not_none(
            geo,
            "adm_level",
            row.get("adm_level"),
        )

    else:
        # Photon itself produced no candidate.
        geo["wb_match_status"] = (
            "not_applicable_no_photon_candidate"
        )

    return geo


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--lookup-db", required=True)

    ap.add_argument(
        "--limit-articles",
        type=int,
        default=0,
        help="Smoke testing only. 0 means full input.",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out = Path(args.output).resolve()
    audit_path = Path(args.audit).resolve()
    lookup_db = Path(args.lookup_db).resolve()

    if not inp.is_file():
        raise SystemExit(f"Missing input: {inp}")

    if not lookup_db.is_file():
        raise SystemExit(
            f"Missing lookup DB: {lookup_db}"
        )

    if out.exists() and not args.overwrite:
        raise SystemExit(
            f"Output exists: {out}. "
            "Use --overwrite if intentional."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # PASS 1
    # Collect the exact lookup keys required by this shard.
    # ---------------------------------------------------------

    articles = 0
    events = 0
    events_with_stated_location = 0
    component_mentions = 0
    unique_keys = set()

    component_count_mismatches = 0

    with inp.open("r", encoding="utf-8") as f:
        for line in f:

            if (
                args.limit_articles
                and articles >= args.limit_articles
            ):
                break

            obj = json.loads(line)
            articles += 1

            predictions = obj.get("predictions") or []

            for event in predictions:
                if not isinstance(event, dict):
                    continue

                events += 1

                comps = event_components(event)

                if comps:
                    events_with_stated_location += 1

                loc = event.get("event_location")
                level = event.get(
                    "event_location_admin_level"
                )

                # Diagnostic only. Do not change raw data.
                if isinstance(loc, str) and ";" in loc:
                    n_loc = len(loc.split(";"))

                    if isinstance(level, str):
                        n_level = len(level.split(";"))
                    else:
                        n_level = 1

                    if n_level not in (1, n_loc):
                        component_count_mismatches += 1

                for comp in comps:
                    component_mentions += 1
                    unique_keys.add((
                        comp["query"],
                        comp["layer"],
                    ))

    print("=== PASS 1 ===")
    print(
        f"articles                    : "
        f"{articles:,}"
    )
    print(
        f"events                      : "
        f"{events:,}"
    )
    print(
        f"events with stated location : "
        f"{events_with_stated_location:,}"
    )
    print(
        f"raw component mentions      : "
        f"{component_mentions:,}"
    )
    print(
        f"unique location/layer keys  : "
        f"{len(unique_keys):,}"
    )
    print(
        f"admin count mismatches      : "
        f"{component_count_mismatches:,}"
    )

    # ---------------------------------------------------------
    # Read only required rows from the native-WSL lookup DB.
    # ---------------------------------------------------------

    con = sqlite3.connect(
        f"file:{lookup_db}?mode=ro",
        uri=True,
    )

    lookup = load_lookup_rows(
        con,
        unique_keys,
    )

    missing = unique_keys - set(lookup)

    print()
    print(
        "lookup rows found  :",
        f"{len(lookup):,}",
    )
    print(
        "missing unique keys:",
        f"{len(missing):,}",
    )

    if missing:
        print()
        print("First missing keys:")

        for key in sorted(missing)[:30]:
            print(repr(key))

        raise RuntimeError(
            "Some location/layer keys are absent "
            "from the final Photon lookup DB."
        )

    # ---------------------------------------------------------
    # PASS 2
    # Preserve each event exactly and ADD geotaxonomy[].
    # ---------------------------------------------------------

    tmp = Path(str(out) + ".tmp")

    if tmp.exists():
        tmp.unlink()

    out_articles = 0
    out_events = 0
    out_components = 0

    photon_status = Counter()
    wb_status = Counter()

    with (
        inp.open("r", encoding="utf-8") as fin,
        tmp.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:

            if (
                args.limit_articles
                and out_articles >= args.limit_articles
            ):
                break

            obj = json.loads(line)
            out_articles += 1

            predictions = obj.get("predictions") or []

            for event in predictions:
                if not isinstance(event, dict):
                    continue

                out_events += 1

                comps = event_components(event)

                if not comps:
                    continue

                geotaxonomy = []

                for comp in comps:
                    key = (
                        comp["query"],
                        comp["layer"],
                    )

                    row = lookup[key]

                    geo = materialize_component(
                        comp,
                        row,
                    )

                    geotaxonomy.append(geo)
                    out_components += 1

                    photon_status[
                        geo.get(
                            "photon_resolution_status"
                        )
                        or "NULL"
                    ] += 1

                    wb_status[
                        geo.get("wb_match_status")
                        or "NULL"
                    ] += 1

                # Additive only.
                # event_location and
                # event_location_admin_level are untouched.
                event["geotaxonomy"] = geotaxonomy

            fout.write(
                json.dumps(
                    obj,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # Accounting
    # ---------------------------------------------------------

    if out_articles != articles:
        raise RuntimeError(
            f"Article mismatch: "
            f"{out_articles:,} != {articles:,}"
        )

    if out_events != events:
        raise RuntimeError(
            f"Event mismatch: "
            f"{out_events:,} != {events:,}"
        )

    if out_components != component_mentions:
        raise RuntimeError(
            f"Component mismatch: "
            f"{out_components:,} != "
            f"{component_mentions:,}"
        )

    os.replace(tmp, out)

    audit = {
        "input": str(inp),
        "output": str(out),
        "lookup_db": str(lookup_db),

        "limited": bool(args.limit_articles),
        "limit_articles": (
            args.limit_articles or None
        ),

        "articles": articles,
        "events": events,

        "events_with_stated_location":
            events_with_stated_location,

        "raw_component_mentions":
            component_mentions,

        "unique_location_layer_keys":
            len(unique_keys),

        "missing_unique_lookup_keys": 0,

        "admin_component_count_mismatches":
            component_count_mismatches,

        "photon_resolution_status":
            dict(photon_status),

        "wb_match_status":
            dict(wb_status),
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=== MATERIALIZATION COMPLETE ===")
    print(f"articles   : {articles:,}")
    print(f"events     : {events:,}")
    print(
        f"components : "
        f"{component_mentions:,}"
    )

    print()
    print("Photon statuses:")

    for k, v in photon_status.most_common():
        print(
            f"  {k:<34} "
            f"{v:>12,}"
        )

    print()
    print("WB statuses:")

    for k, v in wb_status.most_common():
        print(
            f"  {k:<42} "
            f"{v:>12,}"
        )

    print()
    print("output:", out)
    print("audit :", audit_path)

    con.close()


if __name__ == "__main__":
    main()
