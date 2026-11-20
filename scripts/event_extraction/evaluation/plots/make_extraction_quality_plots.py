#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MISSING = {"", "not_stated"}
YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def clean_locations(raw: object) -> list[str]:
    """
    Same conservative location-component cleanup:
      - split on ;
      - trim
      - drop not_stated
      - case-insensitive dedupe within one event
      - preserve first spelling
    """
    if not isinstance(raw, str):
        return []

    seen = set()
    out = []

    for value in raw.split(";"):
        value = value.strip()

        if not value or value.casefold() == "not_stated":
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)
        out.append(value)

    return out


def first_stated_year(raw: object) -> int | None:
    """
    Extract the first explicit 4-digit year from the raw event_time.

    We deliberately do NOT turn ranges/open intervals into artificial exact dates.
    """
    if not isinstance(raw, str):
        return None

    value = raw.strip()

    if not value or value.casefold() == "not_stated":
        return None

    m = YEAR_RE.search(value)

    if not m:
        return None

    year = int(m.group(1))

    # Guard against obviously malformed years without pretending we know
    # the event's exact date.
    if 1000 <= year <= 2199:
        return year

    return None


def compact(n: int | float) -> str:
    n = float(n)

    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"

    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"

    return f"{n:.0f}"


def load_location_counts(path: Path) -> Counter:
    """
    Reuse the already-built location/layer inventory.

    The queue contains one row per (cleaned location, layer) pair.
    Summing across layers gives total cleaned mentions for each location.
    """
    counts = Counter()
    rows = 0
    mentions = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            location = obj["location"]
            count = int(obj["count"])

            counts[location] += count
            rows += 1
            mentions += count

    print("=== LOCATION INVENTORY ===")
    print(f"location/layer rows   : {rows:,}")
    print(f"unique locations      : {len(counts):,}")
    print(f"component mentions    : {mentions:,}")
    print()

    return counts


def iter_metadata(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(
                    f"Invalid metadata JSON: {path}:{line_no}: {e}"
                )


def first_pass(root: Path):
    """
    One full pass over structurally-valid candidate events.

    Counts:
      - global event types
      - event types by language
      - raw event-time years

    candidate_geocoding_input.jsonl and event_metadata.jsonl are consumed
    in lockstep. event_metadata has one row for each structurally-valid event.
    """
    shard_dirs = sorted(root.glob("shard-*"))

    if len(shard_dirs) != 200:
        raise RuntimeError(
            f"Expected 200 shard directories under {root}; found {len(shard_dirs)}"
        )

    event_counts = Counter()

    lang_event_counts = {
        "eng": Counter(),
        "fra": Counter(),
    }

    lang_totals = Counter()

    year_counts = Counter()
    time_not_stated = 0
    time_no_year = 0

    total_events = 0

    for shard_no, shard_dir in enumerate(shard_dirs, 1):

        candidate_path = shard_dir / "candidate_geocoding_input.jsonl"
        metadata_path = shard_dir / "event_metadata.jsonl"

        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)

        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        meta_it = iter_metadata(metadata_path)

        shard_events = 0

        with candidate_path.open(encoding="utf-8") as f:

            for line_no, line in enumerate(f, 1):

                if not line.strip():
                    continue

                obj = json.loads(line)

                article_id = str(obj["id"])
                predictions = obj.get("predictions")

                if not isinstance(predictions, list):
                    raise RuntimeError(
                        f"Invalid predictions: {candidate_path}:{line_no}"
                    )

                for event in predictions:

                    try:
                        meta = next(meta_it)
                    except StopIteration:
                        raise RuntimeError(
                            f"Metadata ended before candidate events in "
                            f"{shard_dir.name}"
                        )

                    if str(meta.get("id")) != article_id:
                        raise RuntimeError(
                            f"Candidate/metadata ID mismatch in {shard_dir.name}: "
                            f"{article_id!r} vs {meta.get('id')!r}"
                        )

                    event_type = event.get("event_type")

                    if not isinstance(event_type, str) or not event_type:
                        raise RuntimeError(
                            f"Unexpected invalid event_type in {candidate_path}"
                        )

                    event_counts[event_type] += 1
                    total_events += 1
                    shard_events += 1

                    language = str(meta.get("language", "")).strip().lower()

                    if language in ("en", "english"):
                        language = "eng"
                    elif language in ("fr", "fre", "french"):
                        language = "fra"

                    if language in lang_event_counts:
                        lang_event_counts[language][event_type] += 1
                        lang_totals[language] += 1

                    raw_time = event.get("event_time")

                    if (
                        not isinstance(raw_time, str)
                        or not raw_time.strip()
                        or raw_time.strip().casefold() == "not_stated"
                    ):
                        time_not_stated += 1
                    else:
                        year = first_stated_year(raw_time)

                        if year is None:
                            time_no_year += 1
                        else:
                            year_counts[year] += 1

        extra_meta = next(meta_it, None)

        if extra_meta is not None:
            raise RuntimeError(
                f"Metadata contains extra structurally-valid events in "
                f"{shard_dir.name}"
            )

        print(
            f"[PASS1 {shard_no:03d}/200] "
            f"{shard_dir.name}: events={shard_events:,}",
            flush=True,
        )

    print()
    print("=== EVENT AUDIT ===")
    print(f"total events       : {total_events:,}")
    print(f"unique event types : {len(event_counts):,}")
    print(f"English events     : {lang_totals['eng']:,}")
    print(f"French events      : {lang_totals['fra']:,}")
    print(f"time not_stated    : {time_not_stated:,}")
    print(f"time no year token : {time_no_year:,}")
    print(f"time with year     : {sum(year_counts.values()):,}")
    print()

    if len(event_counts) != 48:
        print(
            f"WARNING: expected 48 event classes, observed {len(event_counts)}"
        )

    return {
        "event_counts": event_counts,
        "lang_event_counts": lang_event_counts,
        "lang_totals": lang_totals,
        "year_counts": year_counts,
        "time_not_stated": time_not_stated,
        "time_no_year": time_no_year,
        "total_events": total_events,
    }


def rare_location_pass(
    root: Path,
    rare_types: set[str],
):
    """
    Second pass only for globally rare event classes.

    Counts event-location component assignments for those classes.
    """
    shard_dirs = sorted(root.glob("shard-*"))

    counts = {
        event_type: Counter()
        for event_type in rare_types
    }

    component_totals = Counter()

    for shard_no, shard_dir in enumerate(shard_dirs, 1):

        path = shard_dir / "candidate_geocoding_input.jsonl"
        shard_matches = 0

        with path.open(encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                obj = json.loads(line)

                for event in obj.get("predictions", []):

                    event_type = event.get("event_type")

                    if event_type not in rare_types:
                        continue

                    locations = clean_locations(
                        event.get("event_location")
                    )

                    for location in locations:
                        counts[event_type][location] += 1
                        component_totals[event_type] += 1
                        shard_matches += 1

        print(
            f"[PASS2 {shard_no:03d}/200] "
            f"{shard_dir.name}: rare location-components={shard_matches:,}",
            flush=True,
        )

    return counts, component_totals


def choose_local_hotspots(
    rare_types,
    rare_location_counts,
    rare_component_totals,
    location_counts,
    min_location_mentions,
    min_rare_mentions,
):
    """
    For every globally rare event type, identify a location where that event
    is disproportionately common.

    Metric:
        local_share  = rare-event mentions at location / all mentions at location
        global_share = rare-event location-component mentions /
                       all location-component mentions

        enrichment = local_share / global_share
    """
    total_location_mentions = sum(location_counts.values())

    rows = []

    for event_type in rare_types:

        global_components = rare_component_totals[event_type]

        if not global_components:
            continue

        global_share = (
            global_components / total_location_mentions
        )

        candidates = []

        for location, event_count in rare_location_counts[event_type].items():

            location_total = location_counts.get(location, 0)

            if (
                location_total >= min_location_mentions
                and event_count >= min_rare_mentions
            ):
                local_share = event_count / location_total
                enrichment = (
                    local_share / global_share
                    if global_share > 0
                    else 0
                )

                candidates.append(
                    (
                        enrichment,
                        local_share,
                        event_count,
                        location_total,
                        location,
                    )
                )

        # Relax thresholds if a very rare class has no qualifying location.
        if not candidates:

            for location, event_count in rare_location_counts[event_type].items():

                location_total = location_counts.get(location, 0)

                if location_total >= 100 and event_count >= 3:

                    local_share = event_count / location_total

                    enrichment = (
                        local_share / global_share
                        if global_share > 0
                        else 0
                    )

                    candidates.append(
                        (
                            enrichment,
                            local_share,
                            event_count,
                            location_total,
                            location,
                        )
                    )

        if not candidates:
            continue

        (
            enrichment,
            local_share,
            event_count,
            location_total,
            location,
        ) = max(candidates)

        rows.append(
            {
                "event_type": event_type,
                "location": location,
                "global_component_count": global_components,
                "global_share_pct": 100 * global_share,
                "local_event_count": event_count,
                "location_total_mentions": location_total,
                "local_share_pct": 100 * local_share,
                "enrichment": enrichment,
            }
        )

    return rows


def write_event_csv(
    out: Path,
    event_counts,
    lang_event_counts,
    lang_totals,
):
    total = sum(event_counts.values())

    path = out / "event_distribution.csv"

    with path.open("w", encoding="utf-8", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "event_type",
                "count",
                "share_pct",
                "english_count",
                "english_share_pct",
                "french_count",
                "french_share_pct",
            ]
        )

        for event_type, count in event_counts.most_common():

            en = lang_event_counts["eng"][event_type]
            fr = lang_event_counts["fra"][event_type]

            writer.writerow(
                [
                    event_type,
                    count,
                    100 * count / total,
                    en,
                    (
                        100 * en / lang_totals["eng"]
                        if lang_totals["eng"]
                        else 0
                    ),
                    fr,
                    (
                        100 * fr / lang_totals["fra"]
                        if lang_totals["fra"]
                        else 0
                    ),
                ]
            )


def write_location_csv(out: Path, location_counts: Counter):
    path = out / "location_distribution.csv.gz"

    with gzip.open(
        path,
        "wt",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "location",
                "mention_count",
            ]
        )

        for location, count in location_counts.most_common():
            writer.writerow([location, count])


def write_year_csv(
    out: Path,
    year_counts: Counter,
):
    path = out / "time_distribution.csv"

    with path.open("w", encoding="utf-8", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["year", "event_mentions"])

        for year in sorted(year_counts):
            writer.writerow([year, year_counts[year]])


def write_rare_csv(out: Path, rows: list[dict]):
    path = out / "rare_event_location_enrichment.csv"

    fields = [
        "event_type",
        "location",
        "global_component_count",
        "global_share_pct",
        "local_event_count",
        "location_total_mentions",
        "local_share_pct",
        "enrichment",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_event_distribution(
    out: Path,
    event_counts: Counter,
):
    items = event_counts.most_common()
    labels = [x[0] for x in items][::-1]
    values = [x[1] for x in items][::-1]

    fig, ax = plt.subplots(figsize=(12, 15))

    ax.barh(labels, values)
    ax.set_xscale("log")

    ax.set_xlabel("Extracted event mentions (log scale)")
    ax.set_ylabel("Event type")

    ax.set_title(
        "Overall event-type distribution across 48 classes\n"
        "Global V4 structurally-valid event mentions"
    )

    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        out / "01_overall_event_distribution.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_language_distribution(
    out: Path,
    event_counts: Counter,
    lang_event_counts,
    lang_totals,
):
    # Keep the same ordering as the overall distribution.
    ordered = [
        event_type
        for event_type, _ in event_counts.most_common()
    ][::-1]

    en = np.array(
        [
            100
            * lang_event_counts["eng"][x]
            / lang_totals["eng"]
            for x in ordered
        ]
    )

    fr = np.array(
        [
            100
            * lang_event_counts["fra"][x]
            / lang_totals["fra"]
            for x in ordered
        ]
    )

    y = np.arange(len(ordered))
    height = 0.39

    fig, ax = plt.subplots(figsize=(13, 15))

    ax.barh(
        y - height / 2,
        en,
        height=height,
        label="English",
    )

    ax.barh(
        y + height / 2,
        fr,
        height=height,
        label="French",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(ordered)

    ax.set_xlabel(
        "Share of all extracted mentions within language (%)"
    )

    ax.set_ylabel("Event type")

    ax.set_title(
        "English vs French event-type distribution\n"
        "Normalized within each language"
    )

    ax.legend()
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        out / "02_english_vs_french_event_distribution.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_rare_hotspots(
    out: Path,
    rows: list[dict],
):
    if not rows:
        print(
            "WARNING: no rare-event/location hotspots qualified; "
            "plot 03 was not generated."
        )
        return

    # Largest enrichment at top.
    rows = sorted(
        rows,
        key=lambda x: x["enrichment"],
    )

    labels = [
        f"{r['event_type']}  —  {r['location']}"
        for r in rows
    ]

    global_pct = np.array(
        [r["global_share_pct"] for r in rows]
    )

    local_pct = np.array(
        [r["local_share_pct"] for r in rows]
    )

    y = np.arange(len(rows))
    height = 0.38

    fig, ax = plt.subplots(
        figsize=(13, max(7, 0.8 * len(rows)))
    )

    ax.barh(
        y - height / 2,
        global_pct,
        height=height,
        label="Global share",
    )

    ax.barh(
        y + height / 2,
        local_pct,
        height=height,
        label="Share within selected location",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    # Global percentages for rare events can be tiny.
    if np.all(global_pct > 0) and np.all(local_pct > 0):
        ax.set_xscale("log")

    ax.set_xlabel(
        "Share of location-event component mentions (%)"
    )

    ax.set_title(
        "Globally rare events can be locally concentrated\n"
        "For each rare class, the location with strongest qualified enrichment"
    )

    ax.legend()
    ax.grid(axis="x", alpha=0.25)

    for i, row in enumerate(rows):
        ax.text(
            row["local_share_pct"],
            i + height / 2,
            f"  ×{row['enrichment']:.1f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    fig.savefig(
        out / "03_rare_events_by_location.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_location_histogram(
    out: Path,
    location_counts: Counter,
):
    values = np.fromiter(
        location_counts.values(),
        dtype=np.int64,
    )

    maximum = int(values.max())

    bins = np.logspace(
        0,
        math.log10(maximum + 1),
        60,
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.hist(values, bins=bins)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        "Extracted mentions per cleaned location (log scale)"
    )

    ax.set_ylabel(
        "Number of unique cleaned locations (log scale)"
    )

    ax.set_title(
        "Location-frequency distribution\n"
        f"{len(location_counts):,} unique cleaned location strings"
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        out / "04_location_distribution.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_time_distribution(
    out: Path,
    year_counts: Counter,
    time_not_stated: int,
    time_no_year: int,
):
    if not year_counts:
        raise RuntimeError(
            "No explicit year tokens found in event_time."
        )

    years = sorted(year_counts)

    counts = [
        year_counts[y]
        for y in years
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(years, counts, marker="o", markersize=2)

    ax.set_xlabel("First explicit year in raw event_time")
    ax.set_ylabel("Extracted event mentions")

    ax.set_title(
        "Temporal distribution of extracted events\n"
        "No artificial date precision is introduced"
    )

    ax.grid(alpha=0.25)

    note = (
        f"not_stated: {time_not_stated:,}   |   "
        f"stated time without 4-digit year: {time_no_year:,}"
    )

    fig.text(
        0.5,
        0.01,
        note,
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    fig.savefig(
        out / "05_time_distribution.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_low_classes(
    out: Path,
    event_counts: Counter,
    n: int = 15,
):
    """
    Backup plot for questions about low-represented classes.
    """
    items = sorted(
        event_counts.items(),
        key=lambda x: x[1],
    )[:n]

    labels = [x[0] for x in items][::-1]
    values = [x[1] for x in items][::-1]

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.barh(labels, values)

    ax.set_xlabel("Extracted event mentions")
    ax.set_ylabel("Event type")

    ax.set_title(
        f"Backup: {n} lowest-represented event classes"
    )

    ax.grid(axis="x", alpha=0.25)

    for i, value in enumerate(values):
        ax.text(
            value,
            i,
            f"  {value:,}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    fig.savefig(
        out / "backup_low_represented_classes.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=Path,
        default=Path(
            "outputs/geography/global_v4_audit/postprocessing_v1"
        ),
        help="Folder containing shard-000 ... shard-199",
    )

    ap.add_argument(
        "--location-pairs",
        type=Path,
        default=Path(
            "outputs/geography/global_v4_audit/location_layer_pairs.jsonl"
        ),
        help="Previously built cleaned location/layer inventory.",
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/geography/global_v4_audit/extraction_quality_plots"
        ),
    )

    ap.add_argument(
        "--rare-n",
        type=int,
        default=10,
        help="How many globally lowest-frequency event classes to analyze.",
    )

    ap.add_argument(
        "--min-location-mentions",
        type=int,
        default=1000,
        help=(
            "Minimum all-event mentions at a location before "
            "considering it a rare-event hotspot."
        ),
    )

    ap.add_argument(
        "--min-rare-at-location",
        type=int,
        default=20,
        help=(
            "Minimum mentions of a rare event at a location before "
            "considering the enrichment."
        ),
    )

    args = ap.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.root.is_dir():
        raise FileNotFoundError(args.root)

    if not args.location_pairs.is_file():
        raise FileNotFoundError(args.location_pairs)

    # ---------------------------------------------------------
    # Existing cleaned location inventory
    # ---------------------------------------------------------

    location_counts = load_location_counts(
        args.location_pairs
    )

    # ---------------------------------------------------------
    # Full structurally-valid event pass
    # ---------------------------------------------------------

    result = first_pass(args.root)

    event_counts = result["event_counts"]

    # Lowest-frequency event classes globally.
    rare_ordered = [
        event_type
        for event_type, _ in sorted(
            event_counts.items(),
            key=lambda x: x[1],
        )[: args.rare_n]
    ]

    print("=== GLOBALLY RARE EVENT CLASSES ===")

    for rank, event_type in enumerate(
        rare_ordered,
        1,
    ):
        print(
            f"{rank:2d}. "
            f"{event_type:<40} "
            f"{event_counts[event_type]:,}"
        )

    print()

    # ---------------------------------------------------------
    # Second pass only for rare classes × locations
    # ---------------------------------------------------------

    rare_location_counts, rare_component_totals = (
        rare_location_pass(
            args.root,
            set(rare_ordered),
        )
    )

    rare_rows = choose_local_hotspots(
        rare_ordered,
        rare_location_counts,
        rare_component_totals,
        location_counts,
        args.min_location_mentions,
        args.min_rare_at_location,
    )

    # ---------------------------------------------------------
    # Tables
    # ---------------------------------------------------------

    write_event_csv(
        args.out,
        event_counts,
        result["lang_event_counts"],
        result["lang_totals"],
    )

    write_location_csv(
        args.out,
        location_counts,
    )

    write_year_csv(
        args.out,
        result["year_counts"],
    )

    write_rare_csv(
        args.out,
        rare_rows,
    )

    # ---------------------------------------------------------
    # Five presentation plots
    # ---------------------------------------------------------

    print()
    print("Generating plots...")

    plot_event_distribution(
        args.out,
        event_counts,
    )

    plot_language_distribution(
        args.out,
        event_counts,
        result["lang_event_counts"],
        result["lang_totals"],
    )

    plot_rare_hotspots(
        args.out,
        rare_rows,
    )

    plot_location_histogram(
        args.out,
        location_counts,
    )

    plot_time_distribution(
        args.out,
        result["year_counts"],
        result["time_not_stated"],
        result["time_no_year"],
    )

    # Backup requested earlier.
    plot_low_classes(
        args.out,
        event_counts,
        n=min(15, len(event_counts)),
    )

    # ---------------------------------------------------------
    # Compact machine-readable summary
    # ---------------------------------------------------------

    summary = {
        "total_structurally_valid_events": result["total_events"],
        "event_classes": len(event_counts),
        "english_events": result["lang_totals"]["eng"],
        "french_events": result["lang_totals"]["fra"],
        "unique_cleaned_locations": len(location_counts),
        "cleaned_location_component_mentions": sum(
            location_counts.values()
        ),
        "time_not_stated": result["time_not_stated"],
        "time_stated_without_four_digit_year": result["time_no_year"],
        "time_with_explicit_year": sum(
            result["year_counts"].values()
        ),
        "rare_classes": [
            {
                "event_type": x,
                "global_event_mentions": event_counts[x],
            }
            for x in rare_ordered
        ],
        "rare_event_location_hotspots": rare_rows,
    }

    (
        args.out / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=== DONE ===")
    print(f"Output folder: {args.out}")
    print()
    print("Presentation plots:")
    print("  01_overall_event_distribution.png")
    print("  02_english_vs_french_event_distribution.png")
    print("  03_rare_events_by_location.png")
    print("  04_location_distribution.png")
    print("  05_time_distribution.png")
    print()
    print("Backup:")
    print("  backup_low_represented_classes.png")
    print()
    print("Tables:")
    print("  event_distribution.csv")
    print("  location_distribution.csv.gz")
    print("  rare_event_location_enrichment.csv")
    print("  time_distribution.csv")
    print("  summary.json")


if __name__ == "__main__":
    main()
