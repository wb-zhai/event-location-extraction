#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Shared presentation style
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 18,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 18,
})


def compact(n):
    n = float(n)

    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"

    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"

    return f"{n:.0f}"


def save(fig, outdir: Path, stem: str):
    fig.savefig(
        outdir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        outdir / f"{stem}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Load existing summary tables
# ============================================================

def load_events(path: Path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):

            rows.append({
                "event_type": row["event_type"],
                "count": int(row["count"]),
                "share_pct": float(row["share_pct"]),
                "english_count": int(row["english_count"]),
                "english_share_pct": float(
                    row["english_share_pct"]
                ),
                "french_count": int(row["french_count"]),
                "french_share_pct": float(
                    row["french_share_pct"]
                ),
            })

    return rows


def load_locations(path: Path):
    rows = []

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(
        path,
        "rt",
        encoding="utf-8",
    ) as f:

        for row in csv.DictReader(f):

            rows.append({
                "location": row["location"],
                "count": int(row["mention_count"]),
            })

    return rows


def load_rare(path: Path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):

            rows.append({
                "event_type": row["event_type"],
                "location": row["location"],

                "global_component_count":
                    int(row["global_component_count"]),

                "global_share_pct":
                    float(row["global_share_pct"]),

                "local_event_count":
                    int(row["local_event_count"]),

                "location_total_mentions":
                    int(row["location_total_mentions"]),

                "local_share_pct":
                    float(row["local_share_pct"]),

                "enrichment":
                    float(row["enrichment"]),
            })

    return rows


def load_time(path: Path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):

            rows.append({
                "year": int(row["year"]),
                "count": int(row["event_mentions"]),
            })

    return rows


# ============================================================
# 1. EVENT DISTRIBUTION
#
# Gui:
# "event coverage"
# "(1), is a good plot. easy to understand."
# ============================================================

def plot_event_distribution(rows, outdir):

    rows = sorted(
        rows,
        key=lambda x: x["count"],
        reverse=True,
    )

    total = sum(
        x["count"]
        for x in rows
    )

    plot_rows = rows[::-1]

    labels = [
        x["event_type"]
        for x in plot_rows
    ]

    values = [
        x["count"]
        for x in plot_rows
    ]

    fig, ax = plt.subplots(
        figsize=(12, 14)
    )

    ax.barh(
        labels,
        values,
    )

    # Keep log scale because all 48 classes remain visible.
    ax.set_xscale("log")

    ax.set_xlabel(
        "Number of extracted mentions"
    )

    ax.set_ylabel("")

    ax.set_title(
        "Distribution of extracted event types",
        loc="left",
        fontweight="bold",
        pad=22,
    )

    ax.text(
        0,
        1.005,
        (
            f"{compact(total)} mentions across "
            f"{len(rows)} event classes"
        ),
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
    )

    ax.grid(
        axis="x",
        alpha=0.18,
    )

    # Rank in original descending ordering.
    rank = {
        row["event_type"]: i
        for i, row in enumerate(rows)
    }

    n = len(rows)

    # Annotate top 10 and bottom 5 only.
    for i, row in enumerate(plot_rows):

        original_rank = rank[
            row["event_type"]
        ]

        if (
            original_rank < 10
            or original_rank >= n - 5
        ):

            ax.text(
                row["count"],
                i,
                f"  {compact(row['count'])}",
                va="center",
                fontsize=8.5,
            )

    fig.tight_layout()

    save(
        fig,
        outdir,
        "01_event_distribution",
    )


# ============================================================
# 2. ENGLISH VS FRENCH EVENT DISTRIBUTION
#
# Gui:
# "english vs french"
# ============================================================

def plot_english_french(rows, outdir):

    prepared = []

    for row in rows:

        difference = (
            row["french_share_pct"]
            - row["english_share_pct"]
        )

        prepared.append({
            **row,
            "difference": difference,
            "abs_difference": abs(difference),
        })

    # Largest differences toward the top.
    prepared = sorted(
        prepared,
        key=lambda x: x["abs_difference"],
    )

    labels = [
        x["event_type"]
        for x in prepared
    ]

    english = np.array([
        x["english_share_pct"]
        for x in prepared
    ])

    french = np.array([
        x["french_share_pct"]
        for x in prepared
    ])

    y = np.arange(
        len(prepared)
    )

    fig, ax = plt.subplots(
        figsize=(12, 14)
    )

    # Connect each EN / FR pair.
    for i in range(
        len(prepared)
    ):

        ax.plot(
            [
                english[i],
                french[i],
            ],
            [
                i,
                i,
            ],
            linewidth=1,
            alpha=0.35,
        )

    ax.scatter(
        english,
        y,
        s=34,
        label="English",
        zorder=3,
    )

    ax.scatter(
        french,
        y,
        s=34,
        label="French",
        zorder=3,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Share of extracted mentions within each language (%)"
    )

    ax.set_ylabel("")

    ax.set_title(
        "English vs French event distribution",
        loc="left",
        fontweight="bold",
        pad=22,
    )

    ax.text(
        0,
        1.005,
        (
            "Normalized within each language so corpus size "
            "does not drive the comparison"
        ),
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
    )

    ax.grid(
        axis="x",
        alpha=0.18,
    )

    ax.legend(
        loc="lower right",
        frameon=False,
    )

    # Label only the 5 largest language differences.
    biggest = sorted(
        prepared,
        key=lambda x: x["abs_difference"],
        reverse=True,
    )[:5]

    biggest_names = {
        x["event_type"]
        for x in biggest
    }

    for i, row in enumerate(prepared):

        if (
            row["event_type"]
            not in biggest_names
        ):
            continue

        right = max(
            row["english_share_pct"],
            row["french_share_pct"],
        )

        diff = row["difference"]

        sign = (
            "+"
            if diff >= 0
            else ""
        )

        ax.text(
            right,
            i,
            f"  {sign}{diff:.1f} pp FR−EN",
            va="center",
            fontsize=8.5,
        )

    fig.tight_layout()

    save(
        fig,
        outdir,
        "02_english_vs_french_event_distribution",
    )


# ============================================================
# 3. RARE EVENTS BY LOCATION
#
# Gui:
# "another analysis about locations in which rare events
# are appearing more often, demonstrating that rare events
# are not rare in certain locations."
#
# Keep the original global-vs-local comparison style.
# ============================================================

def plot_rare_by_location(rows, outdir):

    if not rows:
        print(
            "WARNING: no rows for rare-event location analysis."
        )
        return

    rows = sorted(
        rows,
        key=lambda x: x["enrichment"],
        reverse=True,
    )

    # Reverse for horizontal plot so largest is at top.
    plot_rows = rows[::-1]

    labels = [
        (
            f"{x['event_type']} — "
            f"{x['location']}"
        )
        for x in plot_rows
    ]

    global_pct = np.array([
        x["global_share_pct"]
        for x in plot_rows
    ])

    local_pct = np.array([
        x["local_share_pct"]
        for x in plot_rows
    ])

    y = np.arange(
        len(plot_rows)
    )

    height = 0.38

    fig, ax = plt.subplots(
        figsize=(13, 8)
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

    ax.set_yticklabels(
        labels
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Share of event mentions (%)"
    )

    ax.set_ylabel("")

    ax.set_title(
        "Rare events by location",
        loc="left",
        fontweight="bold",
        pad=22,
    )

    ax.text(
        0,
        1.005,
        (
            "Globally low-represented events can appear "
            "much more frequently in particular locations"
        ),
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
    )

    ax.grid(
        axis="x",
        alpha=0.18,
    )

    ax.legend(
        loc="lower right",
        frameon=False,
    )

    # Keep enrichment annotation AND provide local n.
    for i, row in enumerate(
        plot_rows
    ):

        ax.text(
            row["local_share_pct"],
            i + height / 2,
            (
                f"  ×{row['enrichment']:.1f}"
                f"  (n={row['local_event_count']:,})"
            ),
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    save(
        fig,
        outdir,
        "03_rare_events_by_location",
    )


# ============================================================
# 4. LOCATION DISTRIBUTION
#
# Gui:
# "geo coverage:
# Just a single histogram, what you put as (5)."
# ============================================================

def plot_location_distribution(rows, outdir):

    counts = np.array(
        [
            x["count"]
            for x in rows
        ],
        dtype=np.int64,
    )

    unique_locations = len(
        counts
    )

    total_mentions = int(
        counts.sum()
    )

    singleton_count = int(
        np.sum(
            counts == 1
        )
    )

    singleton_share = (
        100
        * singleton_count
        / unique_locations
        if unique_locations
        else 0
    )

    ordered = np.sort(
        counts
    )[::-1]

    top100_share = (
        100
        * ordered[:100].sum()
        / total_mentions
        if total_mentions
        else 0
    )

    top1000_share = (
        100
        * ordered[:1000].sum()
        / total_mentions
        if total_mentions
        else 0
    )

    bins = np.logspace(
        0,
        math.log10(
            int(counts.max()) + 1
        ),
        60,
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.hist(
        counts,
        bins=bins,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Number of mentions per location"
    )

    ax.set_ylabel(
        "Number of locations"
    )

    ax.set_title(
        "Location mentions follow a strong long-tail distribution",
        loc="left",
        fontweight="bold",
        pad=22,
    )

    ax.text(
        0,
        1.005,
        (
            f"{unique_locations:,} unique locations"
            f"   •   {singleton_share:.1f}% appear once"
            f"   •   top 100 account for "
            f"{top100_share:.1f}% of mentions"
            f"   •   top 1,000 account for "
            f"{top1000_share:.1f}%"
        ),
        transform=ax.transAxes,
        fontsize=10.5,
        va="bottom",
    )

    ax.grid(
        alpha=0.18,
    )

    fig.tight_layout()

    save(
        fig,
        outdir,
        "04_location_distribution",
    )


# ============================================================
# 5. TIME DISTRIBUTION
#
# Gui:
# "temporal:
# Just put (9) simple distribution."
#
# Still simple distribution, but avoid 1000-2200 crushing
# the useful region.
# ============================================================

def weighted_quantile_range(
    years,
    counts,
    lower=0.005,
    upper=0.995,
):

    order = np.argsort(
        years
    )

    years = years[order]
    counts = counts[order]

    cumulative = np.cumsum(
        counts
    )

    total = cumulative[-1]

    low_idx = np.searchsorted(
        cumulative,
        lower * total,
    )

    high_idx = np.searchsorted(
        cumulative,
        upper * total,
    )

    low_year = int(
        years[
            min(
                low_idx,
                len(years) - 1,
            )
        ]
    )

    high_year = int(
        years[
            min(
                high_idx,
                len(years) - 1,
            )
        ]
    )

    return (
        low_year,
        high_year,
    )


def plot_time_distribution(
    rows,
    outdir,
):

    years = np.array(
        [
            x["year"]
            for x in rows
        ],
        dtype=int,
    )

    counts = np.array(
        [
            x["count"]
            for x in rows
        ],
        dtype=np.int64,
    )

    total = int(
        counts.sum()
    )

    (
        low_year,
        high_year,
    ) = weighted_quantile_range(
        years,
        counts,
    )

    # Small visual padding.
    low_year -= 1
    high_year += 1

    mask = (
        (years >= low_year)
        &
        (years <= high_year)
    )

    display_years = years[
        mask
    ]

    display_counts = counts[
        mask
    ]

    outside_mentions = int(
        counts[~mask].sum()
    )

    outside_share = (
        100
        * outside_mentions
        / total
        if total
        else 0
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        display_years,
        display_counts,
        width=0.85,
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Extracted event mentions"
    )

    ax.set_title(
        "Time distribution",
        loc="left",
        fontweight="bold",
        pad=22,
    )

    ax.text(
        0,
        1.005,
        (
            f"Displayed period: "
            f"{low_year}–{high_year}. "
            f"{outside_share:.2f}% of year-resolved "
            f"mentions fall outside the displayed range."
        ),
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    fig.tight_layout()

    save(
        fig,
        outdir,
        "05_time_distribution",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "global_v4_audit/"
            "extraction_quality_plots"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "global_v4_audit/"
            "gui_5_plots_presentation"
        ),
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_file = (
        args.input
        / "event_distribution.csv"
    )

    location_file = (
        args.input
        / "location_distribution.csv.gz"
    )

    rare_file = (
        args.input
        / "rare_event_location_enrichment.csv"
    )

    time_file = (
        args.input
        / "time_distribution.csv"
    )

    for path in [
        event_file,
        location_file,
        rare_file,
        time_file,
    ]:

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    print(
        "Loading summary tables..."
    )

    events = load_events(
        event_file
    )

    locations = load_locations(
        location_file
    )

    rare = load_rare(
        rare_file
    )

    times = load_time(
        time_file
    )

    print(
        f"event classes       : {len(events):,}"
    )

    print(
        f"unique locations    : {len(locations):,}"
    )

    print(
        f"rare-event rows     : {len(rare):,}"
    )

    print(
        f"years represented   : {len(times):,}"
    )

    print()

    print(
        "[1/5] Event distribution"
    )

    plot_event_distribution(
        events,
        args.out,
    )

    print(
        "[2/5] English vs French event distribution"
    )

    plot_english_french(
        events,
        args.out,
    )

    print(
        "[3/5] Rare events by location"
    )

    plot_rare_by_location(
        rare,
        args.out,
    )

    print(
        "[4/5] Location distribution"
    )

    plot_location_distribution(
        locations,
        args.out,
    )

    print(
        "[5/5] Time distribution"
    )

    plot_time_distribution(
        times,
        args.out,
    )

    print()
    print("=== DONE ===")
    print(
        f"Output folder: {args.out}"
    )

    print()
    print("Gui's five plots:")
    print(
        "  01_event_distribution.png"
    )
    print(
        "  02_english_vs_french_event_distribution.png"
    )
    print(
        "  03_rare_events_by_location.png"
    )
    print(
        "  04_location_distribution.png"
    )
    print(
        "  05_time_distribution.png"
    )

    print()
    print(
        "PDF versions were also generated."
    )


if __name__ == "__main__":
    main()
