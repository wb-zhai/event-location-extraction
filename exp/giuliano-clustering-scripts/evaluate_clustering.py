#!/usr/bin/env python3
"""
Analyze event clusters stored in a Parquet file.

Expected columns:
    cluster_uid
    adm_code
    adm_level
    event_type
    cluster_start_date
    cluster_end_date
    cluster_span_days
    n_mentions
    n_articles

Outputs:
    analysis_output/
        event_type_occurrences.csv
        event_type_admin_level_distribution.csv
        event_type_admin_level_fraction.csv
        event_type_duration_stats.csv
        event_type_summary.csv
        plot_event_type_occurrences.png
        plot_admin_level_fraction_by_event_type.png
        plot_average_event_duration.png

Usage:
    python analyze_events.py input.parquet
    python analyze_events.py input.parquet --top-n 20
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "cluster_uid",
    "adm_code",
    "adm_level",
    "event_type",
    "cluster_start_date",
    "cluster_end_date",
    "cluster_span_days",
    "n_mentions",
    "n_articles",
]


def load_data(path: str) -> pd.DataFrame:
    """Load and perform basic cleaning."""
    # Requires: pip install pandas pyarrow matplotlib
    df = pd.read_parquet(path, engine="pyarrow")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Dates
    df["cluster_start_date"] = pd.to_datetime(
        df["cluster_start_date"], errors="coerce", utc=True
    )
    df["cluster_end_date"] = pd.to_datetime(
        df["cluster_end_date"], errors="coerce", utc=True
    )

    # Numeric columns
    for col in ["adm_level", "cluster_span_days", "n_mentions", "n_articles"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean event labels
    df["event_type"] = (
        df["event_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    )

    # If span_days is missing, reconstruct it from dates.
    missing_span = df["cluster_span_days"].isna()
    df.loc[missing_span, "cluster_span_days"] = (
        df.loc[missing_span, "cluster_end_date"]
        - df.loc[missing_span, "cluster_start_date"]
    ).dt.total_seconds() / 86400 + 1

    # Remove rows without an event type because they cannot be grouped.
    df = df.dropna(subset=["event_type"]).copy()

    return df


def make_tables(df: pd.DataFrame, outdir: Path) -> dict:
    """Create the requested summary tables."""

    # 1. Event types by occurrence.
    occurrence = (
        df.groupby("event_type", dropna=False)
        .agg(
            occurrences=("cluster_uid", "nunique"),
            rows=("cluster_uid", "size"),
            total_mentions=("n_mentions", "sum"),
            total_articles=("n_articles", "sum"),
        )
        .sort_values("occurrences", ascending=False)
        .reset_index()
    )
    occurrence["fraction_of_occurrences"] = (
        occurrence["occurrences"] / occurrence["occurrences"].sum()
    )

    # 2. Admin-level distribution within each event type.
    # First count event clusters by event_type x adm_level.
    admin_counts = (
        df.groupby(["event_type", "adm_level"], dropna=False)
        .agg(occurrences=("cluster_uid", "nunique"))
        .reset_index()
    )

    admin_counts["admin_level_label"] = admin_counts["adm_level"].apply(
        lambda x: "Missing" if pd.isna(x) else f"Adm {int(x)}"
    )

    # Fractions sum to 1 within each event type.
    admin_counts["fraction_within_event_type"] = (
        admin_counts["occurrences"]
        / admin_counts.groupby("event_type")["occurrences"].transform("sum")
    )

    admin_fraction = admin_counts.pivot(
        index="event_type",
        columns="admin_level_label",
        values="fraction_within_event_type",
    ).fillna(0)

    # Put admin levels in natural order where possible.
    def admin_sort_key(x):
        if x == "Missing":
            return 999
        try:
            return int(x.split()[-1])
        except Exception:
            return 998

    admin_fraction = admin_fraction[
        sorted(admin_fraction.columns, key=admin_sort_key)
    ].reset_index()

    # 3. Duration statistics by event type.
    duration = (
        df.groupby("event_type")
        .agg(
            occurrences=("cluster_uid", "nunique"),
            mean_days=("cluster_span_days", "mean"),
            median_days=("cluster_span_days", "median"),
            std_days=("cluster_span_days", "std"),
            p25_days=("cluster_span_days", lambda x: x.quantile(0.25)),
            p75_days=("cluster_span_days", lambda x: x.quantile(0.75)),
            max_days=("cluster_span_days", "max"),
        )
        .sort_values("mean_days", ascending=False)
        .reset_index()
    )

    # One compact summary table.
    summary = occurrence.merge(
        duration[
            [
                "event_type",
                "mean_days",
                "median_days",
                "p25_days",
                "p75_days",
                "max_days",
            ]
        ],
        on="event_type",
        how="left",
    )

    occurrence.to_csv(outdir / "event_type_occurrences.csv", index=False)
    admin_counts.to_csv(
        outdir / "event_type_admin_level_distribution.csv", index=False
    )
    admin_fraction.to_csv(
        outdir / "event_type_admin_level_fraction.csv", index=False
    )
    duration.to_csv(outdir / "event_type_duration_stats.csv", index=False)
    summary.to_csv(outdir / "event_type_summary.csv", index=False)

    return {
        "occurrence": occurrence,
        "admin_counts": admin_counts,
        "admin_fraction": admin_fraction,
        "duration": duration,
        "summary": summary,
    }


def make_plots(tables: dict, outdir: Path, top_n: int = 20) -> None:
    """Create the three requested plots."""

    occurrence = tables["occurrence"]
    admin_fraction = tables["admin_fraction"]
    duration = tables["duration"]

    # -----------------------------
    # Plot 1: most common event types
    # -----------------------------
    top = occurrence.head(top_n).sort_values("occurrences")

    plt.figure(figsize=(10, max(6, 0.35 * len(top))))
    plt.barh(top["event_type"], top["occurrences"])
    plt.xlabel("Number of event occurrences")
    plt.ylabel("Event type")
    plt.title(f"Most common event types — top {top_n}")
    plt.tight_layout()
    plt.savefig(outdir / "plot_event_type_occurrences.png", dpi=200)
    plt.close()

    # ----------------------------------------
    # Plot 2: admin-level fraction by event type
    # ----------------------------------------
    top_event_names = occurrence.head(top_n)["event_type"].tolist()
    admin_plot = admin_fraction[
        admin_fraction["event_type"].isin(top_event_names)
    ].copy()

    # Reorder rows according to occurrence ranking.
    order = {name: i for i, name in enumerate(top_event_names)}
    admin_plot["_order"] = admin_plot["event_type"].map(order)
    admin_plot = admin_plot.sort_values("_order").drop(columns="_order")

    level_cols = [c for c in admin_plot.columns if c != "event_type"]

    plt.figure(figsize=(12, max(6, 0.4 * len(admin_plot))))
    left = np.zeros(len(admin_plot))

    for col in level_cols:
        values = admin_plot[col].to_numpy()
        plt.barh(
            admin_plot["event_type"],
            values,
            left=left,
            label=col,
        )
        left += values

    plt.xlabel("Fraction within event type")
    plt.ylabel("Event type")
    plt.title(f"Administrative-level composition — top {top_n} event types")
    plt.xlim(0, 1)
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    plt.legend(title="Administrative level", bbox_to_anchor=(1.02, 1),
               loc="upper left")
    plt.tight_layout()
    plt.savefig(outdir / "plot_admin_level_fraction_by_event_type.png", dpi=200)
    plt.close()

    # --------------------------------
    # Plot 3: average duration by type
    # --------------------------------
    top_duration = duration.head(top_n).sort_values("mean_days")

    plt.figure(figsize=(10, max(6, 0.35 * len(top_duration))))
    plt.barh(top_duration["event_type"], top_duration["mean_days"])
    plt.xlabel("Average event duration (days)")
    plt.ylabel("Event type")
    plt.title(f"Average event duration — top {top_n} by mean duration")
    plt.tight_layout()
    plt.savefig(outdir / "plot_average_event_duration.png", dpi=200)
    plt.close()


def print_interesting_statistics(df: pd.DataFrame, tables: dict, top_n: int) -> None:
    occurrence = tables["occurrence"]
    duration = tables["duration"]
    admin_fraction = tables["admin_fraction"]

    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Rows:                         {len(df):,}")
    print(f"Unique event clusters:        {df['cluster_uid'].nunique():,}")
    print(f"Unique event types:           {df['event_type'].nunique():,}")
    print(f"Unique administrative codes:  {df['adm_code'].nunique():,}")
    print(
        f"Date range:                   "
        f"{df['cluster_start_date'].min()} -> {df['cluster_end_date'].max()}"
    )

    print("\n" + "=" * 70)
    print("1) MOST IMPORTANT EVENT TYPES BY OCCURRENCE")
    print("=" * 70)

    display = occurrence.head(top_n).copy()
    display["share"] = display["fraction_of_occurrences"].map(lambda x: f"{x:.1%}")
    print(
        display[
            ["event_type", "occurrences", "share", "total_mentions", "total_articles"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("2) ADMINISTRATIVE-LEVEL COMPOSITION")
    print("=" * 70)
    print(
        "Fractions are calculated within each event type, so each event type "
        "sums to approximately 100% across administrative levels."
    )
    print(admin_fraction.head(top_n).to_string(index=False))

    print("\n" + "=" * 70)
    print("3) EVENT DURATION")
    print("=" * 70)

    print(
        duration.head(top_n)[
            [
                "event_type",
                "occurrences",
                "mean_days",
                "median_days",
                "p25_days",
                "p75_days",
                "max_days",
            ]
        ].to_string(index=False)
    )

    # A few useful derived observations.
    print("\n" + "=" * 70)
    print("ADDITIONAL INTERESTING STATISTICS")
    print("=" * 70)

    top_occ = occurrence.iloc[0]
    longest_mean = duration.iloc[0]

    print(
        f"Most frequent event type: {top_occ['event_type']} "
        f"({int(top_occ['occurrences']):,} occurrences; "
        f"{top_occ['fraction_of_occurrences']:.1%} of all occurrences)"
    )

    print(
        f"Event type with the longest average duration: "
        f"{longest_mean['event_type']} "
        f"({longest_mean['mean_days']:.1f} days on average; "
        f"median {longest_mean['median_days']:.1f})"
    )

    # Overall administrative-level distribution.
    overall_admin = (
        df.groupby("adm_level")["cluster_uid"]
        .nunique()
        .sort_values(ascending=False)
    )
    overall_admin_fraction = overall_admin / overall_admin.sum()

    print("\nOverall administrative-level distribution:")
    for level, frac in overall_admin_fraction.items():
        label = "Missing" if pd.isna(level) else f"Adm {int(level)}"
        print(f"  {label}: {frac:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_file", help="Path to the input parquet file")
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Directory for tables and plots",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=48,
        help="Number of event types shown in the plots and console output",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.parquet_file}")
    df = load_data(args.parquet_file)

    print(f"Loaded {len(df):,} rows.")
    tables = make_tables(df, outdir)
    make_plots(tables, outdir, top_n=args.top_n)
    print_interesting_statistics(df, tables, args.top_n)

    print("\n" + "=" * 70)
    print(f"All outputs written to: {outdir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
