#!/usr/bin/env python3

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(
    "/mnt/c/Users/Bonaventure/Downloads/event-location-extraction"
)
DB = ROOT / "outputs/geography/global_v4_audit/location_layer_pairs.sqlite"

OUTDIR = ROOT / "outputs/geography/global_v4_audit/extraction_quality_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "05_location_distribution_by_hierarchy.png"
PDF = OUTDIR / "05_location_distribution_by_hierarchy.pdf"
CSV = OUTDIR / "05_location_hierarchy_summary.csv"

EXPECTED_LOCATIONS = 1_054_032
EXPECTED_MENTIONS = 137_210_352

# More specific level wins only when two layers have exactly the
# same mention count for the same cleaned location.
priority_sql = """
CASE lower(layer)
    WHEN 'district' THEN 1
    WHEN 'county'   THEN 2
    WHEN 'city'     THEN 3
    WHEN 'state'    THEN 4
    WHEN 'country'  THEN 5
    WHEN 'no_layer' THEN 6
    ELSE 7
END
"""

con = sqlite3.connect(DB)

query = f"""
WITH totals AS (
    SELECT
        location,
        SUM(mention_count) AS total_mentions
    FROM pairs
    GROUP BY location
),
ranked AS (
    SELECT
        location,
        layer,
        mention_count,
        ROW_NUMBER() OVER (
            PARTITION BY location
            ORDER BY
                mention_count DESC,
                {priority_sql},
                layer
        ) AS rn
    FROM pairs
)
SELECT
    t.location,
    t.total_mentions,
    r.layer AS dominant_layer
FROM totals t
JOIN ranked r
  ON r.location = t.location
 AND r.rn = 1
"""

print("Reading and aggregating location/layer inventory...")
df = pd.read_sql_query(query, con)
con.close()

df["total_mentions"] = df["total_mentions"].astype(np.int64)

def normalize_layer(x):
    x = str(x or "").strip().lower()

    if x == "country":
        return "Country"
    if x == "state":
        return "State"
    if x == "county":
        return "County"
    if x == "district":
        return "District"
    if x == "city":
        return "City"
    if x in {"no_layer", "", "none", "null"}:
        return "Unspecified"
    return "Other"

df["hierarchy"] = df["dominant_layer"].map(normalize_layer)

n_locations = len(df)
n_mentions = int(df["total_mentions"].sum())

print()
print("=" * 72)
print("PLOT 5 INPUT VALIDATION")
print("=" * 72)
print(f"unique locations       {n_locations:,}")
print(f"component mentions     {n_mentions:,}")

if n_locations != EXPECTED_LOCATIONS:
    raise RuntimeError(
        f"Expected {EXPECTED_LOCATIONS:,} unique locations, "
        f"found {n_locations:,}"
    )

if n_mentions != EXPECTED_MENTIONS:
    raise RuntimeError(
        f"Expected {EXPECTED_MENTIONS:,} mentions, "
        f"found {n_mentions:,}"
    )

order = [
    "District",
    "County",
    "City",
    "State",
    "Country",
    "Unspecified",
    "Other",
]

# Explicit palette because the hierarchy breakdown is the requested
# visual encoding.
colors = {
    "District": "#0072B2",
    "County": "#56B4E9",
    "City": "#009E73",
    "State": "#E69F00",
    "Country": "#D55E00",
    "Unspecified": "#999999",
    "Other": "#CC79A7",
}

summary = (
    df.groupby("hierarchy", as_index=False)
      .agg(
          unique_locations=("location", "size"),
          component_mentions=("total_mentions", "sum"),
      )
)

summary["location_share_pct"] = (
    100 * summary["unique_locations"] / n_locations
)
summary["mention_share_pct"] = (
    100 * summary["component_mentions"] / n_mentions
)

summary["sort_order"] = summary["hierarchy"].map(
    {name: i for i, name in enumerate(order)}
).fillna(999)

summary = (
    summary.sort_values("sort_order")
           .drop(columns="sort_order")
)

summary.to_csv(CSV, index=False)

print()
print("Hierarchy breakdown:")
print(summary.to_string(index=False, formatters={
    "unique_locations": "{:,}".format,
    "component_mentions": "{:,}".format,
    "location_share_pct": "{:.2f}".format,
    "mention_share_pct": "{:.2f}".format,
}))

maximum = int(df["total_mentions"].max())
bins = np.logspace(
    0,
    np.log10(maximum + 1),
    60
)

data = []
labels = []
plot_colors = []

for level in order:
    vals = df.loc[
        df["hierarchy"] == level,
        "total_mentions"
    ].to_numpy(dtype=np.int64)

    if len(vals) == 0:
        continue

    data.append(vals)
    labels.append(level)
    plot_colors.append(colors[level])

fig, ax = plt.subplots(figsize=(11.5, 6.5))

ax.hist(
    data,
    bins=bins,
    stacked=True,
    label=labels,
    color=plot_colors,
    edgecolor="none",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Number of mentions per unique location", fontsize=12)
ax.set_ylabel("Number of unique locations", fontsize=12)

ax.set_title(
    "Location mentions follow a strong long-tail distribution",
    fontsize=15,
    fontweight="bold",
    pad=14,
)

ax.text(
    0.5,
    1.005,
    "Color indicates the dominant administrative hierarchy associated with each location",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
)

ax.grid(
    True,
    which="major",
    axis="both",
    alpha=0.20,
)

ax.legend(
    title="Geographic hierarchy",
    frameon=False,
    loc="upper right",
)

fig.text(
    0.01,
    0.01,
    f"{n_locations:,} unique locations | "
    f"{n_mentions:,} location-component mentions",
    fontsize=9,
)

fig.tight_layout(rect=(0, 0.035, 1, 1))

fig.savefig(
    PNG,
    dpi=220,
    bbox_inches="tight",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)

plt.close(fig)

print()
print("=" * 72)
print("PLOT 5 COMPLETE")
print("=" * 72)
print("PNG:", PNG)
print("PDF:", PDF)
print("CSV:", CSV)
