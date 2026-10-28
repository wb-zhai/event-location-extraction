#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def pearson(x, y):
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)
    return sx.corr(sy, method="pearson")


def spearman(x, y):
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)
    return sx.corr(sy, method="spearman")


def regression_line(x, y):
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)
    slope = sx.cov(sy) / sx.var() if sx.var() != 0 else 0.0
    intercept = sy.mean() - slope * sx.mean()
    return slope, intercept


def pick_labels(df, n_each_side=5):
    pos = df.nlargest(n_each_side, "delta_relaxed_f1")
    neg = df.nsmallest(n_each_side, "delta_relaxed_f1")
    hi_support = df.nlargest(5, "eval_support")
    return pd.concat([pos, neg, hi_support]).drop_duplicates(subset=["event_type"])


def main():
    ap = argparse.ArgumentParser(
        description="Create a clean correlation plot between training-sample change and F1 change."
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to training_sample_change_vs_f1.csv",
    )
    ap.add_argument(
        "--language",
        default="French",
        choices=["French", "English"],
    )
    ap.add_argument(
        "--min-support",
        type=int,
        default=20,
        help="Minimum evaluation support per event type.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output PNG path.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = {
        "language",
        "event_type",
        "eval_support",
        "pct_increase_train_samples",
        "delta_relaxed_f1",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = df[df["language"] == args.language].copy()
    df = df[pd.notna(df["pct_increase_train_samples"])].copy()
    df = df[df["eval_support"] >= args.min_support].copy()

    if df.empty:
        raise SystemExit("No rows left after filtering.")

    df["x"] = df["pct_increase_train_samples"].astype(float)
    df["y"] = df["delta_relaxed_f1"].astype(float)

    # Point size reflects evaluation support, but is compressed for readability.
    df["size"] = 30 + 8 * df["eval_support"].clip(lower=1).pow(0.5)

    r = pearson(df["x"], df["y"])
    rho = spearman(df["x"], df["y"])
    slope, intercept = regression_line(df["x"], df["y"])

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(df["x"], df["y"], s=df["size"], alpha=0.8)

    xmin, xmax = df["x"].min(), df["x"].max()
    if xmin == xmax:
        xs = [xmin, xmax]
    else:
        pad = 0.03 * (xmax - xmin)
        xmin -= pad
        xmax += pad
        xs = [xmin + (xmax - xmin) * i / 100 for i in range(101)]

    ys = [slope * x + intercept for x in xs]
    ax.plot(xs, ys, linewidth=2)

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    label_df = pick_labels(df, n_each_side=5)
    for _, row in label_df.iterrows():
        ax.annotate(
            row["event_type"],
            (row["x"], row["y"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title(f"{args.language}: training increase vs F1 change")
    ax.set_xlabel("Increase in training windows containing the event type (%)")
    ax.set_ylabel("V4 minus V3 relaxed F1 (points)")

    note = (
        f"Eval support >= {args.min_support}\n"
        f"n = {len(df)} labels\n"
        f"Pearson r = {r:.3f}\n"
        f"Spearman rho = {rho:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
