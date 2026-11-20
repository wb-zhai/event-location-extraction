#!/usr/bin/env python3
"""
Create clean correlation heatmaps for the V3 -> V4 training-sample/F1 analysis.

Input:
    data/extraction/eval/v4_reports/training_f1_correlation/
        training_sample_change_vs_f1.csv

Default filter:
    evaluation support >= 20

Outputs:
    correlation_heatmap_french.png
    correlation_heatmap_english.png
    correlation_summary_pearson.png
    correlation_summary_spearman.png

No regression/scatter lines are drawn.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATRIX_COLUMNS = [
    ("delta_train_samples", "Δ train\nwindows"),
    ("pct_increase_train_samples", "% Δ train\nwindows"),
    ("delta_train_mentions", "Δ train\nmentions"),
    ("delta_relaxed_f1", "Δ relaxed\nF1"),
    ("delta_exact_f1", "Δ exact\nF1"),
]

SUMMARY_COMPARISONS = [
    ("delta_train_samples", "delta_relaxed_f1", "Δ windows vs\nΔ relaxed F1"),
    ("pct_increase_train_samples", "delta_relaxed_f1", "% Δ windows vs\nΔ relaxed F1"),
    ("delta_train_samples", "delta_exact_f1", "Δ windows vs\nΔ exact F1"),
    ("pct_increase_train_samples", "delta_exact_f1", "% Δ windows vs\nΔ exact F1"),
]


def correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def prepare(df: pd.DataFrame, language: str, min_support: int) -> pd.DataFrame:
    out = df[
        (df["language"] == language)
        & (df["eval_support"] >= min_support)
    ].copy()

    for col, _ in MATRIX_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def annotate_matrix(ax, matrix: np.ndarray):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            value = matrix[i, j]
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=11)


def plot_language_matrix(
    df: pd.DataFrame,
    language: str,
    min_support: int,
    method: str,
    output: Path,
):
    subset = prepare(df, language, min_support)

    cols = [c for c, _ in MATRIX_COLUMNS]
    labels = [label for _, label in MATRIX_COLUMNS]

    matrix = subset[cols].corr(method=method).to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(matrix, vmin=-1, vmax=1)

    annotate_matrix(ax, matrix)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="x", rotation=35)

    ax.set_title(
        f"{language}: {method.capitalize()} correlation matrix\n"
        f"Event types with evaluation support ≥ {min_support} (n={len(subset)})"
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Correlation coefficient")

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    df: pd.DataFrame,
    min_support: int,
    method: str,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    languages = ["English", "French"]
    columns = [label for _, _, label in SUMMARY_COMPARISONS]

    matrix = []
    ns = []

    for language in languages:
        subset = prepare(df, language, min_support)
        ns.append(len(subset))

        row = []
        for x_col, y_col, _ in SUMMARY_COMPARISONS:
            row.append(correlation(subset[x_col], subset[y_col], method))
        matrix.append(row)

    return np.array(matrix, dtype=float), languages, columns, ns


def plot_summary(
    df: pd.DataFrame,
    min_support: int,
    method: str,
    output: Path,
):
    matrix, languages, columns, ns = build_summary(df, min_support, method)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    image = ax.imshow(matrix, vmin=-1, vmax=1, aspect="auto")

    annotate_matrix(ax, matrix)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(
        [f"{lang} (n={n})" for lang, n in zip(languages, ns)]
    )
    ax.tick_params(axis="x", rotation=20)

    ax.set_title(
        f"{method.capitalize()} correlation: training-sample change vs F1 change\n"
        f"Event types with evaluation support ≥ {min_support}"
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Correlation coefficient")

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "data/extraction/eval/v4_reports/training_f1_correlation/"
            "training_sample_change_vs_f1.csv"
        ),
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "data/extraction/eval/v4_reports/training_f1_correlation/heatmaps"
        ),
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"Missing input CSV: {args.csv}")

    df = pd.read_csv(args.csv)

    required = {
        "language",
        "event_type",
        "eval_support",
        "delta_train_samples",
        "pct_increase_train_samples",
        "delta_train_mentions",
        "delta_relaxed_f1",
        "delta_exact_f1",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Full correlation matrices. Pearson is the main version for Philipp's request.
    plot_language_matrix(
        df,
        "French",
        args.min_support,
        "pearson",
        args.out_dir / "correlation_heatmap_french.png",
    )
    plot_language_matrix(
        df,
        "English",
        args.min_support,
        "pearson",
        args.out_dir / "correlation_heatmap_english.png",
    )

    # Compact report-ready summaries.
    plot_summary(
        df,
        args.min_support,
        "pearson",
        args.out_dir / "correlation_summary_pearson.png",
    )
    plot_summary(
        df,
        args.min_support,
        "spearman",
        args.out_dir / "correlation_summary_spearman.png",
    )

    print(f"Filter: evaluation support >= {args.min_support}")
    print()
    for method in ("pearson", "spearman"):
        matrix, languages, columns, ns = build_summary(
            df, args.min_support, method
        )
        print(f"===== {method.upper()} =====")
        for i, language in enumerate(languages):
            print(f"{language} (n={ns[i]})")
            for j, col in enumerate(columns):
                print(f"  {col.replace(chr(10), ' '):<34} {matrix[i, j]:+.3f}")
        print()

    print("Wrote:")
    for path in sorted(args.out_dir.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
