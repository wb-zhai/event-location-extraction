#!/usr/bin/env python3
"""Compute event_type co-occurrence counts within the same article and plot
them as a heatmap PNG.

Co-occurrence here means: for each article, the set of distinct event_types
present in annotation.events; every unordered pair of distinct types in that
set gets +1, and the diagonal (type, type) is the number of articles
containing at least one event of that type.

Requires matplotlib/numpy. If the project venv doesn't have them, run via uv
without touching the venv:

    uv run --with matplotlib --with numpy \
        python scripts/event_extraction/generation/cooccurrence.py \
        --input dataset/extraction/data/en_5k.relevance.annotated.jsonl \
        --output /tmp/cooccurrence.png \
        --top-n 25

Pass --json to also dump the raw matrix (e.g. to feed an interactive heatmap).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.event_extraction.generation.io_utils import iter_jsonl, resolve_path
from scripts.event_extraction.generation.stats import get_events

# Sequential blue ramp, light -> dark (low -> high magnitude).
_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
_CMAP = LinearSegmentedColormap.from_list("sequential_blue", _RAMP)


def compute_matrix(path: Path, top_n: int | None) -> dict[str, Any]:
    doc_count: Counter = Counter()   # articles containing at least one event of this type
    pair_count: Counter = Counter()  # (type_a, type_b) unordered, a < b -> co-occurrence count
    n_articles_with_events = 0
    n_articles = 0

    for rec in iter_jsonl(path):
        n_articles += 1
        events = get_events(rec)
        types = sorted({(e.get("event_type") or "unknown") for e in events})
        if not types:
            continue
        n_articles_with_events += 1
        for t in types:
            doc_count[t] += 1
        for a, b in itertools.combinations(types, 2):
            pair_count[(a, b)] += 1

    top_types = [t for t, _ in doc_count.most_common(top_n)]  # top_n=None -> all types

    matrix: list[list[int]] = []
    for a in top_types:
        row = []
        for b in top_types:
            if a == b:
                row.append(doc_count[a])
            else:
                key = (a, b) if a < b else (b, a)
                row.append(pair_count[key])
        matrix.append(row)

    return {
        "input": str(path),
        "n_articles": n_articles,
        "n_articles_with_events": n_articles_with_events,
        "labels": top_types,
        "matrix": matrix,
    }


def overlap_matrix(matrix: np.ndarray) -> np.ndarray:
    """count / min(diag_X, diag_Y) per pair -- the overlap coefficient: what
    fraction of the *rarer* type's articles also carry the other type. Bounded
    [0, 1] and symmetric; the diagonal is trivially 1.0 (a type fully overlaps
    itself), so it carries no extra information here.
    """
    diag = np.diag(matrix)
    denom = np.minimum.outer(diag, diag)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(denom > 0, matrix / denom, 0.0)
    np.fill_diagonal(rate, 1.0)
    return rate


def plot_heatmap(data: dict[str, Any], output: Path, dpi: int, metric: str = "count") -> None:
    labels = data["labels"]
    raw = np.array(data["matrix"], dtype=float)
    n = len(labels)

    if metric == "overlap":
        matrix = overlap_matrix(raw)
        scaled = matrix.copy()  # already bounded [0, 1], no need to compress the range
        np.fill_diagonal(scaled, np.nan)  # trivially 1.0 -- gray it out rather than let it dominate as "darkest"
    else:
        matrix = raw
        # sqrt scale so mid-range pairs stay visible next to the dominant pairs
        scaled = np.sqrt(matrix)

    cmap = _CMAP.copy()
    cmap.set_bad(color="#d8d7d0")

    fig, ax = plt.subplots(figsize=(0.42 * n + 3, 0.42 * n + 2.5))
    im = ax.imshow(scaled, cmap=cmap, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)

    # outline the diagonal (self-frequency / trivial self-overlap, not a real co-occurrence)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="white", linewidth=1.2))

    # thin gridlines between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=1)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    # if metric == "overlap":
    #     cbar.set_label("Co-occurrence", fontsize=9)
    #     cbar.ax.tick_params(labelsize=8)
    #     cbar.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    # else:
    #     cbar.set_label("Articles (sqrt scale)", fontsize=9)
    #     # label the colorbar in original counts, not sqrt units
    #     max_val = matrix.max()
    #     ticks = np.linspace(0, np.sqrt(max_val), 5)
    #     cbar.set_ticks(ticks)
    #     cbar.set_ticklabels([f"{int(round(t ** 2))}" for t in ticks])
    #     cbar.ax.tick_params(labelsize=8)

    n_articles = data.get("n_articles")
    n_with_events = data.get("n_articles_with_events")
    subtitle = f"{n_with_events}/{n_articles} articles with ≥1 event" if n_articles else ""
    if metric == "overlap":
        title = (
            f"Event type co-occurrence\n{subtitle}\n"
            # "cell = % of the rarer type's articles that also have the other type; diagonal grayed out (trivially 100%)"
        )
    else:
        title = (
            f"Event type co-occurrence -- raw counts (within-article)\n{subtitle}\n"
            "diagonal outlined = # articles with that type alone"
        )
    ax.set_title(title, fontsize=10, pad=12)

    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-type co-occurrence heatmap (within-article).")
    parser.add_argument("--input", required=True, help="Path to a JSONL file.")
    parser.add_argument("--output", required=True, help="Output image path (.png).")
    parser.add_argument("--top-n", type=int, default=None, help="Number of most-frequent event types to keep (default: all).")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--json", default=None, help="Also write the raw co-occurrence matrix to this JSON path.")
    parser.add_argument(
        "--metric", choices=["count", "overlap"], default="count",
        help="'count' = raw # articles (default); 'overlap' = count / min(diag_X, diag_Y), "
             "i.e. the fraction of the rarer type's articles that also show the other type.",
    )
    args = parser.parse_args()

    path = resolve_path(args.input)
    data = compute_matrix(path, args.top_n)
    print(f"{len(data['labels'])} types, {data['n_articles_with_events']} articles with events "
          f"out of {data['n_articles']}.")

    if args.json:
        json_path = Path(args.json)
        json_path.write_text(json.dumps(data, indent=2))
        print(f"Wrote {json_path}")

    output_path = Path(args.output)
    plot_heatmap(data, output_path, args.dpi, metric=args.metric)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
