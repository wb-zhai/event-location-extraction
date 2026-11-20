#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def load_sft(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"{path}: expected a JSON list")
    return data


def target_events(row: dict) -> list[dict]:
    raw = row.get("output", "")
    obj = raw if isinstance(raw, dict) else json.loads(raw)
    events = obj.get("events") or []
    return [e for e in events if isinstance(e, dict)]


def count_training_exposure(rows: list[dict]):
    sample_count = Counter()
    mention_count = Counter()
    for row in rows:
        labels_in_sample = set()
        for ev in target_events(row):
            label = ev.get("event_type")
            if not label:
                continue
            label = str(label)
            labels_in_sample.add(label)
            mention_count[label] += 1
        for label in labels_in_sample:
            sample_count[label] += 1
    return sample_count, mention_count


def rankdata(values):
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(x, y):
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if den == 0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / den


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def regression_line(x, y):
    mx, my = mean(x), mean(y)
    den = sum((v - mx) ** 2 for v in x)
    if den == 0:
        return 0.0, my
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / den
    return slope, my - slope * mx


def load_metrics(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{path}: no rows")
    return rows


def check_v4_contains_v3_prefix(v3: list[dict], v4: list[dict]):
    if len(v4) < len(v3):
        raise RuntimeError(f"V4 has fewer train rows than V3: {len(v4)} < {len(v3)}")
    keys = ("system", "instruction", "input", "output")
    mismatches = 0
    for i, (a, b) in enumerate(zip(v3, v4)):
        if any(a.get(k) != b.get(k) for k in keys):
            mismatches += 1
            if mismatches <= 3:
                print(f"WARNING: V3/V4 prefix mismatch at train row {i}")
    if mismatches:
        raise RuntimeError(
            f"V4 is not an exact V3 training prefix: {mismatches} mismatched rows."
        )


def build_joined_rows(metrics, v3_samples, v4_samples, v3_mentions, v4_mentions):
    out = []
    for r in metrics:
        label = r["event_type"]
        v3n = int(v3_samples.get(label, 0))
        v4n = int(v4_samples.get(label, 0))
        delta = v4n - v3n
        pct = 100.0 * delta / v3n if v3n else float("nan")

        v3m = int(v3_mentions.get(label, 0))
        v4m = int(v4_mentions.get(label, 0))

        out.append({
            "language": r["language"],
            "event_type": label,
            "eval_support": int(float(r["support"])),
            "v3_train_samples": v3n,
            "v4_train_samples": v4n,
            "delta_train_samples": delta,
            "pct_increase_train_samples": pct,
            "v3_train_mentions": v3m,
            "v4_train_mentions": v4m,
            "delta_train_mentions": v4m - v3m,
            "v3_relaxed_f1": float(r["v3_relaxed_f1"]),
            "v4_relaxed_f1": float(r["v4_relaxed_f1"]),
            "delta_relaxed_f1": float(r["delta_relaxed_f1"]),
            "v3_exact_f1": float(r["v3_exact_f1"]),
            "v4_exact_f1": float(r["v4_exact_f1"]),
            "delta_exact_f1": float(r["delta_exact_f1"]),
        })
    return out


def correlation_stats(rows, xkey, ykey, min_support=None):
    use = [
        r for r in rows
        if (min_support is None or r["eval_support"] >= min_support)
        and math.isfinite(float(r[xkey]))
        and math.isfinite(float(r[ykey]))
    ]
    x = [float(r[xkey]) for r in use]
    y = [float(r[ykey]) for r in use]
    return len(use), pearson(x, y), spearman(x, y)


def scatter(rows, language, xkey, xlabel, out, min_support=20):
    use = [r for r in rows if r["language"] == language and math.isfinite(float(r[xkey]))]
    x = [float(r[xkey]) for r in use]
    y = [float(r["delta_relaxed_f1"]) for r in use]

    r = pearson(x, y)
    rho = spearman(x, y)
    slope, intercept = regression_line(x, y)

    supported = [z for z in use if z["eval_support"] >= min_support]
    x2 = [float(z[xkey]) for z in supported]
    y2 = [float(z["delta_relaxed_f1"]) for z in supported]
    r2 = pearson(x2, y2) if len(supported) >= 2 else float("nan")
    rho2 = spearman(x2, y2) if len(supported) >= 2 else float("nan")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(x, y)

    if x:
        xmin, xmax = min(x), max(x)
        xs = [xmin + (xmax - xmin) * i / 99 for i in range(100)] if xmax > xmin else [xmin]
        ax.plot(xs, [slope * xx + intercept for xx in xs])

    ax.axhline(0, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("V4 minus V3 relaxed F1 (points)")
    ax.set_title(f"{language}: training-sample change vs F1 change")

    for row in sorted(use, key=lambda z: abs(z["delta_relaxed_f1"]), reverse=True)[:10]:
        ax.annotate(
            row["event_type"],
            (float(row[xkey]), float(row["delta_relaxed_f1"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    note = (
        f"All labels: Pearson r={r:.3f}, Spearman rho={rho:.3f}, n={len(use)}\n"
        f"Eval support >= {min_support}: r={r2:.3f}, rho={rho2:.3f}, n={len(supported)}"
    )
    ax.text(
        0.02, 0.98, note,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "alpha": 0.12},
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    fields = [
        "language", "event_type", "eval_support",
        "v3_train_samples", "v4_train_samples", "delta_train_samples",
        "pct_increase_train_samples",
        "v3_train_mentions", "v4_train_mentions", "delta_train_mentions",
        "v3_relaxed_f1", "v4_relaxed_f1", "delta_relaxed_f1",
        "v3_exact_f1", "v4_exact_f1", "delta_exact_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = dict(r)
            if not math.isfinite(float(row["pct_increase_train_samples"])):
                row["pct_increase_train_samples"] = ""
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--v3-train",
        type=Path,
        default=Path("data/v3_tempered_sft_20pct_empty/train.json"),
    )
    ap.add_argument(
        "--v4-train",
        type=Path,
        default=Path("data/extraction/v4_sft_exact_v3_plus_french/train.json"),
    )
    ap.add_argument(
        "--metrics",
        type=Path,
        default=Path("data/extraction/eval/v4_reports/plots/v3_v4_per_label_metrics.csv"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/extraction/eval/v4_reports/training_f1_correlation"),
    )
    args = ap.parse_args()

    for p in (args.v3_train, args.v4_train, args.metrics):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frozen SFT training files...")
    v3 = load_sft(args.v3_train)
    v4 = load_sft(args.v4_train)
    print(f"V3 train windows: {len(v3):,}")
    print(f"V4 train windows: {len(v4):,}")
    print(f"Added V4 train windows: {len(v4)-len(v3):,}")

    print("\nChecking that V4 preserves the exact V3 training prefix...")
    check_v4_contains_v3_prefix(v3, v4)
    print("OK: exact V3 prefix preserved.")

    print("\nCounting per-event-type training exposure...")
    v3_samples, v3_mentions = count_training_exposure(v3)
    v4_samples, v4_mentions = count_training_exposure(v4)

    metrics = load_metrics(args.metrics)
    joined = build_joined_rows(metrics, v3_samples, v4_samples, v3_mentions, v4_mentions)
    write_csv(args.out_dir / "training_sample_change_vs_f1.csv", joined)

    for language in ("French", "English"):
        rows = [r for r in joined if r["language"] == language]

        n, r, rho = correlation_stats(rows, "delta_train_samples", "delta_relaxed_f1")
        nr, rr, rhor = correlation_stats(rows, "pct_increase_train_samples", "delta_relaxed_f1")
        ne, re, rhoe = correlation_stats(rows, "delta_train_samples", "delta_exact_f1")
        nh, rh, rhoh = correlation_stats(
            rows, "delta_train_samples", "delta_relaxed_f1", min_support=20
        )

        print(f"\n===== {language.upper()} =====")
        print(f"Added train windows vs relaxed F1 change: Pearson r={r:.3f}, Spearman rho={rho:.3f}, n={n}")
        print(f"% train-window increase vs relaxed F1 change: Pearson r={rr:.3f}, Spearman rho={rhor:.3f}, n={nr}")
        print(f"Added train windows vs exact F1 change: Pearson r={re:.3f}, Spearman rho={rhoe:.3f}, n={ne}")
        print(f"Added train windows vs relaxed F1 change, eval support >=20: Pearson r={rh:.3f}, Spearman rho={rhoh:.3f}, n={nh}")

    scatter(
        joined,
        "French",
        "delta_train_samples",
        "Added V4 training windows containing event type",
        args.out_dir / "01_french_added_train_samples_vs_f1_change.png",
    )
    scatter(
        joined,
        "French",
        "pct_increase_train_samples",
        "Increase in training windows containing event type (%)",
        args.out_dir / "02_french_relative_train_increase_vs_f1_change.png",
    )
    scatter(
        joined,
        "English",
        "delta_train_samples",
        "Added V4 training windows containing event type",
        args.out_dir / "03_english_added_train_samples_vs_f1_change.png",
    )

    print("\nLargest added training-sample counts:")
    fr_rows = [r for r in joined if r["language"] == "French"]
    for r in sorted(fr_rows, key=lambda z: z["delta_train_samples"], reverse=True)[:10]:
        print(
            f"  {r['event_type']:<38} "
            f"+{r['delta_train_samples']:>4} windows, "
            f"French dF1={r['delta_relaxed_f1']:+.1f}"
        )

    print("\nWrote:")
    for p in sorted(args.out_dir.iterdir()):
        print(f"  {p}")


if __name__ == "__main__":
    main()
