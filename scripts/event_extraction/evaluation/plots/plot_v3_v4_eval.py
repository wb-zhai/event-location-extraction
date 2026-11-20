#!/usr/bin/env python3
"""
Create V3 vs V4 evaluation plots from the frozen English/French Teacher reports.

Run from the event-location-extraction repo root:

    python scripts/event_extraction/evaluation/plots/plot_v3_v4_eval.py

Default inputs:
    data/extraction/eval/v4_reports/v3_english.txt
    data/extraction/eval/v4_reports/v4_english.txt
    data/extraction/eval/v4_reports/v3_french.txt
    data/extraction/eval/v4_reports/v4_french.txt
    data/extraction/eval/v4_reports/v3_vs_v4_english_no_event.txt
    data/extraction/eval/v4_reports/v3_vs_v4_french_no_event.txt

Outputs:
    data/extraction/eval/v4_reports/plots/
        01_english_core_metrics.png
        02_french_core_metrics.png
        03_english_no_event.png
        04_french_no_event.png
        05_english_per_label_f1_delta.png
        06_french_per_label_f1_delta.png
        v3_v4_core_metrics.csv
        v3_v4_no_event_metrics.csv
        v3_v4_per_label_metrics.csv

Notes:
- The legacy no-event comparison script prints its first input as "V2"
  and second input as "V3". Here those are intentionally relabeled as
  V3 and V4.
- "Macro" in eval_v3_sft.py is a per-document macro. For the plots below,
  the main extraction/field charts use the MICRO pooled metrics.
- Per-label macro F1 can be recovered from v3_v4_per_label_metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


MICRO_METRICS = [
    "Event extraction",
    "Location extraction",
    "End-to-end event-location",
    "End-to-end event-time",
    "End-to-end event-admin-level",
]

DISPLAY = {
    "Event extraction": "Event extraction",
    "Location extraction": "Location",
    "End-to-end event-location": "Event + location",
    "End-to-end event-time": "Event + time",
    "End-to-end event-admin-level": "Event + admin",
}


def parse_eval_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    out = {"micro": {}, "per_label": {}}

    # -------------------------
    # MICRO pooled-count table
    # -------------------------
    in_micro = False
    for line in lines:
        if "MICRO  (pooled counts)" in line:
            in_micro = True
            continue
        if in_micro and "MACRO  (per-document average)" in line:
            break
        if not in_micro:
            continue

        for metric in MICRO_METRICS:
            if line.strip().startswith(metric):
                # Six numeric columns: relaxed P/R/F1, exact P/R/F1.
                nums = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)", line)
                # TP/FP/FN counts come after those six numbers, so only take first 6.
                if len(nums) >= 6:
                    p, r, f1, ep, er, ef1 = map(float, nums[:6])
                    out["micro"][metric] = {
                        "relaxed_precision": p,
                        "relaxed_recall": r,
                        "relaxed_f1": f1,
                        "exact_precision": ep,
                        "exact_recall": er,
                        "exact_f1": ef1,
                    }
                break

    # -------------------------
    # PER EVENT TYPE table
    # -------------------------
    start = None
    for i, line in enumerate(lines):
        if "PER EVENT TYPE" in line:
            start = i
            break

    if start is not None:
        # Typical row:
        # civilian casualties 1479 66.6 72.8 69.6 42.7 46.7 44.6 85.4 88.4 86.1
        row_re = re.compile(
            r"^\s*(.*?)\s+"
            r"(\d+)\s+"
            r"(\d+(?:\.\d+)?)\s+"
            r"(\d+(?:\.\d+)?)\s+"
            r"(\d+(?:\.\d+)?)\s+"
            r"(\d+(?:\.\d+)?)\s+"
            r"(\d+(?:\.\d+)?)\s+"
            r"(\d+(?:\.\d+)?)\s+"
            r"(\d+(?:\.\d+)?)\s+"
            r"(\d+(?:\.\d+)?)\s+"
            r"(\d+(?:\.\d+)?)\s*$"
        )
        for line in lines[start + 1 :]:
            if line.startswith("===="):
                break
            m = row_re.match(line)
            if not m:
                continue
            (
                label,
                support,
                rp,
                rr,
                rf1,
                ep,
                er,
                ef1,
                loc,
                time,
                adm,
            ) = m.groups()
            out["per_label"][label.strip()] = {
                "support": int(support),
                "relaxed_precision": float(rp),
                "relaxed_recall": float(rr),
                "relaxed_f1": float(rf1),
                "exact_precision": float(ep),
                "exact_recall": float(er),
                "exact_f1": float(ef1),
                "location_acc": float(loc),
                "time_acc": float(time),
                "admin_acc": float(adm),
            }

    missing = [m for m in MICRO_METRICS if m not in out["micro"]]
    if missing:
        raise RuntimeError(f"{path}: missing MICRO metrics: {missing}")
    if len(out["per_label"]) != 48:
        raise RuntimeError(
            f"{path}: expected 48 event labels, parsed {len(out['per_label'])}"
        )
    return out


def parse_no_event(path: Path) -> dict:
    """
    Parse the two sections emitted by compare_teacher_no_event.py.

    Important:
      first printed block ("V2") = V3 in this experiment
      second printed block ("V3") = V4 in this experiment
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"^===== .*? =====$", text, flags=re.MULTILINE)
    headers = re.findall(r"^===== (.*?) =====$", text, flags=re.MULTILINE)

    parsed = []
    for header, block in zip(headers, blocks[1:]):
        def val(pattern: str) -> float:
            m = re.search(pattern, block)
            if not m:
                raise RuntimeError(f"{path}: could not parse {pattern!r} in {header}")
            return float(m.group(1))

        parsed.append(
            {
                "legacy_header": header,
                "fp_rate": val(r"FP rate on Teacher-empty:\s+([\d.]+)%"),
                "fn_rate": val(r"FN rate on Teacher-positive:\s+([\d.]+)%"),
                "no_event_f1": val(r"no-event F1:\s+([\d.]+)%"),
                "mean_spurious": val(r"mean spurious events / FP empty:\s+([\d.]+)"),
                "total_spurious": val(r"total spurious events on empty:\s+([\d.]+)"),
            }
        )

    if len(parsed) != 2:
        raise RuntimeError(f"{path}: expected 2 no-event blocks, found {len(parsed)}")

    return {"V3": parsed[0], "V4": parsed[1]}


def add_bar_labels(ax, bars, fmt="{:.1f}", fontsize=8):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_core(language: str, v3: dict, v4: dict, out: Path):
    metrics = MICRO_METRICS
    labels = [DISPLAY[m] for m in metrics]

    v3_relaxed = [v3["micro"][m]["relaxed_f1"] for m in metrics]
    v4_relaxed = [v4["micro"][m]["relaxed_f1"] for m in metrics]
    v3_exact = [v3["micro"][m]["exact_f1"] for m in metrics]
    v4_exact = [v4["micro"][m]["exact_f1"] for m in metrics]

    # Keep relaxed and exact together, but visually separated by metric/model labels.
    x = list(range(len(metrics)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6.5))
    b1 = ax.bar([i - 1.5 * width for i in x], v3_relaxed, width, label="V3 relaxed")
    b2 = ax.bar([i - 0.5 * width for i in x], v4_relaxed, width, label="V4 relaxed")
    b3 = ax.bar([i + 0.5 * width for i in x], v3_exact, width, label="V3 exact")
    b4 = ax.bar([i + 1.5 * width for i in x], v4_exact, width, label="V4 exact")

    add_bar_labels(ax, b1)
    add_bar_labels(ax, b2)
    add_bar_labels(ax, b3)
    add_bar_labels(ax, b4)

    ax.set_title(f"V3 vs V4 on the held-out {language} Teacher benchmark")
    ax.set_ylabel("F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, max(v3_relaxed + v4_relaxed + v3_exact + v4_exact) + 12)
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_no_event(language: str, stats: dict, out: Path):
    metrics = [
        ("no_event_f1", "No-event F1"),
        ("fp_rate", "FP rate on\nTeacher-empty"),
        ("fn_rate", "FN rate on\nTeacher-positive"),
    ]
    x = list(range(len(metrics)))
    width = 0.34

    v3 = [stats["V3"][k] for k, _ in metrics]
    v4 = [stats["V4"][k] for k, _ in metrics]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    b1 = ax.bar([i - width / 2 for i in x], v3, width, label="V3")
    b2 = ax.bar([i + width / 2 for i in x], v4, width, label="V4")

    add_bar_labels(ax, b1, "{:.2f}")
    add_bar_labels(ax, b2, "{:.2f}")

    ax.set_title(f"No-event behavior on the held-out {language} benchmark")
    ax.set_ylabel("Percent")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, max(v3 + v4) + 12)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    footer = (
        f"Spurious events on Teacher-empty false positives: "
        f"V3={int(stats['V3']['total_spurious'])}, "
        f"V4={int(stats['V4']['total_spurious'])}"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_label_delta(language: str, v3: dict, v4: dict, out: Path):
    labels = sorted(v3["per_label"])
    rows = []
    for label in labels:
        if label not in v4["per_label"]:
            raise RuntimeError(f"{language}: label missing in V4: {label}")
        d = v4["per_label"][label]["relaxed_f1"] - v3["per_label"][label]["relaxed_f1"]
        rows.append((label, d))

    rows.sort(key=lambda x: x[1])
    names = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(y, deltas)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("V4 minus V3 relaxed F1 (points)")
    ax.set_title(f"Per-event-type F1 change on {language}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_core_csv(path: Path, reports: dict):
    fields = [
        "language",
        "metric",
        "model",
        "relaxed_precision",
        "relaxed_recall",
        "relaxed_f1",
        "exact_precision",
        "exact_recall",
        "exact_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for language, pair in reports.items():
            for model in ("V3", "V4"):
                rep = pair[model]
                for metric in MICRO_METRICS:
                    row = {
                        "language": language,
                        "metric": metric,
                        "model": model,
                        **rep["micro"][metric],
                    }
                    w.writerow(row)


def write_no_event_csv(path: Path, no_event: dict):
    fields = [
        "language",
        "model",
        "no_event_f1",
        "fp_rate",
        "fn_rate",
        "mean_spurious",
        "total_spurious",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for language, pair in no_event.items():
            for model in ("V3", "V4"):
                r = dict(pair[model])
                r.pop("legacy_header", None)
                w.writerow({"language": language, "model": model, **r})


def write_per_label_csv(path: Path, reports: dict):
    fields = [
        "language",
        "event_type",
        "support",
        "v3_relaxed_precision",
        "v3_relaxed_recall",
        "v3_relaxed_f1",
        "v4_relaxed_precision",
        "v4_relaxed_recall",
        "v4_relaxed_f1",
        "delta_relaxed_f1",
        "v3_exact_f1",
        "v4_exact_f1",
        "delta_exact_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for language, pair in reports.items():
            v3 = pair["V3"]["per_label"]
            v4 = pair["V4"]["per_label"]
            for label in sorted(v3):
                a = v3[label]
                b = v4[label]
                w.writerow(
                    {
                        "language": language,
                        "event_type": label,
                        "support": a["support"],
                        "v3_relaxed_precision": a["relaxed_precision"],
                        "v3_relaxed_recall": a["relaxed_recall"],
                        "v3_relaxed_f1": a["relaxed_f1"],
                        "v4_relaxed_precision": b["relaxed_precision"],
                        "v4_relaxed_recall": b["relaxed_recall"],
                        "v4_relaxed_f1": b["relaxed_f1"],
                        "delta_relaxed_f1": round(b["relaxed_f1"] - a["relaxed_f1"], 3),
                        "v3_exact_f1": a["exact_f1"],
                        "v4_exact_f1": b["exact_f1"],
                        "delta_exact_f1": round(b["exact_f1"] - a["exact_f1"], 3),
                    }
                )


def print_summary(reports: dict, no_event: dict):
    print()
    print("===== QUICK SUMMARY =====")
    for language in ("English", "French"):
        v3 = reports[language]["V3"]
        v4 = reports[language]["V4"]
        a = v3["micro"]["Event extraction"]
        b = v4["micro"]["Event extraction"]

        macro_v3 = mean(x["relaxed_f1"] for x in v3["per_label"].values())
        macro_v4 = mean(x["relaxed_f1"] for x in v4["per_label"].values())
        exact_macro_v3 = mean(x["exact_f1"] for x in v3["per_label"].values())
        exact_macro_v4 = mean(x["exact_f1"] for x in v4["per_label"].values())

        gains = sum(
            v4["per_label"][k]["relaxed_f1"] > v3["per_label"][k]["relaxed_f1"]
            for k in v3["per_label"]
        )
        losses = sum(
            v4["per_label"][k]["relaxed_f1"] < v3["per_label"][k]["relaxed_f1"]
            for k in v3["per_label"]
        )
        ties = 48 - gains - losses

        print(f"\n{language}")
        print(
            f"  Event extraction relaxed F1: "
            f"{a['relaxed_f1']:.1f} -> {b['relaxed_f1']:.1f} "
            f"({b['relaxed_f1']-a['relaxed_f1']:+.1f})"
        )
        print(
            f"  Event extraction exact F1:   "
            f"{a['exact_f1']:.1f} -> {b['exact_f1']:.1f} "
            f"({b['exact_f1']-a['exact_f1']:+.1f})"
        )
        print(
            f"  48-label relaxed macro F1:    "
            f"{macro_v3:.2f} -> {macro_v4:.2f} "
            f"({macro_v4-macro_v3:+.2f})"
        )
        print(
            f"  48-label exact macro F1:      "
            f"{exact_macro_v3:.2f} -> {exact_macro_v4:.2f} "
            f"({exact_macro_v4-exact_macro_v3:+.2f})"
        )
        print(f"  Per-label relaxed F1:         {gains} up / {losses} down / {ties} tied")
        print(
            f"  No-event F1:                  "
            f"{no_event[language]['V3']['no_event_f1']:.2f} -> "
            f"{no_event[language]['V4']['no_event_f1']:.2f}"
        )
        print(
            f"  Empty-doc FP rate:            "
            f"{no_event[language]['V3']['fp_rate']:.2f}% -> "
            f"{no_event[language]['V4']['fp_rate']:.2f}%"
        )
        print(
            f"  Positive-doc FN rate:         "
            f"{no_event[language]['V3']['fn_rate']:.2f}% -> "
            f"{no_event[language]['V4']['fn_rate']:.2f}%"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/extraction/eval/v4_reports"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <report-dir>/plots",
    )
    args = ap.parse_args()

    report_dir = args.report_dir
    out_dir = args.out_dir or report_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "English": {
            "V3": report_dir / "v3_english.txt",
            "V4": report_dir / "v4_english.txt",
        },
        "French": {
            "V3": report_dir / "v3_french.txt",
            "V4": report_dir / "v4_french.txt",
        },
    }
    no_event_files = {
        "English": report_dir / "v3_vs_v4_english_no_event.txt",
        "French": report_dir / "v3_vs_v4_french_no_event.txt",
    }

    for language, pair in files.items():
        for model, path in pair.items():
            if not path.exists():
                raise SystemExit(f"Missing {language} {model} report: {path}")
    for language, path in no_event_files.items():
        if not path.exists():
            raise SystemExit(f"Missing {language} no-event report: {path}")

    reports = {
        language: {model: parse_eval_report(path) for model, path in pair.items()}
        for language, pair in files.items()
    }
    no_event = {
        language: parse_no_event(path)
        for language, path in no_event_files.items()
    }

    plot_core(
        "English",
        reports["English"]["V3"],
        reports["English"]["V4"],
        out_dir / "01_english_core_metrics.png",
    )
    plot_core(
        "French",
        reports["French"]["V3"],
        reports["French"]["V4"],
        out_dir / "02_french_core_metrics.png",
    )
    plot_no_event(
        "English",
        no_event["English"],
        out_dir / "03_english_no_event.png",
    )
    plot_no_event(
        "French",
        no_event["French"],
        out_dir / "04_french_no_event.png",
    )
    plot_per_label_delta(
        "English",
        reports["English"]["V3"],
        reports["English"]["V4"],
        out_dir / "05_english_per_label_f1_delta.png",
    )
    plot_per_label_delta(
        "French",
        reports["French"]["V3"],
        reports["French"]["V4"],
        out_dir / "06_french_per_label_f1_delta.png",
    )

    write_core_csv(out_dir / "v3_v4_core_metrics.csv", reports)
    write_no_event_csv(out_dir / "v3_v4_no_event_metrics.csv", no_event)
    write_per_label_csv(out_dir / "v3_v4_per_label_metrics.csv", reports)

    print_summary(reports, no_event)

    print()
    print("Wrote:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p}")


if __name__ == "__main__":
    main()
