#!/usr/bin/env python3
"""
Compare V1/V2/V3 event-extraction outputs on the same article corpus.

Works with either:
  - gs://... prefixes containing shard-*.jsonl
  - local directories containing shard-*.jsonl
  - a single local/GCS JSONL file

It streams JSONL and uses SQLite for article-level joins, so it does not need to
hold 1.38M articles from all three runs in RAM.

Outputs:
  <out>/summary.json
  <out>/model_summary.csv
  <out>/event_type_counts.csv
  <out>/pairwise_article_comparison.csv
  <out>/coverage.csv

Default sources are the current South Sudan V1/V2/V3 runs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator


DEFAULTS = {
    "v1": "gs://zhai-risk-factor-extraction/data/db/predictions/south_sudan_articles_a100/",
    "v2": "gs://zhai-risk-factor-extraction/extraction/output/south-sudan-v2/",
    "v3": "gs://zhai-risk-factor-extraction/extraction/output/south-sudan-v3/",
}


def run_capture(args: list[str]) -> str:
    p = subprocess.run(args, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(args)}\n{p.stderr}"
        )
    return p.stdout


def list_inputs(source: str) -> list[str]:
    if source.startswith("gs://"):
        if source.endswith(".jsonl"):
            return [source]
        prefix = source.rstrip("/")
        out = run_capture(["gcloud", "storage", "ls", f"{prefix}/shard-*.jsonl"])
        files = [x.strip() for x in out.splitlines() if x.strip().endswith(".jsonl")]
        if not files:
            raise RuntimeError(f"No shard-*.jsonl files found under {source}")
        return sorted(files)

    p = Path(source)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(str(x) for x in p.glob("shard-*.jsonl"))
        if not files:
            files = sorted(str(x) for x in p.glob("*.jsonl"))
        if not files:
            raise RuntimeError(f"No JSONL files found under {source}")
        return files

    files = sorted(glob.glob(source))
    if not files:
        raise RuntimeError(f"No inputs matched: {source}")
    return files


@contextmanager
def open_text(path: str):
    if path.startswith("gs://"):
        proc = subprocess.Popen(
            ["gcloud", "storage", "cat", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1024 * 1024,
        )
        assert proc.stdout is not None
        try:
            yield proc.stdout
        finally:
            proc.stdout.close()
            stderr = proc.stderr.read() if proc.stderr else ""
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"gcloud storage cat failed for {path}:\n{stderr}")
    else:
        with open(path, "r", encoding="utf-8") as fh:
            yield fh


def percentile_sorted(xs: list[int], q: float) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def norm_id(v) -> str:
    if v is None:
        return ""
    return str(v)


def model_col(model: str) -> tuple[str, str]:
    return f"n_{model}", f"has_{model}"


def process_model(
    conn: sqlite3.Connection,
    model: str,
    source: str,
) -> dict:
    n_col, has_col = model_col(model)
    files = list_inputs(source)

    total_rows = 0
    bad_json = 0
    missing_id = 0
    duplicate_ids = 0
    total_events = 0
    empty_articles = 0
    event_counts: list[int] = []
    label_counts: Counter[str] = Counter()

    print(f"\n[{model.upper()}] {source}", file=sys.stderr)
    print(f"  {len(files)} input file(s)", file=sys.stderr)

    cur = conn.cursor()
    cur.execute("BEGIN")

    for fi, path in enumerate(files, 1):
        print(f"  [{fi:02d}/{len(files):02d}] {path}", file=sys.stderr)
        with open_text(path) as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad_json += 1
                    continue

                rid = norm_id(rec.get("id"))
                if not rid:
                    missing_id += 1
                    continue

                preds = rec.get("predictions") or []
                if not isinstance(preds, list):
                    preds = []

                n = len(preds)
                has = int(n > 0)
                total_rows += 1
                total_events += n
                empty_articles += int(n == 0)
                event_counts.append(n)

                for ev in preds:
                    if isinstance(ev, dict):
                        label = str(ev.get("event_type") or "").strip()
                        if label:
                            label_counts[label] += 1

                # Create the row if it does not exist, then update this model.
                cur.execute(
                    "INSERT OR IGNORE INTO articles(id) VALUES (?)",
                    (rid,),
                )
                before = conn.total_changes
                cur.execute(
                    f"""
                    UPDATE articles
                       SET {n_col} = ?,
                           {has_col} = ?,
                           seen_{model} = seen_{model} + 1
                     WHERE id = ?
                    """,
                    (n, has, rid),
                )

                # seen_model > 1 means the id appeared more than once in that run.
                cur.execute(
                    f"SELECT seen_{model} FROM articles WHERE id = ?",
                    (rid,),
                )
                seen = cur.fetchone()[0]
                if seen > 1:
                    duplicate_ids += 1

                if total_rows % 100_000 == 0:
                    conn.commit()
                    cur.execute("BEGIN")
                    print(
                        f"    {total_rows:,} rows, {total_events:,} events",
                        file=sys.stderr,
                    )

    conn.commit()

    event_counts.sort()
    n_articles = len(event_counts)
    summary = {
        "model": model,
        "source": source,
        "files": len(files),
        "rows": total_rows,
        "bad_json": bad_json,
        "missing_id": missing_id,
        "duplicate_id_rows": duplicate_ids,
        "articles_with_events": n_articles - empty_articles,
        "empty_articles": empty_articles,
        "empty_rate": (empty_articles / n_articles) if n_articles else 0.0,
        "total_events": total_events,
        "mean_events_per_article": (total_events / n_articles) if n_articles else 0.0,
        "median_events_per_article": statistics.median(event_counts) if event_counts else 0.0,
        "p95_events_per_article": percentile_sorted(event_counts, 0.95),
        "p99_events_per_article": percentile_sorted(event_counts, 0.99),
        "unique_event_types": len(label_counts),
        "label_counts": dict(label_counts),
    }
    return summary


def pairwise(conn: sqlite3.Connection, a: str, b: str) -> dict:
    na, ha = model_col(a)
    nb, hb = model_col(b)
    row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS common,
            SUM(CASE WHEN {na} = {nb} THEN 1 ELSE 0 END) AS same_count,
            AVG(ABS({na} - {nb})) AS mean_abs_count_diff,
            SUM(CASE WHEN {ha}=1 AND {hb}=1 THEN 1 ELSE 0 END) AS both_positive,
            SUM(CASE WHEN {ha}=0 AND {hb}=0 THEN 1 ELSE 0 END) AS both_empty,
            SUM(CASE WHEN {ha}=1 AND {hb}=0 THEN 1 ELSE 0 END) AS a_only_positive,
            SUM(CASE WHEN {ha}=0 AND {hb}=1 THEN 1 ELSE 0 END) AS b_only_positive,
            SUM({na}) AS events_a,
            SUM({nb}) AS events_b
        FROM articles
        WHERE seen_{a} > 0 AND seen_{b} > 0
        """
    ).fetchone()

    common = row[0] or 0
    return {
        "pair": f"{a}_vs_{b}",
        "common_articles": common,
        "same_event_count_articles": row[1] or 0,
        "same_event_count_rate": ((row[1] or 0) / common) if common else 0.0,
        "mean_abs_event_count_difference": row[2] or 0.0,
        "both_positive": row[3] or 0,
        "both_empty": row[4] or 0,
        f"{a}_positive_{b}_empty": row[5] or 0,
        f"{a}_empty_{b}_positive": row[6] or 0,
        f"events_{a}_on_common": row[7] or 0,
        f"events_{b}_on_common": row[8] or 0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=DEFAULTS["v1"])
    ap.add_argument("--v2", default=DEFAULTS["v2"])
    ap.add_argument("--v3", default=DEFAULTS["v3"])
    ap.add_argument("--out", default="outputs/evaluation/comparison_v1_v2_v3")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "article_join.sqlite"

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            n_v1 INTEGER,
            has_v1 INTEGER,
            seen_v1 INTEGER NOT NULL DEFAULT 0,
            n_v2 INTEGER,
            has_v2 INTEGER,
            seen_v2 INTEGER NOT NULL DEFAULT 0,
            n_v3 INTEGER,
            has_v3 INTEGER,
            seen_v3 INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seen ON articles(seen_v1, seen_v2, seen_v3)")
    conn.commit()

    summaries = []
    for model, source in [("v1", args.v1), ("v2", args.v2), ("v3", args.v3)]:
        summaries.append(process_model(conn, model, source))

    pair_rows = [
        pairwise(conn, "v1", "v2"),
        pairwise(conn, "v1", "v3"),
        pairwise(conn, "v2", "v3"),
    ]

    coverage_row = conn.execute("""
        SELECT
            COUNT(*) AS union_articles,
            SUM(CASE WHEN seen_v1>0 THEN 1 ELSE 0 END) AS v1_articles,
            SUM(CASE WHEN seen_v2>0 THEN 1 ELSE 0 END) AS v2_articles,
            SUM(CASE WHEN seen_v3>0 THEN 1 ELSE 0 END) AS v3_articles,
            SUM(CASE WHEN seen_v1>0 AND seen_v2>0 AND seen_v3>0 THEN 1 ELSE 0 END) AS common_all_three,
            SUM(CASE WHEN seen_v1=0 AND (seen_v2>0 OR seen_v3>0) THEN 1 ELSE 0 END) AS missing_v1,
            SUM(CASE WHEN seen_v2=0 AND (seen_v1>0 OR seen_v3>0) THEN 1 ELSE 0 END) AS missing_v2,
            SUM(CASE WHEN seen_v3=0 AND (seen_v1>0 OR seen_v2>0) THEN 1 ELSE 0 END) AS missing_v3
        FROM articles
    """).fetchone()

    coverage = [{
        "union_articles": coverage_row[0] or 0,
        "v1_articles": coverage_row[1] or 0,
        "v2_articles": coverage_row[2] or 0,
        "v3_articles": coverage_row[3] or 0,
        "common_all_three": coverage_row[4] or 0,
        "missing_v1": coverage_row[5] or 0,
        "missing_v2": coverage_row[6] or 0,
        "missing_v3": coverage_row[7] or 0,
    }]

    flat_summaries = []
    label_rows = []
    for s in summaries:
        labels = s.pop("label_counts")
        flat_summaries.append(s)
        for label, count in sorted(labels.items(), key=lambda x: (-x[1], x[0])):
            label_rows.append({
                "model": s["model"],
                "event_type": label,
                "count": count,
                "share_of_events": count / s["total_events"] if s["total_events"] else 0.0,
            })
        s["label_counts"] = labels

    payload = {
        "coverage": coverage[0],
        "models": summaries,
        "pairwise": pair_rows,
    }

    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    write_csv(out / "model_summary.csv", flat_summaries)
    write_csv(out / "event_type_counts.csv", label_rows)
    write_csv(out / "pairwise_article_comparison.csv", pair_rows)
    write_csv(out / "coverage.csv", coverage)

    print("\n=== COVERAGE ===")
    for k, v in coverage[0].items():
        print(f"{k:24s} {v:,}" if isinstance(v, int) else f"{k:24s} {v}")

    print("\n=== MODEL SUMMARY ===")
    for s in summaries:
        print(
            f"{s['model'].upper()}: rows={s['rows']:,}  "
            f"events={s['total_events']:,}  "
            f"positive_articles={s['articles_with_events']:,}  "
            f"empty={s['empty_rate']:.2%}  "
            f"mean_events/article={s['mean_events_per_article']:.3f}  "
            f"labels={s['unique_event_types']}"
        )

    print("\n=== PAIRWISE ===")
    for r in pair_rows:
        print(
            f"{r['pair']}: common={r['common_articles']:,}  "
            f"same_count={r['same_event_count_rate']:.2%}  "
            f"mean_abs_diff={r['mean_abs_event_count_difference']:.3f}"
        )

    print(f"\nWrote comparison artifacts to: {out}")
    print("Keep article_join.sqlite until the comparison is finished.")

    conn.close()


if __name__ == "__main__":
    main()
