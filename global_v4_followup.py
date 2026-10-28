#!/usr/bin/env python3
"""Read-only Global V4 follow-up: missing-record diagnosis and temporal profiling.

Uses Python standard library and the authenticated gcloud CLI. Never changes
remote objects, starts inference, or modifies the original audit database.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

MANIFEST = "gs://zhai-risk-factor-extraction/extraction/manifests/global-v4-relevant-en-fr"
OUTPUT = "gs://zhai-risk-factor-extraction/extraction/output/global-v4-relevant-en-fr"
EXPECTED = 34390924
SHARDS = 200
VERSION = "1.0"


def command(*args, attempts=3):
    last = None
    for attempt in range(attempts):
        p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0:
            return p.stdout
        last = p.stderr.strip()
        if attempt + 1 < attempts:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{' '.join(args)} failed: {last[-1800:]}")


def download(uri, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if uri.startswith("gs://"):
        command("gcloud", "storage", "cp", uri, str(dest))
    else:
        import shutil
        shutil.copyfile(uri, dest)
    return dest


def object_metadata(uri):
    if not uri.startswith("gs://"):
        p = Path(uri)
        digest = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        return {"size": p.stat().st_size, "sha256": digest.hexdigest()}
    return json.loads(command("gcloud", "storage", "objects", "describe", uri, "--format=json"))


def snapshot(meta):
    return {k: meta.get(k) for k in ("generation", "size", "crc32c", "md5Hash")}


def normalized_id(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("invalid article ID")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("invalid article ID")


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def source_diagnostic(path):
    raw = Path(path).read_bytes()
    result = {"downloaded_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if not raw.strip():
        return {**result, "status": "empty_object"}
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        return {**result, "status": "not_json", "error": str(e)[:160]}
    if not isinstance(obj, dict):
        return {**result, "status": "unexpected_json_root", "root_type": type(obj).__name__}
    source = obj.get("source") if isinstance(obj.get("source"), dict) else obj
    result["root_keys"] = sorted(obj)
    result["source_keys"] = sorted(source)
    result["source_status"] = obj.get("status")
    result["source_id"] = obj.get("id", source.get("id"))
    title = source.get("title")
    result["title_chars"] = len(title) if isinstance(title, str) else None
    candidates = {}
    for key in ("text", "body", "content", "article_text"):
        val = source.get(key)
        if isinstance(val, str):
            candidates[key] = len(val.strip())
    result["text_field_lengths"] = candidates
    if candidates:
        result["status"] = "nonempty_text" if any(candidates.values()) else "empty_text"
    else:
        result["status"] = "text_shape_unrecognized"
    return result


def diagnose_missing(args):
    work = Path(args.workdir).resolve()
    out = work / "missing_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    with (work / "missing_article_ids.csv").open(newline="", encoding="utf-8") as f:
        missing = list(csv.DictReader(f))
    if len(missing) != len({normalized_id(r["id"]) for r in missing}):
        raise RuntimeError("Missing-ID CSV contains duplicate IDs.")
    if args.expected_missing is not None and len(missing) != args.expected_missing:
        raise RuntimeError(f"Expected {args.expected_missing} missing IDs, found {len(missing)}.")
    if args.audit_db:
        db = sqlite3.connect(f"file:{Path(args.audit_db).resolve().as_posix()}?mode=ro", uri=True)
        try:
            n = db.execute("SELECT COUNT(*) FROM manifest").fetchone()[0]
            if n != args.expected:
                raise RuntimeError(f"Audit manifest contains {n} IDs, expected {args.expected}.")
            for row in missing:
                known = db.execute("SELECT shard,language FROM manifest WHERE id=?", (normalized_id(row["id"]),)).fetchone()
                if known != (int(row["manifest_shard"]), row["language"]):
                    raise RuntimeError(f"Audit manifest disagrees for {row['id']}.")
                if db.execute("SELECT 1 FROM observed WHERE id=?", (normalized_id(row["id"]),)).fetchone():
                    raise RuntimeError(f"Missing ID {row['id']} is actually in observed.")
        finally:
            db.close()
    groups = collections.defaultdict(dict)
    for row in missing:
        groups[int(row["manifest_shard"])][normalized_id(row["id"])] = row
    found = {}
    source_meta = {}
    with tempfile.TemporaryDirectory(prefix="v4-missing-") as temp:
        for shard, wanted in sorted(groups.items()):
            uri = f"{args.manifest.rstrip('/')}/manifest-{shard:03d}.jsonl"
            meta_before = object_metadata(uri)
            local = download(uri, Path(temp) / f"manifest-{shard:03d}.jsonl")
            with local.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    aid = normalized_id(row.get("id"))
                    if aid in wanted:
                        if aid in found:
                            raise RuntimeError(f"Duplicate manifest entry for {aid}.")
                        found[aid] = row
                        source_meta[aid] = {"manifest_shard": shard, "manifest_line": line_no}
            if snapshot(meta_before) != snapshot(object_metadata(uri)):
                raise RuntimeError(f"Manifest changed during inspection: {uri}")
            print(f"Manifest {shard}: {len(wanted)} requested, {sum(x in found for x in wanted)} found", flush=True)
        if set(found) != {normalized_id(r["id"]) for r in missing}:
            raise RuntimeError("Some missing IDs could not be recovered from their manifests.")
        with (out / "missing_manifest.jsonl").open("w", encoding="utf-8") as f:
            for row in missing:
                f.write(json.dumps(found[normalized_id(row["id"])], ensure_ascii=False) + "\n")
        results = []
        for index, row in enumerate(missing, 1):
            aid = normalized_id(row["id"])
            item = found[aid]
            uri = item.get("gcs_path") or item.get("cloud_uri")
            result = {**row, **source_meta[aid], "gcs_path": uri, "publish_date": item.get("publish_date", "")}
            try:
                if not isinstance(uri, str) or not uri.startswith("gs://"):
                    raise ValueError("Missing or non-GCS source path")
                meta = object_metadata(uri)
                result["object_metadata"] = snapshot(meta)
                size = int(meta.get("size", 0))
                if size > args.max_source_bytes:
                    result["source_status"] = "too_large_to_inspect"
                else:
                    local = download(uri, Path(temp) / f"source-{index}.json")
                    result["source_diagnostic"] = source_diagnostic(local)
                    result["source_status"] = result["source_diagnostic"]["status"]
                    local.unlink()
                if snapshot(meta) != snapshot(object_metadata(uri)):
                    raise RuntimeError("Source object changed during inspection")
            except Exception as e:
                result["source_status"] = "read_error"
                result["error"] = str(e)[-500:]
            results.append(result)
            print(f"[{index}/{len(missing)}] {aid}: {result['source_status']}", flush=True)
    write_json(out / "source_diagnostics.json", results)
    fields = ["id", "manifest_shard", "manifest_line", "language", "publish_date",
              "gcs_path", "source_status", "error"]
    write_csv(out / "source_diagnostics.csv", results, fields)
    counts = collections.Counter(r["source_status"] for r in results)
    summary = {"requested": len(missing), "manifest_entries_found": len(found),
               "source_status_counts": dict(counts),
               "note": "Source health is diagnostic, not proof of the original skip cause. No inference was run."}
    write_json(out / "summary.json", summary)
    (out / "summary.txt").write_text(
        "GLOBAL V4 MISSING-RECORD DIAGNOSTIC\n"
        + f"Requested: {len(missing)}\nManifest entries recovered: {len(found)}\n"
        + "".join(f"{k}: {v}\n" for k, v in sorted(counts.items()))
        + "\nNo inference was run. Source health does not establish the original skip cause.\n",
        encoding="utf-8")
    print("\n" + (out / "summary.txt").read_text(encoding="utf-8"))
    print("Reports:", out)
    return 0


# Temporal profiling: strictly descriptive. No date conversion is applied.
DATE_YEAR = re.compile(r"^\d{4}$")
DATE_MONTH = re.compile(r"^\d{4}-\d{2}$")
DATE_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
YEARISH = r"\d{4}(?:-\d{2}(?:-\d{2})?)?"
RANGE = re.compile(rf"^({YEARISH})\s*(?:/|–|—| - | to | au | à )\s*({YEARISH})$", re.I)
TRAILING = re.compile(rf"^{YEARISH}[-/]+$")
PARTIAL = re.compile(r"^\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?$")
MISSING = {"", "not_stated", "null", "none", "n/a", "na", "unknown"}


def time_category(value):
    if not isinstance(value, str):
        return "missing_or_non_string", "missing_or_non_string"
    s = value.strip()
    if s.lower() in MISSING:
        return "not_stated", "not_stated"
    if DATE_YEAR.fullmatch(s):
        return ("valid_year", "iso") if 1 <= int(s) <= 9999 else ("invalid_date", "year")
    if DATE_MONTH.fullmatch(s):
        try:
            dt.date.fromisoformat(s + "-01")
            return "valid_month", "iso"
        except ValueError:
            return "invalid_date", "month"
    if DATE_DAY.fullmatch(s):
        try:
            dt.date.fromisoformat(s)
            return "valid_day", "iso"
        except ValueError:
            return "invalid_date", "day"
    if re.match(r"^\d{4}-\d{2}-\d{2}[T ]", s):
        try:
            dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return "valid_datetime", "iso"
        except ValueError:
            return "invalid_date", "datetime"
    if RANGE.fullmatch(s):
        return "range_candidate", "two_date_tokens"
    if TRAILING.fullmatch(s):
        return "truncated_date", "trailing_separator"
    if PARTIAL.fullmatch(s):
        return "non_iso_numeric_date", "numeric"
    if re.search(r"\d{4}", s) and re.search(r"[/;,–—]|\bto\b|\bau\b|\bà\b", s, re.I):
        return "complex_date_expression", "multiple_or_range"
    if re.search(r"\d{4}", s):
        return "text_with_year", "text"
    if re.search(r"[A-Za-zÀ-ÿ]", s):
        return "relative_or_textual", "text"
    return "unclassified", "other"


def temporal_db(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS progress (shard INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL, stats TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS values_count (category TEXT NOT NULL, value TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY(category,value))")
    db.execute("CREATE TABLE IF NOT EXISTS samples (category TEXT NOT NULL, sample_key TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(category,sample_key))")
    return db


def profile_shard(path, shard, db, sample_limit):
    counts = collections.Counter()
    values = collections.Counter()
    samples = collections.defaultdict(dict)
    def flush():
        if values:
            db.executemany("INSERT INTO values_count VALUES (?,?,?) ON CONFLICT(category,value) DO UPDATE SET n=n+excluded.n",
                           [(a,b,n) for (a,b),n in values.items()])
            values.clear()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            counts["article_rows"] += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("predictions"), list):
                    raise ValueError("Invalid output record")
            except (ValueError, TypeError):
                counts["invalid_rows"] += 1
                continue
            aid = row.get("id")
            for index, ev in enumerate(row["predictions"]):
                counts["raw_mentions"] += 1
                value = ev.get("event_time") if isinstance(ev, dict) else None
                category, pattern = time_category(value)
                counts[category] += 1
                if category.startswith("valid_") or category == "not_stated":
                    continue
                key = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
                values[(category, key)] += 1
                if len(values) >= 20000:
                    flush()
                sample_key = hashlib.sha256(f"{aid}:{index}:{key}".encode()).hexdigest()
                if len(samples[category]) < sample_limit:
                    samples[category][sample_key] = {
                        "id": aid, "shard": shard, "line": line_no, "event_index": index,
                        "category": category, "pattern": pattern, "event_time": value,
                        "event_type": ev.get("event_type") if isinstance(ev, dict) else None,
                        "time_status": ev.get("time_status") if isinstance(ev, dict) else None,
                        "grounding_quote": ev.get("grounding_quote") if isinstance(ev, dict) else None,
                        "event_location": ev.get("event_location") if isinstance(ev, dict) else None}
    flush()
    for category, group in samples.items():
        db.executemany("INSERT OR IGNORE INTO samples VALUES (?,?,?)",
                       [(category,k,json.dumps(v,ensure_ascii=False)) for k,v in group.items()])
    return counts


def profile_time(args):
    work = Path(args.workdir).resolve() / "time_profile"
    work.mkdir(parents=True, exist_ok=True)
    db = temporal_db(work / "profile.sqlite")
    try:
        with tempfile.TemporaryDirectory(prefix="v4-time-") as temp:
            for shard in range(args.shards):
                uri = f"{args.output.rstrip('/')}/shard-{shard}.jsonl"
                meta = object_metadata(uri)
                fingerprint = json.dumps(snapshot(meta), sort_keys=True)
                old = db.execute("SELECT fingerprint FROM progress WHERE shard=?", (shard,)).fetchone()
                if old:
                    if old[0] != fingerprint:
                        raise RuntimeError(f"Shard {shard} changed since the previous profile.")
                    print(f"[{shard+1}/{args.shards}] already complete", flush=True)
                    continue
                print(f"[{shard+1}/{args.shards}] downloading and profiling", flush=True)
                local = download(uri, Path(temp) / f"shard-{shard}.jsonl")
                with db:
                    counts = profile_shard(local, shard, db, args.sample_limit)
                    if snapshot(object_metadata(uri)) != snapshot(meta):
                        raise RuntimeError(f"Shard {shard} changed during profiling.")
                    db.execute("INSERT INTO progress VALUES (?,?,?)", (shard, fingerprint, json.dumps(counts)))
                local.unlink()
                print(f"  {counts['article_rows']:,} articles, {counts['raw_mentions']:,} mentions", flush=True)
                if (shard + 1) % 10 == 0:
                    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        total = collections.Counter()
        for (text,) in db.execute("SELECT stats FROM progress"):
            total.update(json.loads(text))
        n_values = db.execute("SELECT COUNT(*) FROM values_count").fetchone()[0]
        fields = ["category", "value", "count"]
        def value_records(limit=None):
            sql = "SELECT category,value,n FROM values_count ORDER BY n DESC, category,value"
            if limit is not None:
                sql += " LIMIT " + str(int(limit))
            for category, value, count in db.execute(sql):
                yield {"category":category,"value":value,"count":count}
        # Stream the full distribution from SQLite. Do not load millions of values into RAM.
        write_csv(work / "time_values.csv", value_records(), fields)
        write_csv(work / "time_values_top.csv", value_records(1000), fields)
        patterns = [{"category":a,"unique_values":b,"mentions":c} for a,b,c in db.execute(
            "SELECT category,COUNT(*),SUM(n) FROM values_count GROUP BY category ORDER BY SUM(n) DESC")]
        write_csv(work / "time_pattern_summary.csv", patterns,
                  ["category","unique_values","mentions"])
        # Keep the report small even when the full value distribution is large.
        samples = [json.loads(x[0]) for x in db.execute("SELECT payload FROM samples ORDER BY category,sample_key LIMIT 1000")]
        write_json(work / "time_samples.json", samples)
        write_json(work / "report.json", {"profile_version": VERSION, "processed_shards": args.shards,
                   "counts": dict(total), "unique_review_values": n_values,
                   "review_patterns": patterns,
                   "note": "Descriptive only. No dates were normalized; publication-date grounding and accuracy remain unverified."})
        text = "GLOBAL V4 TEMPORAL PROFILE\n" + "".join(f"{k}: {v:,}\n" for k,v in sorted(total.items()))
        text += f"Unique review values: {n_values:,}\nNo dates were normalized.\n"
        (work / "summary.txt").write_text(text,encoding="utf-8")
        print("\n" + text + "Reports: " + str(work))
        return 0
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["missing", "time"], help="Diagnostic to run")
    p.add_argument("--workdir", default="global_v4_audit")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--expected", type=int, default=EXPECTED)
    p.add_argument("--expected-missing", type=int, default=36)
    p.add_argument("--audit-db", default=None, help="Optional original audit.sqlite, opened read-only")
    p.add_argument("--shards", type=int, default=SHARDS)
    p.add_argument("--max-source-bytes", type=int, default=10_000_000)
    p.add_argument("--sample-limit", type=int, default=20)
    args = p.parse_args()
    return diagnose_missing(args) if args.mode == "missing" else profile_time(args)


if __name__ == "__main__":
    sys.exit(main())
