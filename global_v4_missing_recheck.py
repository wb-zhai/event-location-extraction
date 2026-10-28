#!/usr/bin/env python3
"""Read-only recheck of the 36 missing Global V4 source articles.

Reads the original missing_manifest.jsonl and inspects the actual top-level
article body. Does not rerun inference, modify GCS, or change the audit database.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time


def gcloud(*args):
    last = ""
    for attempt in range(3):
        result = subprocess.run(
            ["gcloud", *args], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout
        last = result.stderr
        if attempt < 2:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last[-1200:])


def metadata(uri):
    return json.loads(gcloud("storage", "objects", "describe", uri, "--format=json"))


def snapshot(meta):
    return {k: meta.get(k) for k in ("generation", "size", "crc32c", "md5Hash")}


def inspect_article(obj, expected_id, min_chars):
    if not isinstance(obj, dict):
        return {"status": "non_object_json", "root_type": type(obj).__name__}
    # Event Registry article JSON contains body at the root. "source" is
    # publisher metadata, not the article body.
    body = obj.get("body")
    title = obj.get("title")
    result = {
        "root_keys": sorted(obj.keys()),
        "root_uri": obj.get("uri"),
        "id_matches_source_uri": str(obj.get("uri")) == str(expected_id),
        "body_type": type(body).__name__,
        "body_chars": len(body) if isinstance(body, str) else None,
        "body_chars_stripped": len(body.strip()) if isinstance(body, str) else None,
        "title_type": type(title).__name__,
        "title_chars": len(title) if isinstance(title, str) else None,
    }
    if not isinstance(body, str):
        result["status"] = "body_missing_or_non_string"
    elif not body.strip():
        result["status"] = "empty_body"
    elif len(body.strip()) < min_chars:
        result["status"] = "short_body"
    else:
        result["status"] = "body_meets_basic_length_check"
    return result


def run(args):
    work = Path(args.workdir).resolve()
    source = work / "missing_diagnostic" / "missing_manifest.jsonl"
    target = work / "missing_diagnostic" / "corrected_recheck"
    if not source.is_file():
        raise SystemExit(f"Missing input: {source}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(r["id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate IDs in recovery manifest.")
    if len(rows) != args.expected:
        raise SystemExit(f"Expected {args.expected} records, found {len(rows)}.")
    results = []
    with tempfile.TemporaryDirectory(prefix="v4-recheck-") as scratch:
        for index, row in enumerate(rows, 1):
            uri = row.get("gcs_path")
            result = {
                "id": str(row["id"]),
                "manifest_shard": row.get("manifest_shard", ""),
                "language": row.get("language", ""),
                "publish_date": row.get("publish_date", ""),
                "gcs_path": uri,
            }
            try:
                if not isinstance(uri, str) or not uri.startswith("gs://"):
                    raise ValueError("Missing or invalid GCS source path")
                before = metadata(uri)
                result["source_generation"] = before.get("generation")
                result["source_bytes"] = before.get("size")
                if int(before.get("size", 0)) > args.max_bytes:
                    result["status"] = "source_exceeds_inspection_limit"
                else:
                    local = Path(scratch) / f"source-{index}.json"
                    gcloud("storage", "cp", uri, str(local))
                    raw = local.read_bytes()
                    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
                    obj = json.loads(raw)
                    result.update(inspect_article(obj, row["id"], args.min_chars))
                if snapshot(before) != snapshot(metadata(uri)):
                    raise RuntimeError("Source object changed during inspection")
            except Exception as exc:
                result["status"] = "read_or_parse_error"
                result["error"] = str(exc)[-600:]
            results.append(result)
            print(f"[{index}/{len(rows)}] {row['id']}: {result['status']}", flush=True)

    target.mkdir(parents=True, exist_ok=True)
    (target / "source_diagnostics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["id", "manifest_shard", "language", "publish_date", "gcs_path",
              "status", "body_type", "body_chars", "body_chars_stripped",
              "title_chars", "id_matches_source_uri", "source_bytes",
              "source_generation", "error"]
    with (target / "source_diagnostics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    counts = collections.Counter(r["status"] for r in results)
    summary = {
        "checked": len(results), "status_counts": dict(counts),
        "basic_min_body_chars": args.min_chars,
        "note": "This is a source-body diagnostic, not an exact replay of the inference loader. Original skip causes remain to be confirmed from code/logs."
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = "GLOBAL V4 SOURCE RECHECK\n" + "\n".join(
        f"{key}: {value}" for key, value in sorted(counts.items())) + "\n"
    text += "No inference was run. No original outputs were changed.\n"
    (target / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text + "Reports: " + str(target))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default="global_v4_audit")
    parser.add_argument("--expected", type=int, default=36)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-bytes", type=int, default=10_000_000)
    raise SystemExit(run(parser.parse_args()))
