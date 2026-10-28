#!/usr/bin/env python3
"""Global V4 structural postprocessing, full-corpus and resumable.

Requires the existing global_v4_temporal_parse.py and completed audit files.
Preserves the original seven-field prediction contract. Writes only local files.
Does not run inference, geocode, verify grounding, or write to GCS/SQL.
Python 3.10+; standard library and an authenticated gcloud CLI only.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

VERSION = "1.0.0"
DEFAULT_MANIFEST = "gs://zhai-risk-factor-extraction/extraction/manifests/global-v4-relevant-en-fr"
DEFAULT_OUTPUT = "gs://zhai-risk-factor-extraction/extraction/output/global-v4-relevant-en-fr"
FIELDS = ("event_type", "grounding_quote", "event_location", "event_location_admin_level", "event_time", "time_status", "severity")
ADMIN = {"country", "state", "county", "city", "district", "not_stated"}
ENUMS = {"time_status": {"past", "ongoing", "forecast", "not_stated"},
         "severity": {"low", "medium", "high", "extreme", "not_stated"}}
TOKEN = r"\d{4}(?:-\d{2}(?:-\d{2})?)?"
RANGE = re.compile(rf"^({TOKEN})\s+(?:to|au|à)\s+({TOKEN})$", re.I)
YEAR_RANGE = re.compile(r"^(\d{4})-(\d{4})$")
OPEN_END = re.compile(rf"^({TOKEN})-$")
OPEN_START = re.compile(rf"^-({TOKEN})$")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_line(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def command(*argv):
    last = ""
    for attempt in range(3):
        p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0:
            return p.stdout
        last = p.stderr.strip()
        if attempt < 2:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Command failed: {' '.join(argv)}\n{last[-1500:]}")


def metadata(uri):
    if not uri.startswith("gs://"):
        p = Path(uri)
        return {"size": p.stat().st_size, "sha256": sha256(p)}
    obj = json.loads(command("gcloud", "storage", "objects", "describe", uri, "--format=json"))
    return {k: obj.get(k) for k in ("generation", "size", "crc32c", "md5Hash")}


def obtain(uri, scratch):
    if not uri.startswith("gs://"):
        p = Path(uri)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    dest = Path(scratch) / Path(uri).name
    command("gcloud", "storage", "cp", uri, str(dest))
    return dest


def aid(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("Invalid article ID")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("Invalid article ID")


def load_parser(path):
    spec = importlib.util.spec_from_file_location("global_v4_temporal_parser", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "parse_time", None)):
        raise ValueError("Temporal parser must export parse_time(value).")
    return module


def temporal(value, parser):
    parsed = parser.parse_time(value)
    if parsed["parse_status"] == "parsed_candidate" or not isinstance(value, str):
        return parsed
    s = value.strip()
    match = OPEN_END.fullmatch(s)
    if match:
        candidate = parser.parse_time(match.group(1) + "/")
        if candidate["parse_status"] == "parsed_candidate":
            candidate["raw_value"] = value
            return candidate
    match = OPEN_START.fullmatch(s)
    if match:
        candidate = parser.parse_time("/" + match.group(1))
        if candidate["parse_status"] == "parsed_candidate":
            candidate["raw_value"] = value
            return candidate
    match = YEAR_RANGE.fullmatch(s) or RANGE.fullmatch(s)
    if match:
        candidate = parser.parse_time(match.group(1) + "/" + match.group(2))
        if candidate["parse_status"] == "parsed_candidate":
            candidate["raw_value"] = value
            candidate["review_reason"] = "interval_semantics_require_source_review"
            return candidate
    return parsed


def validate_event(event, labels, aliases):
    if not isinstance(event, dict):
        return None, ["event_not_object"], []
    errors, changes, clean = [], [], {}
    for field in FIELDS:
        value = event.get(field)
        if field not in event:
            errors.append("missing_field." + field)
        elif not isinstance(value, str):
            errors.append("non_string_field." + field)
        elif not value.strip():
            errors.append("blank_field." + field)
        else:
            clean[field] = value
    if errors:
        return None, errors, changes
    label = clean["event_type"]
    if label not in labels:
        if label in aliases:
            clean["event_type"] = aliases[label]
            changes.append({"field": "event_type", "from": label, "to": aliases[label], "rule": "approved_exact_alias"})
        else:
            errors.append("off_ontology")
    for field, allowed in ENUMS.items():
        if clean[field] not in allowed:
            errors.append("invalid_enum." + field)
    locations = clean["event_location"]
    levels = clean["event_location_admin_level"]
    if locations == "not_stated":
        if levels != "not_stated":
            errors.append("missing_location_admin_mismatch")
    else:
        places = [v.strip() for v in locations.split(";")]
        kinds = [v.strip() for v in levels.split(";")]
        if len(places) != len(kinds) or any(not v for v in places + kinds):
            errors.append("location_admin_count_mismatch")
        if any(v not in ADMIN for v in kinds):
            errors.append("invalid_admin_level_component")
    return (None if errors else clean), sorted(set(errors)), changes


def load_manifest(path):
    result = {}
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"Blank manifest line {line_no}")
            row = json.loads(line)
            key = aid(row["id"])
            if key in result:
                raise ValueError(f"Duplicate manifest ID: {key}")
            result[key] = {"publish_date": row.get("publish_date"), "language": row.get("language"),
                           "gcs_path": row.get("gcs_path") or row.get("cloud_uri")}
    return result


def load_inputs(args):
    audit = json.loads(Path(args.audit_report).read_text(encoding="utf-8"))
    labels = audit.get("canonical_labels")
    if not isinstance(labels, list) or len(labels) != 48 or len(set(labels)) != 48:
        raise ValueError("Audit report must contain exactly 48 unique canonical labels.")
    if audit.get("expected_manifest_rows") != 34390924 or audit.get("expected_shards") != 200:
        raise ValueError("Audit is not the expected frozen Global V4 run.")
    aliases = {}
    if args.aliases:
        aliases = json.loads(Path(args.aliases).read_text(encoding="utf-8"))
        if not isinstance(aliases, dict) or not all(isinstance(k, str) and isinstance(v, str) and v in labels for k, v in aliases.items()):
            raise ValueError("Aliases must map exact raw labels to canonical labels.")
    expected = {}
    with Path(args.shard_stats).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shard = int(row["shard"])
            if shard in expected:
                raise ValueError("Duplicate shard-stat row")
            expected[shard] = row
    if set(expected) != set(range(200)):
        raise ValueError("Expected 200 shard-stat rows numbered 0 through 199.")
    exclusions = {}
    with Path(args.missing_ids).open(newline="", encoding="utf-8") as f:
        missing_ids = {aid(row["id"]) for row in csv.DictReader(f)}
    if len(missing_ids) != 36:
        raise ValueError("Expected 36 unique missing IDs from the frozen audit.")
    with Path(args.exclusions).open(encoding="utf-8") as f:
        text = f.read()
    records = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list):
        raise ValueError("Exclusion input must contain an array or JSONL records.")
    for row in records:
        key = aid(row["id"])
        if key in exclusions:
            raise ValueError("Duplicate exclusion ID")
        if row.get("resolution") == "excluded_empty_source_body":
            if row.get("source_body_chars") != 0:
                raise ValueError("Exclusion ledger lacks verified empty-body evidence")
            exclusions[key] = row
        elif row.get("status") == "empty_body":
            if row.get("body_type") != "str" or row.get("body_chars_stripped") != 0 or row.get("id_matches_source_uri") is not True:
                raise ValueError(f"Invalid empty-body diagnostic: {key}")
            exclusions[key] = {
                "id": key, "resolution": "excluded_empty_source_body",
                "source_gcs_path": row.get("gcs_path"),
                "source_generation": row.get("source_generation"),
                "source_sha256": row.get("source_sha256"),
                "source_body_chars": 0, "inference_performed": False,
                "reason": "The original source object has an empty body.",
            }
        else:
            raise ValueError(f"Unapproved exclusion resolution: {key}")
    if set(exclusions) != missing_ids:
        raise ValueError("The verified exclusions do not exactly match the audit's 36 missing IDs.")
    parser = load_parser(args.parser)
    return set(labels), aliases, expected, exclusions, parser


def output_checksums(folder):
    return {p.name: sha256(p) for p in sorted(Path(folder).glob("*.jsonl"))}


def verify_complete(folder, expected_hash):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["config_hash"] != expected_hash:
        raise RuntimeError(f"Existing results use a different configuration: {folder}")
    if output_checksums(folder) != summary["output_checksums"]:
        raise RuntimeError(f"Existing results have missing or modified files: {folder}")
    return summary["counts"]


def process_shard(shard, args, labels, aliases, expected, exclusions, parser, config_files):
    stem = f"shard-{shard:03d}"
    final = Path(args.out) / stem
    output_uri = f"{args.output.rstrip('/')}/shard-{shard}.jsonl"
    manifest_uri = f"{args.manifest.rstrip('/')}/manifest-{shard:03d}.jsonl"
    snapshot = {"output": metadata(output_uri), "manifest": metadata(manifest_uri)}
    config = {"version": VERSION, "shard": shard, "uris": [output_uri, manifest_uri],
              "source_snapshot": snapshot, "file_hashes": config_files}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if final.exists():
        counts = verify_complete(final, config_hash)
        print(f"{stem}: already complete", flush=True)
        return counts

    with tempfile.TemporaryDirectory(prefix=f".{stem}-", dir=args.out) as scratch_name:
        scratch = Path(scratch_name)
        manifest_path = obtain(manifest_uri, scratch)
        input_path = obtain(output_uri, scratch)
        manifest = load_manifest(manifest_path)
        stage = scratch / "result"
        stage.mkdir()
        counts = collections.Counter()
        seen = set()
        with (stage / "candidate_geocoding_input.jsonl").open("w", encoding="utf-8") as clean, \
             (stage / "event_metadata.jsonl").open("w", encoding="utf-8") as sidecar, \
             (stage / "quarantine.jsonl").open("w", encoding="utf-8") as quarantine, \
             (stage / "article_status.jsonl").open("w", encoding="utf-8") as article_status:
            with input_path.open(encoding="utf-8") as inp:
                for line_no, line in enumerate(inp, 1):
                    if not line.strip():
                        raise ValueError(f"Blank output line: {stem}:{line_no}")
                    row = json.loads(line)
                    key = aid(row["id"])
                    if key not in manifest or key in seen or key in exclusions:
                        raise ValueError(f"Unexpected, duplicate, or excluded ID: {key}")
                    predictions = row.get("predictions")
                    if not isinstance(predictions, list):
                        raise ValueError(f"Invalid prediction list: {key}")
                    seen.add(key)
                    info = manifest[key]
                    kept = []
                    counts["articles"] += 1
                    counts["raw_events"] += len(predictions)
                    if not predictions:
                        counts["genuine_no_event_articles"] += 1
                    for index, event in enumerate(predictions):
                        event_key = hashlib.sha256(f"{VERSION}:{key}:{index}".encode()).hexdigest()
                        provenance = {"event_key": event_key, "id": key, "shard": shard,
                                      "event_index": index, "language": info["language"],
                                      "publish_date": info["publish_date"], "source_gcs_path": info["gcs_path"]}
                        valid, errors, changes = validate_event(event, labels, aliases)
                        if valid is None:
                            counts["quarantined_events"] += 1
                            counts.update("issue." + e for e in errors)
                            quarantine.write(json_line({**provenance, "reason_codes": errors, "raw_event": event}))
                            continue
                        parsed = temporal(valid["event_time"], parser)
                        counts["structurally_valid_events"] += 1
                        counts["time." + parsed["syntax_kind"]] += 1
                        if parsed["parse_status"] != "parsed_candidate":
                            counts["temporal_unresolved_events"] += 1
                        elif parsed["syntax_kind"] != "single_date":
                            counts["interval_candidate_events"] += 1
                        counts["approved_alias_changes"] += len(changes)
                        kept.append(valid)
                        sidecar.write(json_line({**provenance, "schema_status": "structurally_valid",
                                                 "grounding_status": "not_verified", "changes": changes,
                                                 "raw_event_time": event["event_time"], "temporal": parsed,
                                                 "clustering_time_status": "source_review_required"}))
                    # Never turn an all-invalid article into a false no-event record.
                    if not predictions or kept:
                        clean.write(json_line({"id": row["id"], "predictions": kept}))
                        counts["candidate_articles"] += 1
                    if not predictions:
                        status = "no_event"
                    elif len(kept) == len(predictions):
                        status = "valid"
                    elif kept:
                        status = "partially_quarantined"
                    else:
                        status = "all_events_quarantined"
                    counts["article_status." + status] += 1
                    article_status.write(json_line({"id": key, "shard": shard, "raw_events": len(predictions),
                                                     "valid_events": len(kept), "quarantined_events": len(predictions)-len(kept),
                                                     "status": status}))
        missing = set(manifest) - seen
        expected_missing = {key for key in missing if key in exclusions}
        if missing != expected_missing:
            raise RuntimeError(f"{stem}: {len(missing)} missing manifest IDs, {len(expected_missing)} verified empty-source exclusions")
        counts["manifest_rows"] = len(manifest)
        counts["excluded_empty_source_articles"] = len(expected_missing)
        for key in sorted(expected_missing):
            with (stage / "article_status.jsonl").open("a", encoding="utf-8") as f:
                f.write(json_line({"id": key, "shard": shard, "raw_events": 0, "valid_events": 0,
                                   "quarantined_events": 0, "status": "excluded_empty_source_body"}))
        reference = expected[shard]
        for field, actual in (("manifest_rows", len(manifest)), ("output_rows", counts["articles"]),
                              ("raw_events", counts["raw_events"]), ("no_event_articles", counts["genuine_no_event_articles"])):
            if int(reference[field]) != actual:
                raise RuntimeError(f"{stem}: {field} expected {reference[field]}, got {actual}")
        if counts["raw_events"] != counts["structurally_valid_events"] + counts["quarantined_events"]:
            raise AssertionError("Event accounting mismatch")
        if counts["candidate_articles"] + counts["article_status.all_events_quarantined"] != counts["articles"]:
            raise AssertionError("Candidate article accounting mismatch")
        if snapshot != {"output": metadata(output_uri), "manifest": metadata(manifest_uri)}:
            raise RuntimeError(f"Source changed during processing: {stem}")
        summary = {"config": config, "config_hash": config_hash, "counts": dict(counts),
                   "status": "structural_processing_complete", "source_grounding_verified": False,
                   "temporal_semantics_verified": False, "geocoding_verified": False,
                   "database_ingested": False, "raw_outputs_modified": False,
                   "output_checksums": output_checksums(stage)}
        write_json(stage / "summary.json", summary)
        # Publish only a complete shard. An existing result is never overwritten.
        if final.exists():
            raise RuntimeError(f"Output appeared concurrently: {final}")
        os.rename(stage, final)
        print(f"{stem}: {counts['articles']:,} articles, {counts['structurally_valid_events']:,} valid, {counts['quarantined_events']:,} quarantined", flush=True)
        return dict(counts)


def aggregate(args, expected, exclusions):
    totals = collections.Counter()
    completed = []
    for shard in range(200):
        folder = Path(args.out) / f"shard-{shard:03d}"
        if not (folder / "summary.json").is_file():
            continue
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if any(not (folder / name).is_file() for name in summary["output_checksums"]):
            raise RuntimeError(f"Missing completed output: {folder}")
        totals.update(summary["counts"])
        completed.append(shard)
    full = len(completed) == 200
    if full:
        expected_totals = {k: sum(int(row[k]) for row in expected.values())
                           for k in ("manifest_rows", "output_rows", "raw_events", "no_event_articles")}
        for field, actual in (("manifest_rows", totals["manifest_rows"]), ("output_rows", totals["articles"]),
                              ("raw_events", totals["raw_events"]), ("no_event_articles", totals["genuine_no_event_articles"])):
            if actual != expected_totals[field]:
                raise RuntimeError(f"Global reconciliation failed: {field}")
        if exclusions and totals["excluded_empty_source_articles"] != len(exclusions):
            raise RuntimeError("Global exclusion reconciliation failed")
    report = {"version": VERSION, "completed_shards": completed, "all_200_shards_complete": full,
              "counts": dict(totals), "source_grounding_verified": False,
              "temporal_semantics_verified": False, "geocoding_verified": False,
              "database_ingested": False, "raw_outputs_modified": False}
    write_json(Path(args.out) / "run_summary.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", action="store_true", help="Explicit full-corpus mode; no article limit.")
    p.add_argument("--shard", type=int, help="Process one complete shard, with no article limit.")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--audit-report", default="global_v4_audit/report.json")
    p.add_argument("--shard-stats", default="global_v4_audit/shard_stats.csv")
    p.add_argument("--parser", default="global_v4_temporal_parse.py")
    p.add_argument("--aliases", help="Optional manually reviewed exact-label alias JSON.")
    p.add_argument("--missing-ids", default="global_v4_audit/missing_article_ids.csv")
    p.add_argument("--exclusions", default="global_v4_audit/missing_diagnostic/corrected_recheck/source_diagnostics.json",
                   help="Existing corrected diagnostic JSON, or a verified 36-record resolution JSONL.")
    p.add_argument("--out", default="global_v4_audit/postprocessing_v1")
    args = p.parse_args()
    if args.all == (args.shard is not None):
        p.error("Choose exactly one of --all or --shard N.")
    if args.shard is not None and not 0 <= args.shard < 200:
        p.error("Shard must be in 0..199.")
    if Path(args.out).resolve() == Path(args.audit_report).resolve().parent:
        p.error("The output directory must not be the original audit directory.")
    labels, aliases, expected, exclusions, parser = load_inputs(args)
    files = {"processor": sha256(__file__), "audit_report": sha256(args.audit_report),
             "shard_stats": sha256(args.shard_stats), "parser": sha256(args.parser),
             "aliases": sha256(args.aliases) if args.aliases else None,
             "exclusions": sha256(args.exclusions), "missing_ids": sha256(args.missing_ids)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ledger = Path(args.out) / "source_exclusions.jsonl"
    ledger_data = "".join(json_line(exclusions[key]) for key in sorted(exclusions))
    if ledger.exists() and ledger.read_text(encoding="utf-8") != ledger_data:
        raise RuntimeError("Existing exclusion ledger differs. Use a new --out.")
    if not ledger.exists():
        ledger.write_text(ledger_data, encoding="utf-8")
    for shard in (range(200) if args.all else [args.shard]):
        process_shard(shard, args, labels, aliases, expected, exclusions, parser, files)
    report = aggregate(args, expected, exclusions)
    print(f"Completed {len(report['completed_shards'])}/200 shards. Reports: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
