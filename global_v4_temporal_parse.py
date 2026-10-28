#!/usr/bin/env python3
"""Conservative, read-only syntax parser for Global V4 temporal audit values.

Reads time_values.csv from the completed audit. Produces a value-level parsing
table and summary without modifying predictions, GCS, or any database.
No source-grounding, incident-date inference, or automatic event-date repair.
"""
from __future__ import annotations

import argparse
import calendar
import collections
import csv
import datetime as dt
import json
from pathlib import Path
import re

VERSION = "1.0"
TOKEN = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")
MISSING = {"", "not_stated", "null", "none", "n/a", "na", "unknown"}
FIELDS = [
    "raw_value", "count", "syntax_kind", "parse_status", "review_reason",
    "start_token", "end_token", "start_precision", "end_precision",
    "lower_bound_date", "upper_bound_date", "source_verified",
]


def date_token(value):
    """Return the exact token and its calendar bounds, never an inferred event date."""
    match = TOKEN.fullmatch(value)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else None
    day = int(match.group(3)) if match.group(3) else None
    if not 1 <= year <= 9999:
        return None
    try:
        if month is None:
            return {"token": value, "precision": "year",
                    "lower": dt.date(year, 1, 1), "upper": dt.date(year, 12, 31)}
        if day is None:
            last = calendar.monthrange(year, month)[1]
            return {"token": value, "precision": "month",
                    "lower": dt.date(year, month, 1), "upper": dt.date(year, month, last)}
        date = dt.date(year, month, day)
        return {"token": value, "precision": "day", "lower": date, "upper": date}
    except ValueError:
        return None


def parse_time(value):
    result = {key: "" for key in FIELDS}
    result["raw_value"] = value
    result["source_verified"] = "false"
    if not isinstance(value, str):
        result.update(syntax_kind="missing_or_non_string", parse_status="unresolved",
                      review_reason="missing_or_non_string")
        return result
    s = value.strip()
    if s.lower() in MISSING:
        result.update(syntax_kind="not_stated", parse_status="unresolved",
                      review_reason="no_temporal_value")
        return result

    # A single exact ISO token has one known precision. Calendar bounds do not
    # imply the event occurred on a particular day within a month or year.
    point = date_token(s)
    if point:
        result.update(syntax_kind="single_date", parse_status="parsed_candidate",
                      start_token=s, end_token=s,
                      start_precision=point["precision"], end_precision=point["precision"],
                      lower_bound_date=point["lower"].isoformat(),
                      upper_bound_date=point["upper"].isoformat())
        return result

    # Only parse precisely one slash. Additional separators or non-ISO tokens
    # must be reviewed rather than silently stripped or guessed.
    if s.count("/") == 1:
        left, right = (part.strip() for part in s.split("/", 1))
        start = date_token(left) if left else None
        end = date_token(right) if right else None
        if start and end:
            if start["lower"] > end["upper"]:
                result.update(syntax_kind="reversed_interval", parse_status="unresolved",
                              review_reason="start_after_end")
                return result
            kind = "closed_interval"
        elif start and not right:
            kind = "open_end"
        elif not left and end:
            kind = "open_start"
        else:
            result.update(syntax_kind="invalid_or_ambiguous_interval",
                          parse_status="unresolved", review_reason="invalid_or_missing_endpoint")
            return result
        result.update(syntax_kind=kind, parse_status="parsed_candidate",
                      review_reason="interval_semantics_require_source_review")
        if start:
            result["start_token"] = start["token"]
            result["start_precision"] = start["precision"]
            result["lower_bound_date"] = start["lower"].isoformat()
        if end:
            result["end_token"] = end["token"]
            result["end_precision"] = end["precision"]
            result["upper_bound_date"] = end["upper"].isoformat()
        return result

    if TOKEN.fullmatch(s) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        reason = "invalid_calendar_date"
    elif "/" in s:
        reason = "multiple_or_unrecognized_separators"
    else:
        reason = "unsupported_temporal_expression"
    result.update(syntax_kind="unresolved", parse_status="unresolved", review_reason=reason)
    return result


def run(args):
    source = Path(args.input)
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    unique = collections.Counter()
    examples = collections.defaultdict(list)
    total = 0
    with source.open(encoding="utf-8", newline="") as inp, \
         (target / "time_parsed_values.csv").open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(inp)
        if not {"category", "value", "count"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected the audit time_values.csv with category,value,count.")
        writer = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            n = int(row["count"])
            if n < 0:
                raise ValueError("Negative frequency.")
            parsed = parse_time(row["value"])
            parsed["count"] = n
            writer.writerow(parsed)
            kind = parsed["syntax_kind"]
            counts[kind] += n
            unique[kind] += 1
            total += n
            if len(examples[kind]) < 8:
                examples[kind].append({"raw_value": row["value"], "count": n,
                                       "review_reason": parsed["review_reason"]})
    summary = {
        "version": VERSION,
        "scope": "Flagged values in the completed Global V4 temporal audit; ordinary valid dates are excluded.",
        "review_mentions": total,
        "unique_review_values": sum(unique.values()),
        "weighted_syntax_counts": dict(counts),
        "unique_syntax_counts": dict(unique),
        "examples": dict(examples),
        "policy": {
            "raw_values_preserved": True,
            "source_grounding_verified": False,
            "interval_semantics_verified": False,
            "calendar_bounds_are_not_exact_event_dates": True,
            "open_bounds_remain_open": True,
            "reversed_and_ambiguous_intervals_remain_unresolved": True,
            "predictions_modified": False,
        },
    }
    (target / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["GLOBAL V4 TEMPORAL SYNTAX PARSE", "",
             f"Review mentions: {total:,}",
             f"Unique review values: {sum(unique.values()):,}", ""]
    lines += [f"{key}: {n:,}" for key, n in counts.most_common()]
    lines += ["", "Calendar syntax only. No predictions or original event dates were changed.",
              "Interval semantics and grounding require source review.", ""]
    text = "\n".join(lines)
    (target / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print("Reports:", target)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="global_v4_audit/time_profile/time_values.csv")
    parser.add_argument("--out", default="global_v4_audit/time_normalization_review")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
