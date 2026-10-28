#!/usr/bin/env python3
"""Build a Cloud Batch manifest from the frozen Teacher dev articles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def first(*values):
    for v in values:
        if v is not None and str(v).strip():
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = missing_path = missing_id = 0

    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source") if isinstance(row.get("source"), dict) else {}

            article_id = first(row.get("id"), source.get("id"))
            gcs_path = first(
                row.get("gcs_path"),
                row.get("cloud_uri"),
                source.get("gcs_path"),
                source.get("cloud_uri"),
            )
            publish_date = first(
                row.get("publish_date"),
                row.get("published_at"),
                source.get("publish_date"),
                source.get("published_at"),
                "",
            )
            language = first(
                row.get("language"),
                source.get("language"),
                "eng",
            )

            if article_id is None:
                missing_id += 1
                continue
            if not gcs_path:
                missing_path += 1
                continue

            dst.write(json.dumps({
                "id": article_id,
                "gcs_path": gcs_path,
                "publish_date": str(publish_date),
                "language": str(language),
            }, ensure_ascii=False) + "\n")
            n += 1

    print(f"manifest rows: {n:,}")
    print(f"missing id:    {missing_id:,}")
    print(f"missing GCS:   {missing_path:,}")
    if missing_id or missing_path:
        raise SystemExit("ERROR: manifest is incomplete; do not launch inference")


if __name__ == "__main__":
    main()
