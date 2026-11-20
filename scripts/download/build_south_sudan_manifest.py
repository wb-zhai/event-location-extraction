#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from contextlib import ExitStack
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Build 20 Cloud Batch manifest shards from the fixed South Sudan JSONL.")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-shards", type=int, default=20)
    p.add_argument("--default-language", default="eng")
    return p.parse_args()

def main():
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = [0] * args.num_shards
    total = skipped = 0

    with ExitStack() as stack:
        writers = [
            stack.enter_context((out_dir / f"manifest-{i:03d}.jsonl").open("w", encoding="utf-8"))
            for i in range(args.num_shards)
        ]

        with inp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"Invalid JSON on line {line_no}: {e}") from e

                source = rec.get("source") or {}
                article_id = rec.get("id")
                gcs_path = source.get("cloud_uri") or rec.get("cloud_uri")
                publish_date = (
                    source.get("published_at")
                    or source.get("publish_date")
                    or rec.get("published_at")
                    or rec.get("publish_date")
                )
                language = rec.get("language") or source.get("language") or args.default_language

                if article_id is None or not gcs_path or not publish_date:
                    skipped += 1
                    continue

                row = {
                    "id": article_id,
                    "gcs_path": gcs_path,
                    "publish_date": str(publish_date),
                    "language": language,
                }
                shard = total % args.num_shards
                writers[shard].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[shard] += 1
                total += 1

    print(f"Wrote {total:,} rows across {args.num_shards} shards")
    print(f"Skipped {skipped:,} rows missing id/cloud_uri/publish_date")
    print(f"Shard min={min(counts):,}, max={max(counts):,}")

if __name__ == "__main__":
    main()
