#!/usr/bin/env python3
"""
Convert the plain TSV unique-location inventory into a newline-safe JSONL queue.

Why this exists:
The original TSV writer replaced tabs but did not escape embedded newlines in
event_location strings. A small number of locations therefore span multiple
physical TSV lines. Because every real TSV record starts with "<integer>\\t"
and tabs inside locations were already replaced, we can reconstruct the exact
records deterministically.

Output schema, one JSON object per line:
  {"count": 6090231, "location": "Gaza"}

No normalization is applied to the location string.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_repaired_records(path: Path):
    current_count = None
    current_parts = None

    with path.open("r", encoding="utf-8", errors="strict", newline="") as f:
        header = f.readline()
        if not header:
            raise SystemExit(f"Empty input: {path}")
        if not header.startswith("count\tlocation"):
            raise SystemExit(
                f"Unexpected header in {path!s}: {header[:100]!r}. "
                "Expected 'count\\tlocation'."
            )

        for physical_line_no, raw in enumerate(f, start=2):
            # Remove only the physical line terminator. Preserve all other text.
            line = raw[:-1] if raw.endswith("\n") else raw
            if line.endswith("\r"):
                line = line[:-1]

            left, sep, right = line.partition("\t")
            is_new_record = bool(sep) and left.isdigit()

            if is_new_record:
                if current_count is not None:
                    yield current_count, "\n".join(current_parts)
                current_count = int(left)
                current_parts = [right]
            else:
                if current_count is None:
                    raise SystemExit(
                        f"Malformed continuation before first record at "
                        f"physical line {physical_line_no}: {line[:120]!r}"
                    )
                current_parts.append(line)

    if current_count is not None:
        yield current_count, "\n".join(current_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_tsv", type=Path)
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("global_v4_unique_locations.jsonl"),
    )
    ap.add_argument("--expected-unique", type=int, default=2_191_166)
    ap.add_argument("--expected-mentions", type=int, default=120_001_356)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    mentions = 0
    multiline = 0
    max_count = -1
    previous_count = None

    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for count, location in iter_repaired_records(args.input_tsv):
            n += 1
            mentions += count
            if "\n" in location:
                multiline += 1

            # The source inventory was frequency-sorted. Verify that property
            # because the Photon cache builder uses it to maximize coverage early.
            if previous_count is not None and count > previous_count:
                raise SystemExit(
                    f"Input is not count-descending at record {n}: "
                    f"{count} > {previous_count}"
                )
            previous_count = count
            max_count = max(max_count, count)

            out.write(
                json.dumps(
                    {"count": count, "location": location},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    print(f"records             : {n:,}")
    print(f"mention_count_sum   : {mentions:,}")
    print(f"multiline_locations : {multiline:,}")
    print(f"max_frequency       : {max_count:,}")
    print(f"output              : {args.output}")

    if n != args.expected_unique:
        raise SystemExit(
            f"ERROR: unique count mismatch: {n:,} != {args.expected_unique:,}"
        )
    if mentions != args.expected_mentions:
        raise SystemExit(
            f"ERROR: mention sum mismatch: {mentions:,} != {args.expected_mentions:,}"
        )

    print("AUDIT: PASS")


if __name__ == "__main__":
    main()
