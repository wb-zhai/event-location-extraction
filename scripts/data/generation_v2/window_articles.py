"""Exact-offset article windowing CLI for generation_v2."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v2.io_utils import append_jsonl_row, iter_jsonl, load_records, resolve_path, write_jsonl


@dataclass(frozen=True)
class ArticleWindow:
    article_id: str
    window_index: int
    window_count: int
    start: int
    end: int
    core_start: int
    core_end: int
    text: str


@dataclass(frozen=True)
class TextUnit:
    start: int
    end: int


BOUNDARY_RE = re.compile(r"(?:\n\s*\n+)|(?<=[.!?;:])\s+")


def trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def split_units(text: str, max_chars: int) -> list[TextUnit]:
    units: list[TextUnit] = []
    start = 0
    for match in BOUNDARY_RE.finditer(text):
        value = trim(text, start, match.start())
        if value is not None:
            units.extend(split_long_unit(text, value[0], value[1], max_chars))
        start = match.end()
    value = trim(text, start, len(text))
    if value is not None:
        units.extend(split_long_unit(text, value[0], value[1], max_chars))
    return units


def split_long_unit(text: str, start: int, end: int, max_chars: int) -> list[TextUnit]:
    units: list[TextUnit] = []
    cursor = start
    while cursor < end:
        limit = min(cursor + max_chars, end)
        split_at = limit
        if limit < end:
            split_at = max(text.rfind(" ", cursor, limit), text.rfind("\n", cursor, limit))
            if split_at <= cursor:
                split_at = limit
        value = trim(text, cursor, split_at)
        if value is not None:
            units.append(TextUnit(*value))
        cursor = split_at
        while cursor < end and text[cursor].isspace():
            cursor += 1
    return units


def build_windows(
    article_id: str,
    text: str,
    *,
    target_chars: int = 6000,
    max_chars: int = 9000,
    overlap_sentences: int = 2,
) -> list[ArticleWindow]:
    units = split_units(text, max_chars=max_chars)
    if not units:
        return []

    groups: list[tuple[int, int]] = []
    start_index = 0
    while start_index < len(units):
        end_index = start_index + 1
        while end_index < len(units):
            if units[end_index].end - units[start_index].start > target_chars:
                break
            end_index += 1
        groups.append((start_index, end_index))
        start_index = end_index

    windows: list[ArticleWindow] = []
    for index, (core_start_index, core_end_index) in enumerate(groups):
        window_start_index = core_start_index
        window_end_index = core_end_index
        for _ in range(overlap_sentences):
            if window_start_index <= 0:
                break
            proposed_start = units[window_start_index - 1].start
            proposed_end = units[window_end_index - 1].end
            if proposed_end - proposed_start > max_chars:
                break
            window_start_index -= 1
        for _ in range(overlap_sentences):
            if window_end_index >= len(units):
                break
            proposed_start = units[window_start_index].start
            proposed_end = units[window_end_index].end
            if proposed_end - proposed_start > max_chars:
                break
            window_end_index += 1

        start = units[window_start_index].start
        end = units[window_end_index - 1].end
        core_start = units[core_start_index].start
        core_end = units[core_end_index - 1].end
        windows.append(
            ArticleWindow(
                article_id=article_id,
                window_index=index,
                window_count=len(groups),
                start=start,
                end=end,
                core_start=core_start,
                core_end=core_end,
                text=text[start:end],
            )
        )
    return windows


def window_record(record: dict, *, target_chars: int, max_chars: int, overlap_sentences: int) -> list[dict]:
    text = str(record.get("text") or "")
    article_id = str(record.get("id"))
    windows = build_windows(
        article_id,
        text,
        target_chars=target_chars,
        max_chars=max_chars,
        overlap_sentences=overlap_sentences,
    )
    return [
        {
            "id": f"{article_id}::w{window.window_index}",
            "article_id": article_id,
            "title": record.get("title", ""),
            "text": window.text,
            "article_text": text,
            "window": {
                "window_index": window.window_index,
                "window_count": window.window_count,
                "start": window.start,
                "end": window.end,
                "core_start": window.core_start,
                "core_end": window.core_end,
            },
            "source_url": record.get("source_url"),
            "publish_date": record.get("publish_date"),
            "source_bucket": record.get("source_bucket"),
        }
        for window in windows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split articles into exact-offset windows.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-chars", type=int, default=6000)
    parser.add_argument("--max-chars", type=int, default=9000)
    parser.add_argument("--overlap-sentences", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append-missing",
        action="store_true",
        help="Append windows only for sampled article ids not already present in the output.",
    )
    return parser.parse_args()


def completed_article_ids(output_path: Path) -> set[str]:
    completed: set[str] = set()
    for row in iter_jsonl(output_path):
        article_id = row.get("article_id") or str(row.get("id", "")).split("::w", 1)[0]
        if article_id:
            completed.add(str(article_id))
    return completed


def main() -> int:
    args = parse_args()
    output_path = resolve_path(args.output)
    if output_path.exists() and not args.overwrite and not args.append_missing:
        print(f"Output already exists, skipping: {output_path}")
        return 0
    records = load_records(resolve_path(args.input))
    if output_path.exists() and not args.overwrite and args.append_missing:
        completed = completed_article_ids(output_path)
        records = [record for record in records if str(record.get("id")) not in completed]
        if not records:
            print(f"No new sampled articles to window: {output_path}")
            return 0
        print(f"Appending windows for {len(records)} new sampled articles to {output_path}")
    else:
        write_jsonl(output_path, [], overwrite=True)
    row_count = 0
    for record in tqdm(records, desc="Windowing"):
        for row in window_record(
            record,
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            overlap_sentences=args.overlap_sentences,
        ):
            append_jsonl_row(output_path, row)
            row_count += 1
    print(f"Wrote {row_count} windows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
