"""Sentence-aware article windows for event extraction.

The generator asks the LLM for offsets relative to each window.  This module
therefore keeps every window as an exact slice of the original article so local
offsets can be projected back to article offsets without text normalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ArticleWindow:
    window_index: int
    window_count: int
    start_char: int
    end_char: int
    text: str
    overlap_prev: bool
    overlap_next: bool
    core_start_char: int
    core_end_char: int


@dataclass(frozen=True)
class TextUnit:
    start_char: int
    end_char: int


BOUNDARY_RE = re.compile(r"(?:\n\s*\n+)|(?<=[.!?;:])\s+")


def trim_span_to_non_whitespace(
    text: str, start_char: int, end_char: int
) -> tuple[int, int] | None:
    """Return a non-empty span after trimming surrounding whitespace."""
    while start_char < end_char and text[start_char].isspace():
        start_char += 1
    while end_char > start_char and text[end_char - 1].isspace():
        end_char -= 1
    if start_char >= end_char:
        return None
    return start_char, end_char


def split_long_unit(
    text: str, start_char: int, end_char: int, max_chars: int
) -> list[TextUnit]:
    """Split one pathological long unit on whitespace without changing text."""
    units: list[TextUnit] = []
    cursor = start_char
    while cursor < end_char:
        limit = min(cursor + max_chars, end_char)
        split_at = limit
        if limit < end_char:
            whitespace = text.rfind(" ", cursor, limit)
            newline = text.rfind("\n", cursor, limit)
            split_at = max(whitespace, newline)
            if split_at <= cursor:
                split_at = limit
        trimmed = trim_span_to_non_whitespace(text, cursor, split_at)
        if trimmed is not None:
            units.append(TextUnit(*trimmed))
        cursor = split_at
        while cursor < end_char and text[cursor].isspace():
            cursor += 1
    return units


def sentence_units(text: str, max_chars: int = 9000) -> list[TextUnit]:
    """Split text into sentence-like units with original character offsets."""
    if not text:
        return []

    units: list[TextUnit] = []
    start_char = 0
    for match in BOUNDARY_RE.finditer(text):
        trimmed = trim_span_to_non_whitespace(text, start_char, match.start())
        if trimmed is not None:
            unit_start, unit_end = trimmed
            if unit_end - unit_start > max_chars:
                units.extend(split_long_unit(text, unit_start, unit_end, max_chars))
            else:
                units.append(TextUnit(unit_start, unit_end))
        start_char = match.end()

    trimmed = trim_span_to_non_whitespace(text, start_char, len(text))
    if trimmed is not None:
        unit_start, unit_end = trimmed
        if unit_end - unit_start > max_chars:
            units.extend(split_long_unit(text, unit_start, unit_end, max_chars))
        else:
            units.append(TextUnit(unit_start, unit_end))
    return units


def _core_groups(units: list[TextUnit], target_chars: int) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start_index = 0
    while start_index < len(units):
        end_index = start_index + 1
        while end_index < len(units):
            proposed_start = units[start_index].start_char
            proposed_end = units[end_index].end_char
            if proposed_end - proposed_start > target_chars:
                break
            end_index += 1
        groups.append((start_index, end_index))
        start_index = end_index
    return groups


def _expand_with_overlap(
    units: list[TextUnit],
    core_start_index: int,
    core_end_index: int,
    overlap_sentences: int,
    max_chars: int,
) -> tuple[int, int]:
    window_start_index = core_start_index
    window_end_index = core_end_index

    for _ in range(overlap_sentences):
        if window_start_index <= 0:
            break
        proposed_start = units[window_start_index - 1].start_char
        proposed_end = units[window_end_index - 1].end_char
        if proposed_end - proposed_start > max_chars:
            break
        window_start_index -= 1

    for _ in range(overlap_sentences):
        if window_end_index >= len(units):
            break
        proposed_start = units[window_start_index].start_char
        proposed_end = units[window_end_index].end_char
        if proposed_end - proposed_start > max_chars:
            break
        window_end_index += 1

    return window_start_index, window_end_index


def build_article_windows(
    text: str,
    target_chars: int = 6000,
    max_chars: int = 9000,
    overlap_sentences: int = 2,
) -> list[ArticleWindow]:
    """Build extraction windows with sentence boundaries and overlap."""
    if target_chars < 1:
        raise ValueError("target_chars must be >= 1")
    if max_chars < target_chars:
        raise ValueError("max_chars must be >= target_chars")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    units = sentence_units(text, max_chars=max_chars)
    if not units:
        return []

    groups = _core_groups(units, target_chars)
    windows: list[ArticleWindow] = []
    for window_index, (core_start_index, core_end_index) in enumerate(groups):
        window_start_index, window_end_index = _expand_with_overlap(
            units,
            core_start_index,
            core_end_index,
            overlap_sentences,
            max_chars,
        )
        start_char = units[window_start_index].start_char
        end_char = units[window_end_index - 1].end_char
        core_start_char = units[core_start_index].start_char
        core_end_char = units[core_end_index - 1].end_char
        windows.append(
            ArticleWindow(
                window_index=window_index,
                window_count=len(groups),
                start_char=start_char,
                end_char=end_char,
                text=text[start_char:end_char],
                overlap_prev=start_char < core_start_char,
                overlap_next=end_char > core_end_char,
                core_start_char=core_start_char,
                core_end_char=core_end_char,
            )
        )
    return windows
