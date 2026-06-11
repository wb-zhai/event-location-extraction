"""Convert zhai-annotated JSONL to LlamaFactory Alpaca SFT format.

Supports paragraph-based windowing (default) so each training example fits within
a small model's context window.  Use --no-window to fall back to one record per
whole article.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Optional

HERE = pathlib.Path(__file__).parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_IGNORE = {"document_relevance", "event_location_text", "event_time_text", "affected_group", "affected_entity"}


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")


def split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char offsets of every non-empty paragraph in *text*."""
    paras: list[tuple[int, int]] = []
    cursor = 0
    for m in _BLANK_LINE_RE.finditer(text):
        seg = text[cursor:m.start()]
        if seg.strip():
            paras.append((cursor, m.start()))
        cursor = m.end()
    tail = text[cursor:]
    if tail.strip():
        paras.append((cursor, len(text)))
    return paras


# ---------------------------------------------------------------------------
# Event span location
# ---------------------------------------------------------------------------

def _all_occurrences(text: str, span: str) -> list[tuple[int, int]]:
    """Return all (start, end) positions of *span* in *text* (exact match)."""
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(span, start)
        if idx == -1:
            break
        out.append((idx, idx + len(span)))
        start = idx + 1
    return out


def locate_event_span(text: str, event: dict) -> tuple[int, int] | None:
    """Return the minimal (start, end) char span that covers all grounding texts.

    Strategy:
    1. Anchor on grounding_quote via TextAnchorResolver (exact→lesser→fuzzy).
    2. For event_location_text and event_time_text (when not empty/'not_stated'),
       pick the occurrence nearest the grounding anchor.
    3. Return the union span, or None if grounding_quote can't be located at all.
    """
    from src.inference.text_anchor import TextAnchorResolver

    resolver = TextAnchorResolver()

    gq = event.get("grounding_quote") or ""
    anchor = resolver.resolve(text, gq)
    if anchor.start is None or anchor.end is None:
        # Last resort: first occurrence of grounding_quote
        idx = text.find(gq) if gq else -1
        if idx == -1:
            return None
        anchor_start, anchor_end = idx, idx + len(gq)
    else:
        anchor_start, anchor_end = int(anchor.start), int(anchor.end)

    span_start, span_end = anchor_start, anchor_end

    for key in ("event_location_text", "event_time_text"):
        val = event.get(key) or ""
        if not val or val == "not_stated":
            continue
        occurrences = _all_occurrences(text, val)
        if not occurrences:
            continue
        # Pick the occurrence whose midpoint is closest to the anchor midpoint
        anchor_mid = (anchor_start + anchor_end) / 2
        best = min(occurrences, key=lambda se: abs((se[0] + se[1]) / 2 - anchor_mid))
        span_start = min(span_start, best[0])
        span_end = max(span_end, best[1])

    return (span_start, span_end)


# ---------------------------------------------------------------------------
# Window building
# ---------------------------------------------------------------------------

def build_paragraph_windows(
    paras: list[tuple[int, int]],
    *,
    max_chars: int,
    max_paras: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Return a list of (lo, hi) paragraph *index* ranges (hi exclusive).

    Each window covers paras[lo:hi].  Windows are greedy: accumulate whole
    paragraphs until the next one would exceed max_chars *or* max_paras is
    already reached.  An oversized single paragraph always forms its own window.
    """
    if not paras:
        return []

    windows: list[tuple[int, int]] = []
    lo = 0
    while lo < len(paras):
        hi = lo + 1  # always include at least one paragraph
        while hi < len(paras):
            if hi - lo >= max_paras:
                break
            next_len = paras[hi][1] - paras[lo][0]
            if next_len > max_chars:
                break
            hi += 1
        windows.append((lo, hi))
        # Stop as soon as the window covers all remaining paragraphs
        if hi >= len(paras):
            break
        # Advance start, ensuring overlap but always making forward progress
        advance = max(1, hi - lo - overlap)
        lo += advance

    return windows


# ---------------------------------------------------------------------------
# SFT record building
# ---------------------------------------------------------------------------

def filter_annotation(annotation: dict, ignore: set[str]) -> dict:
    result = {k: v for k, v in annotation.items() if k not in ignore}
    if "events" in result:
        result["events"] = [
            {k: v for k, v in event.items() if k not in ignore}
            for event in result["events"]
        ]
    return result


def build_user_message(template: str, publish_date: str, article_text: str) -> str:
    return (
        template
        .replace("{{PUBLISH_DATE}}", publish_date)
        .replace("{{ARTICLE_TEXT}}", article_text)
    )


def _build_windowed_records(
    row: dict,
    system_prompt: str,
    user_template: str,
    ignore: set[str],
    max_chars: int,
    max_paras: int,
    overlap: int,
) -> list[dict]:
    source = row.get("source", {})
    annotation = row.get("annotation") or {}
    text = source.get("text", "")
    publish_date = source.get("publish_date", "")

    paras = split_paragraphs(text)
    if not paras:
        # Degenerate: treat whole text as one paragraph
        paras = [(0, len(text))] if text.strip() else []
    if not paras:
        return []

    windows = build_paragraph_windows(
        paras, max_chars=max_chars, max_paras=max_paras, overlap=overlap
    )

    events: list[dict] = annotation.get("events") or []

    # Locate each event's required char span
    event_spans: list[tuple[int, int] | None] = []
    for event in events:
        span = locate_event_span(text, event)
        event_spans.append(span)

    # Build base window char ranges
    win_char_ranges = [
        (paras[lo][0], paras[hi - 1][1])
        for lo, hi in windows
    ]

    # Expansion pass: for each event not covered by any window, widen its
    # home window (the first one whose start para contains the grounding anchor).
    expanded_ranges = list(win_char_ranges)
    expanded_windows = list(windows)

    for event, span in zip(events, event_spans):
        if span is None:
            continue
        ev_start, ev_end = span

        # Check if any window already covers this event
        already_covered = any(
            ws <= ev_start and ev_end <= we
            for ws, we in expanded_ranges
        )
        if already_covered:
            continue

        # Find home window: first window whose char range contains the anchor
        gq = event.get("grounding_quote") or ""
        anchor_idx = text.find(gq) if gq else -1

        home = None
        for w_idx, (ws, we) in enumerate(expanded_ranges):
            if ws <= anchor_idx < we:
                home = w_idx
                break
        if home is None:
            # Fall back to first window
            home = 0

        # Minimally expand: extend para range to include ev_start and ev_end
        lo, hi = expanded_windows[home]
        # Find the paragraph that contains ev_start
        new_lo = lo
        for p_idx in range(lo - 1, -1, -1):
            if paras[p_idx][0] <= ev_start:
                new_lo = p_idx
                break
            if ev_start >= paras[p_idx][0]:
                new_lo = p_idx
                break
        # Find the paragraph that contains ev_end
        new_hi = hi
        for p_idx in range(hi, len(paras)):
            if paras[p_idx][1] >= ev_end:
                new_hi = p_idx + 1
                break
        # Only expand (never shrink)
        new_lo = min(new_lo, lo)
        new_hi = max(new_hi, hi)
        expanded_windows[home] = (new_lo, new_hi)
        expanded_ranges[home] = (paras[new_lo][0], paras[new_hi - 1][1])

    # Assignment pass: each event goes into every window that fully contains it
    records = []
    for (lo, hi), (ws, we) in zip(expanded_windows, expanded_ranges):
        window_text = text[ws:we]

        assigned = []
        for event, span in zip(events, event_spans):
            if span is None:
                # Can't locate span — assign to first window only
                if lo == expanded_windows[0][0]:
                    assigned.append(event)
                continue
            ev_start, ev_end = span
            if ws <= ev_start and ev_end <= we:
                assigned.append(event)

        window_annotation = {**annotation, "events": assigned}
        user_msg = build_user_message(user_template, publish_date, window_text)
        assistant_msg = json.dumps(filter_annotation(window_annotation, ignore), ensure_ascii=False)

        records.append({
            "system": system_prompt,
            "instruction": "",
            "input": user_msg,
            "output": assistant_msg,
            "_window_chars": len(window_text),
            "_window_paras": hi - lo,
            "_window_events": len(assigned),
        })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output JSON file (LlamaFactory Alpaca)")
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=list(DEFAULT_IGNORE),
        metavar="KEY",
        help="Annotation keys to strip (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing system_prompt.txt and user_prompt.txt "
            "(default: prompts/student relative to this script)"
        ),
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable windowing — one record per whole article",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Soft character cap per window (default: 3000)",
    )
    parser.add_argument(
        "--max-paras",
        type=int,
        default=15,
        help="Hard paragraph cap per window (default: 15)",
    )
    parser.add_argument(
        "--overlap-paras",
        type=int,
        default=1,
        help="Paragraphs shared between consecutive windows (default: 1)",
    )
    parser.add_argument(
        "--tokenizer",
        metavar="NAME_OR_PATH",
        help="HuggingFace tokenizer to use for token count statistics",
    )
    args = parser.parse_args()

    ignore = set(args.ignore)

    prompt_dir = pathlib.Path(args.prompt_dir) if args.prompt_dir else HERE / "prompts" / "student"
    system_prompt = (prompt_dir / "system_prompt.txt").read_text()
    user_template = (prompt_dir / "user_prompt.txt").read_text()

    records = []
    with open(args.input) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {lineno}: {e}", file=sys.stderr)
                continue

            annotation = row.get("annotation")
            if annotation is None:
                print(f"Skipping line {lineno}: no annotation", file=sys.stderr)
                continue

            if args.no_window:
                source = row.get("source", {})
                user_msg = build_user_message(
                    user_template,
                    source.get("publish_date", ""),
                    source.get("text", ""),
                )
                assistant_msg = json.dumps(filter_annotation(annotation, ignore), ensure_ascii=False)
                records.append({
                    "system": system_prompt,
                    "instruction": "",
                    "input": user_msg,
                    "output": assistant_msg,
                    "_window_chars": len(source.get("text", "")),
                    "_window_paras": None,
                    "_window_events": len((annotation.get("events") or [])),
                })
            else:
                records.extend(
                    _build_windowed_records(
                        row,
                        system_prompt,
                        user_template,
                        ignore,
                        max_chars=args.max_chars,
                        max_paras=args.max_paras,
                        overlap=args.overlap_paras,
                    )
                )

    # Strip internal metadata before writing
    output_records = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in records
    ]
    with open(args.output, "w") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(output_records)} records to {args.output}")
    _print_window_stats(records)

    if args.tokenizer and records:
        _print_token_stats(output_records, args.tokenizer)


def _print_window_stats(records: list[dict]) -> None:
    if not records:
        return
    chars = [r["_window_chars"] for r in records]
    events = [r["_window_events"] for r in records]
    paras = [r["_window_paras"] for r in records if r["_window_paras"] is not None]

    def _stats(vals: list, label: str) -> str:
        if not vals:
            return f"{label}: no data"
        avg = sum(vals) / len(vals)
        return f"{label}: avg={avg:.0f}  min={min(vals)}  max={max(vals)}"

    print(f"\n=== Window statistics ({len(records)} windows) ===")
    print(_stats(chars, "chars/window"))
    if paras:
        print(_stats(paras, "paras/window"))
    print(_stats(events, "events/window"))


def _print_token_stats(records: list[dict], tokenizer_name: str) -> None:
    from transformers import AutoTokenizer

    print(f"\nLoading tokenizer: {tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    input_counts, output_counts = [], []
    for r in records:
        prompt = "\n\n".join(p for p in [r["system"], r["input"]] if p)
        input_counts.append(len(tok.encode(prompt)))
        output_counts.append(len(tok.encode(r["output"])))

    def _stats(counts: list[int]) -> str:
        return f"avg={sum(counts)/len(counts):.0f}  min={min(counts)}  max={max(counts)}"

    print(f"Input  tokens: {_stats(input_counts)}")
    print(f"Output tokens: {_stats(output_counts)}")
    print(f"Total  tokens: {_stats([i + o for i, o in zip(input_counts, output_counts)])}")


if __name__ == "__main__":
    main()
