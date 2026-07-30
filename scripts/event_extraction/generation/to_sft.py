"""Convert zhai-annotated JSONL to LlamaFactory Alpaca SFT format.

Supports paragraph-based windowing (default) so each training example fits
within a small model's context window.  Oversized paragraphs are split before
windowing.  Use --no-window to fall back to one record per whole article.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys

HERE = pathlib.Path(__file__).parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_ONTOLOGY = REPO_ROOT / "ontologies" / "zhai" / "bona.v4.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_IGNORE = {
    "event_location_text",
    "event_time_text",
}

LLAMAFACTORY_COLUMNS = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
}


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")
_SENTENCE_TERMINATORS = {".", "!", "?", "。", "！", "？"}
_CLOSING_PUNCTUATION = {'"', "'", ")", "]", "}", "”", "’"}


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


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    sent_start = start
    idx = start
    while idx < end:
        if text[idx] not in _SENTENCE_TERMINATORS:
            idx += 1
            continue

        sent_end = idx + 1
        while sent_end < end and text[sent_end] in _CLOSING_PUNCTUATION:
            sent_end += 1
        if sent_end == end or text[sent_end].isspace():
            trimmed = _trim_span(text, sent_start, sent_end)
            if trimmed[0] < trimmed[1]:
                spans.append(trimmed)
            sent_start = sent_end
            while sent_start < end and text[sent_start].isspace():
                sent_start += 1
            idx = sent_start
            continue

        idx += 1

    trimmed = _trim_span(text, sent_start, end)
    if trimmed[0] < trimmed[1]:
        spans.append(trimmed)
    return spans


def split_oversized_paragraphs(
    text: str, paras: list[tuple[int, int]], *, max_chars: int
) -> list[tuple[int, int]]:
    """Split oversized paragraph spans into complete sentence spans.

    Normal paragraphs remain intact. Oversized paragraphs are split into
    complete sentence spans so windows prefer semantic boundaries. A sentence
    longer than *max_chars* remains intact because there is no smaller sentence
    boundary to choose. Offsets always refer to *text*.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")

    split_paras: list[tuple[int, int]] = []
    for para_start, para_end in paras:
        para_start, para_end = _trim_span(text, para_start, para_end)
        if para_start >= para_end:
            continue
        if para_end - para_start <= max_chars:
            split_paras.append((para_start, para_end))
            continue

        split_paras.extend(_sentence_spans(text, para_start, para_end))

    return split_paras


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

    Each window covers paras[lo:hi].  Windows are greedy: accumulate spans until
    the next one would exceed max_chars *or* max_paras is already reached.
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


def _window_span(
    paras: list[tuple[int, int]], window: tuple[int, int]
) -> tuple[int, int]:
    lo, hi = window
    return paras[lo][0], paras[hi - 1][1]


def _window_chars(paras: list[tuple[int, int]], window: tuple[int, int]) -> int:
    start, end = _window_span(paras, window)
    return end - start


def _merge_window_ranges(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[int, int]:
    return min(a[0], b[0]), max(a[1], b[1])


def coalesce_short_windows(
    paras: list[tuple[int, int]],
    windows: list[tuple[int, int]],
    *,
    min_chars: int,
    max_chars: int,
    max_paras: int,
) -> list[tuple[int, int]]:
    """Merge short windows into a neighbor when the merged window still fits."""
    if min_chars <= 0 or len(windows) <= 1:
        return windows

    coalesced: list[tuple[int, int]] = []
    i = 0
    while i < len(windows):
        window = windows[i]
        if _window_chars(paras, window) >= min_chars:
            coalesced.append(window)
            i += 1
            continue

        if i + 1 < len(windows):
            merged = _merge_window_ranges(window, windows[i + 1])
            if (
                merged[1] - merged[0] <= max_paras
                and _window_chars(paras, merged) <= max_chars
            ):
                coalesced.append(merged)
                i += 2
                continue

        if coalesced:
            merged = _merge_window_ranges(coalesced[-1], window)
            if (
                merged[1] - merged[0] <= max_paras
                and _window_chars(paras, merged) <= max_chars
            ):
                coalesced[-1] = merged
                i += 1
                continue

        coalesced.append(window)
        i += 1

    return coalesced


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


def load_ontology_labels(path: pathlib.Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("events"), dict):
        return list(payload["events"].keys())
    if isinstance(payload, dict):
        return list(payload.keys())
    if isinstance(payload, list):
        return [str(item) for item in payload]
    raise ValueError(f"Unsupported ontology format at {path}")


def load_ontology_descriptions(path: pathlib.Path) -> dict[str, str]:
    """Return {label: description} for ontologies that carry descriptions.

    Only the ``{"events": {label: description}}`` shape has them; every other
    supported shape is a bare label list, so this returns ``{}``.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("events"), dict):
        return {
            str(label): str(desc)
            for label, desc in payload["events"].items()
            if isinstance(desc, str) and desc.strip()
        }
    return {}


def _normalize_labels(raw_labels: object, *, key: str) -> list[str]:
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]
    if not isinstance(raw_labels, list):
        raise ValueError(f"{key} must be a string or list of strings")

    labels: list[str] = []
    seen: set[str] = set()
    for index, label in enumerate(raw_labels):
        if not isinstance(label, str):
            raise ValueError(f"{key}[{index}] must be a string")
        label = label.strip()
        if not label:
            raise ValueError(f"{key}[{index}] must be a non-empty string")
        if label not in seen:
            labels.append(label)
            seen.add(label)
    if not labels:
        raise ValueError(f"{key} must contain at least one label")
    return labels


def row_labels(row: dict, default_labels: list[str], top_k: int | None) -> list[str]:
    if "candidate" in row:
        labels = _normalize_labels(row["candidate"], key="candidate")
        if top_k is not None:
            labels = labels[:top_k]
    elif "candidates" in row:
        labels = _normalize_labels(row["candidates"], key="candidates")
        if top_k is not None:
            labels = labels[:top_k]
    else:
        labels = list(default_labels)

    if not labels:
        raise ValueError("top-k removed all labels")
    return labels


def row_language(row: dict) -> str:
    """Return "fra" or "eng" for *row*, inferred from its "language" field.

    Follows the 3-letter code convention used by the download scripts
    (e.g. scripts/download/from_db_matrix.py). Defaults to "eng" when the
    field is absent or unrecognized.
    """
    lang = str(row.get("language") or "eng").strip().lower()
    return "fra" if lang in ("fra", "fr", "french", "français", "francais") else "eng"


def render_system_prompt(
    system_prompt: str,
    labels: list[str],
    descriptions: dict[str, str] | None = None,
) -> str:
    if descriptions:
        block = "\n".join(
            f"{label}: {descriptions[label]}" if label in descriptions else label
            for label in labels
        )
    else:
        block = "\n".join(labels)
    rendered, count = re.subn(
        r"\{\{ALLOWED_EVENT_TYPES\}\}",
        block,
        system_prompt,
        count=1,
    )
    if count != 1:
        raise ValueError("system prompt must contain one {{ALLOWED_EVENT_TYPES}} placeholder")
    return rendered


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
    min_chars: int = 0,
) -> list[dict]:
    source = row.get("source", {})
    annotation = row.get("annotation") or {}
    text = source.get("text", "")
    publish_date = source.get("publish_date", "")

    paras = split_paragraphs(text)
    if not paras:
        # Degenerate: treat whole text as one paragraph
        paras = [(0, len(text))] if text.strip() else []
    paras = split_oversized_paragraphs(text, paras, max_chars=max_chars)
    if not paras:
        return []

    windows = build_paragraph_windows(
        paras, max_chars=max_chars, max_paras=max_paras, overlap=overlap
    )
    windows = coalesce_short_windows(
        paras,
        windows,
        min_chars=min_chars,
        max_chars=max_chars,
        max_paras=max_paras,
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
        # Only expand (never shrink), and keep the configured bounds hard.
        new_lo = min(new_lo, lo)
        new_hi = max(new_hi, hi)
        expanded_start, expanded_end = paras[new_lo][0], paras[new_hi - 1][1]
        if (
            expanded_end - expanded_start <= max_chars
            and new_hi - new_lo <= max_paras
        ):
            expanded_windows[home] = (new_lo, new_hi)
            expanded_ranges[home] = (expanded_start, expanded_end)

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

        if min_chars > 0 and len(window_text) < min_chars and not assigned:
            continue

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

def _load_rows(paths: list[pathlib.Path]) -> list[tuple[str, dict]]:
    """Read JSONL *paths*, returning (source_file_name, row) tuples."""
    rows: list[tuple[str, dict]] = []
    for path in paths:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping {path.name}:{lineno}: {e}", file=sys.stderr)
                    continue
                rows.append((path.name, row))
    return rows


def _row_has_events(row: dict) -> bool:
    annotation = row.get("annotation") or {}
    return bool(annotation.get("events"))


def _split_rows(
    rows: list[tuple[str, dict]], dev_ratio: float, rng: random.Random
) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    """Stratified article-level split, balanced by (source file, has-events).

    Each (source_file, has_events) group is shuffled and split independently so
    both dimensions stay proportionally represented in train and dev. Splitting
    at the row/article level (rather than per-window) keeps overlapping windows
    from the same article on the same side of the split.
    """
    groups: dict[tuple[str, bool], list[tuple[str, dict]]] = {}
    for source_name, row in rows:
        key = (source_name, _row_has_events(row))
        groups.setdefault(key, []).append((source_name, row))

    train_rows: list[tuple[str, dict]] = []
    dev_rows: list[tuple[str, dict]] = []
    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        dev_count = round(len(shuffled) * dev_ratio)
        dev_rows.extend(shuffled[:dev_count])
        train_rows.extend(shuffled[dev_count:])

    return train_rows, dev_rows


def _rows_to_records(
    rows: list[tuple[str, dict]],
    args: argparse.Namespace,
    ignore: set[str],
    default_labels: list[str],
    descriptions: dict[str, str],
    templates: dict[str, tuple[str, str]],
) -> list[dict]:
    records = []
    for source_name, row in rows:
        annotation = row.get("annotation")
        if annotation is None:
            print(f"Skipping row from {source_name} (id: {row.get('id')}): no annotation", file=sys.stderr)
            continue

        lang = row_language(row)
        if lang not in templates:
            print(
                f"Skipping row from {source_name} (id: {row.get('id')}): "
                f"no prompt templates available for language {lang!r}",
                file=sys.stderr,
            )
            continue
        system_template, user_template = templates[lang]

        try:
            system_prompt = render_system_prompt(
                system_template,
                row_labels(row, default_labels, args.top_k_candidates),
                descriptions,
            )
        except ValueError as e:
            print(f"Skipping row from {source_name} (id: {row.get('id')}): {e}", file=sys.stderr)
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
                    min_chars=args.min_chars,
                )
            )

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Input JSONL file(s)",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory (writes train.json and dev.json)",
    )
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
        "--ontology",
        type=pathlib.Path,
        default=DEFAULT_ONTOLOGY,
        help="Default label ontology JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=None,
        metavar="N",
        help="Limit row candidate labels to the first N labels",
    )
    parser.add_argument(
        "--ontology-descriptions",
        choices=("none", "all"),
        default="none",
        help=(
            "Render each allowed_event_types entry as 'label: description' using the "
            "ontology's descriptions ('all'), or as a bare label ('none'). Descriptions "
            "add roughly 2.2k tokens to every example with bona.v4.json, so 'none' is the "
            "default; train and serve with the same setting (default: %(default)s)"
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
        help="Target character cap per window; complete sentences are not cut (default: 3000)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Drop no-event windows shorter than this after merge attempts (default: 200; 0 disables)",
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
    parser.add_argument(
        "--max-empty-ratio",
        type=float,
        default=None,
        metavar="RATIO",
        help="Randomly drop empty windows so they make up at most this fraction of all windows (0–1). Default: keep all.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        metavar="RATIO",
        help=(
            "Fraction of articles held out for dev, balanced by (source file, "
            "has-events) so both stay proportionally represented (default: 0.1)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the train/dev split and --max-empty-ratio sampling (default: non-deterministic)",
    )
    parser.add_argument(
        "--dataset-train-name",
        default="train",
        metavar="NAME",
        help="Dataset key for the train split in dataset_info.json (default: train)",
    )
    parser.add_argument(
        "--dataset-dev-name",
        default="dev",
        metavar="NAME",
        help="Dataset key for the dev split in dataset_info.json (default: dev)",
    )
    args = parser.parse_args()

    if args.top_k_candidates is not None and args.top_k_candidates < 1:
        parser.error("--top-k-candidates must be >= 1")
    if args.max_chars < 1:
        parser.error("--max-chars must be >= 1")
    if args.min_chars < 0:
        parser.error("--min-chars must be >= 0")
    if args.max_paras < 1:
        parser.error("--max-paras must be >= 1")
    if args.overlap_paras < 0:
        parser.error("--overlap-paras must be >= 0")
    if args.max_empty_ratio is not None and not (0.0 <= args.max_empty_ratio <= 1.0):
        parser.error("--max-empty-ratio must be between 0 and 1")
    if not (0.0 <= args.dev_ratio <= 1.0):
        parser.error("--dev-ratio must be between 0 and 1")

    ignore = set(args.ignore)
    default_labels = load_ontology_labels(args.ontology)
    descriptions: dict[str, str] = {}
    if args.ontology_descriptions == "all":
        descriptions = load_ontology_descriptions(args.ontology)
        if not descriptions:
            parser.error(
                f"--ontology-descriptions all: {args.ontology} carries no descriptions "
                '(expected {"events": {label: description}})'
            )

    prompt_dir = pathlib.Path(args.prompt_dir) if args.prompt_dir else HERE / "prompts" / "student"
    templates: dict[str, tuple[str, str]] = {
        "eng": (
            (prompt_dir / "system_prompt.txt").read_text(),
            (prompt_dir / "user_prompt.txt").read_text(),
        ),
    }
    system_prompt_fr = prompt_dir / "system_prompt.fr.txt"
    user_prompt_fr = prompt_dir / "user_prompt.fr.txt"
    if system_prompt_fr.exists() and user_prompt_fr.exists():
        templates["fra"] = (system_prompt_fr.read_text(), user_prompt_fr.read_text())

    input_paths = [pathlib.Path(p) for p in args.input]
    rows = _load_rows(input_paths)

    rng = random.Random(args.seed)
    train_rows, dev_rows = _split_rows(rows, args.dev_ratio, rng)

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in (("train", train_rows), ("dev", dev_rows)):
        records = _rows_to_records(
            split_rows, args, ignore, default_labels, descriptions, templates
        )

        if args.max_empty_ratio is not None:
            records = _drop_empty_windows(records, args.max_empty_ratio, rng)

        # Strip internal metadata before writing
        output_records = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in records
        ]
        output_path = output_dir / f"{split_name}.json"
        with open(output_path, "w") as f:
            json.dump(output_records, f, ensure_ascii=False, indent=2)

        print(f"Wrote {len(output_records)} records to {output_path}")
        _print_window_stats(records, label=split_name)

        if args.tokenizer and records:
            _print_token_stats(output_records, args.tokenizer, label=split_name)

    dataset_info_path = _write_dataset_info(
        output_dir, args.dataset_train_name, args.dataset_dev_name
    )
    print(f"Wrote {dataset_info_path}")


def _write_dataset_info(
    output_dir: pathlib.Path, train_name: str, dev_name: str
) -> pathlib.Path:
    """Write a LlamaFactory dataset_info.json pointing at train.json/dev.json."""
    info = {
        train_name: {"file_name": "train.json", "columns": LLAMAFACTORY_COLUMNS},
        dev_name: {"file_name": "dev.json", "columns": LLAMAFACTORY_COLUMNS},
    }
    path = output_dir / "dataset_info.json"
    with open(path, "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return path


def _drop_empty_windows(
    records: list[dict], max_empty_ratio: float, rng: random.Random
) -> list[dict]:
    """Randomly drop empty windows so their share doesn't exceed *max_empty_ratio*."""
    empty_indices = [i for i, r in enumerate(records) if r["_window_events"] == 0]
    non_empty_count = len(records) - len(empty_indices)

    if max_empty_ratio >= 1.0 or not empty_indices:
        return records

    # solve: max_keep / (non_empty + max_keep) <= max_empty_ratio
    max_keep = int(max_empty_ratio * non_empty_count / (1.0 - max_empty_ratio))

    if len(empty_indices) <= max_keep:
        return records

    to_drop = set(rng.sample(empty_indices, len(empty_indices) - max_keep))
    dropped = len(to_drop)
    result = [r for i, r in enumerate(records) if i not in to_drop]
    print(f"Dropped {dropped} empty windows to meet --max-empty-ratio {max_empty_ratio}", file=sys.stderr)
    return result


def _print_window_stats(records: list[dict], *, label: str = "") -> None:
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

    prefix = f"{label} " if label else ""
    print(f"\n=== {prefix}window statistics ({len(records)} windows) ===")
    print(_stats(chars, "chars/window"))
    if paras:
        print(_stats(paras, "paras/window"))
    print(_stats(events, "events/window"))
    no_event_count = sum(1 for count in events if count == 0)
    no_event_pct = no_event_count / len(records) * 100
    print(f"windows without events: {no_event_count} ({no_event_pct:.1f}%)")


def _print_token_stats(records: list[dict], tokenizer_name: str, *, label: str = "") -> None:
    from transformers import AutoTokenizer

    prefix = f"{label} " if label else ""
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    input_counts, output_counts = [], []
    for r in records:
        prompt = "\n\n".join(p for p in [r["system"], r["input"]] if p)
        input_counts.append(len(tok.encode(prompt)))
        output_counts.append(len(tok.encode(r["output"])))

    def _stats(counts: list[int]) -> str:
        return f"avg={sum(counts)/len(counts):.0f}  min={min(counts)}  max={max(counts)}"

    print(f"{prefix}input  tokens: {_stats(input_counts)}")
    print(f"{prefix}output tokens: {_stats(output_counts)}")
    print(f"{prefix}total  tokens: {_stats([i + o for i, o in zip(input_counts, output_counts)])}")


if __name__ == "__main__":
    main()
