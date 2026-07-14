"""Flatten JSONL event-extraction records into a CSV/TSV for review in Google
Sheets or Excel — no Argilla server needed.

Each article becomes a block of rows:

    id  title  published_at  source_url  risk_factors  text  event_type_reference
    <id> <title> <published_at> <source_url> <risk_factors> <text> <event type + description list>
    event_index  event_type  grounding_quote  event_location_text  event_location  event_time_text  event_time  time_status  severity
    1  <event 1 fields...>
    2  <event 2 fields...>
    (blank row)

Both header rows repeat above every article's block, so the sheet reads
top-to-bottom without needing frozen headers.

See scripts/annotations/README.md for the companion Argilla workflow.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.annotations.events_argilla import (
    EVENT_FIELDS,
    extract_title_text,
    iter_jsonl,
    keep_by_relevance,
    load_allowed_event_types,
    render_event_type_reference,
)
from scripts.data.relevance.relevance_filter import _record_key

# Human-facing explanation of this sheet's layout, written for annotators
# opening the export in Google Sheets/Excel (not for developers — see the
# module docstring above for that). Bundled into the guidelines sheet by
# guidelines_to_csv.py so annotators get it alongside the annotation criteria.
SHEET_GUIDE = """This sheet lists articles and the risk events extracted from each one, for
review outside Argilla.

Each article is a repeating block of rows:

1. A header row (id, title, published_at, source_url, risk_factors, text,
   event_type_reference) followed by one row of values for that article.
   event_type_reference repeats the allowed event types + descriptions from
   the labels table below, for quick reference without switching tabs.
2. A header row for events (event_index, event_type, grounding_quote,
   event_location_text, event_location, event_time_text, event_time,
   time_status, severity), followed by one row per extracted event.
3. A blank row separating this article's block from the next.

Both header rows repeat above every article's block, so the sheet reads
top-to-bottom without needing frozen headers.

How to edit:
- To correct an event, edit the values directly in its row.
- To reject an invalid event, delete its row.
- To add an event the model missed, insert a new row under the event header
  with the same columns filled in (leave event_index as the next number in
  sequence; it isn't used downstream).
- If an article shows a single "(no events extracted)" row, leave it as-is
  unless you find a valid event, in which case replace it with event rows.
- Leave id, title, and the other article-metadata columns unchanged.
"""

ARTICLE_META_FIELDS = [
    "id",
    "title",
    "published_at",
    "source_url",
    "risk_factors",
    "text",
    "event_type_reference",
]
EVENT_ROW_FIELDS = ["event_index", *EVENT_FIELDS]

DELIMITERS = {"tab": "\t", "comma": ","}


def build_article_meta_row(
    rec: dict[str, Any], max_chars: int, event_type_reference: str
) -> list[str]:
    source = rec.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    title, text = extract_title_text(rec)
    if max_chars:
        text = text[:max_chars]
    published_at = str(source.get("published_at") or source.get("publish_date") or "")
    source_url = str(source.get("source_url") or "")
    risk_factors = "; ".join(str(r) for r in (rec.get("risk_factors") or []))
    return [
        _record_key(rec),
        title,
        published_at,
        source_url,
        risk_factors,
        text,
        event_type_reference,
    ]


def build_event_rows(rec: dict[str, Any]) -> list[list[str]]:
    annotation = rec.get("annotation") or {}
    if not isinstance(annotation, dict):
        annotation = {}
    events = [ev for ev in (annotation.get("events") or []) if isinstance(ev, dict)]
    rows = []
    for i, ev in enumerate(events, start=1):
        rows.append([str(i), *[str(ev.get(f, "")) for f in EVENT_FIELDS]])
    return rows


def write_sheet(records: list[dict[str, Any]], output_path: Path, delimiter: str, max_chars: int) -> None:
    event_type_reference = render_event_type_reference(load_allowed_event_types())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        for rec in records:
            writer.writerow(ARTICLE_META_FIELDS)
            writer.writerow(build_article_meta_row(rec, max_chars, event_type_reference))
            writer.writerow(EVENT_ROW_FIELDS)
            event_rows = build_event_rows(rec)
            if event_rows:
                writer.writerows(event_rows)
            else:
                writer.writerow(["(no events extracted)"])
            writer.writerow([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--delimiter", choices=list(DELIMITERS), default="tab")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=0, help="Truncate article text to N chars (0 = no truncation).")
    parser.add_argument(
        "--only-relevant",
        action="store_true",
        help="Drop records whose relevance.decision is not relevant/partially_relevant.",
    )
    args = parser.parse_args()

    records = iter_jsonl(args.input)
    if args.only_relevant:
        records = [r for r in records if keep_by_relevance(r)]
    if args.limit:
        records = records[: args.limit]

    write_sheet(records, args.output, DELIMITERS[args.delimiter], args.max_chars)
    print(f"[write] {len(records)} articles -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
