"""Push articles into Argilla for human event-annotation review, and export the
resulting human-corrected event structures alongside the model's predictions.

The annotator reads the article, highlights each event mention in the text and
tags it with an event_type as a visual grounding aid. The exported event
structures are edited in a tab-separated table (`event_details_table`), one row
per event, so annotators can paste into a spreadsheet instead of repairing a
large hand-edited JSON list.

Older datasets that still have the previous `event_details_json` question are
supported on export as a legacy fallback via `merge_spans_and_details`.

See scripts/annotations/README.md for setup and usage.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import argilla as rg

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from scripts.annotations.relevance_argilla import normalize_decision
from scripts.data.relevance.relevance_filter import _record_key

RELEVANT_DECISIONS = {"relevant", "partially_relevant"}

RELEVANCE_QUESTION_NAME = "document_relevance"
EVENT_SPANS_QUESTION_NAME = "event_spans"
EVENT_DETAILS_TABLE_QUESTION_NAME = "event_details_table"
EVENT_DETAILS_JSON_QUESTION_NAME = "event_details_json"
EVENT_DETAILS_QUESTION_NAME = EVENT_DETAILS_TABLE_QUESTION_NAME
DEFAULT_WORKSPACE = "default"

# The 9 event fields not captured by the event_spans highlight (which covers
# event_type + grounding_quote). Order here is the order used in exported/
# reconstructed event dicts.
DETAIL_FIELDS = [
    "event_location_text",
    "event_location",
    "event_time_text",
    "event_time",
    "time_status",
    "affected_entity",
    "affected_group",
    "severity",
    "modality",
]
EVENT_FIELDS = ["event_type", "grounding_quote", *DETAIL_FIELDS]
CHOICE_DETAIL_FIELDS = {
    "time_status": ["past", "ongoing", "forecast", "not_stated"],
    "severity": ["low", "medium", "high", "extreme", "not_stated"],
    "modality": ["asserted", "projected"],
}
VERBATIM_FIELDS = ["event_location_text", "event_time_text"]
MISSING_TEXT_VALUES = {"", "not_stated"}
EVENT_TABLE_FIELDS = EVENT_FIELDS
EVENT_TABLE_HEADER = "\t".join(EVENT_TABLE_FIELDS)
EVENT_TABLE_TEMPLATE = f"""```text
{EVENT_TABLE_HEADER}
drought	Hammered by four droughts in a row	in the Horn of Africa	not_stated	not_stated	not_stated	past	not_stated	not_stated	not_stated	asserted
```"""

SYSTEM_PROMPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "data"
    / "generation_v3"
    / "prompts"
    / "teacher"
    / "system_prompt.txt"
)


def load_allowed_event_types() -> list[str]:
    if not SYSTEM_PROMPT_PATH.exists():
        raise SystemExit(f"Missing {SYSTEM_PROMPT_PATH} (needed for the allowed event_type list).")
    text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    match = re.search(r"<allowed_event_types>\n(.*?)\n</allowed_event_types>", text, re.S)
    if not match:
        raise SystemExit(f"Could not find <allowed_event_types> block in {SYSTEM_PROMPT_PATH}.")
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


# Argilla caps dataset guidelines at 10,000 characters; the full teacher system
# prompt (with few-shot examples etc.) runs ~22,000. This condensed version
# keeps the schema/field rules and explains the span+table workflow, and points
# annotators at the full file for edge cases.
GUIDELINES_TEMPLATE = """Validate, correct, and add to the extracted events for this article.
See {prompt_path} for the full extraction spec (few-shot examples, edge-case rules)
this condensed version is drawn from.

<workflow>
1. In the article text, highlight each event mention ({event_spans_question!r} question) \
and tag it with its event_type from the dropdown (must be one of allowed_event_types below).
   The highlighted span becomes that event's grounding_quote — copy/typing it is not needed.
2. In {event_details_question!r}, edit the tab-separated event table. The first line must stay \
the header. Each following line is one exported event. You can paste the table into a spreadsheet, \
edit cells, then paste it back.
3. To remove an event, delete its table row. To add an event, add a row. Highlighting its \
grounding_quote in {event_spans_question!r} is useful for review but export reads the table rows.
4. Submit only the header line for {event_details_question!r} if the document has no valid events.
</workflow>

<output_schema (per event, after export parses the table)>
{{
  "event_type": "string (must exactly match one of allowed_event_types below)",
  "grounding_quote": "exact contiguous quote from the article that justifies event_type",
  "event_location_text": "shortest verbatim quote with location evidence, or not_stated",
  "event_location": "geocodable place name(s) derived from event_location_text, ';'-separated, or not_stated",
  "event_time_text": "shortest verbatim quote with time evidence, or not_stated",
  "event_time": "ISO 8601 (YYYY / YYYY-MM / YYYY-MM-DD), range as X/Y, open-ended as X/ or /Y, or not_stated",
  "time_status": "past | ongoing | forecast | not_stated",
  "affected_entity": "thing/system/asset/economy/resource affected, or not_stated",
  "affected_group": "people/population group affected (never a country or org), or not_stated",
  "severity": "low | medium | high | extreme | not_stated",
  "modality": "asserted | projected"
}}
</output_schema>

<critical_rules>
Use only the article text (and publish date) as evidence — never world knowledge, geography knowledge, or outside metadata.
Never invent locations, dates, affected groups, severity, or event types not supported by the text.
Extract one event per distinct, explicitly stated event; do not merge distinct events or duplicate repeated mentions without new information.
</critical_rules>

<field_notes>
time_status: past = completed/historical; ongoing = current/continuing/worsening; forecast = expected/projected/predicted/planned/warned about.
severity: infer only from explicit severity language (severe, major, catastrophic, etc.) — worsening/trajectory language alone (intensifies, escalates) does not imply "high". Never infer severity from a number alone.
modality: asserted = stated as fact (including attributed claims); projected = expected/predicted/forecast/possible/rumored/unconfirmed.
event_location: strip vague directional prefixes ("southwestern Bangladesh" -> "Bangladesh") but keep proper administrative names ("North Darfur state" -> "North Darfur"). Broad scopes ("world", "many countries") are not_stated. Do not put countries/places in affected_group.
event_time: resolve relative expressions (e.g. "last month") using the article's publish date; drop vague qualifiers (early/mid/late/season) to just the year.
</field_notes>

<allowed_event_types>
{allowed_event_types}
</allowed_event_types>
"""


def load_guidelines(allowed_event_types: list[str]) -> str:
    return GUIDELINES_TEMPLATE.format(
        prompt_path=SYSTEM_PROMPT_PATH.relative_to(REPO_ROOT),
        event_spans_question=EVENT_SPANS_QUESTION_NAME,
        event_details_question=EVENT_DETAILS_QUESTION_NAME,
        allowed_event_types="\n".join(allowed_event_types),
    )


DOCUMENT_RELEVANCE_LABELS = {
    "relevant": "Relevant",
    "not_relevant": "Not relevant",
}


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_no}: {exc}"
                    ) from exc
    return records


def get_client():

    api_url = os.environ.get("ARGILLA_API_URL")
    api_key = os.environ.get("ARGILLA_API_KEY")
    if not api_url or not api_key:
        raise SystemExit(
            "ARGILLA_API_URL and ARGILLA_API_KEY must be set (see .env or "
            "scripts/annotations/README.md)."
        )
    return rg.Argilla(api_url=api_url, api_key=api_key)


def build_settings():

    allowed_event_types = load_allowed_event_types()
    return rg.Settings(
        guidelines=load_guidelines(allowed_event_types),
        fields=[
            rg.TextField(name="title"),
            rg.TextField(name="text"),
        ],
        questions=[
            rg.LabelQuestion(
                name=RELEVANCE_QUESTION_NAME,
                labels=DOCUMENT_RELEVANCE_LABELS,
                title="Is this document relevant (does it contain at least one valid event)?",
            ),
            rg.SpanQuestion(
                name=EVENT_SPANS_QUESTION_NAME,
                field="text",
                labels=allowed_event_types,
                allow_overlapping=True,
                required=False,
                title="Highlight each event mention and tag its event_type",
                description=(
                    "Optional visual aid for review. The exported events come from the "
                    f"{EVENT_DETAILS_TABLE_QUESTION_NAME!r} table below."
                ),
            ),
            rg.TextQuestion(
                name=EVENT_DETAILS_TABLE_QUESTION_NAME,
                title="Event table (TSV, one event per row)",
                description=(
                    "Keep the header line. Edit one tab-separated row per event; paste into a "
                    "spreadsheet if that is easier. Delete rows to remove events, add rows to "
                    "add events, and use 'not_stated' for missing values.\n\n"
                    f"{EVENT_TABLE_TEMPLATE}\n\n"
                    "Export writes validation errors for missing/extra columns, invalid choices, "
                    "off-ontology event_type values, and non-verbatim quote fields."
                ),
                use_markdown=True,
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="adm0_code"),
            rg.TermsMetadataProperty(name="risk_factors"),
            rg.TermsMetadataProperty(name="generation_llm"),
            rg.FloatMetadataProperty(name="quality_score"),
        ],
    )


def get_or_create_workspace(client, name: str):

    workspace = client.workspaces(name)
    if workspace is not None:
        print(f"[workspace] found existing workspace {name!r}.", flush=True)
        return workspace
    print(f"[workspace] {name!r} not found, creating...", flush=True)
    workspace = rg.Workspace(name=name, client=client)
    workspace.create()
    print(f"[workspace] created {name!r}.", flush=True)
    return workspace


def get_or_create_dataset(client, name: str, workspace: str, update_settings: bool = False):

    get_or_create_workspace(client, workspace)
    dataset = client.datasets(name=name, workspace=workspace)
    if dataset is not None:
        print(f"[dataset] found existing dataset {name!r} in workspace {workspace!r}.", flush=True)
        if update_settings:
            print(f"[dataset] updating schema + guidelines for {name!r}...", flush=True)
            dataset.settings = build_settings()
            dataset.update()
            print(f"[dataset] updated {name!r}.", flush=True)
        else:
            print(
                "[dataset] existing schema was left unchanged; use --update-settings "
                "to apply the event table UI.",
                flush=True,
            )
        return dataset
    print(f"[dataset] {name!r} not found in workspace {workspace!r}, creating (schema + guidelines)...", flush=True)
    dataset = rg.Dataset(name=name, workspace=workspace, settings=build_settings())
    dataset.create()
    print(f"[dataset] created {name!r}.", flush=True)
    return dataset


def extract_title_text(record: dict[str, Any]) -> tuple[str, str]:
    source = record.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    title = str(record.get("title") or source.get("title", ""))
    text = str(record.get("text") or source.get("text", ""))
    return title, text


def keep_by_relevance(rec: dict[str, Any]) -> bool:
    """Filter on rec["relevance"]["decision"] (not annotation.document_relevance):
    keep relevant/partially_relevant, and keep records with no relevance label at all."""
    relevance = rec.get("relevance")
    if not isinstance(relevance, dict):
        return True
    decision = normalize_decision(relevance.get("decision"))
    if decision is None:
        return True
    return decision in RELEVANT_DECISIONS


def model_name(rec: dict[str, Any]) -> str:
    generation_llm = rec.get("generation_llm")
    if isinstance(generation_llm, dict):
        return str(generation_llm.get("model") or "model")
    return str(generation_llm or "model")


def split_matched_unmatched(
    events: list[Any], text: str
) -> tuple[list[tuple[int, int, str, dict[str, Any]]], list[dict[str, Any]]]:
    """Split model events into (matched, unmatched) by whether grounding_quote is
    found verbatim in text. matched entries are (start, end, event_type, detail_dict)."""
    matched = []
    unmatched = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        quote = str(ev.get("grounding_quote") or "")
        event_type = str(ev.get("event_type") or "")
        detail = {field: ev.get(field, "not_stated") for field in DETAIL_FIELDS}
        idx = text.find(quote) if quote and event_type else -1
        if idx >= 0:
            matched.append((idx, idx + len(quote), event_type, detail))
        else:
            unmatched.append({"event_type": event_type, "grounding_quote": quote, **detail})
    matched.sort(key=lambda m: m[0])
    return matched, unmatched


def clean_table_cell(value: Any) -> str:
    return str(value if value is not None else "not_stated").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def format_events_table(events: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=EVENT_TABLE_FIELDS,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for event in events:
        writer.writerow({field: clean_table_cell(event.get(field, "not_stated")) for field in EVENT_TABLE_FIELDS})
    return output.getvalue().rstrip("\n")


def parse_events_table(value: str | None) -> tuple[list[dict[str, str]], list[str]]:
    if value is None or not value.strip():
        return [], []

    reader = csv.DictReader(io.StringIO(value), delimiter="\t")
    if reader.fieldnames is None:
        return [], []

    fieldnames = [field.strip() if field is not None else "" for field in reader.fieldnames]
    errors: list[str] = []
    missing = [field for field in EVENT_TABLE_FIELDS if field not in fieldnames]
    extra = [field for field in fieldnames if field and field not in EVENT_TABLE_FIELDS]
    for field in missing:
        errors.append(f"{EVENT_DETAILS_TABLE_QUESTION_NAME}: missing column {field!r}")
    for field in extra:
        errors.append(f"{EVENT_DETAILS_TABLE_QUESTION_NAME}: unexpected column {field!r}")

    events: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        if row.get(None):
            errors.append(
                f"{EVENT_DETAILS_TABLE_QUESTION_NAME}: row {row_number} has too many cells; "
                "check for extra tabs"
            )
        event = {field: (row.get(field) or "").strip() for field in EVENT_TABLE_FIELDS}
        if not any(event.values()):
            continue
        events.append(event)
    return events, errors


def build_record(rec: dict[str, Any], max_chars: int):

    title, text = extract_title_text(rec)
    if max_chars:
        text = text[:max_chars]
    annotation = rec.get("annotation") or {}
    if not isinstance(annotation, dict):
        annotation = {}

    document_relevance = annotation.get("document_relevance")
    events = annotation.get("events") or []
    matched, unmatched = split_matched_unmatched(events, text)

    metadata: dict[str, Any] = {}
    adm0_code = rec.get("adm0_code")
    if adm0_code:
        metadata["adm0_code"] = str(adm0_code)
    risk_factors = rec.get("risk_factors")
    if risk_factors:
        metadata["risk_factors"] = [str(r) for r in risk_factors]
    if rec.get("generation_llm"):
        metadata["generation_llm"] = model_name(rec)
    quality_score = rec.get("quality_score")
    if quality_score is not None:
        metadata["quality_score"] = float(quality_score)

    suggestions = []
    if document_relevance in DOCUMENT_RELEVANCE_LABELS:
        suggestions.append(
            rg.Suggestion(
                RELEVANCE_QUESTION_NAME,
                value=document_relevance,
                agent=model_name(rec),
            )
        )
    if matched:
        suggestions.append(
            rg.Suggestion(
                EVENT_SPANS_QUESTION_NAME,
                value=[
                    {"start": start, "end": end, "label": event_type}
                    for start, end, event_type, _ in matched
                ],
                agent=model_name(rec),
            )
        )
    table_events = [
        {
            "event_type": event_type,
            "grounding_quote": text[start:end],
            **detail,
        }
        for start, end, event_type, detail in matched
    ] + unmatched
    suggestions.append(
        rg.Suggestion(
            EVENT_DETAILS_TABLE_QUESTION_NAME,
            value=format_events_table(table_events),
            agent=model_name(rec),
        )
    )

    return rg.Record(
        fields={
            "title": title,
            "text": text,
        },
        metadata=metadata,
        suggestions=suggestions,
        id=_record_key(rec) or None,
    )


def push(args: argparse.Namespace) -> None:
    print("[connect] authenticating to Argilla...", flush=True)
    client = get_client()
    print("[connect] connected.", flush=True)

    dataset = get_or_create_dataset(
        client,
        args.dataset_name,
        args.workspace,
        update_settings=args.update_settings,
    )

    print(f"[load] reading {args.input}...", flush=True)
    records = iter_jsonl(Path(args.input))
    total = len(records)
    records = [rec for rec in records if keep_by_relevance(rec)]
    skipped = total - len(records)
    if args.limit:
        records = records[: args.limit]
    print(f"[load] {len(records)} record(s) to push ({skipped} skipped as not_relevant).", flush=True)

    rg_records = [build_record(rec, args.max_chars) for rec in records]
    print(f"[upload] sending {len(rg_records)} record(s)...", flush=True)
    dataset.records.log(rg_records)
    print(
        f"[done] pushed {len(rg_records)} records to dataset {args.dataset_name!r}.",
        flush=True,
    )


def merge_spans_and_details(
    text: str, spans: list[dict[str, Any]], details: list[Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reconstruct full event dicts from a SpanQuestion response and the
    companion detail JSON list. Positional entries (no explicit event_type/
    grounding_quote keys) are zipped in span order; entries with both those
    keys are treated as explicit (unmatched-at-push-time) events kept as-is."""
    warnings: list[str] = []
    spans_sorted = sorted(spans, key=lambda s: s.get("start", 0))

    positional: list[dict[str, Any]] = []
    explicit: list[dict[str, Any]] = []
    for entry in details:
        if isinstance(entry, dict) and "event_type" in entry and "grounding_quote" in entry:
            explicit.append(entry)
        elif isinstance(entry, dict):
            positional.append(entry)
        else:
            warnings.append(f"skipped non-object detail entry: {entry!r}")

    if len(positional) != len(spans_sorted):
        warnings.append(
            f"{len(spans_sorted)} highlighted span(s) but {len(positional)} positional "
            "detail entry/entries — extra/missing entries may be misaligned."
        )

    events: list[dict[str, Any]] = []
    for i, span in enumerate(spans_sorted):
        detail = positional[i] if i < len(positional) else {}
        events.append(
            {
                "event_type": span.get("label"),
                "grounding_quote": text[span["start"] : span["end"]],
                **{field: detail.get(field, "not_stated") for field in DETAIL_FIELDS},
            }
        )
    for detail in positional[len(spans_sorted) :]:
        warnings.append("extra positional detail entry with no matching span; kept under _orphan_detail")
        events.append({"_orphan_detail": detail})
    for entry in explicit:
        events.append(
            {
                "event_type": entry.get("event_type"),
                "grounding_quote": entry.get("grounding_quote"),
                **{field: entry.get(field, "not_stated") for field in DETAIL_FIELDS},
            }
        )
    return events, warnings


def validate_detail_dictionaries(details: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(details, list):
        return [f"{EVENT_DETAILS_JSON_QUESTION_NAME} must be a JSON list; got {type(details).__name__}"]

    detail_keys = set(DETAIL_FIELDS)
    event_keys = set(EVENT_FIELDS)
    for i, entry in enumerate(details):
        if not isinstance(entry, dict):
            errors.append(f"detail {i}: must be an object")
            continue

        has_event_type = "event_type" in entry
        has_grounding_quote = "grounding_quote" in entry
        if has_event_type != has_grounding_quote:
            errors.append(
                f"detail {i}: include both event_type and grounding_quote for an explicit "
                "unhighlighted event, or neither for a positional highlighted-span detail"
            )

        required_keys = event_keys if has_event_type and has_grounding_quote else detail_keys
        keys = set(entry.keys())
        for field in sorted(required_keys - keys):
            errors.append(f"detail {i}: missing required key {field!r}")
        for field in sorted(keys - required_keys):
            errors.append(f"detail {i}: unexpected key {field!r}")

        for field in sorted(keys & required_keys):
            value = entry[field]
            if not isinstance(value, str):
                errors.append(f"detail {i}: {field} must be a string; got {type(value).__name__}")
            elif value == "":
                errors.append(f"detail {i}: {field} is empty; use 'not_stated' when missing")

        for field, choices in CHOICE_DETAIL_FIELDS.items():
            value = entry.get(field)
            if isinstance(value, str) and value not in choices:
                errors.append(f"detail {i}: {field} must be one of {choices}; got {value!r}")
    return errors


def validate_events_annotation(
    document_relevance: str | None,
    events: list[dict[str, Any]],
    text: str,
    allowed_event_types: set[str],
) -> list[str]:
    errors: list[str] = []
    relevance_choices = list(DOCUMENT_RELEVANCE_LABELS)
    if document_relevance not in relevance_choices:
        errors.append(
            f"document_relevance must be one of {relevance_choices}; got {document_relevance!r}"
        )
    if document_relevance == "not_relevant" and events:
        errors.append("document_relevance is not_relevant but events is not empty")
    if document_relevance == "relevant" and not events:
        errors.append("document_relevance is relevant but events is empty")

    required_keys = set(EVENT_FIELDS)
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event {i}: must be an object")
            continue

        keys = set(event.keys())
        for field in EVENT_FIELDS:
            if field not in keys:
                errors.append(f"event {i}: missing required key {field!r}")
        for field in sorted(keys - required_keys):
            errors.append(f"event {i}: unexpected key {field!r}")

        for field in EVENT_FIELDS:
            if field not in event:
                continue
            value = event[field]
            if not isinstance(value, str):
                errors.append(f"event {i}: {field} must be a string; got {type(value).__name__}")
            elif value == "":
                errors.append(f"event {i}: {field} is empty; use 'not_stated' when missing")

        event_type = event.get("event_type")
        if isinstance(event_type, str) and event_type not in allowed_event_types:
            errors.append(f"event {i}: event_type {event_type!r} is not in the ontology")

        grounding_quote = event.get("grounding_quote")
        if not isinstance(grounding_quote, str) or grounding_quote in MISSING_TEXT_VALUES:
            errors.append(f"event {i}: grounding_quote is required and must be an article span")
        elif grounding_quote not in text:
            errors.append(
                f"event {i} [{event_type}]: grounding_quote={grounding_quote!r} not found verbatim in article"
            )

        for field in VERBATIM_FIELDS:
            value = event.get(field)
            if (
                isinstance(value, str)
                and value not in MISSING_TEXT_VALUES
                and value not in text
            ):
                errors.append(
                    f"event {i} [{event_type}]: {field}={value!r} not found verbatim in article"
                )

        for field, choices in CHOICE_DETAIL_FIELDS.items():
            value = event.get(field)
            if isinstance(value, str) and value not in choices:
                errors.append(f"event {i}: {field} must be one of {choices}; got {value!r}")
    return errors


def default_invalid_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.invalid{output_path.suffix}")


def export(args: argparse.Namespace) -> None:
    print("[connect] authenticating to Argilla...", flush=True)
    client = get_client()
    print("[connect] connected.", flush=True)
    allowed_event_types = set(load_allowed_event_types())

    if client.workspaces(args.workspace) is None:
        raise SystemExit(f"Workspace {args.workspace!r} not found.")
    dataset = client.datasets(name=args.dataset_name, workspace=args.workspace)
    if dataset is None:
        raise SystemExit(
            f"Dataset {args.dataset_name!r} not found in workspace {args.workspace!r}."
        )
    print(f"[fetch] downloading records from {args.dataset_name!r}...", flush=True)

    def submitted_value(record, question_name: str):
        # record.responses[name] is a defaultdict(list) — safe for missing keys.
        try:
            responses = list(record.responses[question_name])
        except KeyError:
            return None
        submitted = [
            r for r in responses if getattr(r, "status", "submitted") == "submitted"
        ]
        chosen = submitted[0] if submitted else (responses[0] if responses else None)
        return chosen.value if chosen else None

    def get_suggestion(record, question_name: str):
        # record.suggestions has no __contains__; iterating yields Suggestion
        # objects (not names), so "in" is always False. __getitem__ raises
        # KeyError for a missing question instead.
        try:
            return record.suggestions[question_name]
        except KeyError:
            return None

    rows = []
    invalid_rows = []
    parse_errors = 0
    merge_warning_count = 0
    validation_error_count = 0
    for record in dataset.records(with_suggestions=True, with_responses=True):
        spans_value = submitted_value(record, EVENT_SPANS_QUESTION_NAME) or []
        table_value = submitted_value(record, EVENT_DETAILS_TABLE_QUESTION_NAME)
        legacy_json_value = submitted_value(record, EVENT_DETAILS_JSON_QUESTION_NAME)
        relevance_value = submitted_value(record, RELEVANCE_QUESTION_NAME)

        if args.only_submitted and table_value is None and legacy_json_value is None:
            continue

        text = record.fields.get("text", "")

        details_parsed = None
        parse_error = None
        events: list[dict[str, Any]] = []
        merge_warnings: list[str] = []
        detail_errors: list[str] = []
        table_errors: list[str] = []
        if table_value is not None:
            events, table_errors = parse_events_table(table_value)
        elif legacy_json_value is not None:
            try:
                details_parsed = json.loads(legacy_json_value)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
                parse_errors += 1
            else:
                detail_errors = validate_detail_dictionaries(details_parsed)
                if isinstance(details_parsed, list):
                    events, merge_warnings = merge_spans_and_details(
                        text, spans_value, details_parsed
                    )
                    if merge_warnings:
                        merge_warning_count += 1

        model_table_suggestion = get_suggestion(record, EVENT_DETAILS_TABLE_QUESTION_NAME)
        model_details_suggestion = get_suggestion(record, EVENT_DETAILS_JSON_QUESTION_NAME)
        model_spans_suggestion = get_suggestion(record, EVENT_SPANS_QUESTION_NAME)
        model_events = None
        if model_table_suggestion is not None:
            model_events, _ = parse_events_table(model_table_suggestion.value)
        elif model_details_suggestion is not None:
            model_events, _ = merge_spans_and_details(
                text,
                model_spans_suggestion.value if model_spans_suggestion else [],
                json.loads(model_details_suggestion.value),
            )

        validation_errors = []
        if parse_error is not None:
            validation_errors.append(
                f"{EVENT_DETAILS_JSON_QUESTION_NAME} is invalid JSON: {parse_error}"
            )
        validation_errors.extend(table_errors)
        validation_errors.extend(detail_errors)
        validation_errors.extend(merge_warnings)
        validation_errors.extend(
            validate_events_annotation(
                relevance_value, events, text, allowed_event_types
            )
        )
        if validation_errors:
            validation_error_count += 1

        row = {
            "id": record.id,
            "title": record.fields.get("title"),
            "text": text,
            "annotation": {
                "document_relevance": relevance_value,
                "events": events,
            },
            "events_parse_error": parse_error,
            "events_table_errors": table_errors or None,
            "events_detail_errors": detail_errors or None,
            "events_merge_warnings": merge_warnings or None,
            "events_validation_errors": validation_errors or None,
            "model_annotation": {
                "events": model_events,
            },
        }
        if validation_errors:
            invalid_rows.append(
                {
                    **row,
                    EVENT_DETAILS_TABLE_QUESTION_NAME: table_value,
                    EVENT_DETAILS_JSON_QUESTION_NAME: legacy_json_value,
                }
            )
        if not args.valid_only or not validation_errors:
            rows.append(row)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    invalid_output_path = (
        Path(args.invalid_output)
        if args.invalid_output
        else default_invalid_output_path(output_path)
    )
    if invalid_rows:
        with invalid_output_path.open("w", encoding="utf-8") as handle:
            for row in invalid_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} records to {output_path}.")
    if invalid_rows:
        print(f"Wrote {len(invalid_rows)} invalid record(s) to {invalid_output_path}.")
    if parse_errors:
        print(
            f"Warning: {parse_errors} record(s) had invalid JSON in "
            f"{EVENT_DETAILS_JSON_QUESTION_NAME!r}; see events_parse_error in the output."
        )
    if merge_warning_count:
        print(
            f"Warning: {merge_warning_count} record(s) had span/detail count mismatches; "
            "see events_merge_warnings in the output."
        )
    if validation_error_count:
        print(
            f"Warning: {validation_error_count} record(s) had validation errors; "
            "see events_validation_errors in the output."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push articles to Argilla for event annotation, or export human-corrected events."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Path to a .env file with ARGILLA_API_URL / ARGILLA_API_KEY (default: repo .env).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_parser = subparsers.add_parser(
        "push", help="Load a JSONL file of articles into Argilla."
    )
    push_parser.add_argument(
        "--input", required=True, type=str, help="Input JSONL file"
    )
    push_parser.add_argument("--dataset-name", required=True, type=str)
    push_parser.add_argument("--workspace", type=str, default=DEFAULT_WORKSPACE)
    push_parser.add_argument("--limit", type=int, default=None)
    push_parser.add_argument(
        "--update-settings",
        action="store_true",
        help="For an existing dataset, update its schema/guidelines to the current event table UI.",
    )
    push_parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Truncate article text to this many characters (0 = no truncation, the default — "
        "span offsets are computed against the text actually shown, so truncating can push "
        "later events' grounding_quote out of reach).",
    )
    push_parser.set_defaults(func=push)

    export_parser = subparsers.add_parser(
        "export",
        help="Export human-corrected events (and model suggestions) to JSONL.",
    )
    export_parser.add_argument("--dataset-name", required=True, type=str)
    export_parser.add_argument("--workspace", type=str, default=DEFAULT_WORKSPACE)
    export_parser.add_argument("--output", required=True, type=str)
    export_parser.add_argument(
        "--invalid-output",
        type=str,
        default=None,
        help="Where to write invalid rows for review (default: <output stem>.invalid<suffix>).",
    )
    export_parser.add_argument(
        "--only-submitted",
        action="store_true",
        help="Only export records with a submitted human response.",
    )
    export_parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Write only valid rows to --output; invalid rows still go to --invalid-output.",
    )
    export_parser.set_defaults(func=export)

    args = parser.parse_args()
    load_dotenv(args.env_file)
    args.func(args)


if __name__ == "__main__":
    main()
