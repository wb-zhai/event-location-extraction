"""Push articles into Label Studio for human event-annotation review, and export the
resulting human-corrected event structures alongside the model's predictions.

Unlike Argilla (see events_argilla.py), Label Studio supports per-region structured
fields natively: highlighting a span with the `event_type` Labels control opens a
details panel (TextArea/Choices controls, `perRegion="true"`) that annotators fill in
directly for that span. There is no hand-typed JSON blob and no positional
span/detail matching to get wrong.

The two evidence-quote fields (`event_location_text`, `event_time_text`) are free-text
fields in that panel, pre-filled from the model's quote. The flat UI can't enforce that
they stay verbatim article text, so export() keeps invalid rows out of the accepted
JSONL and writes them to a review file with validation errors.

See scripts/annotations/README.md for setup and usage.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.annotations.events_argilla import (
    DETAIL_FIELDS,
    SYSTEM_PROMPT_PATH,
    extract_title_text,
    iter_jsonl,
    keep_by_relevance,
    model_name,
    split_matched_unmatched,
)
from scripts.data.relevance.relevance_filter import _record_key

DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "ontologies" / "zhai" / "science.json"

TEXT_NAME = "text"
EVENT_TYPE_NAME = "event_type"
DOCUMENT_RELEVANCE_NAME = "document_relevance"
EVENT_FIELDS = ["event_type", "grounding_quote", *DETAIL_FIELDS]

# The 6 detail fields edited as free text vs. the 3 edited as a fixed choice set
# (event_type + grounding_quote come from the highlighted span itself).
TEXT_DETAIL_FIELDS = [
    "event_location_text",
    "event_location",
    "event_time_text",
    "event_time",
    "affected_entity",
    "affected_group",
]
CHOICE_DETAIL_FIELDS = {
    "time_status": ["past", "ongoing", "forecast", "not_stated"],
    "severity": ["low", "medium", "high", "extreme", "not_stated"],
    "modality": ["asserted", "projected"],
}
DOCUMENT_RELEVANCE_CHOICES = ["relevant", "not_relevant"]

# event_location_text / event_time_text must be verbatim quotes from the article.
# The flat UI can't enforce that at entry time, so export() validates them.
VERBATIM_FIELDS = ["event_location_text", "event_time_text"]
MISSING_TEXT_VALUES = {"", "not_stated"}

EVENT_TIME_HINT = (
    "ISO-8601: YYYY | YYYY-MM | YYYY-MM-DD | range X/Y | open-ended X/ or /Y | not_stated"
)
# Per-field placeholders for the free-text detail controls (mirror the prompt spec).
FIELD_PLACEHOLDERS = {
    "event_location_text": "verbatim quote from the article with location evidence, or not_stated",
    "event_location": "geocodable place name(s) from the location quote, ';'-separated, or not_stated",
    "event_time_text": "verbatim quote from the article with time evidence, or not_stated",
    "event_time": EVENT_TIME_HINT,
    "affected_entity": "thing/system/asset/economy/resource affected, or not_stated",
    "affected_group": "people/population group affected (never a country/org), or not_stated",
}


def load_ontology_event_types(path: Path) -> list[str]:
    """Load event_type labels from an ontology file shaped as {"events": {...}}."""
    with path.open("r", encoding="utf-8") as handle:
        ontology = json.load(handle)
    if not isinstance(ontology, dict):
        raise ValueError(f"Ontology at {path} must be a JSON object.")
    events = ontology.get("events")
    if not isinstance(events, dict):
        raise ValueError(f"Ontology at {path} must contain an object at key 'events'.")
    event_types = [str(label) for label in events.keys() if str(label).strip()]
    if not event_types:
        raise ValueError(f"Ontology at {path} does not define any event types.")
    return event_types


GUIDELINES_TEMPLATE = """Validate, correct, and add to the extracted events for this article.
See {prompt_path} for the full extraction spec (few-shot examples, edge-case rules)
this condensed version is drawn from.

<workflow>
1. Highlight each event mention in the article text and tag it with its event_type from
   the list on the left (use the filter box to search; must be one of allowed_event_types
   below). The highlighted span becomes that event's grounding_quote.
2. Click the highlighted span to select it, then fill in its 9 detail fields in the panel
   below the text: event_location_text, event_location, event_time_text, event_time,
   time_status, affected_entity, affected_group, severity, modality.
   event_location_text and event_time_text must be copied verbatim from the article
   (values not found in the text are flagged on export); event_time uses {event_time_hint}.
3. To remove an event, delete its highlighted span (region list on the right).
4. Set document_relevance at the bottom, then submit — even if the document has no valid
   events (delete any spans and leave the detail fields blank).
</workflow>

<output_schema (per event)>
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
        event_time_hint=EVENT_TIME_HINT,
        allowed_event_types="\n".join(allowed_event_types),
    )


def build_label_config(allowed_event_types: list[str]) -> str:
    labels_xml = "\n".join(
        f"    <Label value={quoteattr(t)}/>" for t in allowed_event_types
    )
    choice_blocks = []
    for field, options in CHOICE_DETAIL_FIELDS.items():
        options_xml = "\n".join(f'      <Choice value="{o}"/>' for o in options)
        choice_blocks.append(
            f'    <Choices name="{field}" toName="{TEXT_NAME}" choice="single" perRegion="true">\n'
            f"{options_xml}\n"
            f"    </Choices>"
        )
    text_blocks = "\n".join(
        f'    <TextArea name="{field}" toName="{TEXT_NAME}" perRegion="true" '
        f"rows=\"1\" editable=\"true\" "
        f"placeholder={quoteattr(FIELD_PLACEHOLDERS.get(field, f'{field} (or not_stated)'))}/>"
        for field in TEXT_DETAIL_FIELDS
    )
    return f"""<View>
  <Text name="title_field" value="$title"/>
  <Text name="unmatched_note" value="$unmatched_note"/>

  <Filter name="filter" toName="{EVENT_TYPE_NAME}" minlength="0" hotkey="shift+f" placeholder="Filter event types"/>
  <Labels name="{EVENT_TYPE_NAME}" toName="{TEXT_NAME}" showInline="false">
{labels_xml}
  </Labels>

  <Text name="{TEXT_NAME}" value="$text"/>

  <View visibleWhen="region-selected" whenTagName="{EVENT_TYPE_NAME}">
    <Header value="Selected event — fill in details" size="5"/>
    <Header value="event_time format — {EVENT_TIME_HINT}" size="6"/>
{text_blocks}
{chr(10).join(choice_blocks)}
  </View>

  <Header value="Document relevance" size="5"/>
  <Choices name="{DOCUMENT_RELEVANCE_NAME}" toName="{TEXT_NAME}" choice="single">
    <Choice value="relevant"/>
    <Choice value="not_relevant"/>
  </Choices>
</View>
"""


def get_env() -> tuple[str, str]:
    url = os.environ.get("LABEL_STUDIO_URL")
    token = os.environ.get("LABEL_STUDIO_API_KEY")
    if not url or not token:
        raise SystemExit(
            "LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY must be set (see .env or "
            "scripts/annotations/README.md)."
        )
    return url.rstrip("/"), token


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Token {token}"}


def _get(url: str, token: str, path: str, **kwargs: Any) -> Any:
    resp = requests.get(f"{url}{path}", headers=_headers(token), timeout=60, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _post(url: str, token: str, path: str, json_body: Any, **kwargs: Any) -> Any:
    resp = requests.post(
        f"{url}{path}", headers=_headers(token), json=json_body, timeout=120, **kwargs
    )
    resp.raise_for_status()
    return resp.json()


def find_project(url: str, token: str, title: str) -> dict[str, Any] | None:
    data = _get(url, token, "/api/projects/", params={"title": title})
    results = data.get("results", []) if isinstance(data, dict) else data
    for project in results:
        if project.get("title") == title:
            return project
    return None


def get_or_create_project(
    url: str, token: str, title: str, label_config: str, instructions: str
) -> dict[str, Any]:
    project = find_project(url, token, title)
    if project is not None:
        print(f"[project] found existing project {title!r} (id={project['id']}).", flush=True)
        return project
    print(f"[project] {title!r} not found, creating...", flush=True)
    project = _post(
        url,
        token,
        "/api/projects/",
        {
            "title": title,
            "label_config": label_config,
            "expert_instruction": f'<div style="white-space:pre-wrap">{escape(instructions)}</div>',
            "show_instruction": True,
        },
    )
    print(f"[project] created {title!r} (id={project['id']}).", flush=True)
    return project


def _list_tasks(
    url: str, token: str, project_id: int, fields: str = "task_only", page_size: int = 200
) -> list[dict[str, Any]]:
    all_tasks: list[dict[str, Any]] = []
    page = 1
    while True:
        data = _get(
            url,
            token,
            "/api/tasks/",
            params={
                "project": project_id,
                "page": page,
                "page_size": page_size,
                "fields": fields,
            },
        )
        tasks = data.get("tasks", [])
        all_tasks.extend(tasks)
        total = data.get("total", len(all_tasks))
        if not tasks or len(all_tasks) >= total:
            break
        page += 1
    return all_tasks


def build_task(
    rec: dict[str, Any], max_chars: int, allowed_event_types: set[str] | None = None
) -> dict[str, Any]:

    title, text = extract_title_text(rec)
    if max_chars:
        text = text[:max_chars]
    annotation = rec.get("annotation") or {}
    if not isinstance(annotation, dict):
        annotation = {}

    document_relevance = annotation.get("document_relevance")
    events = annotation.get("events") or []
    matched, unmatched = split_matched_unmatched(events, text)
    if allowed_event_types is not None:
        valid_matched = []
        invalid_type = []
        for start, end, event_type, detail in matched:
            if event_type in allowed_event_types:
                valid_matched.append((start, end, event_type, detail))
            else:
                invalid_type.append(
                    {
                        "event_type": event_type,
                        "grounding_quote": text[start:end],
                        **detail,
                    }
                )
        matched = valid_matched
        unmatched = unmatched + invalid_type

    data: dict[str, Any] = {
        "title": title,
        "text": text,
        "unmatched_note": (
            (
                f"{len(unmatched)} model event(s) could not be pre-highlighted "
                "(quote not found verbatim, or event_type not in the ontology); "
                "add them manually if still valid:\n"
                + "\n".join(
                    f"- [{e['event_type']}] {e['grounding_quote']!r}" for e in unmatched
                )
            )
            if unmatched
            else ""
        ),
    }
    key = _record_key(rec)
    if key:
        data["item_id"] = key
    adm0_code = rec.get("adm0_code")
    if adm0_code:
        data["adm0_code"] = str(adm0_code)
    risk_factors = rec.get("risk_factors")
    if risk_factors:
        data["risk_factors"] = [str(r) for r in risk_factors]
    if rec.get("generation_llm"):
        data["generation_llm"] = model_name(rec)
    quality_score = rec.get("quality_score")
    if quality_score is not None:
        data["quality_score"] = float(quality_score)

    result: list[dict[str, Any]] = []
    for start, end, event_type, detail in matched:
        region_id = f"r{start}_{end}"
        result.append(
            {
                "id": region_id,
                "from_name": EVENT_TYPE_NAME,
                "to_name": TEXT_NAME,
                "type": "labels",
                "value": {"start": start, "end": end, "text": text[start:end], "labels": [event_type]},
            }
        )
        for field in TEXT_DETAIL_FIELDS:
            result.append(
                {
                    "id": region_id,
                    "from_name": field,
                    "to_name": TEXT_NAME,
                    "type": "textarea",
                    "value": {"text": [str(detail.get(field, "not_stated"))]},
                }
            )
        for field, options in CHOICE_DETAIL_FIELDS.items():
            value = str(detail.get(field, ""))
            if value not in options:
                continue
            result.append(
                {
                    "id": region_id,
                    "from_name": field,
                    "to_name": TEXT_NAME,
                    "type": "choices",
                    "value": {"choices": [value]},
                }
            )
    if document_relevance in DOCUMENT_RELEVANCE_CHOICES:
        result.append(
            {
                "from_name": DOCUMENT_RELEVANCE_NAME,
                "to_name": TEXT_NAME,
                "type": "choices",
                "value": {"choices": [document_relevance]},
            }
        )

    task: dict[str, Any] = {"data": data}
    if result:
        task["predictions"] = [{"model_version": model_name(rec), "result": result}]
    return task


def push(args: argparse.Namespace) -> None:
    url, token = get_env()
    print(f"[connect] using Label Studio at {url}.", flush=True)

    allowed_event_types = load_ontology_event_types(args.ontology)
    allowed_event_type_set = set(allowed_event_types)
    label_config = build_label_config(allowed_event_types)
    guidelines = load_guidelines(allowed_event_types)
    project = get_or_create_project(url, token, args.project_name, label_config, guidelines)

    print(f"[load] reading {args.input}...", flush=True)
    records = iter_jsonl(Path(args.input))
    total = len(records)
    records = [rec for rec in records if keep_by_relevance(rec)]
    skipped = total - len(records)
    if args.limit:
        records = records[: args.limit]
    print(f"[load] {len(records)} record(s) to push ({skipped} skipped as not_relevant).", flush=True)

    existing_ids: set[str] = set()
    if not args.force:
        for task in _list_tasks(url, token, project["id"], fields="task_only"):
            item_id = (task.get("data") or {}).get("item_id")
            if item_id:
                existing_ids.add(item_id)

    tasks = []
    dup_skipped = 0
    for rec in records:
        key = _record_key(rec)
        if key and key in existing_ids:
            dup_skipped += 1
            continue
        tasks.append(build_task(rec, args.max_chars, allowed_event_type_set))
    if dup_skipped:
        print(
            f"[load] {dup_skipped} record(s) already present in project, skipping "
            "(use --force to re-push anyway).",
            flush=True,
        )
    if not tasks:
        print("[done] nothing new to push.", flush=True)
        return

    print(f"[upload] importing {len(tasks)} task(s)...", flush=True)
    _post(url, token, f"/api/projects/{project['id']}/import", tasks)
    print(f"[done] pushed {len(tasks)} task(s) to project {args.project_name!r}.", flush=True)


def parse_annotation_result(result: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """Reconstruct full event dicts from a Label Studio annotation/prediction
    `result` list. Per-region controls (event_type span + its 9 detail fields)
    share the same `id`, so they're grouped by id rather than matched by
    position — unlike the Argilla version, there's no span/detail count to
    get out of sync."""
    regions: dict[str, dict[str, Any]] = {}
    document_relevance = None
    for item in result:
        from_name = item.get("from_name")
        value = item.get("value") or {}
        if from_name == DOCUMENT_RELEVANCE_NAME:
            choices = value.get("choices") or []
            document_relevance = choices[0] if choices else None
            continue
        region_id = item.get("id")
        if region_id is None:
            continue
        region = regions.setdefault(region_id, {"start": 0})
        if from_name == EVENT_TYPE_NAME:
            labels = value.get("labels") or []
            region["event_type"] = labels[0] if labels else None
            region["grounding_quote"] = value.get("text", "")
            region["start"] = value.get("start", 0)
        elif from_name in TEXT_DETAIL_FIELDS:
            texts = value.get("text") or []
            region[from_name] = texts[0] if texts else "not_stated"
        elif from_name in CHOICE_DETAIL_FIELDS:
            choices = value.get("choices") or []
            region[from_name] = choices[0] if choices else "not_stated"

    regions_with_events = [r for r in regions.values() if "event_type" in r]
    regions_with_events.sort(key=lambda r: r["start"])
    events = [
        {
            "event_type": r.get("event_type"),
            "grounding_quote": r.get("grounding_quote", ""),
            **{field: r.get(field, "not_stated") for field in DETAIL_FIELDS},
        }
        for r in regions_with_events
    ]
    return events, document_relevance


def check_verbatim_fields(events: list[dict[str, Any]], text: str) -> list[str]:
    """Flag any VERBATIM_FIELDS value (event_location_text / event_time_text)
    that isn't 'not_stated' and isn't found verbatim in the article — the flat
    UI lets annotators type these freely, so this is the verbatim guardrail."""
    warnings: list[str] = []
    for i, event in enumerate(events):
        for field in VERBATIM_FIELDS:
            value = str(event.get(field, "not_stated"))
            if value and value != "not_stated" and value not in text:
                warnings.append(f"event {i} [{event.get('event_type')}]: {field}={value!r} not found verbatim in article")
    return warnings


def validate_events_annotation(
    document_relevance: str | None,
    events: list[dict[str, Any]],
    text: str,
    allowed_event_types: set[str],
) -> list[str]:
    errors: list[str] = []
    if document_relevance not in DOCUMENT_RELEVANCE_CHOICES:
        errors.append(
            f"document_relevance must be one of {DOCUMENT_RELEVANCE_CHOICES}; got {document_relevance!r}"
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
    url, token = get_env()
    print(f"[connect] using Label Studio at {url}.", flush=True)
    allowed_event_types = set(load_ontology_event_types(args.ontology))

    project = find_project(url, token, args.project_name)
    if project is None:
        raise SystemExit(f"Project {args.project_name!r} not found.")
    print(f"[fetch] downloading tasks from {args.project_name!r}...", flush=True)
    tasks = _list_tasks(url, token, project["id"], fields="all")

    rows = []
    invalid_rows = []
    for task in tasks:
        data = task.get("data") or {}
        text = data.get("text", "")
        annotations = [a for a in (task.get("annotations") or []) if not a.get("was_cancelled")]
        if args.only_submitted and not annotations:
            continue

        events: list[dict[str, Any]] = []
        document_relevance = None
        if annotations:
            events, document_relevance = parse_annotation_result(annotations[-1].get("result") or [])

        predictions = task.get("predictions") or []
        model_events = None
        if predictions:
            model_events, _ = parse_annotation_result(predictions[0].get("result") or [])

        row = {
            "id": data.get("item_id") or task.get("id"),
            "title": data.get("title"),
            "text": text,
            "annotation": {
                "document_relevance": document_relevance,
                "events": events,
            },
            "model_annotation": {"events": model_events},
        }
        validation_errors = validate_events_annotation(
            document_relevance, events, text, allowed_event_types
        )
        if validation_errors:
            invalid_rows.append({**row, "events_validation_errors": validation_errors})
        else:
            rows.append(row)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    invalid_output_path = Path(args.invalid_output) if args.invalid_output else default_invalid_output_path(output_path)
    if invalid_rows:
        with invalid_output_path.open("w", encoding="utf-8") as handle:
            for row in invalid_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} valid record(s) to {output_path}.")
    if invalid_rows:
        print(f"Wrote {len(invalid_rows)} invalid record(s) to {invalid_output_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push articles to Label Studio for event annotation, or export human-corrected events."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Path to a .env file with LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY (default: repo .env).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_parser = subparsers.add_parser(
        "push", help="Load a JSONL file of articles into Label Studio."
    )
    push_parser.add_argument("--input", required=True, type=str, help="Input JSONL file")
    push_parser.add_argument("--project-name", required=True, type=str)
    push_parser.add_argument(
        "--ontology",
        type=Path,
        default=DEFAULT_ONTOLOGY_PATH,
        help="Ontology JSON with event labels under events (default: ontologies/zhai/science.json).",
    )
    push_parser.add_argument("--limit", type=int, default=None)
    push_parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Truncate article text to this many characters (0 = no truncation, the default — "
        "span offsets are computed against the text actually shown, so truncating can push "
        "later events' grounding_quote out of reach).",
    )
    push_parser.add_argument(
        "--force",
        action="store_true",
        help="Push all matching records even if their id is already present in the project "
        "(default: skip already-pushed records, detected via the task's item_id).",
    )
    push_parser.set_defaults(func=push)

    export_parser = subparsers.add_parser(
        "export",
        help="Export human-corrected events (and model predictions) to JSONL.",
    )
    export_parser.add_argument("--project-name", required=True, type=str)
    export_parser.add_argument("--output", required=True, type=str)
    export_parser.add_argument(
        "--invalid-output",
        type=str,
        default=None,
        help="Where to write invalid rows for review (default: <output stem>.invalid<suffix>).",
    )
    export_parser.add_argument(
        "--ontology",
        type=Path,
        default=DEFAULT_ONTOLOGY_PATH,
        help="Ontology JSON with event labels under events (default: ontologies/zhai/science.json).",
    )
    export_parser.add_argument(
        "--only-submitted",
        action="store_true",
        help="Only export tasks with at least one (non-cancelled) annotation.",
    )
    export_parser.set_defaults(func=export)

    args = parser.parse_args()
    load_dotenv(args.env_file)
    args.func(args)


if __name__ == "__main__":
    main()
