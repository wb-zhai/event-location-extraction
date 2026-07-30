"""Push articles into Argilla for human event-annotation review, and export the
resulting human-corrected event structures alongside the model's predictions.

The annotator reads the article and edits a single JSON field
(`event_details_json`) holding the full list of event objects for that
article — no span highlighting, no separate table. This is the simplest UI
for annotators: one field to read, one field to edit.

See scripts/annotations/README.md for setup and usage.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import argilla as rg

from dotenv import load_dotenv

from scripts.annotations.relevance_argilla import normalize_decision
from scripts.relevance.relevance_filter import _record_key

RELEVANT_DECISIONS = {"relevant", "partially_relevant"}

EVENT_DETAILS_JSON_QUESTION_NAME = "event_details_json"
DEFAULT_WORKSPACE = "default"

DETAIL_FIELDS = [
    "event_location_text",
    "event_location",
    "event_location_admin_level",
    "event_time_text",
    "event_time",
    "time_status",
    "severity",
]
EVENT_FIELDS = ["event_type", "grounding_quote", *DETAIL_FIELDS]
CHOICE_DETAIL_FIELDS = {
    "time_status": ["past", "ongoing", "forecast", "not_stated"],
    "severity": ["low", "medium", "high", "extreme", "not_stated"],
}
ADMIN_LEVEL_CHOICES = ["country", "state", "county", "city", "district", "not_stated"]
VERBATIM_FIELDS = ["event_location_text", "event_time_text"]
MISSING_TEXT_VALUES = {"", "not_stated"}

EVENT_TYPE_REFERENCE_FIELD = "event_type_reference"
REASONING_TRACE_FIELD = "reasoning_trace"

# One-line, human-readable description per allowed event_type, condensed from the
# teacher prompt's <event_type_rules>. Shown to annotators as a reference field
# under the article so they can pick the single most specific type. Keyed by the
# exact ontology label; any allowed type missing here falls back to just its name.
EVENT_TYPE_DESCRIPTIONS = {
    "crop failure": "Reduced or failed crop yields, declining farm output (tag the cause — drought, pest, flooding — separately).",
    "agricultural infrastructure damage": "Damage to irrigation systems, grain storage, mills, or rural roads.",
    "land degradation": "Deforestation, soil erosion, desertification, farmland loss.",
    "livestock loss": "Animal disease, herd losses, distress sale of livestock, fodder or pasture shortage.",
    "agricultural input shortage": "Shortage or unaffordability of fertilizer, seeds, fuel, pesticides, or animal feed.",
    "pest infestation": "Locusts, armyworm, or other crop-damaging pest invasions.",
    "armed violence": "Warfare, armed clashes, conflict outbreak or escalation, attacks by military forces (default active-conflict label).",
    "aerial bombardment": "Bombing, shelling, airstrikes, artillery fire.",
    "supply route blockade": "Physical blockade of roads, corridors, or transport routes.",
    "militant activity": "Presence, operations, or attacks of insurgent, militant, or terrorist groups (prefer armed violence if two organized forces fight).",
    "looting": "Robbery, banditry, plundering, organized theft of property.",
    "human rights violation": "Repression, persecution, abuse of human or civil rights.",
    "civilian casualties": "Civilian deaths or injuries, including mass-casualty incidents (tag the causing event separately).",
    "protests": "Demonstrations, riots, strikes, food riots, civil unrest.",
    "infrastructure destruction": "Destruction or severe damage of critical infrastructure — roads, bridges, power plants, water systems, hospitals, buildings.",
    "internal displacement": "People forced to flee or leave their homes within their own country.",
    "refugee movement": "Refugees crossing international borders, or refugee arrivals and situations in host countries.",
    "migrant influx": "Arrival or inflow of migrants into an area, framed from the receiving side.",
    "shelter loss": "Lack or loss of adequate shelter or housing (tag the cause — fire, flooding, armed violence — separately).",
    "economic decline": "Macro-level economic downturn: recession, sector strain, economic crisis, currency depreciation.",
    "poverty": "Rising poverty rates, worsening household economic hardship.",
    "food price inflation": "Rising food prices, food becoming unaffordable, staple price spikes.",
    "trade disruption": "Embargoes, sanctions, import/export bans, trade or border restrictions with a stated trade mechanism.",
    "environmental degradation": "General degradation of the natural environment not captured by a more specific label.",
    "resource depletion": "Depletion or exhaustion of natural resources — minerals, groundwater, forests as a resource stock.",
    "pollution": "Air, soil, or general environmental pollution (water-specific contamination -> water contamination).",
    "ecosystem collapse": "Destruction or collapse of terrestrial or marine ecosystems, depleted fish stocks, biodiversity loss.",
    "power outage": "Electricity supply failures without physical destruction.",
    "fire": "House or building fires, urban fires, wildfires (tag consequences separately).",
    "food scarcity": "Physical shortage or lack of food in markets or households, food deficit, hunger increase.",
    "famine": "Declared or reported famine conditions, extreme and widespread food deprivation.",
    "aid reduction": "Cuts or reductions in humanitarian aid programs or delivery volumes.",
    "aid obstruction": "Obstruction, restriction, or impediment of humanitarian aid access and delivery.",
    "aid diversion": "Theft or diversion of humanitarian aid, stolen food rations, misappropriated relief supplies.",
    "funding shortfall": "Decline or shortfall in donor funding for humanitarian programs.",
    "governance breakdown": "Political instability, misgovernance, breakdown of state functions within a country.",
    "geopolitical tension": "Tensions, disputes, or instability between countries or in international relations.",
    "disease spread": "Epidemics, outbreaks, virus transmission, elevated disease risk (only when a human disease is explicitly named).",
    "health service disruption": "Disruption of health services or care access.",
    "water contamination": "Contamination of water sources: polluted drinking water, sewage overflow into water systems.",
    "flooding": "River floods, flash floods, coastal or seawater inundation.",
    "drought": "Drought conditions, prolonged precipitation deficits.",
    "water shortage": "Reduced water availability: declining river flows, drained or depleted sources, water supply scarcity.",
    "abnormal rainfall": "Abnormally heavy or increased rainfall, including monsoon rains (also tag flooding if it floods).",
    "extreme heat": "Heat waves, abnormal temperature increases.",
    "cyclone": "Tropical cyclones, hurricanes, major storms, destructive high winds.",
    "earthquake": "Earthquakes and seismic events.",
    "landslide": "Landslides and mass ground movement, often following heavy rain or earthquakes.",
}


def render_event_type_reference(allowed_event_types: list[str]) -> str:
    lines = [
        "Pick the single **most specific** type each event supports. If nothing here fits, the event is out of scope.",
        "",
    ]
    for event_type in allowed_event_types:
        description = EVENT_TYPE_DESCRIPTIONS.get(event_type)
        if description:
            lines.append(f"- **{event_type}** — {description}")
        else:
            lines.append(f"- **{event_type}**")
    return "\n".join(lines)

EVENT_JSON_TEMPLATE = """```json
[
  {
    "event_type": "crop failure",
    "grounding_quote": "The 1995 grain harvest in Shandong province suffered a significant decline, falling by 2.7 million tons",
    "event_location_text": "in Shandong province",
    "event_location": "Shandong province",
    "event_location_admin_level": "state",
    "event_time_text": "The 1995 grain harvest",
    "event_time": "1995",
    "time_status": "past",
    "severity": "medium"
  }
]
```"""

SYSTEM_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "event_extraction"
    / "generation"
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


# Human-readable rewrite of the teacher system prompt (which is written as an
# LLM system prompt) so annotators apply the same criteria as the extraction
# model. Formatted as markdown since Argilla renders the guidelines field as
# markdown. Argilla caps guidelines at 10,000 characters; the full teacher
# prompt (with few-shot examples etc.) runs ~22,000, so this condensed version
# keeps the schema/field rules. The full allowed-type list with descriptions is
# shown as a reference field under each article, not repeated here.
GUIDELINES_TEMPLATE = """Review and correct the risk events extracted from this article.

Each article arrives **pre-filled with the model's extraction** as a suggestion in the
**Events** field. Your job is to validate and fix that extraction, not to label from
scratch: confirm the events the model got right, correct the ones it got wrong, delete
anything that isn't a valid event, and add any events it missed.

## Workflow

1. Read the article (title and text).
2. Open the **Events** field — it holds a JSON list of event objects, one per event, seeded
   from the model's prediction.
3. Go through it: fix wrong field values, delete entries that aren't valid events, and add
   new entries for events the model missed. Keep every object to exactly the nine keys below.
4. Submit `[]` if the article contains no valid events.

## What counts as an event

- Extract **one event per distinct, explicitly stated** risk event that matches an allowed
  type below. Don't merge distinct events, and don't duplicate repeated mentions of the same
  event unless a later mention adds new information.
- Use **only the article text and its publish date** as evidence — never world knowledge,
  geography knowledge, source names, URLs, or outside metadata.
- **Never invent** locations, dates, severity, or event types the text doesn't support. When
  something isn't stated, use `not_stated`.
- Historical or background events still count if they are concrete and explicitly stated.
- Skip vague implications, and skip anything whose most specific fitting type isn't in the
  **Allowed event types** reference shown under the article.

## The nine fields (per event)

- **event_type** — must exactly match one of the allowed types below. Use the most specific
  one the text directly supports.
- **grounding_quote** — the shortest exact, contiguous quote from the article that justifies
  the event_type. Must appear verbatim in the text.
- **event_location_text** — the shortest verbatim quote carrying the location evidence, or
  `not_stated`. It can come from a nearby sentence, not just the grounding quote — scan the
  whole paragraph before deciding a location is absent.
- **event_location** — a geocodable place name derived from event_location_text; multiple
  places are `;`-separated in article order, or `not_stated`. Strip vague directional/zonal
  prefixes ("southwestern Bangladesh" -> "Bangladesh") but keep proper administrative names
  ("North Darfur state" -> "North Darfur"). Broad scopes ("world", "many countries", "globally")
  are `not_stated`. Livelihood/pastoral zones aren't geocodable — use the named region/country
  containing them, or `not_stated`.
- **event_location_admin_level** — the administrative level of each place in event_location:
  `country` · `state` (province/region) · `county` (district/prefecture within a state) ·
  `city` · `district` (a neighborhood/borough within a city) · `not_stated`. For multiple
  places, give one `;`-separated value per place in the same order as event_location (e.g.
  event_location `"Germany; Italy"` -> event_location_admin_level `"country; country"`).
  `not_stated` whenever event_location is `not_stated`, or the place doesn't cleanly fit one
  of these levels (e.g. a multi-country region like "South Asia").
- **event_time_text** — the shortest verbatim quote carrying the time evidence, copied in the
  article's own words (e.g. "last month", not the resolved date), or `not_stated`.
- **event_time** — ISO 8601 derived from event_time_text: `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`;
  a range as `X/Y`; open-ended as `X/` or `/Y`; else `not_stated`. Resolve relative expressions
  ("last month", "today", "currently") against the publish date; drop vague qualifiers
  (early/mid/late/season) to just the year.
- **time_status** — `past` (completed/historical) · `ongoing` (current/continuing/worsening) ·
  `forecast` (expected/projected/predicted/planned/warned about) · `not_stated`.
- **severity** — `low` · `medium` · `high` · `extreme` · `not_stated`. `not_stated` is the
  **default**, not a last resort — assign a level **only** when the text uses explicit
  severity language (severe, major, catastrophic, etc.) describing that event's impact.
  Worsening or trajectory language alone (intensifies, escalates, deteriorates) does **not**
  imply high. Never infer severity from a number alone, and never pick `low`/`medium` as a
  safer-sounding guess when explicit language is missing — use `not_stated` instead.

## Picking event_type — common distinctions

- Two organized forces fighting -> `armed violence`; insurgent/terrorist group presence or
  attacks -> `militant activity`; bombing/shelling/airstrikes -> `aerial bombardment`.
- Tag consequences as **separate** events: civilian deaths/injuries -> `civilian casualties`;
  destroyed bridges/hospitals/power plants -> `infrastructure destruction` (farm-specific
  facilities -> `agricultural infrastructure damage`; electricity supply failure without
  physical destruction -> `power outage`).
- Reported shortage/deficit of food -> `food scarcity`; declared/extreme famine conditions ->
  `famine`.
- Precipitation deficit -> `drought`; reduced water availability/supply -> `water shortage`;
  heavy/increased rainfall -> `abnormal rainfall` (also tag `flooding` if it floods).
- A spreading human disease itself -> `disease spread`; disruption of care/health services ->
  `health service disruption`.
- Macro downturn/recession/currency depreciation -> `economic decline`; embargoes/sanctions/
  import-export bans with a stated trade mechanism -> `trade disruption`.

See the **Allowed event types** reference under the article for the full list of types and
what each one covers.
"""


def load_guidelines() -> str:
    return GUIDELINES_TEMPLATE


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
    print(f"[connect] using Argilla server {api_url!r}...", flush=True)
    return rg.Argilla(api_url=api_url, api_key=api_key)


def build_settings():

    return rg.Settings(
        guidelines=load_guidelines(),
        fields=[
            rg.TextField(name="title"),
            rg.TextField(name="text"),
            rg.TextField(
                name=EVENT_TYPE_REFERENCE_FIELD,
                title="Allowed event types",
                use_markdown=True,
            ),
            rg.TextField(
                name=REASONING_TRACE_FIELD,
                title="Model reasoning trace",
                use_markdown=True,
                required=False,
            ),
        ],
        questions=[
            rg.TextQuestion(
                name=EVENT_DETAILS_JSON_QUESTION_NAME,
                title="Events (JSON list, one object per event)",
                description=(
                    "Edit the JSON list directly: fix values, delete invalid entries, add "
                    "missing events. Submit `[]` if there are no valid events.\n\n"
                    f"{EVENT_JSON_TEMPLATE}\n\n"
                    "Export writes validation errors for invalid JSON, extra keys, invalid "
                    "choices, off-ontology event_type values, and non-verbatim quote fields "
                    "(missing keys are written as warnings, not errors)."
                ),
                use_markdown=False,
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="id"),
            rg.TermsMetadataProperty(name="adm0_code"),
            rg.TermsMetadataProperty(name="risk_factors"),
            rg.TermsMetadataProperty(name="generation_llm"),
            rg.FloatMetadataProperty(name="quality_score"),
            rg.TermsMetadataProperty(name="publish_date"),
        ],
    )


def sync_dataset_settings(dataset) -> None:
    """Add any fields/questions/metadata from build_settings() that are missing
    from an existing dataset, and refresh guidelines.

    Only additions are applied. Wholesale-replacing dataset.settings (the
    previous approach) hands Argilla fresh Field/Question/Metadata objects with
    no server-side id, so its update path calls create() on every one of them
    -- including ones that already exist -- and the server rejects the
    duplicate create() with a Conflict error.
    """
    target = build_settings()
    dataset.settings.guidelines = target.guidelines
    for collection_name in ("fields", "questions", "metadata"):
        existing_names = {item.name for item in getattr(dataset.settings, collection_name)}
        for item in getattr(target, collection_name):
            if item.name not in existing_names:
                getattr(dataset.settings, collection_name).add(item)
    dataset.update()


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
            sync_dataset_settings(dataset)
            print(f"[dataset] updated {name!r}.", flush=True)
        else:
            print(
                "[dataset] existing schema was left unchanged; use --update-settings "
                "to apply the JSON events UI.",
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
    """Filter on rec["relevance"]["decision"] (the upstream Gemini relevance gate):
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


def normalize_event(ev: dict[str, Any]) -> dict[str, str]:
    return {field: str(ev.get(field) or "not_stated") for field in EVENT_FIELDS}


def reasoning_trace_text(rec: dict[str, Any]) -> str:
    """Render the model's thinking-mode thought summaries (generation_llm.metadata
    .thought_summaries), if any, as markdown for the reference field."""
    generation_llm = rec.get("generation_llm")
    metadata = generation_llm.get("metadata") if isinstance(generation_llm, dict) else None
    summaries = metadata.get("thought_summaries") if isinstance(metadata, dict) else None
    if not summaries:
        return "_No reasoning trace available._"
    return "\n\n---\n\n".join(str(s) for s in summaries)


def build_record(rec: dict[str, Any], max_chars: int, event_type_reference: str):

    title, text = extract_title_text(rec)
    if max_chars:
        text = text[:max_chars]
    annotation = rec.get("annotation") or {}
    if not isinstance(annotation, dict):
        annotation = {}

    events = [normalize_event(ev) for ev in (annotation.get("events") or []) if isinstance(ev, dict)]

    record_id = _record_key(rec) or None

    metadata: dict[str, Any] = {}
    if record_id:
        metadata["id"] = record_id
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
    source = rec.get("source") or {}
    publish_date = (source.get("publish_date") or source.get("published_at")) if isinstance(source, dict) else None
    if publish_date:
        metadata["publish_date"] = str(publish_date)

    suggestions = [
        rg.Suggestion(
            EVENT_DETAILS_JSON_QUESTION_NAME,
            value=json.dumps(events, indent=2, ensure_ascii=False),
            agent=model_name(rec),
        )
    ]

    return rg.Record(
        fields={
            "title": title,
            "text": text,
            EVENT_TYPE_REFERENCE_FIELD: event_type_reference,
            REASONING_TRACE_FIELD: reasoning_trace_text(rec),
        },
        metadata=metadata,
        suggestions=suggestions,
        id=record_id,
    )


def push(args: argparse.Namespace) -> None:
    if args.replace and args.limit:
        raise SystemExit(
            "--replace and --limit cannot be combined: --replace would then delete "
            "the records that --limit excluded from this push."
        )

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
    if args.filter_relevance:
        records = [rec for rec in records if keep_by_relevance(rec)]
    skipped = total - len(records)
    if args.limit:
        records = records[: args.limit]
    print(f"[load] {len(records)} record(s) to push ({skipped} skipped as not_relevant).", flush=True)

    event_type_reference = render_event_type_reference(load_allowed_event_types())
    rg_records = [
        build_record(rec, args.max_chars, event_type_reference) for rec in records
    ]
    print(f"[upload] sending {len(rg_records)} record(s)...", flush=True)
    dataset.records.log(rg_records)
    print(
        f"[done] pushed {len(rg_records)} records to dataset {args.dataset_name!r}.",
        flush=True,
    )

    if args.replace:
        new_ids = {r.id for r in rg_records if r.id is not None}
        existing_ids = {r.id for r in dataset.records()}
        stale_ids = existing_ids - new_ids
        if stale_ids:
            print(f"[replace] deleting {len(stale_ids)} stale record(s)...", flush=True)
            dataset.records.delete([rg.Record(id=i) for i in stale_ids])
            print(
                f"[replace] deleted {len(stale_ids)} record(s) not present in {args.input!r}.",
                flush=True,
            )


def validate_events_annotation(
    events: list[dict[str, Any]],
    text: str,
    allowed_event_types: set[str],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required_keys = set(EVENT_FIELDS)
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event {i}: must be an object")
            continue

        keys = set(event.keys())
        for field in EVENT_FIELDS:
            if field not in keys:
                warnings.append(f"event {i}: missing key {field!r}")
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

        admin_level = event.get("event_location_admin_level")
        if isinstance(admin_level, str):
            admin_level_parts = [p.strip() for p in admin_level.split(";")]
            bad_parts = [p for p in admin_level_parts if p not in ADMIN_LEVEL_CHOICES]
            if bad_parts:
                errors.append(
                    f"event {i}: event_location_admin_level parts must be one of "
                    f"{ADMIN_LEVEL_CHOICES}; got {admin_level!r}"
                )
            else:
                location = event.get("event_location")
                location_parts = (
                    [p.strip() for p in location.split(";")]
                    if isinstance(location, str)
                    else []
                )
                if isinstance(location, str) and len(admin_level_parts) != len(location_parts):
                    errors.append(
                        f"event {i}: event_location_admin_level has "
                        f"{len(admin_level_parts)} ';'-separated value(s) but event_location "
                        f"has {len(location_parts)}"
                    )
    return errors, warnings


def default_invalid_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.invalid{output_path.suffix}")


def export(args: argparse.Namespace) -> None:
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

    usernames_by_id = {str(user.id): user.username for user in client.users}

    def chosen_response(record, question_name: str):
        # record.responses[name] is a defaultdict(list) — safe for missing keys.
        try:
            responses = list(record.responses[question_name])
        except KeyError:
            return None
        submitted = [
            r for r in responses if getattr(r, "status", "submitted") == "submitted"
        ]
        return submitted[0] if submitted else (responses[0] if responses else None)

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
    validation_error_count = 0
    validation_warning_count = 0
    for record in dataset.records(with_suggestions=True, with_responses=True):
        response = chosen_response(record, EVENT_DETAILS_JSON_QUESTION_NAME)
        json_value = response.value if response else None
        annotated_by = (
            usernames_by_id.get(str(response.user_id)) if response else None
        )

        if args.only_submitted and json_value is None:
            continue

        title = record.fields.get("title", "")
        text = record.fields.get("text", "")

        parse_error = None
        events: list[dict[str, Any]] = []
        if json_value is not None:
            try:
                parsed = json.loads(json_value)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
                parse_errors += 1
            else:
                if isinstance(parsed, list):
                    events = parsed
                else:
                    parse_error = (
                        f"{EVENT_DETAILS_JSON_QUESTION_NAME} must be a JSON list; "
                        f"got {type(parsed).__name__}"
                    )
                    parse_errors += 1

        model_suggestion = get_suggestion(record, EVENT_DETAILS_JSON_QUESTION_NAME)
        model_events = None
        if model_suggestion is not None:
            model_events = json.loads(model_suggestion.value)

        validation_errors = []
        if parse_error is not None:
            validation_errors.append(
                f"{EVENT_DETAILS_JSON_QUESTION_NAME} is invalid: {parse_error}"
            )
        event_errors, validation_warnings = validate_events_annotation(
            events, f"{title}\n\n{text}", allowed_event_types
        )
        validation_errors.extend(event_errors)
        if validation_errors:
            validation_error_count += 1
        if validation_warnings:
            validation_warning_count += 1

        row = {
            "id": record.id,
            "title": record.fields.get("title"),
            "text": text,
            "annotation": {
                "events": events,
                "annotated_by": annotated_by,
                "original_annotation": {
                    "events": model_events,
                },
            },
            "events_parse_error": parse_error,
            "events_validation_errors": validation_errors or None,
            "events_validation_warnings": validation_warnings or None,
        }
        if validation_errors:
            invalid_rows.append(
                {
                    **row,
                    EVENT_DETAILS_JSON_QUESTION_NAME: json_value,
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
    if validation_error_count:
        print(
            f"Warning: {validation_error_count} record(s) had validation errors; "
            "see events_validation_errors in the output."
        )
    if validation_warning_count:
        print(
            f"Note: {validation_warning_count} record(s) had validation warnings "
            "(e.g. missing keys); see events_validation_warnings in the output."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push articles to Argilla for event annotation, or export human-corrected events."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=".env",
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
        "--filter-relevance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip records the upstream relevance gate marked not_relevant "
            "(default: off; pass --filter-relevance to enable)."
        ),
    )
    push_parser.add_argument(
        "--update-settings",
        action="store_true",
        help="For an existing dataset, update its schema/guidelines to the current JSON events UI.",
    )
    push_parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Truncate article text to this many characters (0 = no truncation, the default).",
    )
    push_parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "After pushing, delete any existing records not present in --input, "
            "making the dataset an exact mirror of the file (same dataset/URL, "
            "content fully replaced). Not compatible with --limit."
        ),
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
