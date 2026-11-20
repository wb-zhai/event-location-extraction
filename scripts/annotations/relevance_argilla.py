"""Push articles into Argilla for human relevance annotation, and export the
resulting human labels alongside the Gemini gate's predictions.

See scripts/annotations/README.md for setup and usage.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import argilla as rg

from dotenv import load_dotenv

from scripts.relevance.relevance_filter import _record_key

RELEVANCE_QUESTION_NAME = "relevance"
DEFAULT_WORKSPACE = "default"

# Human-facing label set. Runs using the legacy 2-label prompt only ever
# produce "relevant"/"irrelevant" in relevance.decision; "irrelevant" is
# normalized to "not_relevant" so both prompt versions map onto the same
# underlying values. This full set is used for free-text formatting (e.g.
# gemini_assessment) regardless of which label set the dataset's question
# offers to annotators; see LABEL_DISPLAY_2LABEL for the binary-prompt case.
LABEL_DISPLAY = {
    "relevant": "Relevant",
    "partially_relevant": "Partially relevant",
    "not_relevant": "Not relevant",
}

# Definitions lifted from DEFAULT_RELEVANCE_SYSTEM_PROMPT_3LABEL's <labels> block,
# shown to annotators so they apply the same criteria as the Gemini gate.
LABEL_DESCRIPTIONS = {
    "relevant": (
        "The article reports a real, current, concrete instance of at least one "
        "event category above — whether that's the article's main subject or a "
        'clearly factual side mention within a story about something else (e.g. a '
        'football report noting "amid extreme rainfall affecting the region").'
    ),
    "partially_relevant": (
        "The article's connection to an event category is genuinely ambiguous or "
        "underspecified — e.g. a very early/developing situation with too little "
        "detail to confirm, or content that mixes clearly in-scope and "
        "out-of-scope material such that scope is unclear."
    ),
    "not_relevant": (
        "The article is not about any event category, or category-related terms "
        "appear only in a quote, anecdote, historical aside, hypothetical, or "
        "rhetorical comparison rather than in reporting on a real, current "
        "occurrence."
    ),
}

RELEVANCE_QUESTION_DESCRIPTION = "\n".join(
    f"- {LABEL_DISPLAY[label]}: {LABEL_DESCRIPTIONS[label]}" for label in LABEL_DISPLAY
)

# Binary label set for datasets produced by the legacy 2-label prompt
# (DEFAULT_RELEVANCE_SYSTEM_PROMPT in relevance_filter.py), which only ever
# emits "relevant"/"irrelevant" — no "partially_relevant" middle ground.
# Offering a 3-way question over such a dataset would let annotators pick a
# label the source model never had access to, so datasets detected as
# binary (see detect_two_label) get this narrower question instead.
LABEL_DISPLAY_2LABEL = {
    "relevant": "Relevant",
    "not_relevant": "Not relevant",
}


def label_descriptions_2label(title_only: bool = False) -> dict[str, str]:
    """Descriptions for the binary label set, adapted from
    DEFAULT_RELEVANCE_SYSTEM_PROMPT's <policy> block (the binary prompt).

    The not_relevant wording names what's actually shown to the annotator —
    title only under --title-only, title+preview otherwise — so it doesn't
    reference material they can't see.
    """
    if title_only:
        not_relevant = (
            "The title gives no indication of any real, current, concrete "
            "instance of any event category — i.e. all category-related "
            "language is rhetorical, historical, hypothetical, or quoted "
            "without describing a genuine current occurrence."
        )
    else:
        not_relevant = (
            "The title and preview give no indication of any real, current, "
            "concrete instance of any event category — i.e. all category-related "
            "language is rhetorical, historical, hypothetical, or quoted without "
            "describing a genuine current occurrence."
        )
    return {
        "relevant": LABEL_DESCRIPTIONS["relevant"],
        "not_relevant": not_relevant,
    }


def relevance_question_description_2label(title_only: bool = False) -> str:
    descriptions = label_descriptions_2label(title_only)
    return "\n".join(
        f"- {LABEL_DISPLAY_2LABEL[label]}: {descriptions[label]}"
        for label in LABEL_DISPLAY_2LABEL
    )

# Event categories the extraction pipeline looks for, from
# DEFAULT_RELEVANCE_SYSTEM_PROMPT_3LABEL's <event_categories> block.
EVENT_CATEGORIES = [
    "agricultural issues",
    "conflict and security",
    "displacement and migration",
    "economic stress",
    "environmental issues",
    "food insecurity",
    "humanitarian disruption",
    "political instability",
    "public health",
    "weather and natural hazards",
]

# Human-readable rewrite of DEFAULT_RELEVANCE_SYSTEM_PROMPT_3LABEL (which is
# written as an LLM system prompt) so annotators apply the same criteria as
# the Gemini gate. Formatted as markdown since Argilla renders the guidelines
# field as markdown, not as XML-tagged prompt instructions.
GUIDELINES_TEMPLATE = """Decide whether this article should be sent on to the full risk-event extraction pipeline.

## Workflow

{workflow}

## Event categories

The pipeline only extracts events in these categories — judge relevance against this list, not against food security or crisis reporting in general:

{categories}

## Labels

{labels}

## Policy

{policy}
"""

WORKFLOW_FULL = """1. Read the title and article preview.
2. Check the **gemini_assessment** field, if present — it shows the automated gate's label, confidence, and reasoning. You are confirming or correcting that call, not labeling from scratch.
3. Pick one label for the **relevance** question below."""

# Used under --title-only: the article body and gemini_assessment aren't
# pushed as fields at all, so those workflow steps are dropped and this is
# an independent human judgment rather than a confirm/correct pass.
WORKFLOW_TITLE_ONLY = """1. Read the title only — the article body and Gemini's assessment are not shown for this dataset.
2. Pick one label for the **relevance** question below, using only the title and no outside knowledge of the story."""

POLICY_3LABEL_HEAD = """- **Favor recall over precision**: what matters is whether a real, current occurrence of a category is reported at all, not how prominent it is in the article. A brief, factual side mention of a real event should be Relevant, not Partially relevant — reserve Partially relevant for genuine ambiguity or lack of detail, not for prominence.
- **Opinion/analysis pieces** are Relevant if they factually reference a concrete, current event in one of the categories, even briefly, and Not relevant if they only use category language rhetorically (e.g. domestic politics, culture, sports, entertainment, or personal profiles that merely borrow a related term or metaphor)."""

# Adapted from DEFAULT_RELEVANCE_SYSTEM_PROMPT's <policy> block (the binary
# prompt) for datasets detected as 2-label — no "partially relevant" middle
# ground is offered, so borderline cases resolve straight to Relevant.
POLICY_2LABEL_HEAD = """- **Favor recall over precision**: what matters is whether a real, current occurrence of a category is reported at all, not how prominent it is in the article. A brief, factual side mention of a real event should be Relevant.
- **Opinion/analysis pieces** are Relevant if they factually reference a concrete, current event in one of the categories, even briefly, and Not relevant if they only use category language rhetorically (e.g. domestic politics, culture, sports, entertainment, or personal profiles that merely borrow a related term or metaphor)."""


def _borderline_bullet(two_label: bool = False, title_only: bool = False) -> str:
    ambiguous = "the title is ambiguous" if title_only else "the article is ambiguous or only partially visible in the preview"
    if two_label:
        return f"- **When borderline**: if {ambiguous}, prefer Relevant over Not relevant."
    return (
        f"- **When borderline**: if {ambiguous}, prefer Partially relevant over "
        "Not relevant; prefer Relevant over Partially relevant when a real, "
        "current in-scope occurrence is clearly described, however briefly."
    )


def _use_only_bullet(title_only: bool = False) -> str:
    if title_only:
        return (
            "- **Use only the title**: judge from the title alone, not outside "
            "knowledge of the event — the article body is not shown for this dataset."
        )
    return (
        "- **Use only what's shown**: judge from the title and article preview "
        "here, not outside knowledge of the event."
    )


def build_policy(two_label: bool = False, title_only: bool = False) -> str:
    head = POLICY_2LABEL_HEAD if two_label else POLICY_3LABEL_HEAD
    return "\n".join(
        [
            head,
            _borderline_bullet(two_label, title_only),
            _use_only_bullet(title_only),
        ]
    )


def load_guidelines(two_label: bool = False, title_only: bool = False) -> str:
    label_display = LABEL_DISPLAY_2LABEL if two_label else LABEL_DISPLAY
    label_descriptions = (
        label_descriptions_2label(title_only) if two_label else LABEL_DESCRIPTIONS
    )
    labels = "\n".join(
        f"- **{label_display[label]}**: {label_descriptions[label]}"
        for label in label_display
    )
    return GUIDELINES_TEMPLATE.format(
        workflow=WORKFLOW_TITLE_ONLY if title_only else WORKFLOW_FULL,
        categories="\n".join(f"- {c}" for c in EVENT_CATEGORIES),
        labels=labels,
        policy=build_policy(two_label, title_only),
    )


def normalize_decision(decision: Any) -> str | None:
    """Map a raw relevance.decision value onto the 3-way label set."""
    if not decision:
        return None
    value = str(decision).strip().lower()
    if value == "irrelevant":
        value = "not_relevant"
    return value if value in LABEL_DISPLAY else None


def detect_two_label(records: list[dict[str, Any]]) -> bool:
    """True if every relevance.decision in records comes from the legacy
    2-label prompt (relevant/irrelevant only, never partially_relevant).

    Datasets with no relevance decisions at all (e.g. pure human-annotation
    input) default to the full 3-label question, matching prior behavior.
    """
    decisions = set()
    for rec in records:
        relevance = rec.get("relevance")
        if not isinstance(relevance, dict):
            continue
        raw = relevance.get("decision")
        if raw:
            decisions.add(str(raw).strip().lower())
    if not decisions:
        return False
    return "partially_relevant" not in decisions and decisions <= {
        "relevant",
        "irrelevant",
        "not_relevant",
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


def build_settings(two_label: bool = False, title_only: bool = False):

    fields = [rg.TextField(name="title")]
    if not title_only:
        fields.append(rg.TextField(name="text"))
        fields.append(rg.TextField(name="gemini_assessment", required=False))

    description = (
        relevance_question_description_2label(title_only)
        if two_label
        else RELEVANCE_QUESTION_DESCRIPTION
    )

    return rg.Settings(
        guidelines=load_guidelines(two_label, title_only),
        fields=fields,
        questions=[
            rg.LabelQuestion(
                name=RELEVANCE_QUESTION_NAME,
                labels=LABEL_DISPLAY_2LABEL if two_label else LABEL_DISPLAY,
                title="Is this article relevant to food-security risk-event extraction?",
                description=description,
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="id"),
            rg.TermsMetadataProperty(name="adm0_code"),
            rg.TermsMetadataProperty(name="risk_factors"),
            rg.TermsMetadataProperty(name="gemini_decision"),
            rg.FloatMetadataProperty(name="gemini_confidence"),
        ],
    )


def get_or_create_workspace(client, name: str):

    workspace = client.workspaces(name)
    if workspace is not None:
        return workspace
    workspace = rg.Workspace(name=name, client=client)
    workspace.create()
    return workspace


def get_or_create_dataset(
    client, name: str, workspace: str, two_label: bool = False, title_only: bool = False
):

    get_or_create_workspace(client, workspace)
    dataset = client.datasets(name=name, workspace=workspace)
    if dataset is not None:
        return dataset
    dataset = rg.Dataset(
        name=name, workspace=workspace, settings=build_settings(two_label, title_only)
    )
    dataset.create()
    return dataset


def extract_title_text(record: dict[str, Any]) -> tuple[str, str]:
    source = record.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    title = str(record.get("title") or source.get("title", ""))
    text = str(record.get("text") or source.get("text", ""))
    return title, text


def gemini_assessment_str(relevance: dict[str, Any]) -> str:
    label = normalize_decision(relevance.get("decision"))
    if label is None:
        return ""
    confidence = float(relevance.get("confidence", 0.0) or 0.0)
    reason = relevance.get("reason", "")
    return f"gemini: {LABEL_DISPLAY[label]} ({confidence:.2f}) — {reason}"


def build_record(rec: dict[str, Any], max_chars: int, title_only: bool = False):

    title, text = extract_title_text(rec)
    relevance = rec.get("relevance") or {}
    if not isinstance(relevance, dict):
        relevance = {}

    label = normalize_decision(relevance.get("decision"))

    metadata: dict[str, Any] = {}
    input_id = rec.get("id")
    if input_id:
        metadata["id"] = str(input_id)
    adm0_code = rec.get("adm0_code")
    if adm0_code:
        metadata["adm0_code"] = str(adm0_code)
    risk_factors = rec.get("risk_factors")
    if risk_factors:
        metadata["risk_factors"] = [str(r) for r in risk_factors]
    if label is not None:
        metadata["gemini_decision"] = label
        metadata["gemini_confidence"] = float(relevance.get("confidence", 0.0) or 0.0)

    suggestions = []
    # Title-only annotation is an independent human judgment, not a
    # confirm/correct pass over Gemini's call, so don't pre-fill its answer.
    if label is not None and not title_only:
        suggestions.append(
            rg.Suggestion(
                RELEVANCE_QUESTION_NAME,
                value=label,
                agent=str(relevance.get("model") or "gemini"),
                score=float(relevance.get("confidence", 0.0) or 0.0),
            )
        )

    fields = {"title": title}
    if not title_only:
        fields["text"] = text[:max_chars]
        fields["gemini_assessment"] = gemini_assessment_str(relevance)

    return rg.Record(
        fields=fields,
        metadata=metadata,
        suggestions=suggestions,
        id=_record_key(rec) or None,
    )


def push(args: argparse.Namespace) -> None:
    if args.replace and args.limit:
        raise SystemExit(
            "--replace and --limit cannot be combined: --replace would then delete "
            "the records that --limit excluded from this push."
        )

    client = get_client()
    records = iter_jsonl(Path(args.input))
    two_label = detect_two_label(records)
    if two_label:
        print("Detected 2-label relevance data — using binary Relevant/Not relevant question.")
    if args.title_only:
        print("Title-only mode: annotators will see only the title (no article text, no Gemini suggestion).")
    dataset = get_or_create_dataset(
        client, args.dataset_name, args.workspace, two_label, args.title_only
    )

    if args.limit:
        records = records[: args.limit]

    rg_records = [build_record(rec, args.max_chars, args.title_only) for rec in records]
    dataset.records.log(rg_records)
    print(f"Pushed {len(rg_records)} records to dataset {args.dataset_name!r}.")

    if args.replace:
        new_by_id = {r.id: r for r in rg_records if r.id is not None}
        existing = list(dataset.records(with_responses=True))
        existing_by_id = {r.id: r for r in existing}

        # Argilla's upsert (log) never overwrites a record's *fields* (title/text/
        # gemini_assessment) once the record exists -- only metadata, suggestions,
        # and responses can be updated in place. That means re-pushing with a
        # wider --max-chars has no effect on already-created records. Refresh
        # fields on any record that has no submitted human response yet by
        # deleting and re-logging it; leave annotated records untouched so their
        # responses aren't lost.
        to_refresh = []
        for record in existing:
            if record.id not in new_by_id:
                continue
            responses = list(record.responses[RELEVANCE_QUESTION_NAME])
            has_submitted = any(
                getattr(r, "status", "submitted") == "submitted" for r in responses
            )
            if not has_submitted:
                to_refresh.append(record.id)

        if to_refresh:
            # rg.Record(id=...) alone fails Argilla's own validation (it requires
            # at least one of fields/metadata/vectors/responses/suggestions), so
            # delete using the actual Record objects fetched from the server.
            dataset.records.delete([existing_by_id[i] for i in to_refresh])
            dataset.records.log([new_by_id[i] for i in to_refresh])
            print(
                f"Refreshed fields on {len(to_refresh)} unannotated record(s) "
                "(e.g. so a wider --max-chars takes effect)."
            )

        stale_ids = set(existing_by_id) - set(new_by_id)
        if stale_ids:
            dataset.records.delete([existing_by_id[i] for i in stale_ids])
            print(
                f"Deleted {len(stale_ids)} stale record(s) not present in {args.input!r}."
            )


def export(args: argparse.Namespace) -> None:
    client = get_client()
    if client.workspaces(args.workspace) is None:
        raise SystemExit(f"Workspace {args.workspace!r} not found.")
    dataset = client.datasets(name=args.dataset_name, workspace=args.workspace)
    if dataset is None:
        raise SystemExit(
            f"Dataset {args.dataset_name!r} not found in workspace {args.workspace!r}."
        )

    rows = []
    for record in dataset.records(with_suggestions=True, with_responses=True):
        # record.responses/record.suggestions only implement __iter__ and
        # __getitem__, not __contains__, so `name in record.responses` falls
        # back to iterating and comparing each Response/Suggestion object to
        # the string `name` -- always False. Index directly instead:
        # responses is a defaultdict(list) (safe on a missing key), while
        # suggestions is a plain dict (raises KeyError on a missing key).
        responses = list(record.responses[RELEVANCE_QUESTION_NAME])
        submitted = [
            r for r in responses if getattr(r, "status", "submitted") == "submitted"
        ]
        chosen = submitted[0] if submitted else (responses[0] if responses else None)
        human_value = chosen.value if chosen else None
        if args.only_submitted and human_value is None:
            continue

        try:
            suggestion = record.suggestions[RELEVANCE_QUESTION_NAME]
        except KeyError:
            suggestion = None
        gemini_value = suggestion.value if suggestion else None

        rows.append(
            {
                "id": record.id,
                "title": record.fields.get("title"),
                "human_relevance": human_value,
                "human_is_relevant": (
                    (human_value != "not_relevant") if human_value else None
                ),
                "gemini_relevance": gemini_value,
                "gemini_is_relevant": (
                    (gemini_value != "not_relevant") if gemini_value else None
                ),
                "agreement": (
                    (human_value == gemini_value)
                    if human_value and gemini_value
                    else None
                ),
            }
        )

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} records to {output_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push articles to Argilla for relevance annotation, or export human labels."
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
    push_parser.add_argument("--max-chars", type=int, default=10000)
    push_parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "After pushing, delete any existing records not present in --input, "
            "making the dataset an exact mirror of the file (same dataset/URL, "
            "content fully replaced). Not compatible with --limit."
        ),
    )
    push_parser.add_argument(
        "--title-only",
        action="store_true",
        help=(
            "Only show annotators the title — no article text or gemini_assessment "
            "field, and no pre-filled Gemini suggestion. Only takes effect when "
            "creating a new dataset."
        ),
    )
    push_parser.set_defaults(func=push)

    export_parser = subparsers.add_parser(
        "export",
        help="Export human relevance responses (and Gemini suggestions) to JSONL.",
    )
    export_parser.add_argument("--dataset-name", required=True, type=str)
    export_parser.add_argument("--workspace", type=str, default=DEFAULT_WORKSPACE)
    export_parser.add_argument("--output", required=True, type=str)
    export_parser.add_argument(
        "--only-submitted",
        action="store_true",
        help="Only export records with a submitted human response.",
    )
    export_parser.set_defaults(func=export)

    args = parser.parse_args()
    load_dotenv(args.env_file)
    args.func(args)


if __name__ == "__main__":
    main()
