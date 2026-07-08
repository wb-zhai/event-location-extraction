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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from scripts.data.relevance.relevance_filter import (
    DEFAULT_RELEVANCE_SYSTEM_PROMPT_3LABEL,
    _record_key,
)

RELEVANCE_QUESTION_NAME = "relevance"
DEFAULT_WORKSPACE = "default"

# Human-facing label set. Runs using the legacy 2-label prompt only ever
# produce "relevant"/"irrelevant" in relevance.decision; "irrelevant" is
# normalized to "not_relevant" so both prompt versions map onto one question.
LABEL_DISPLAY = {
    "relevant": "Relevant",
    "partially_relevant": "Partially relevant",
    "not_relevant": "Not relevant",
}

# Definitions lifted from DEFAULT_RELEVANCE_SYSTEM_PROMPT_3LABEL's <labels> block,
# shown to annotators so they apply the same criteria as the Gemini gate.
LABEL_DESCRIPTIONS = {
    "relevant": (
        "The article substantively reports on a current, concrete instance of "
        "at least one event category (agricultural production issues, conflicts "
        "and violence, economic issues, environmental issues, food crisis, "
        "forced displacement, humanitarian aid, land-related issues, pests and "
        "diseases, political instability, or weather shocks)."
    ),
    "partially_relevant": (
        "The article touches on an event category, but only partially, "
        "ambiguously, or as secondary context to a different main subject "
        "(e.g. a brief mention within a broader story, an early/developing "
        "situation with limited detail, or content mixing in-scope and "
        "out-of-scope material)."
    ),
    "not_relevant": (
        "The article is not about any event category, or category-related "
        "terms appear only in a quote, anecdote, historical aside, or "
        "rhetorical comparison rather than in reporting on a real, current "
        "event."
    ),
}

RELEVANCE_QUESTION_DESCRIPTION = "\n".join(
    f"- {LABEL_DISPLAY[label]}: {LABEL_DESCRIPTIONS[label]}" for label in LABEL_DISPLAY
)


def normalize_decision(decision: Any) -> str | None:
    """Map a raw relevance.decision value onto the 3-way label set."""
    if not decision:
        return None
    value = str(decision).strip().lower()
    if value == "irrelevant":
        value = "not_relevant"
    return value if value in LABEL_DISPLAY else None


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

    return rg.Settings(
        guidelines=DEFAULT_RELEVANCE_SYSTEM_PROMPT_3LABEL,
        fields=[
            rg.TextField(name="title"),
            rg.TextField(name="text"),
            rg.TextField(name="gemini_assessment", required=False),
        ],
        questions=[
            rg.LabelQuestion(
                name=RELEVANCE_QUESTION_NAME,
                labels=LABEL_DISPLAY,
                title="Is this article relevant to food-security risk-event extraction?",
                description=RELEVANCE_QUESTION_DESCRIPTION,
            ),
        ],
        metadata=[
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


def get_or_create_dataset(client, name: str, workspace: str):

    get_or_create_workspace(client, workspace)
    dataset = client.datasets(name=name, workspace=workspace)
    if dataset is not None:
        return dataset
    dataset = rg.Dataset(name=name, workspace=workspace, settings=build_settings())
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


def build_record(rec: dict[str, Any], max_chars: int):

    title, text = extract_title_text(rec)
    relevance = rec.get("relevance") or {}
    if not isinstance(relevance, dict):
        relevance = {}

    label = normalize_decision(relevance.get("decision"))

    metadata: dict[str, Any] = {}
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
    if label is not None:
        suggestions.append(
            rg.Suggestion(
                RELEVANCE_QUESTION_NAME,
                value=label,
                agent=str(relevance.get("model") or "gemini"),
                score=float(relevance.get("confidence", 0.0) or 0.0),
            )
        )

    return rg.Record(
        fields={
            "title": title,
            "text": text[:max_chars],
            "gemini_assessment": gemini_assessment_str(relevance),
        },
        metadata=metadata,
        suggestions=suggestions,
        id=_record_key(rec) or None,
    )


def push(args: argparse.Namespace) -> None:
    client = get_client()
    dataset = get_or_create_dataset(client, args.dataset_name, args.workspace)

    records = iter_jsonl(Path(args.input))
    if args.limit:
        records = records[: args.limit]

    rg_records = [build_record(rec, args.max_chars) for rec in records]
    dataset.records.log(rg_records)
    print(f"Pushed {len(rg_records)} records to dataset {args.dataset_name!r}.")


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
        responses = (
            list(record.responses[RELEVANCE_QUESTION_NAME])
            if RELEVANCE_QUESTION_NAME in record.responses
            else []
        )
        submitted = [
            r for r in responses if getattr(r, "status", "submitted") == "submitted"
        ]
        chosen = submitted[0] if submitted else (responses[0] if responses else None)
        human_value = chosen.value if chosen else None
        if args.only_submitted and human_value is None:
            continue

        suggestion = (
            record.suggestions[RELEVANCE_QUESTION_NAME]
            if RELEVANCE_QUESTION_NAME in record.suggestions
            else None
        )
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
    push_parser.add_argument("--max-chars", type=int, default=2000)
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
