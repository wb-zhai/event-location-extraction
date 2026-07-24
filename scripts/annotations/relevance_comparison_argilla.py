"""Push two relevance-labeled JSONL files (same articles, two different
relevance predictions e.g. from different prompt versions) into Argilla so a
human can pick which prediction they prefer, and export the resulting
preferences.

By default only records where the two predictions *disagree* are pushed --
that's the interesting case to review. Pass --include-agreements to push
everything.

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

from scripts.annotations.relevance_argilla import LABEL_DISPLAY, normalize_decision
from scripts.relevance.relevance_filter import _record_key

PREFERENCE_QUESTION_NAME = "preference"
DEFAULT_WORKSPACE = "default"

GUIDELINES_TEMPLATE = """Compare two relevance predictions for the same article and say which one you agree with.

## Workflow

1. Read the title and article preview.
2. Read **Prediction A** ({name_a}) and **Prediction B** ({name_b}) — each shows the
   predicted label, confidence, and reasoning.
3. Pick one option for the **{question_title}** question below, based on which prediction
   you think is actually correct for this article (not which one sounds more confident).

## Options

- **Prefer A**: Prediction A is correct (or closer to correct).
- **Prefer B**: Prediction B is correct (or closer to correct).
- **Tie**: Both are equally good (or equally wrong in the same way).
- **Neither**: Both are wrong.
"""


def load_guidelines(name_a: str, name_b: str) -> str:
    return GUIDELINES_TEMPLATE.format(
        name_a=name_a, name_b=name_b, question_title=PREFERENCE_QUESTION_NAME
    )


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


def build_settings(name_a: str, name_b: str):

    return rg.Settings(
        guidelines=load_guidelines(name_a, name_b),
        fields=[
            rg.TextField(name="title"),
            rg.TextField(name="text"),
            rg.TextField(name="prediction_a", title=f"Prediction A: {name_a}"),
            rg.TextField(name="prediction_b", title=f"Prediction B: {name_b}"),
        ],
        questions=[
            rg.LabelQuestion(
                name=PREFERENCE_QUESTION_NAME,
                labels={
                    "prefer_a": f"Prefer A ({name_a})",
                    "prefer_b": f"Prefer B ({name_b})",
                    "tie": "Tie / both equally good",
                    "neither": "Neither / both wrong",
                },
                title="Which relevance prediction do you agree with?",
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="id"),
            rg.TermsMetadataProperty(name="adm0_code"),
            rg.TermsMetadataProperty(name="risk_factors"),
            rg.TermsMetadataProperty(name="decision_a"),
            rg.TermsMetadataProperty(name="decision_b"),
            rg.FloatMetadataProperty(name="confidence_a"),
            rg.FloatMetadataProperty(name="confidence_b"),
            rg.TermsMetadataProperty(name="agreement"),
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
    client,
    name: str,
    workspace: str,
    name_a: str,
    name_b: str,
    update_settings: bool = False,
):

    get_or_create_workspace(client, workspace)
    dataset = client.datasets(name=name, workspace=workspace)
    if dataset is not None:
        if update_settings:
            dataset.settings = build_settings(name_a, name_b)
            dataset.update()
        return dataset
    dataset = rg.Dataset(
        name=name, workspace=workspace, settings=build_settings(name_a, name_b)
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


def prediction_str(relevance: dict[str, Any]) -> str:
    label = normalize_decision(relevance.get("decision"))
    if label is None:
        return "(no prediction)"
    confidence = float(relevance.get("confidence", 0.0) or 0.0)
    reason = relevance.get("reason", "")
    return f"{LABEL_DISPLAY[label]} (confidence: {confidence:.2f})\n\n{reason}"


def build_record(
    rec_a: dict[str, Any],
    rec_b: dict[str, Any],
    max_chars: int,
    agree: bool,
):
    title, text = extract_title_text(rec_a)
    if not title and not text:
        title, text = extract_title_text(rec_b)

    relevance_a = rec_a.get("relevance") or {}
    relevance_b = rec_b.get("relevance") or {}
    if not isinstance(relevance_a, dict):
        relevance_a = {}
    if not isinstance(relevance_b, dict):
        relevance_b = {}

    label_a = normalize_decision(relevance_a.get("decision"))
    label_b = normalize_decision(relevance_b.get("decision"))

    metadata: dict[str, Any] = {"agreement": "agree" if agree else "disagree"}
    input_id = rec_a.get("id") or rec_b.get("id")
    if input_id:
        metadata["id"] = str(input_id)
    adm0_code = rec_a.get("adm0_code") or rec_b.get("adm0_code")
    if adm0_code:
        metadata["adm0_code"] = str(adm0_code)
    risk_factors = rec_a.get("risk_factors") or rec_b.get("risk_factors")
    if risk_factors:
        metadata["risk_factors"] = [str(r) for r in risk_factors]
    if label_a is not None:
        metadata["decision_a"] = label_a
        metadata["confidence_a"] = float(relevance_a.get("confidence", 0.0) or 0.0)
    if label_b is not None:
        metadata["decision_b"] = label_b
        metadata["confidence_b"] = float(relevance_b.get("confidence", 0.0) or 0.0)

    return rg.Record(
        fields={
            "title": title,
            "text": text[:max_chars] if max_chars else text,
            "prediction_a": prediction_str(relevance_a),
            "prediction_b": prediction_str(relevance_b),
        },
        metadata=metadata,
        id=_record_key(rec_a) or _record_key(rec_b) or None,
    )


def push(args: argparse.Namespace) -> None:
    if args.replace and args.limit:
        raise SystemExit(
            "--replace and --limit cannot be combined: --replace would then delete "
            "the records that --limit excluded from this push."
        )

    client = get_client()
    dataset = get_or_create_dataset(
        client,
        args.dataset_name,
        args.workspace,
        args.name_a,
        args.name_b,
        update_settings=args.update_settings,
    )

    records_a = iter_jsonl(Path(args.input_a))
    records_b = iter_jsonl(Path(args.input_b))
    by_id_a = {_record_key(r): r for r in records_a if _record_key(r)}
    by_id_b = {_record_key(r): r for r in records_b if _record_key(r)}

    common_ids = [rid for rid in by_id_a if rid in by_id_b]
    only_a = len(by_id_a) - len(common_ids)
    only_b = len(by_id_b) - len(common_ids)
    if only_a or only_b:
        print(
            f"Skipping {only_a} record(s) only in --input-a and {only_b} only in "
            "--input-b (need to be present in both to compare)."
        )

    rg_records = []
    no_prediction = 0
    for rid in common_ids:
        rec_a, rec_b = by_id_a[rid], by_id_b[rid]
        relevance_a = rec_a.get("relevance") or {}
        relevance_b = rec_b.get("relevance") or {}
        label_a = normalize_decision(
            relevance_a.get("decision") if isinstance(relevance_a, dict) else None
        )
        label_b = normalize_decision(
            relevance_b.get("decision") if isinstance(relevance_b, dict) else None
        )
        if label_a is None or label_b is None:
            no_prediction += 1
            continue

        agree = label_a == label_b
        if agree and not args.include_agreements:
            continue

        rg_records.append(build_record(rec_a, rec_b, args.max_chars, agree))

    if no_prediction:
        print(
            f"Skipped {no_prediction} record(s) missing a relevance prediction in "
            "--input-a or --input-b."
        )

    if args.limit:
        rg_records = rg_records[: args.limit]

    dataset.records.log(rg_records)
    print(f"Pushed {len(rg_records)} records to dataset {args.dataset_name!r}.")

    if args.replace:
        new_ids = {r.id for r in rg_records if r.id is not None}
        existing_ids = {r.id for r in dataset.records()}
        stale_ids = existing_ids - new_ids
        if stale_ids:
            dataset.records.delete([rg.Record(id=i) for i in stale_ids])
            print(
                f"Deleted {len(stale_ids)} stale record(s) not present in this push."
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
    for record in dataset.records(with_responses=True):
        responses = (
            list(record.responses[PREFERENCE_QUESTION_NAME])
            if PREFERENCE_QUESTION_NAME in record.responses
            else []
        )
        submitted = [
            r for r in responses if getattr(r, "status", "submitted") == "submitted"
        ]
        chosen = submitted[0] if submitted else (responses[0] if responses else None)
        preference = chosen.value if chosen else None
        if args.only_submitted and preference is None:
            continue

        metadata = record.metadata or {}
        rows.append(
            {
                "id": record.id,
                "title": record.fields.get("title"),
                "preference": preference,
                "decision_a": metadata.get("decision_a"),
                "confidence_a": metadata.get("confidence_a"),
                "decision_b": metadata.get("decision_b"),
                "confidence_b": metadata.get("confidence_b"),
            }
        )

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} records to {output_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Push two relevance-labeled JSONL files to Argilla for side-by-side "
            "preference annotation, or export the resulting preferences."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Path to a .env file with ARGILLA_API_URL / ARGILLA_API_KEY (default: repo .env).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_parser = subparsers.add_parser(
        "push",
        help="Load two JSONL files (same articles, two relevance predictions) into Argilla.",
    )
    push_parser.add_argument(
        "--input-a", required=True, type=str, help="First relevance-labeled JSONL file."
    )
    push_parser.add_argument(
        "--input-b", required=True, type=str, help="Second relevance-labeled JSONL file."
    )
    push_parser.add_argument(
        "--name-a",
        type=str,
        default=None,
        help="Display name for --input-a's predictions (default: file stem).",
    )
    push_parser.add_argument(
        "--name-b",
        type=str,
        default=None,
        help="Display name for --input-b's predictions (default: file stem).",
    )
    push_parser.add_argument("--dataset-name", required=True, type=str)
    push_parser.add_argument("--workspace", type=str, default=DEFAULT_WORKSPACE)
    push_parser.add_argument("--limit", type=int, default=None)
    push_parser.add_argument("--max-chars", type=int, default=10000)
    push_parser.add_argument(
        "--include-agreements",
        action="store_true",
        help=(
            "Also push records where both inputs agree on the relevance decision "
            "(default: only push disagreements)."
        ),
    )
    push_parser.add_argument(
        "--update-settings",
        action="store_true",
        help="For an existing dataset, update its schema/guidelines (e.g. after changing --name-a/--name-b).",
    )
    push_parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "After pushing, delete any existing records not present in this push, "
            "making the dataset an exact mirror of the current comparison. Not "
            "compatible with --limit."
        ),
    )
    push_parser.set_defaults(func=push)

    export_parser = subparsers.add_parser(
        "export", help="Export human preferences to JSONL."
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
    if args.command == "push":
        args.name_a = args.name_a or Path(args.input_a).stem
        args.name_b = args.name_b or Path(args.input_b).stem
    args.func(args)


if __name__ == "__main__":
    main()
