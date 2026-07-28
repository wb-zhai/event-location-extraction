"""Recover a --batch-api --cascade relevance_filter.py run that crashed locally
after already submitting batch jobs to Gemini.

Batch jobs run server-side and keep going (and stay downloadable) even if the
local script/terminal dies while polling. This script re-lists jobs by
display-name prefix, downloads whichever chunks reached JOB_STATE_SUCCEEDED,
reconstructs the same output records relevance_filter.py would have written,
and appends them to --output. Any record still missing a final decision
(no first-pass result yet, or first-pass said relevant but no cascade result
yet) is left out of --output entirely, so a subsequent

    relevance_filter.py --resume --batch-api --cascade ...

will pick it back up and reprocess it through the normal pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from src.llms.llm_client import GeminiLLMClient  # noqa: E402

from scripts.relevance.relevance_filter import (  # noqa: E402
    _batch_response_metadata,
    _batch_response_text,
    _record_key,
    clean_relevance_decision,
    should_filter_by_relevance,
)


def _parse_part_index(display_name: str) -> int | None:
    marker = "-part-"
    idx = display_name.rfind(marker)
    if idx == -1:
        return None
    try:
        return int(display_name[idx + len(marker) :])
    except ValueError:
        return None


def _download_results(client: GeminiLLMClient, job) -> dict[str, dict[str, Any]]:
    """Returns {key: relevance_info} for one succeeded batch job."""
    if not job.dest or not job.dest.file_name:
        return {}
    data = client.client.files.download(file=job.dest.file_name)
    model = job.model.removeprefix("models/") if job.model else ""
    out: dict[str, dict[str, Any]] = {}
    for raw in data.decode("utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        line = json.loads(raw)
        key = str(line.get("key") or (line.get("metadata") or {}).get("key") or "")
        if not key:
            continue
        try:
            response = line.get("response")
            if not isinstance(response, dict):
                raise ValueError(str(line.get("error") or "Missing batch response."))
            parsed = json.loads(_batch_response_text(response))
            decision = clean_relevance_decision(parsed)
            metadata = _batch_response_metadata(response)
            out[key] = {
                "decision": "relevant" if decision["is_relevant"] else "irrelevant",
                "is_relevant": decision["is_relevant"],
                "confidence": decision["confidence"],
                "reason": decision["reason"],
                "model": model,
                "metadata": metadata,
            }
        except Exception as exc:
            out[key] = {"error": str(exc), "filtered": False, "model": model}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover a crashed --batch-api --cascade relevance_filter.py run "
        "from already-submitted Gemini batch jobs."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Original input JSONL (same file "
        "passed to relevance_filter.py --input)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL to write recovered "
        "records to (same file passed to relevance_filter.py --output)",
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="First-pass model")
    parser.add_argument(
        "--cascade-model", default="gemini-3.1-pro-preview", help="Escalation model"
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--filter-only", action="store_true")
    parser.add_argument(
        "--job-name-prefix",
        default=None,
        help="Batch job display-name "
        "prefix to match (default: --output's filename stem, same as relevance_filter.py "
        "uses when submitting jobs)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    prefix = args.job_name_prefix or output_path.stem

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    by_key = {_record_key(r): r for r in records}
    print(f"Loaded {len(records)} input records.")

    client = GeminiLLMClient(model_name=args.model, system_prompt=None)

    all_jobs = list(client.client.batches.list())
    matching = [
        j
        for j in all_jobs
        if j.display_name and j.display_name.startswith(f"{prefix}-part-")
    ]
    matching.sort(key=lambda j: j.create_time)

    first_pass_jobs = [j for j in matching if j.model == f"models/{args.model}"]
    escalation_jobs = [j for j in matching if j.model == f"models/{args.cascade_model}"]

    print(
        f"Found {len(first_pass_jobs)} first-pass job(s) and "
        f"{len(escalation_jobs)} escalation job(s) matching prefix '{prefix}'."
    )

    def collect(jobs, label) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for j in jobs:
            part = _parse_part_index(j.display_name)
            if j.state.name != "JOB_STATE_SUCCEEDED":
                print(
                    f"  [{label}] part-{part}: {j.state.name} (skipping, not done yet)"
                )
                continue
            results = _download_results(client, j)
            print(f"  [{label}] part-{part}: {j.state.name}, {len(results)} results")
            merged.update(results)
        return merged

    first_pass_results = collect(first_pass_jobs, "first-pass")
    escalation_results = collect(escalation_jobs, "cascade")

    final_records: list[dict[str, Any]] = []
    pending_first_pass = 0
    pending_escalation = 0

    for rec in records:
        key = _record_key(rec)
        fp = first_pass_results.get(key)
        if fp is None:
            pending_first_pass += 1
            continue

        if fp.get("error") or not fp.get("is_relevant"):
            relevance_info = {
                **fp,
                "filtered": (
                    should_filter_by_relevance(fp, args.confidence_threshold)
                    if not fp.get("error")
                    else False
                ),
                "threshold": args.confidence_threshold,
                "cascade_escalated": False,
            }
            out = dict(rec)
            out["relevance"] = relevance_info
            final_records.append(out)
            continue

        esc = escalation_results.get(key)
        if esc is None:
            pending_escalation += 1
            continue

        relevance_info = {
            **esc,
            "filtered": (
                should_filter_by_relevance(esc, args.confidence_threshold)
                if not esc.get("error")
                else False
            ),
            "threshold": args.confidence_threshold,
            "cascade_escalated": True,
            "cascade_first_pass": {
                "model": fp.get("model"),
                "decision": fp.get("decision"),
                "confidence": fp.get("confidence"),
                "metadata": fp.get("metadata"),
            },
        }
        out = dict(rec)
        out["relevance"] = relevance_info
        final_records.append(out)

    write_mode = "a" if output_path.exists() else "w"
    with output_path.open(write_mode, encoding="utf-8") as fh:
        for r in final_records:
            if args.filter_only and r.get("relevance", {}).get("filtered", False):
                continue
            fh.write(json.dumps(r) + "\n")

    print(f"\nRecovered {len(final_records)} finalized records -> {output_path}")
    print(f"  Still missing first-pass result: {pending_first_pass}")
    print(f"  First-pass said relevant, awaiting cascade result: {pending_escalation}")
    if pending_first_pass or pending_escalation:
        print(
            "\nRun relevance_filter.py again with --resume --batch-api --cascade "
            "(same --input/--output) to reprocess the remaining records."
        )


if __name__ == "__main__":
    main()
