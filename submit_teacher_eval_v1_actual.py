#!/usr/bin/env python3
"""
Submit the V1 Teacher evaluation using the actual June 2026 South Sudan
production execution semantics recovered from the historical Batch launcher.

Important: the historical Batch template propagated only:
  INPUT_GCS, OUTPUT_GCS_PREFIX, MODEL_GCS,
  MAX_NEW_TOKENS, BATCH_SIZE,
  BATCH_TASK_INDEX, BATCH_TASK_COUNT.

Therefore this helper intentionally does NOT set retriever, quantization,
max_model_len, top_k_candidates, GPU-memory, temperature, or windowing env vars.
Those fall back to the historical image's /entrypoint.sh defaults, matching the
successful production path.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


DEFAULT_IMAGE = (
    "us-central1-docker.pkg.dev/zerohungerai/event-extraction/"
    "event-infer@sha256:962f378a8d683fad72eac2ab365c5f818d03c8262861fb2b30e8722d414c5f47"
)

DEFAULT_INPUT = (
    "gs://zhai-risk-factor-extraction/extraction/eval/"
    "v3_teacher_dev_articles.jsonl"
)

DEFAULT_MODEL = (
    "gs://zhai-risk-factor-extraction/vertexai/experiments/"
    "qwen3_5-lora-a100-40gb-20260617-151840/merged"
)

DEFAULT_OUTPUT_PREFIX = (
    "gs://zhai-risk-factor-extraction/extraction/eval/"
    "v1_teacher_predictions_actual"
)


def q(s: str) -> str:
    return shlex.quote(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--input-gcs", default=DEFAULT_INPUT)
    ap.add_argument("--model-gcs", default=DEFAULT_MODEL)
    ap.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument(
        "--template",
        default="vertexai/inference/event-extraction/batch_job.spot-a100.json",
    )
    ap.add_argument("--project", default="zerohungerai")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--expected-rows", type=int, default=2198)

    # These are the only inference knobs the historical Batch launcher propagated.
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=500)

    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")

    with template_path.open(encoding="utf-8") as f:
        spec = json.load(f)

    groups = spec.get("taskGroups")
    if not isinstance(groups, list) or not groups:
        raise SystemExit("Template has no taskGroups[0]")

    group = groups[0]
    task_spec = group.get("taskSpec")
    if not isinstance(task_spec, dict):
        raise SystemExit("Template has no taskGroups[0].taskSpec")

    group["taskCount"] = 1
    group["parallelism"] = 1
    task_spec["maxRunDuration"] = "86400s"

    allocation = spec.get("allocationPolicy") or {}
    instances = allocation.get("instances") or []
    if not instances:
        raise SystemExit("Template has no allocationPolicy.instances[0]")

    inst = instances[0]
    inst["installGpuDrivers"] = True

    policy = inst.get("policy") or {}
    boot_disk = policy.get("bootDisk")
    if isinstance(boot_disk, dict):
        boot_disk.pop("image", None)

    output_prefix = args.output_prefix.rstrip("/")
    remote_output = output_prefix + "/shard-0.jsonl"

    shell = f"""
set -euo pipefail

export INPUT_GCS={q(args.input_gcs)}
export OUTPUT_GCS_PREFIX={q(output_prefix)}
export MODEL_GCS={q(args.model_gcs)}

# Historical production launcher propagated only these inference settings.
export MAX_NEW_TOKENS={args.max_new_tokens}
export BATCH_SIZE={args.batch_size}

export BATCH_TASK_INDEX=0
export BATCH_TASK_COUNT=1

# Make sure reconstruction-only variables are absent so the historical
# entrypoint uses its own defaults, exactly as in the successful production path.
unset MAX_MODEL_LEN TOP_K_CANDIDATES QUANTIZATION GPU_MEMORY_UTIL TEMPERATURE
unset RETRIEVER_MODEL_NAME RETRIEVER_INDEX RETRIEVER_GPU_MEM_UTIL
unset RETRIEVER_QUERY_MODE RETRIEVER_MAX_MODEL_LEN
unset MAX_CHARS MIN_CHARS MAX_PARAS OVERLAP_PARAS

echo "[v1-teacher] actual-production semantics"
echo "[v1-teacher] input: $INPUT_GCS"
echo "[v1-teacher] output prefix: $OUTPUT_GCS_PREFIX"
echo "[v1-teacher] max_new_tokens: $MAX_NEW_TOKENS"
echo "[v1-teacher] batch_size: $BATCH_SIZE"

/entrypoint.sh

echo "[v1-teacher] validating output"
CHECK=/tmp/v1_teacher_check.jsonl
gsutil cp {q(remote_output)} "$CHECK"

N=$(wc -l < "$CHECK")
echo "[v1-teacher] output rows: $N"
test "$N" -eq {args.expected_rows}

python3 - "$CHECK" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    rows = [json.loads(next(f)) for _ in range(5)]

for i, row in enumerate(rows):
    for k in ("annotation", "predictions", "source"):
        if k not in row:
            raise SystemExit(f"row {{i}} missing {{k}}")
    src = row.get("source") or {{}}
    if not src.get("publish_date"):
        raise SystemExit(f"row {{i}} lost source.publish_date")

print("[v1-teacher] annotation/source preservation: OK")
print("[v1-teacher] sample prediction counts:",
      [len(r.get("predictions") or []) for r in rows])
PY

echo "[v1-teacher] validation passed"
""".strip()

    task_spec["runnables"] = [
        {
            "container": {
                "imageUri": args.image,
                "entrypoint": "/bin/bash",
                "commands": ["-lc", shell],
            }
        }
    ]

    out_dir = Path("data/extraction/eval/jobs")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / f"{args.job_id}.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print(f"Job ID:         {args.job_id}")
    print(f"Image:          {args.image}")
    print(f"Input:          {args.input_gcs}")
    print(f"Model:          {args.model_gcs}")
    print(f"Output:         {remote_output}")
    print(f"Expected rows:  {args.expected_rows}")
    print(f"Config:         {config_path}")
    print()
    print("Historical propagated settings only:")
    print(f"  max_new_tokens={args.max_new_tokens}")
    print(f"  batch_size={args.batch_size}")
    print("  retriever=OFF (not propagated by production Batch launcher)")
    print("  all other inference values = historical /entrypoint.sh defaults")
    print()

    cmd = [
        "gcloud", "batch", "jobs", "submit", args.job_id,
        f"--project={args.project}",
        f"--location={args.region}",
        f"--config={config_path}",
    ]

    print("Submit command:")
    print(" ".join(shlex.quote(x) for x in cmd))

    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
