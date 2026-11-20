#!/usr/bin/env python3
"""
Submit a one-task Cloud Batch Teacher-evaluation inference job.

Uses the repo's existing spot-A100 batch template for allocation/service-account
settings, but overrides the runnable so vllm_infer.py reads the raw Teacher JSONL
directly. This intentionally does NOT use --gcs_input and does NOT use
--lean_output, so the output preserves Teacher `annotation` alongside predictions.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


DEFAULT_IMAGE = (
    "us-central1-docker.pkg.dev/zerohungerai/"
    "event-extraction/event-infer:latest"
)
DEFAULT_INPUT = (
    "gs://zhai-risk-factor-extraction/extraction/eval/"
    "v3_teacher_dev_articles.jsonl"
)


def q(s: str) -> str:
    return shlex.quote(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--model-gcs", required=True)
    ap.add_argument("--output-gcs", required=True)
    ap.add_argument("--input-gcs", default=DEFAULT_INPUT)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument(
        "--template",
        default="vertexai/inference/event-extraction/batch_job.spot-a100.json",
    )
    ap.add_argument("--project", default="zerohungerai")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--expected-rows", type=int, default=2198)

    # Frozen inference settings used for V2/V3 South Sudan serving.
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-paras", type=int, default=15)
    ap.add_argument("--overlap-paras", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.0)
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

    # One GPU, one task, exact same article set.
    group["taskCount"] = 1
    group["parallelism"] = 1
    task_spec["maxRunDuration"] = "86400s"

    model_glob = args.model_gcs.rstrip("/") + "/*"

    shell = f"""
set -euo pipefail

LOCAL_INPUT=/tmp/teacher_eval.jsonl
LOCAL_MODEL=/tmp/model
LOCAL_OUTPUT=/tmp/predictions.jsonl

mkdir -p "$LOCAL_MODEL"

echo "[teacher-eval] downloading Teacher input"
gsutil cp {q(args.input_gcs)} "$LOCAL_INPUT"

N=$(wc -l < "$LOCAL_INPUT")
echo "[teacher-eval] input rows: $N"
test "$N" -eq {args.expected_rows}

echo "[teacher-eval] downloading merged model"
gsutil -m cp -r {q(model_glob)} "$LOCAL_MODEL/"

echo "[teacher-eval] starting inference"
cd /app
python3 /app/scripts/event_extraction/inference/vllm_infer.py \
  --model_name_or_path "$LOCAL_MODEL" \
  --input "$LOCAL_INPUT" \
  --output "$LOCAL_OUTPUT" \
  --shard_index 0 \
  --num_shards 1 \
  --max_new_tokens {args.max_new_tokens} \
  --max_chars {args.max_chars} \
  --min_chars {args.min_chars} \
  --max_paras {args.max_paras} \
  --overlap_paras {args.overlap_paras} \
  --batch_size {args.batch_size} \
  --gpu_memory_utilization {args.gpu_memory_utilization} \
  --temperature {args.temperature}

OUT_N=$(wc -l < "$LOCAL_OUTPUT")
echo "[teacher-eval] output rows: $OUT_N"
test "$OUT_N" -eq {args.expected_rows}

echo "[teacher-eval] uploading full Teacher+prediction output"
gsutil cp "$LOCAL_OUTPUT" {q(args.output_gcs)}

echo "[teacher-eval] done"
""".strip()

    # Override only the runnable. Preserve the proven spot-A100 allocation policy,
    # network, service account, logs policy, compute resource, etc. from the repo template.
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

    print(f"Job ID:       {args.job_id}")
    print(f"Image:        {args.image}")
    print(f"Input:        {args.input_gcs}")
    print(f"Model:        {args.model_gcs}")
    print(f"Output:       {args.output_gcs}")
    print(f"Config:       {config_path}")
    print("lean_output:  FALSE")
    print("gcs_input:    FALSE")
    print("guided decode: FALSE")
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
