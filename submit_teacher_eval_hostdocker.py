#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
from pathlib import Path

DEFAULT_IMAGE = "us-central1-docker.pkg.dev/zerohungerai/event-extraction/event-infer:latest"
DEFAULT_INPUT = "gs://zhai-risk-factor-extraction/extraction/eval/v3_teacher_dev_articles.jsonl"

def q(s: str) -> str:
    return shlex.quote(s)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--model-gcs", required=True)
    ap.add_argument("--output-gcs", required=True)
    ap.add_argument("--input-gcs", default=DEFAULT_INPUT)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--template", default="vertexai/inference/event-extraction/batch_job.spot-a100.json")
    ap.add_argument("--project", default="zerohungerai")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--expected-rows", type=int, default=2198)
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

    spec = json.loads(template_path.read_text(encoding="utf-8"))
    group = spec["taskGroups"][0]
    task_spec = group["taskSpec"]
    group["taskCount"] = 1
    group["parallelism"] = 1
    task_spec["maxRunDuration"] = "86400s"

    model_glob = args.model_gcs.rstrip("/") + "/*"

    payload = f"""
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

    payload_b64 = base64.b64encode(payload.encode()).decode()

    host_script = f"""#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[host] OS:"
cat /etc/os-release

if ! command -v docker >/dev/null 2>&1; then
  echo "[host] installing docker.io from Ubuntu repositories"
  apt-get update -qq
  apt-get install -y -qq docker.io
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
  nvidia-ctk runtime configure --runtime=docker || true
fi

systemctl restart docker || service docker restart || true
docker --version
nvidia-smi

TOKEN=$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

echo "$TOKEN" | docker login -u oauth2accesstoken --password-stdin us-central1-docker.pkg.dev

IMAGE={q(args.image)}
docker pull "$IMAGE"

PAYLOAD_B64={q(payload_b64)}
docker run --rm \
  --gpus all \
  --shm-size=4g \
  -e PAYLOAD_B64="$PAYLOAD_B64" \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc 'echo "$PAYLOAD_B64" | base64 -d | bash'
"""

    # Critical difference from the failed submitter:
    # use a SCRIPT runnable, not a Batch container runnable.
    task_spec["runnables"] = [{"script": {"text": host_script}}]

    out_dir = Path("data/extraction/eval/jobs")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / f"{args.job_id}.json"
    config_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    print("Job ID:      ", args.job_id)
    print("Input:       ", args.input_gcs)
    print("Model:       ", args.model_gcs)
    print("Output:      ", args.output_gcs)
    print("Config:      ", config_path)
    print("Runnable:     host script -> docker run")
    print("lean_output:  FALSE")
    print("guided decode: FALSE")

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
