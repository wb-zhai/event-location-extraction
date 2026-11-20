#!/usr/bin/env python3
"""
Submit V1 Teacher evaluation on Cloud Batch using the recovered June 22, 2026
historical event-infer image and its own /entrypoint.sh.

The script preserves the proven current Batch allocation/network/service-account
template, removes the problematic pinned boot image, enables automatic GPU driver
installation, and overrides only the runnable.

Historical V1 inference settings are frozen to the South Sudan production command.
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

DEFAULT_RETRIEVER_MODEL = (
    "gs://zhai-risk-factor-extraction/vertexai/retrievers/"
    "harrier-oss-v1-0.6b"
)

DEFAULT_RETRIEVER_INDEX = (
    "gs://zhai-risk-factor-extraction/vertexai/index/science/index/"
    "harrier-oss-v1-0.6b"
)

DEFAULT_OUTPUT_PREFIX = (
    "gs://zhai-risk-factor-extraction/extraction/eval/"
    "v1_teacher_predictions_raw"
)

DEFAULT_RUNTIME_SRC = (
    "gs://zhai-risk-factor-extraction/extraction/eval/"
    "v1_runtime_src_21055.tar.gz"
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
    ap.add_argument("--retriever-model", default=DEFAULT_RETRIEVER_MODEL)
    ap.add_argument("--retriever-index", default=DEFAULT_RETRIEVER_INDEX)
    ap.add_argument("--runtime-src-gcs", default=DEFAULT_RUNTIME_SRC)

    ap.add_argument(
        "--template",
        default="vertexai/inference/event-extraction/batch_job.spot-a100.json",
    )
    ap.add_argument("--project", default="zerohungerai")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--expected-rows", type=int, default=2198)

    # Exact historical South Sudan V1 production settings.
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=12238)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-paras", type=int, default=15)
    ap.add_argument("--overlap-paras", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--top-k-candidates", type=int, default=90)
    ap.add_argument("--quantization", default="fp8")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retriever-gpu-memory-utilization", type=float, default=0.15)
    ap.add_argument("--retriever-query-mode", default="per_window")
    ap.add_argument("--retriever-max-model-len", type=int, default=4096)

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

    shell = f'''
set -euo pipefail

export INPUT_GCS={q(args.input_gcs)}
export OUTPUT_GCS_PREFIX={q(output_prefix)}
export MODEL_GCS={q(args.model_gcs)}

export MAX_NEW_TOKENS={args.max_new_tokens}
export MAX_MODEL_LEN={args.max_model_len}
export MAX_CHARS={args.max_chars}
export MIN_CHARS={args.min_chars}
export MAX_PARAS={args.max_paras}
export OVERLAP_PARAS={args.overlap_paras}
export BATCH_SIZE={args.batch_size}
export TOP_K_CANDIDATES={args.top_k_candidates}
export QUANTIZATION={q(args.quantization)}
export GPU_MEMORY_UTIL={args.gpu_memory_utilization}
export TEMPERATURE={args.temperature}

export RETRIEVER_MODEL_NAME={q(args.retriever_model)}
export RETRIEVER_INDEX={q(args.retriever_index)}
export RETRIEVER_GPU_MEM_UTIL={args.retriever_gpu_memory_utilization}
export RETRIEVER_QUERY_MODE={q(args.retriever_query_mode)}
export RETRIEVER_MAX_MODEL_LEN={args.retriever_max_model_len}

export BATCH_TASK_INDEX=0
export BATCH_TASK_COUNT=1

# The recovered historical image contains the inference script but not the repo's
# top-level `src` package required by the retriever. Restore exactly the `src/`
# tree from historical commit 21055... and expose it on PYTHONPATH.
RUNTIME_SRC_DIR=/tmp/v1_runtime_src
RUNTIME_SRC_TAR=/tmp/v1_runtime_src.tar.gz
mkdir -p "$RUNTIME_SRC_DIR"

echo "[v1-teacher] restoring historical src package"
gsutil cp {q(args.runtime_src_gcs)} "$RUNTIME_SRC_TAR"
tar -xzf "$RUNTIME_SRC_TAR" -C "$RUNTIME_SRC_DIR"
export PYTHONPATH="$RUNTIME_SRC_DIR:${{PYTHONPATH:-}}"

python3 - <<'PY'
from src.index.inmemory import InMemoryIndexer
print("[v1-teacher] historical src import: OK")
PY

echo "[v1-teacher] image runtime: historical June 22 event-infer"
echo "[v1-teacher] input: $INPUT_GCS"
echo "[v1-teacher] output prefix: $OUTPUT_GCS_PREFIX"
echo "[v1-teacher] expected rows: {args.expected_rows}"

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
    row = json.loads(next(f))

required = ["annotation", "predictions", "source"]
missing = [k for k in required if k not in row]
if missing:
    raise SystemExit("Missing expected output keys: " + ", ".join(missing))

src = row.get("source") or {{}}
if not src.get("publish_date"):
    raise SystemExit("First output row lost source.publish_date")

print("[v1-teacher] first-row keys:", sorted(row))
print("[v1-teacher] source.publish_date:", src.get("publish_date"))
print("[v1-teacher] annotation preserved: yes")
print("[v1-teacher] predictions:", len(row.get("predictions") or []))
PY

echo "[v1-teacher] validation passed"
'''.strip()

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

    print(f"Job ID:           {args.job_id}")
    print(f"Image:            {args.image}")
    print(f"Input:            {args.input_gcs}")
    print(f"Model:            {args.model_gcs}")
    print(f"Retriever model:  {args.retriever_model}")
    print(f"Retriever index:  {args.retriever_index}")
    print(f"Runtime src:      {args.runtime_src_gcs}")
    print(f"Output:           {remote_output}")
    print(f"Expected rows:    {args.expected_rows}")
    print(f"Config:           {config_path}")
    print()
    print("Frozen V1 settings:")
    print(
        f"  max_new_tokens={args.max_new_tokens} "
        f"max_model_len={args.max_model_len} "
        f"max_chars={args.max_chars}"
    )
    print(
        f"  top_k_candidates={args.top_k_candidates} "
        f"quantization={args.quantization} "
        f"gpu_memory_utilization={args.gpu_memory_utilization}"
    )
    print(
        f"  retriever_query_mode={args.retriever_query_mode} "
        f"retriever_gpu_memory_utilization={args.retriever_gpu_memory_utilization} "
        f"retriever_max_model_len={args.retriever_max_model_len}"
    )
    print()

    cmd = [
        "gcloud",
        "batch",
        "jobs",
        "submit",
        args.job_id,
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
