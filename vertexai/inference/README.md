# GCP Cloud Batch — batch inference runbook

Runs `scripts/train/inference/vllm_infer.py` as an N-task array job on GCP Cloud Batch.
Each task is an independent GPU VM (one shard). Outputs are merged after all tasks complete.

## Prerequisites

- `gcloud` CLI authenticated — run `gcloud auth login`
- GCS bucket for model weights, input data, and outputs
- GPU quota in your target region for your chosen tier (check before large jobs)

---

## Step 1 — Stage model and data to GCS

```bash
# Upload the merged model (only once per model version)
gsutil -m cp -r models/new_schema/qwen3-lora-a100-40gb-20260616-172821/merged/ \
    gs://BUCKET/models/qwen3-4b-merged/

# Upload your input JSONL
gsutil cp /path/to/input.jsonl gs://BUCKET/data/input.jsonl
```

---

## Step 2 — Build and push the Docker image

Run the setup script from the **repo root** (one-time; idempotent):

```bash
bash vertexai/inference/setup.sh
```

This creates the Artifact Registry repo if needed, then submits a Cloud Build job to build and push the image.
Reads `vertexai/inference/.env` automatically; override with `--project`, `--region`, or `--tag`.

Add `--local` to build with local Docker instead of Cloud Build.

---

## Step 3 — Submit the job

```bash
cd vertexai/inference
chmod +x submit_batch.sh merge_shards.sh

./submit_batch.sh \
    --tier    spot-l4 \          # spot-l4 | spot-a100 | flex-l4 | flex-a100 | l4 | a100
    --shards  20 \               # number of parallel GPU tasks
    --image   "${IMAGE}" \
    --input   gs://BUCKET/data/input.jsonl \
    --output  gs://BUCKET/output/run-001 \
    --model   gs://BUCKET/models/qwen3-4b-merged \
    --project "${PROJECT}" \
    --region  "${REGION}"
```

### Optional inference flags

| Flag | Default | Description |
|---|---|---|
| `--max-new-tokens` | `4096` | Max tokens to generate per window |
| `--max-model-len` | vLLM auto | Total context length (prompt + output); set to limit VRAM |
| `--max-chars` | `3000` | Max characters per text window |
| `--top-k-candidates` | all | Number of ontology labels to include per article |
| `--quantization` | none | vLLM quantization mode (e.g. `fp8`) |
| `--batch-size` | `500` | Articles per vLLM batch |

Example with all tuning flags:

```bash
./submit_batch.sh \
    --tier              spot-l4 \
    --shards            1 \
    --input             gs://BUCKET/data/input.jsonl \
    --output            gs://BUCKET/output/run-001 \
    --model             gs://BUCKET/models/qwen3-4b-merged \
    --max-new-tokens    2048 \
    --max-model-len     8192 \
    --max-chars         3000 \
    --top-k-candidates  70 \
    --quantization      fp8
```

### Choosing `--shards` (at ~1.3 s/article on L4)

| Articles | 20 shards | 45 shards | 90 shards |
|---|---|---|---|
| 300 K | ~5.4 h | ~2.4 h | ~1.2 h |
| 1 M | ~18 h | ~8 h | ~4 h |

Cost is flat in N (you pay total GPU-hours regardless). Use more shards to finish faster.

### GPU tiers

| `--tier` | Machine | GPU | Spot? | ~$/GPU-h | Notes |
|---|---|---|---|---|---|
| `spot-l4` | g2-standard-8 | 1× L4 24GB | Yes | ~$0.30 | Cheapest; auto-retries on preemption |
| `spot-a100` | a2-highgpu-1g | 1× A100 40GB | Yes | ~$1.10 | Fast + cheap; preemptible, auto-retries |
| `flex-l4` | g2-standard-8 | 1× L4 24GB | Flex | ~$0.54 | Lower interruption risk than spot; GCP queues until capacity available |
| `flex-a100` | a2-highgpu-1g | 1× A100 40GB | Flex | ~$2.20 | Best A100 availability; lower interruption risk than spot |
| `l4` | g2-standard-8 | 1× L4 24GB | No | ~$0.85 | No interruptions |
| `a100` | a2-highgpu-1g | 1× A100 40GB | No | ~$3.50 | Faster per-GPU, fewer nodes needed |

---

## Step 4 — Monitor

```bash
gcloud batch jobs describe infer-YYYYMMDD-HHMMSS \
    --project="${PROJECT}" --location="${REGION}"

# Live logs from all tasks
gcloud logging read \
    "resource.labels.job_uid=infer-YYYYMMDD-HHMMSS" \
    --project="${PROJECT}" --limit=200 --format="value(textPayload)"
```

---

## Step 5 — Merge shards

```bash
./merge_shards.sh \
    --output  gs://BUCKET/output/run-001 \
    --dest    combined.jsonl \
    --shards  20                # optional validation
```

---

## Spot preemption / recovery

The entrypoint syncs partial output to GCS every 90 seconds and resumes from it on retry.
`vllm_infer.py` also has built-in recovery (skips already-processed articles by SHA-256 of text).
Both layers together mean a preempted task loses at most ~90 s of work and retries cleanly
(`maxRetryCount: 3` is set in `batch_job.spot-l4.json`).

---

## Troubleshooting

**"Quota exceeded"** — request quota for `NVIDIA_L4_GPUS` (or `NVIDIA_A100_GPUS` / `PREEMPTIBLE_NVIDIA_A100_GPUS` for spot-a100) in your region via IAM & Admin → Quotas.

**Image not found** — confirm the Artifact Registry repo exists and the image was pushed successfully.

**Model load fails** — verify the GCS path contains the full merged model (should have `config.json`, `tokenizer.json`, `model-*.safetensors`).

**Shard count mismatch after merge** — check if any tasks failed via `gcloud batch jobs describe`; re-run failed tasks or resubmit with the same `--output` prefix (recovery will skip completed articles).