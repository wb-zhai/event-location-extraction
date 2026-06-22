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
| `--gpu-memory-util` | `0.95` | Fraction of GPU VRAM reserved for the main model |
| `--retriever-model-name` | none | HF model id for the retriever (e.g. `microsoft/harrier-oss-v1-0.6b`) |
| `--retriever-index` | none | Path to the retriever index on the container (must be baked into the image or mounted) |
| `--retriever-gpu-mem-util` | none | GPU memory fraction for the retriever |
| `--retriever-query-mode` | none | Retriever query mode (e.g. `per_window`) |
| `--retriever-max-model-len` | none | Max context length for the retriever |

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

### Choosing `--shards`

#### L4 (~1.3 s/article)

| Articles | 20 shards | 45 shards | 90 shards |
|---|---|---|---|
| 300 K | ~5.4 h | ~2.4 h | ~1.2 h |
| 1 M | ~18 h | ~8 h | ~4 h |

#### A100 (~0.25 s/article)

| Articles | 20 shards | 45 shards | 90 shards |
|---|---|---|---|
| 300 K | ~1.0 h | ~28 min | ~14 min |
| 1 M | ~3.5 h | ~1.5 h | ~46 min |

Cost is flat in N (you pay total GPU-hours regardless). Use more shards to finish faster.

### Cost for 1 M articles by tier

Total GPU-hours: L4 ≈ 361 h (1M × 1.3 s), A100 ≈ 69 h (1M × 0.25 s).

| `--tier` | GPU | GPU-h | $/GPU-h | **Total cost** |
|---|---|---|---|---|
| `spot-l4` | L4 | 361 | $0.30 | **~$108** |
| `spot-a100` | A100 | 69 | $1.10 | **~$76** |
| `flex-l4` | L4 | 361 | $0.54 | **~$195** |
| `flex-a100` | A100 | 69 | $2.20 | **~$153** |
| `l4` | L4 | 361 | $0.85 | **~$307** |
| `a100` | A100 | 69 | $3.50 | **~$243** |

`spot-a100` is the cheapest option (~$76) thanks to A100 speed offsetting the higher $/h.

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

## Scripts

```
bash vertexai/inference/submit_batch.sh \
    --tier              flex-a100 \
    --shards            20 \
    --job-id            south-sudan-a100-$(date -u +%Y%m%d-%H%M%S) \
    --input             gs://zhai-risk-factor-extraction/data/db/south_sudan_articles.jsonl \
    --output            gs://zhai-risk-factor-extraction/data/db/predictions/south_sudan_articles \
    --model             gs://zhai-risk-factor-extraction/vertexai/experiments/qwen3_5-lora-a100-40gb-20260617-151840/merged \
    --max-new-tokens           4096 \
    --max-model-len            12238 \
    --max-chars                3000 \
    --top-k-candidates         90 \
    --quantization             fp8 \
    --gpu-memory-util          0.75 \
    --retriever-model-name     microsoft/harrier-oss-v1-0.6b \
    --retriever-index          zhai-risk-factor-extraction/vertexai/index/science/index/harrier-oss-v1-0.6b \
    --retriever-gpu-mem-util   0.15 \
    --retriever-query-mode     per_window \
    --retriever-max-model-len  4096
```

```
bash vertexai/inference/submit_batch.sh \
    --tier    flex-a100 \
    --shards  1 \
    --job-id  infer-test-a100-$(date -u +%Y%m%d-%H%M%S) \
    --input   gs://zhai-risk-factor-extraction/data/db/download_20000.sample.jsonl \
    --output  gs://zhai-risk-factor-extraction/data/db/predictions/download_20000.sample.smoke.a100 \
    --model   gs://zhai-risk-factor-extraction/vertexai/experiments/qwen3_5-lora-a100-40gb-20260617-151840/merged \
    --max-new-tokens           4096 \
    --max-model-len            12238 \
    --max-chars                3000 \
    --top-k-candidates         90 \
    --quantization             fp8 \
    --gpu-memory-util          0.75 \
    --retriever-model-name     gs://zhai-risk-factor-extraction/vertexai/retrievers/harrier-oss-v1-0.6b \
    --retriever-index          gs://zhai-risk-factor-extraction/vertexai/index/science/index/harrier-oss-v1-0.6b \
    --retriever-gpu-mem-util   0.15 \
    --retriever-query-mode     per_window \
    --retriever-max-model-len  4096
```

> **flex-a100 note:** `FLEX_START` VMs require a finite per-task wall-clock cap
> (GCE rejects a flex-start VM that has a `maxRunDuration` but no termination
> action, and Batch's implicit 7-day default has none). `submit_batch.sh`
> sets this automatically via `--max-run-duration` (default `86400s`, max
> `604800s`). Without it the job fails at scheduling time with
> `CODE_GCE_BAD_REQUEST` and produces **no task logs**.

```
bash vertexai/inference/submit_batch.sh \
    --tier    flex-a100 \
    --shards  20 \
    --image   "${IMAGE}" \
    --input   gs://BUCKET/data/zhai-v3-science-dev.jsonl \
    --output  gs://BUCKET/output/run-15062025 \
    --model   gs://BUCKET/models/qwen3-4b-merged \
    --project "${PROJECT}" \
    --region  "${REGION}" \
    --max-new-tokens           4096 \
    --max-model-len            12238 \
    --max-chars                3000 \
    --top-k-candidates         90 \
    --quantization             fp8 \
    --gpu-memory-util          0.75 \
    --retriever-model-name     microsoft/harrier-oss-v1-0.6b \
    --retriever-index          ontologies/zhai/index/harrier-oss-v1-0.6b \
    --retriever-gpu-mem-util   0.15 \
    --retriever-query-mode     per_window \
    --retriever-max-model-len  4096
```