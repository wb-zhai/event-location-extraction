# Vertex AI Training

Run LlamaFactory LoRA fine-tuning on Vertex AI custom jobs.

This folder is self-contained: the image clones upstream
[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) at a pinned commit
during the build, so nothing outside this folder is needed to build or run it.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Container image — NGC PyTorch base + pinned LlamaFactory clone + `gsutil` |
| `entrypoint.sh` | Container startup: downloads data and config from GCS, runs training, syncs checkpoints back |
| `requirements-extra.txt` | Flash-linear-attention deps needed for Qwen3.5 (not in upstream LlamaFactory requirements) |
| `config/*.yaml` | Training hyperparameter configs — edit and upload to GCS, no rebuild needed |
| `build_push.sh` | Builds and pushes the image to Artifact Registry |
| `submit_job.sh` | Submits a Vertex AI custom job via `gcloud` |
| `cloudbuild.yaml` | Cloud Build config for building the image without a local Docker pull |
| `.env.example` | Template for secrets — copy to `.env` (gitignored) |

## Prerequisites

### GCP setup (one-time)

Set `GCP_PROJECT` and `GCP_REGION` first (or source your `.env`):

```bash
set -a && source vertexai/train/event-extraction/.env && set +a

# Enable required APIs
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
    --project="${GCP_PROJECT}"

# Create Artifact Registry repo
gcloud artifacts repositories create llamafactory \
    --repository-format=docker \
    --location="${GCP_REGION}" \
    --project="${GCP_PROJECT}"

# Authenticate Docker to push images (only needed for local builds)
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev"

# Grant Cloud Build permission to push to Artifact Registry (for --cloud builds)
PROJECT_NUMBER=$(gcloud projects describe "${GCP_PROJECT}" --format="value(projectNumber)")
gcloud artifacts repositories add-iam-policy-binding llamafactory \
    --location="${GCP_REGION}" \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/artifactregistry.writer" \
    --project="${GCP_PROJECT}"

# Grant the default Vertex AI service account read/write on your GCS bucket
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

### GCS layout

| Variable | Role | Example value |
| --- | --- | --- |
| `GCS_BUCKET` | Bucket name | `my-bucket` |
| `GCS_DATA_PREFIX` | Dataset directory inside the bucket | `vertexai/data` |
| `GCS_SAVE_PREFIX` | Checkpoint output root inside the bucket | `vertexai/experiments` |
| `GCS_TRAIN_CONFIG` | Path to the training YAML inside the bucket | `vertexai/configs/train_config.yaml` |

`GCS_DATA_PREFIX` must contain:

- `dataset_info.json` with entries for the dataset(s) referenced by the config you use
- the actual dataset files referenced by those entries

Checkpoints are written to `gs://$GCS_BUCKET/$GCS_SAVE_PREFIX/<job_name>/` (synced every 10 min and on exit).

### Local credentials

```bash
cp vertexai/train/event-extraction/.env.example vertexai/train/event-extraction/.env
# Edit .env and fill in GCP_PROJECT, GCP_REGION, GCS_BUCKET, etc.
```

## Workflow

### 1. Build and push the image

Run from anywhere — the scripts resolve their own folder and the build context is this
folder itself (`vertexai/train/event-extraction/`).

**Option A — Cloud Build (recommended):** builds entirely in GCP, no local download of the NGC base image (~20 GB).

```bash
set -a && source vertexai/train/event-extraction/.env && set +a
bash vertexai/train/event-extraction/build_push.sh --cloud latest
```

Monitor progress:

```bash
gcloud beta builds list --project="${GCP_PROJECT}" --limit=5
gcloud beta builds log $(gcloud beta builds list --project="${GCP_PROJECT}" --limit=1 --format="value(id)") --project="${GCP_PROJECT}"
```

**Option B — local Docker:** pulls the NGC base image locally first.

```bash
set -a && source vertexai/train/event-extraction/.env && set +a
bash vertexai/train/event-extraction/build_push.sh latest
```

Both options print the `IMAGE` at the end — paste it into `.env` as `IMAGE=...`.

Rebuild only when `Dockerfile`, the pinned `LF_COMMIT`, or `requirements-extra.txt` change.
The same image can be reused across many job submissions — **changing hyperparameters does
not require a rebuild**.

### 1b. Upload the train config

Training parameters live in `config/*.yaml`. After editing one, upload it to GCS:

```bash
gsutil cp vertexai/train/event-extraction/config/qwen3_5_4b_base_new_schema.yaml \
    "gs://${GCS_BUCKET}/${GCS_TRAIN_CONFIG}"
```

`GCS_TRAIN_CONFIG` is a path inside the bucket (same convention as `GCS_DATA_PREFIX`):

```bash
export GCS_TRAIN_CONFIG="vertexai/configs/train_config.yaml"
```

### 2. Submit a training job

```bash
# A100 40 GB (default)
bash vertexai/train/event-extraction/submit_job.sh

# L4 24 GB  (8 vCPU | 32 GB host RAM)
GPU=l4 bash vertexai/train/event-extraction/submit_job.sh

# Debug run: T4 + 10 steps (~$0.01, tests the full pipeline end-to-end)
MAX_STEPS=10 GPU=t4 bash vertexai/train/event-extraction/submit_job.sh
```

The job name is auto-generated as `<JOB_NAME_PREFIX>-<gpu>-<timestamp>`. Override the prefix via `JOB_NAME_PREFIX` in `.env` or inline:

```bash
JOB_NAME_PREFIX=my-experiment bash vertexai/train/event-extraction/submit_job.sh
```

If the training config uses `finetuning_type: lora`, the LoRA adapter is automatically merged into the base model at the end of training. The merged model is saved to `<output_dir>/merged/` and included in the final GCS sync.

### 3. Monitor the job

```bash
# List recent jobs
gcloud ai custom-jobs list --project=$GCP_PROJECT --region=$GCP_REGION

# Stream logs (replace JOB_ID)
gcloud ai custom-jobs stream-logs JOB_ID --project=$GCP_PROJECT --region=$GCP_REGION
```

W&B logs appear in your W&B dashboard under the run name matching the job name.

### 4. Retrieve output

```bash
gsutil ls gs://$GCS_BUCKET/saves/
gsutil -m cp -r gs://$GCS_BUCKET/saves/<job_name> ./saves/
```

## GPU options

| `GPU=` | Machine type | vCPU | Host RAM | GPU VRAM | Approx. cost |
| --- | --- | --- | --- | --- | --- |
| `a100-40gb` (default) | `a2-highgpu-1g` | 12 | 85 GB | 40 GB | ~$3.67/hr |
| `l4` | `g2-standard-8` | 8 | 32 GB | 24 GB | ~$0.70/hr |
| `t4` | `n1-standard-4` | 4 | 15 GB | 16 GB | ~$0.35/hr |

> **Debug tip:** use `MAX_STEPS=10 GPU=t4` to test the full pipeline (data download → model load → training step → GCS sync) for ~$0.01.
> **L4/T4 note:** with `cutoff_len 16384` and `per_device_train_batch_size 2` you may hit OOM. Try reducing `--cutoff_len` to `8192` or `--per_device_train_batch_size` to `1` if training fails to start.

## Container image

Base: `nvcr.io/nvidia/pytorch:26.05-py3` (PyTorch 2.12 | CUDA 12.9 | Flash Attention pre-built).
LlamaFactory is cloned from upstream at the commit pinned in `ARG LF_COMMIT` in the Dockerfile —
bump that commit to pick up upstream changes.

To use a different NGC tag, change the `ARG BASE_IMAGE` line in `Dockerfile`:

```
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
```
