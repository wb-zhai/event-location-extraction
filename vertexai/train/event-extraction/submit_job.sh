#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present (values are used as defaults; export/CLI vars override them)
if [[ -f "${HERE}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a && source "${HERE}/.env" && set +a
fi

# Required variables — set these in vertexai/train/event-extraction/.env or export before running
: "${GCP_PROJECT:?GCP_PROJECT must be set}"
: "${GCP_REGION:?GCP_REGION must be set}"
: "${IMAGE:?IMAGE must be set}"
: "${GCS_BUCKET:?GCS_BUCKET must be set}"
: "${GCS_DATA_PREFIX:?GCS_DATA_PREFIX must be set}"
: "${GCS_SAVE_PREFIX:?GCS_SAVE_PREFIX must be set}"
: "${GCS_TRAIN_CONFIG:?GCS_TRAIN_CONFIG must be set}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

# GPU profile: a100-40gb (default), l4, or t4
#   a100-40gb  →  a2-highgpu-1g           (12 vCPU | 85 GB RAM | 1× A100 40 GB) ~$3.67/hr
#   l4         →  g2-standard-8           ( 8 vCPU | 32 GB RAM | 1× L4  24 GB) ~$0.70/hr
#   t4         →  n1-standard-4 + 1× T4  ( 4 vCPU | 15 GB RAM | 1× T4  16 GB) ~$0.35/hr  ← debug
GPU="${GPU:-a100-40gb}"
case "${GPU}" in
    a100-40gb)
        MACHINE_TYPE="a2-highgpu-1g"
        ACCEL_YAML="      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1"
        ;;
    l4)
        MACHINE_TYPE="g2-standard-8"
        ACCEL_YAML="      acceleratorType: NVIDIA_L4
      acceleratorCount: 1"
        ;;
    t4)
        MACHINE_TYPE="n1-standard-4"
        ACCEL_YAML="      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1"
        ;;
    *) echo "Unknown GPU profile '${GPU}'. Valid options: a100-40gb, l4, t4" >&2; exit 1 ;;
esac

# MAX_STEPS: set to a small number (e.g. 10) for a cheap debug run.
MAX_STEPS="${MAX_STEPS:-}"
HF_TOKEN="${HF_TOKEN:-}"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-qwen3_5-4b-base-lora}"
JOB_NAME="${JOB_NAME_PREFIX}-${GPU}-$(date +%Y%m%d-%H%M%S)"
JOB_SPEC=$(mktemp /tmp/vertex_job_XXXXXX.yaml)
trap 'rm -f "${JOB_SPEC}"' EXIT

cat > "${JOB_SPEC}" <<YAML
workerPoolSpecs:
  - replicaCount: 1
    machineSpec:
      machineType: ${MACHINE_TYPE}
${ACCEL_YAML}
    containerSpec:
      imageUri: ${IMAGE}
      env:
        - name: GCS_BUCKET
          value: "${GCS_BUCKET}"
        - name: GCS_DATA_PREFIX
          value: "${GCS_DATA_PREFIX}"
        - name: GCS_SAVE_PREFIX
          value: "${GCS_SAVE_PREFIX}"
        - name: GCS_TRAIN_CONFIG
          value: "${GCS_TRAIN_CONFIG}"
        - name: JOB_NAME
          value: "${JOB_NAME}"
        - name: WANDB_API_KEY
          value: "${WANDB_API_KEY}"
YAML

[[ -n "${HF_TOKEN}" ]] && cat >> "${JOB_SPEC}" <<YAML
        - name: HF_TOKEN
          value: "${HF_TOKEN}"
YAML

[[ -n "${MAX_STEPS}" ]] && cat >> "${JOB_SPEC}" <<YAML
        - name: MAX_STEPS
          value: "${MAX_STEPS}"
YAML

echo "Submitting Vertex AI job: ${JOB_NAME}"
echo "  GPU:       ${GPU} (${MACHINE_TYPE})"
[[ -n "${MAX_STEPS}" ]] && echo "  MAX_STEPS: ${MAX_STEPS} (debug run)"
echo "  Image:     ${IMAGE}"
echo "  Data:      gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}"
echo "  Config:    gs://${GCS_BUCKET}/${GCS_TRAIN_CONFIG}"
echo "  Output:    gs://${GCS_BUCKET}/${GCS_SAVE_PREFIX}/${JOB_NAME}"

gcloud ai custom-jobs create \
    --project="${GCP_PROJECT}" \
    --region="${GCP_REGION}" \
    --display-name="${JOB_NAME}" \
    --config="${JOB_SPEC}"

echo ""
echo "Job submitted. Monitor with:"
echo "  gcloud ai custom-jobs list --project=${GCP_PROJECT} --region=${GCP_REGION}"
