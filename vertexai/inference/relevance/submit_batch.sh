#!/usr/bin/env bash
# Submit a Cloud Batch relevance-classification job.
#
# Usage:
#   ./submit_batch.sh \
#     --tier        spot-l4          # spot-l4 | spot-a100 | flex-l4 | flex-a100 | l4 | a100
#     --shards      100              # number of parallel GPU tasks (= number of manifest files)
#     --image       REGION-docker.pkg.dev/PROJECT/REPO/relevance-infer:TAG
#     --manifest    gs://BUCKET/relevance/manifests   # dir with manifest-000.jsonl ...
#     --output      gs://BUCKET/relevance/output/run-001
#     --model       gs://BUCKET/models/relevance-modernbert
#     [--project    my-gcp-project]  # default: gcloud config get project
#     [--region     us-central1]     # default: us-central1
#     [--job-id     relevance-run-001]
#     [--max-chars 4000] [--max-length 2048] [--batch-size 2000]
#     [--gcs-read-concurrency 64] [--gpu-memory-util 0.9] [--title-only]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present (values are used as defaults; CLI flags override them)
if [[ -f "${HERE}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a && source "${HERE}/.env" && set +a
fi

# ---- Defaults (env vars take precedence over these hardcoded fallbacks) ----
TIER="${TIER:-spot-l4}"
SHARDS="${SHARDS:-100}"
IMAGE="${IMAGE:-}"
MANIFEST_GCS="${MANIFEST_GCS_PREFIX:-}"
OUTPUT_GCS="${OUTPUT_GCS_PREFIX:-}"
MODEL_GCS="${MODEL_GCS:-}"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
JOB_ID="${JOB_ID:-relevance-$(date -u +%Y%m%d-%H%M%S)}"
MAX_CHARS="${MAX_CHARS:-4000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-2000}"
GCS_READ_CONCURRENCY="${GCS_READ_CONCURRENCY:-64}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"
TITLE_ONLY="${TITLE_ONLY:-false}"
# Wall-clock cap per task. REQUIRED for FLEX_START tiers (GCE rejects a flex-start VM
# that has a maxRunDuration but no instance-termination action, and Batch's implicit
# 7-day default has none). Max for flex-start is 604800 (7 days).
MAX_RUN_DURATION="${MAX_RUN_DURATION:-86400s}"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)                 TIER="$2";                 shift 2 ;;
        --shards)               SHARDS="$2";               shift 2 ;;
        --image)                IMAGE="$2";                shift 2 ;;
        --manifest)             MANIFEST_GCS="$2";         shift 2 ;;
        --output)               OUTPUT_GCS="$2";           shift 2 ;;
        --model)                MODEL_GCS="$2";            shift 2 ;;
        --project)              PROJECT="$2";              shift 2 ;;
        --region)               REGION="$2";               shift 2 ;;
        --job-id)               JOB_ID="$2";               shift 2 ;;
        --max-chars)            MAX_CHARS="$2";            shift 2 ;;
        --max-length)           MAX_LENGTH="$2";           shift 2 ;;
        --batch-size)           BATCH_SIZE="$2";           shift 2 ;;
        --gcs-read-concurrency) GCS_READ_CONCURRENCY="$2"; shift 2 ;;
        --gpu-memory-util)      GPU_MEMORY_UTIL="$2";      shift 2 ;;
        --title-only)           TITLE_ONLY="true";         shift ;;
        --max-run-duration)     MAX_RUN_DURATION="$2";     shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---- Validate required args ----
_require() { [[ -z "$2" ]] && { echo "ERROR: $1 is required" >&2; exit 1; } || true; }
_require --image    "${IMAGE}"
_require --manifest "${MANIFEST_GCS}"
_require --output   "${OUTPUT_GCS}"
_require --model    "${MODEL_GCS}"
[[ -z "${PROJECT}" ]] && { echo "ERROR: --project required (gcloud config has no project set)" >&2; exit 1; }

TEMPLATE="${HERE}/batch_job.${TIER}.json"
[[ -f "${TEMPLATE}" ]] || { echo "ERROR: unknown tier '${TIER}' (no template ${TEMPLATE})" >&2; exit 1; }

# ---- Validate shard count matches the number of manifest files ----
if command -v gsutil >/dev/null 2>&1; then
    MANIFEST_COUNT="$(gsutil ls "${MANIFEST_GCS}/manifest-*.jsonl" 2>/dev/null | grep -c 'manifest-' || true)"
    if [[ -n "${MANIFEST_COUNT}" && "${MANIFEST_COUNT}" -gt 0 && "${MANIFEST_COUNT}" -ne "${SHARDS}" ]]; then
        echo "ERROR: --shards ${SHARDS} but found ${MANIFEST_COUNT} manifest-*.jsonl files at ${MANIFEST_GCS}." >&2
        echo "       Re-run build_manifest.py with --num-shards ${SHARDS}, or set --shards ${MANIFEST_COUNT}." >&2
        exit 1
    fi
fi

# ---- Render the template into a temporary config ----
TMP_CONFIG="$(mktemp /tmp/batch_job_XXXXXX.json)"
trap "rm -f ${TMP_CONFIG}" EXIT

sed \
    -e "s|\"TASK_COUNT\"|\"${SHARDS}\"|g" \
    -e "s|IMAGE_URI|${IMAGE}|g" \
    -e "s|VAL_MAX_RUN_DURATION|${MAX_RUN_DURATION}|g" \
    -e "s|VAL_MANIFEST_GCS_PREFIX|${MANIFEST_GCS}|g" \
    -e "s|VAL_OUTPUT_GCS_PREFIX|${OUTPUT_GCS}|g" \
    -e "s|VAL_MODEL_GCS|${MODEL_GCS}|g" \
    -e "s|VAL_MAX_CHARS|${MAX_CHARS}|g" \
    -e "s|VAL_MAX_LENGTH|${MAX_LENGTH}|g" \
    -e "s|VAL_BATCH_SIZE|${BATCH_SIZE}|g" \
    -e "s|VAL_GCS_READ_CONCURRENCY|${GCS_READ_CONCURRENCY}|g" \
    -e "s|VAL_GPU_MEMORY_UTIL|${GPU_MEMORY_UTIL}|g" \
    -e "s|VAL_TITLE_ONLY|${TITLE_ONLY}|g" \
    "${TEMPLATE}" > "${TMP_CONFIG}"

echo "=== Job config (${TIER}, ${SHARDS} shards) ==="
cat "${TMP_CONFIG}"
echo ""

# ---- Submit ----
echo "Submitting job: ${JOB_ID} (project=${PROJECT}, region=${REGION})"
gcloud batch jobs submit "${JOB_ID}" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --config="${TMP_CONFIG}"

echo ""
echo "Job submitted. Monitor with:"
echo "  gcloud batch jobs describe ${JOB_ID} --project=${PROJECT} --location=${REGION}"
echo "  gcloud logging read 'resource.labels.job_uid=${JOB_ID}' --project=${PROJECT} --limit=100"
echo ""
echo "Outputs (no merge): ${OUTPUT_GCS}/shard-*.csv  — read downstream via wildcard."
