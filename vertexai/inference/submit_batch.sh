#!/usr/bin/env bash
# Submit a Cloud Batch inference job.
#
# Usage:
#   ./submit_batch.sh \
#     --tier        spot-l4          # spot-l4 | spot-a100 | flex-l4 | flex-a100 | l4 | a100
#     --shards      20               # number of parallel GPU tasks
#     --image       REGION-docker.pkg.dev/PROJECT/REPO/event-infer:TAG
#     --input       gs://BUCKET/data/input.jsonl
#     --output      gs://BUCKET/output/run-001
#     --model       gs://BUCKET/models/qwen3-4b-merged
#     [--project    my-gcp-project]  # default: gcloud config get project
#     [--region     us-central1]     # default: us-central1
#     [--job-id     infer-run-001]   # default: infer-YYYYMMDD-HHMMSS
#     [--max-new-tokens  4096]
#     [--batch-size      500]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present (values are used as defaults; CLI flags override them)
if [[ -f "${HERE}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a && source "${HERE}/.env" && set +a
fi

# ---- Defaults (env vars take precedence over these hardcoded fallbacks) ----
TIER="${TIER:-spot-l4}"
SHARDS="${SHARDS:-20}"
IMAGE="${IMAGE:-}"
INPUT_GCS="${INPUT_GCS:-}"
OUTPUT_GCS="${OUTPUT_GCS_PREFIX:-}"
MODEL_GCS="${MODEL_GCS:-}"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
JOB_ID="${JOB_ID:-infer-$(date -u +%Y%m%d-%H%M%S)}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-500}"
MAX_CHARS="${MAX_CHARS:-3000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
TOP_K_CANDIDATES="${TOP_K_CANDIDATES:-}"
QUANTIZATION="${QUANTIZATION:-}"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)            TIER="$2";            shift 2 ;;
        --shards)          SHARDS="$2";          shift 2 ;;
        --image)           IMAGE="$2";           shift 2 ;;
        --input)           INPUT_GCS="$2";       shift 2 ;;
        --output)          OUTPUT_GCS="$2";      shift 2 ;;
        --model)           MODEL_GCS="$2";       shift 2 ;;
        --project)         PROJECT="$2";         shift 2 ;;
        --region)          REGION="$2";          shift 2 ;;
        --job-id)          JOB_ID="$2";          shift 2 ;;
        --max-new-tokens)     MAX_NEW_TOKENS="$2";     shift 2 ;;
        --batch-size)         BATCH_SIZE="$2";         shift 2 ;;
        --max-chars)          MAX_CHARS="$2";          shift 2 ;;
        --max-model-len)      MAX_MODEL_LEN="$2";      shift 2 ;;
        --top-k-candidates)   TOP_K_CANDIDATES="$2";   shift 2 ;;
        --quantization)       QUANTIZATION="$2";       shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---- Validate required args ----
for var in IMAGE INPUT_GCS OUTPUT_GCS MODEL_GCS; do
    [[ -z "${!var}" ]] && { echo "ERROR: --${var//_/-} is required" >&2; exit 1; }
done
[[ -z "${PROJECT}" ]] && { echo "ERROR: --project required (gcloud config has no project set)" >&2; exit 1; }

TEMPLATE="${HERE}/batch_job.${TIER}.json"
[[ -f "${TEMPLATE}" ]] || { echo "ERROR: unknown tier '${TIER}' (no template ${TEMPLATE})" >&2; exit 1; }

# ---- Render the template into a temporary config ----
TMP_CONFIG="$(mktemp /tmp/batch_job_XXXXXX.json)"
trap "rm -f ${TMP_CONFIG}" EXIT

sed \
    -e "s|TASK_COUNT|${SHARDS}|g" \
    -e "s|IMAGE_URI|${IMAGE}|g" \
    -e "s|VAL_INPUT_GCS|${INPUT_GCS}|g" \
    -e "s|VAL_OUTPUT_GCS_PREFIX|${OUTPUT_GCS}|g" \
    -e "s|VAL_MODEL_GCS|${MODEL_GCS}|g" \
    -e "s|VAL_MAX_NEW_TOKENS|${MAX_NEW_TOKENS}|g" \
    -e "s|VAL_BATCH_SIZE|${BATCH_SIZE}|g" \
    -e "s|VAL_MAX_CHARS|${MAX_CHARS}|g" \
    -e "s|VAL_MAX_MODEL_LEN|${MAX_MODEL_LEN}|g" \
    -e "s|VAL_TOP_K_CANDIDATES|${TOP_K_CANDIDATES}|g" \
    -e "s|VAL_QUANTIZATION|${QUANTIZATION}|g" \
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
echo "Outputs will appear at: ${OUTPUT_GCS}/shard-*.jsonl"
echo "Once complete, merge with:  ./merge_shards.sh --output ${OUTPUT_GCS} --dest combined.jsonl"
