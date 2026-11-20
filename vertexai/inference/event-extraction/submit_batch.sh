#!/usr/bin/env bash
# Submit a Cloud Batch inference job.
#
# Usage:
#   ./submit_batch.sh \
#     --tier        spot-a100        # spot-l4 | spot-a100 | flex-l4 | flex-a100 | l4 | a100
#     --shards      200              # number of parallel GPU tasks (= number of manifest files)
#     --image       REGION-docker.pkg.dev/PROJECT/REPO/event-infer:TAG
#     --manifest    gs://BUCKET/extraction/manifests   # holds manifest-000.jsonl ...
#     --output      gs://BUCKET/output/run-001
#     --model       gs://BUCKET/models/qwen3-4b-merged
#     [--project    my-gcp-project]  # default: gcloud config get project
#     [--region     us-central1]     # default: us-central1
#     [--job-id     infer-run-001]   # default: infer-YYYYMMDD-HHMMSS
#     [--max-new-tokens  4096]
#     [--batch-size      500]
#     [--temperature     0.0]
#     [--use-guided-decoding]        # constrain event_type to the ontology enum
#     [--ontology-descriptions all]  # required by *-desc models
#
# Build the manifest first:
#   python build_manifest.py --output-prefix gs://BUCKET/extraction/manifests --num-shards 200

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
MANIFEST_GCS="${MANIFEST_GCS_PREFIX:-}"
OUTPUT_GCS="${OUTPUT_GCS_PREFIX:-}"
MODEL_GCS="${MODEL_GCS:-}"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
JOB_ID="${JOB_ID:-infer-$(date -u +%Y%m%d-%H%M%S)}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-500}"
# Wall-clock cap per task (taskSpec.maxRunDuration). NOTE: this does NOT make the
# FLEX_START tiers work — GCE also needs an instance termination action, which Cloud
# Batch's InstancePolicy cannot express, so flex-l4/flex-a100 fail at scheduling with
# CODE_GCE_BAD_REQUEST regardless of this value. See README. Use the spot tiers.
MAX_RUN_DURATION="${MAX_RUN_DURATION:-86400s}"
MAX_CHARS="${MAX_CHARS:-3000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
TOP_K_CANDIDATES="${TOP_K_CANDIDATES:-}"
QUANTIZATION="${QUANTIZATION:-}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-}"
GCS_READ_CONCURRENCY="${GCS_READ_CONCURRENCY:-64}"
LEAN_OUTPUT="${LEAN_OUTPUT:-true}"
MIN_CHARS="${MIN_CHARS:-200}"
MAX_PARAS="${MAX_PARAS:-15}"
OVERLAP_PARAS="${OVERLAP_PARAS:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
USE_GUIDED_DECODING="${USE_GUIDED_DECODING:-false}"
GUIDED_DECODING_BACKEND="${GUIDED_DECODING_BACKEND:-}"
ONTOLOGY="${ONTOLOGY:-}"
ONTOLOGY_DESCRIPTIONS="${ONTOLOGY_DESCRIPTIONS:-}"
SKIP_MANIFEST_CHECK="${SKIP_MANIFEST_CHECK:-false}"
RETRIEVER_MODEL_NAME="${RETRIEVER_MODEL_NAME:-}"
RETRIEVER_INDEX="${RETRIEVER_INDEX:-}"
RETRIEVER_GPU_MEM_UTIL="${RETRIEVER_GPU_MEM_UTIL:-}"
RETRIEVER_QUERY_MODE="${RETRIEVER_QUERY_MODE:-}"
RETRIEVER_MAX_MODEL_LEN="${RETRIEVER_MAX_MODEL_LEN:-}"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)            TIER="$2";            shift 2 ;;
        --shards)          SHARDS="$2";          shift 2 ;;
        --image)           IMAGE="$2";           shift 2 ;;
        --manifest)        MANIFEST_GCS="$2";    shift 2 ;;
        --output)          OUTPUT_GCS="$2";      shift 2 ;;
        --model)           MODEL_GCS="$2";       shift 2 ;;
        --project)         PROJECT="$2";         shift 2 ;;
        --region)          REGION="$2";          shift 2 ;;
        --job-id)          JOB_ID="$2";          shift 2 ;;
        --max-new-tokens)     MAX_NEW_TOKENS="$2";     shift 2 ;;
        --batch-size)         BATCH_SIZE="$2";         shift 2 ;;
        --max-run-duration)   MAX_RUN_DURATION="$2";   shift 2 ;;
        --max-chars)          MAX_CHARS="$2";          shift 2 ;;
        --max-model-len)      MAX_MODEL_LEN="$2";      shift 2 ;;
        --top-k-candidates)   TOP_K_CANDIDATES="$2";   shift 2 ;;
        --quantization)            QUANTIZATION="$2";            shift 2 ;;
        --gpu-memory-util)         GPU_MEMORY_UTIL="$2";         shift 2 ;;
        --gcs-read-concurrency)    GCS_READ_CONCURRENCY="$2";    shift 2 ;;
        --min-chars)               MIN_CHARS="$2";               shift 2 ;;
        --max-paras)               MAX_PARAS="$2";               shift 2 ;;
        --overlap-paras)           OVERLAP_PARAS="$2";           shift 2 ;;
        --temperature)             TEMPERATURE="$2";             shift 2 ;;
        --use-guided-decoding)     USE_GUIDED_DECODING="true";   shift 1 ;;
        --guided-decoding-backend) GUIDED_DECODING_BACKEND="$2"; shift 2 ;;
        --ontology)                ONTOLOGY="$2";                shift 2 ;;
        --ontology-descriptions)   ONTOLOGY_DESCRIPTIONS="$2";   shift 2 ;;
        --lean-output)             LEAN_OUTPUT="$2";             shift 2 ;;
        --skip-manifest-check)     SKIP_MANIFEST_CHECK="true";   shift 1 ;;
        --retriever-model-name)    RETRIEVER_MODEL_NAME="$2";    shift 2 ;;
        --retriever-index)         RETRIEVER_INDEX="$2";         shift 2 ;;
        --retriever-gpu-mem-util)  RETRIEVER_GPU_MEM_UTIL="$2";  shift 2 ;;
        --retriever-query-mode)    RETRIEVER_QUERY_MODE="$2";    shift 2 ;;
        --retriever-max-model-len) RETRIEVER_MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---- Validate required args ----
[[ -z "${IMAGE}"        ]] && { echo "ERROR: --image is required" >&2; exit 1; }
[[ -z "${MANIFEST_GCS}" ]] && { echo "ERROR: --manifest is required" >&2; exit 1; }
[[ -z "${OUTPUT_GCS}"   ]] && { echo "ERROR: --output is required" >&2; exit 1; }
[[ -z "${MODEL_GCS}"    ]] && { echo "ERROR: --model is required" >&2; exit 1; }
[[ -z "${PROJECT}" ]] && { echo "ERROR: --project required (gcloud config has no project set)" >&2; exit 1; }

TEMPLATE="${HERE}/batch_job.${TIER}.json"
[[ -f "${TEMPLATE}" ]] || { echo "ERROR: unknown tier '${TIER}' (no template ${TEMPLATE})" >&2; exit 1; }

# ---- Manifest/shard count must match ----
# Task N reads manifest-NNN.jsonl. Too few manifests and those tasks die on the gsutil cp;
# too many and the extra shards are silently never processed. Neither fails loudly at
# submit time, so check here.
if [[ "${SKIP_MANIFEST_CHECK}" != "true" ]]; then
    echo "Checking manifest count under ${MANIFEST_GCS} ..."
    MANIFEST_COUNT="$(gsutil ls "${MANIFEST_GCS}/manifest-*.jsonl" 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${MANIFEST_COUNT}" -eq 0 ]]; then
        echo "ERROR: no manifest-*.jsonl found under ${MANIFEST_GCS}" >&2
        echo "       build one first: python build_manifest.py --output-prefix ${MANIFEST_GCS} --num-shards ${SHARDS}" >&2
        exit 1
    fi
    if [[ "${MANIFEST_COUNT}" -ne "${SHARDS}" ]]; then
        echo "ERROR: --shards is ${SHARDS} but ${MANIFEST_COUNT} manifest files exist under ${MANIFEST_GCS}" >&2
        echo "       They must match: task N reads manifest-NNN.jsonl." >&2
        echo "       Rebuild the manifest with --num-shards ${SHARDS}, or submit with --shards ${MANIFEST_COUNT}." >&2
        exit 1
    fi
    echo "  ${MANIFEST_COUNT} manifest files, matches --shards ${SHARDS}"
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
    -e "s|VAL_MAX_NEW_TOKENS|${MAX_NEW_TOKENS}|g" \
    -e "s|VAL_BATCH_SIZE|${BATCH_SIZE}|g" \
    -e "s|VAL_MAX_CHARS|${MAX_CHARS}|g" \
    -e "s|VAL_MAX_MODEL_LEN|${MAX_MODEL_LEN}|g" \
    -e "s|VAL_TOP_K_CANDIDATES|${TOP_K_CANDIDATES}|g" \
    -e "s|VAL_QUANTIZATION|${QUANTIZATION}|g" \
    -e "s|VAL_GPU_MEMORY_UTIL|${GPU_MEMORY_UTIL}|g" \
    -e "s|VAL_GCS_READ_CONCURRENCY|${GCS_READ_CONCURRENCY}|g" \
    -e "s|VAL_LEAN_OUTPUT|${LEAN_OUTPUT}|g" \
    -e "s|VAL_MIN_CHARS|${MIN_CHARS}|g" \
    -e "s|VAL_MAX_PARAS|${MAX_PARAS}|g" \
    -e "s|VAL_OVERLAP_PARAS|${OVERLAP_PARAS}|g" \
    -e "s|VAL_TEMPERATURE|${TEMPERATURE}|g" \
    -e "s|VAL_USE_GUIDED_DECODING|${USE_GUIDED_DECODING}|g" \
    -e "s|VAL_GUIDED_DECODING_BACKEND|${GUIDED_DECODING_BACKEND}|g" \
    -e "s|VAL_ONTOLOGY_DESCRIPTIONS|${ONTOLOGY_DESCRIPTIONS}|g" \
    -e "s|VAL_ONTOLOGY|${ONTOLOGY}|g" \
    -e "s|VAL_RETRIEVER_MODEL_NAME|${RETRIEVER_MODEL_NAME}|g" \
    -e "s|VAL_RETRIEVER_INDEX|${RETRIEVER_INDEX}|g" \
    -e "s|VAL_RETRIEVER_GPU_MEM_UTIL|${RETRIEVER_GPU_MEM_UTIL}|g" \
    -e "s|VAL_RETRIEVER_QUERY_MODE|${RETRIEVER_QUERY_MODE}|g" \
    -e "s|VAL_RETRIEVER_MAX_MODEL_LEN|${RETRIEVER_MAX_MODEL_LEN}|g" \
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
if [[ "${SHARDS}" -le 32 ]]; then
    echo "Once complete, merge with:  ./merge_shards.sh --output ${OUTPUT_GCS} --dest combined.jsonl"
else
    echo "NOTE: ${SHARDS} shards is past the 32-object 'gsutil compose' limit, so merge_shards.sh"
    echo "      would fall back to downloading every shard locally. Consume ${OUTPUT_GCS}/shard-*.jsonl"
    echo "      by wildcard instead (as the relevance pipeline does)."
fi
