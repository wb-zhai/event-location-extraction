#!/usr/bin/env bash
# Cloud Batch entrypoint — one task = one GPU shard = one manifest file.
# Required env vars (set in the batch job spec):
#   MANIFEST_GCS_PREFIX  gs://bucket/path/manifests   (contains manifest-000.jsonl ...)
#   OUTPUT_GCS_PREFIX    gs://bucket/path/output/run-001
#   MODEL_GCS            gs://bucket/path/models/relevance-modernbert
#
# Optional env vars (defaults match relevance_vllm_infer.py):
#   MAX_CHARS            (default 4000)
#   MAX_LENGTH           (default 2048)
#   BATCH_SIZE           (default 2000)
#   GCS_READ_CONCURRENCY (default 64)
#   GPU_MEMORY_UTIL      (default 0.9)
#
# Cloud Batch injects:
#   BATCH_TASK_INDEX   0-based task index
#   BATCH_TASK_COUNT   total number of tasks (must equal the number of manifest files)

set -euo pipefail

TASK_INDEX=${BATCH_TASK_INDEX:-0}
TASK_COUNT=${BATCH_TASK_COUNT:-1}
SHARD=$(printf '%03d' "${TASK_INDEX}")

echo "[entrypoint] task ${TASK_INDEX}/${TASK_COUNT} starting (shard ${SHARD})"

LOCAL_MODEL=/local/model
LOCAL_INPUT=/local/manifest-${SHARD}.jsonl
LOCAL_OUTPUT=/local/shard-${TASK_INDEX}.csv
REMOTE_INPUT="${MANIFEST_GCS_PREFIX}/manifest-${SHARD}.jsonl"
REMOTE_OUTPUT="${OUTPUT_GCS_PREFIX}/shard-${TASK_INDEX}.csv"

# LOCAL_MODEL must exist as a directory before the multi-file `gsutil cp` below,
# otherwise gsutil treats it as a single-file destination and errors out.
mkdir -p "${LOCAL_MODEL}"

# --- 1. Pull model ---
echo "[entrypoint] downloading model from ${MODEL_GCS}"
gsutil -m cp -r "${MODEL_GCS}/*" "${LOCAL_MODEL}/"
echo "[entrypoint] model ready"

# --- 2. Pull this task's manifest shard only ---
echo "[entrypoint] downloading manifest ${REMOTE_INPUT}"
gsutil cp "${REMOTE_INPUT}" "${LOCAL_INPUT}"
echo "[entrypoint] manifest ready ($(wc -l < ${LOCAL_INPUT}) rows)"

# --- 3. Resume: pull any partial output from a previous Spot retry ---
echo "[entrypoint] checking for partial output at ${REMOTE_OUTPUT}"
gsutil cp "${REMOTE_OUTPUT}" "${LOCAL_OUTPUT}" 2>/dev/null && \
    echo "[entrypoint] resuming from $(wc -l < ${LOCAL_OUTPUT}) already-labeled rows" || \
    echo "[entrypoint] no partial output found, starting fresh"

# --- 4. Background sync: upload output every 90 s so Spot preemptions lose < 90 s ---
_sync_loop() {
    while true; do
        sleep 90
        if [[ -f "${LOCAL_OUTPUT}" ]]; then
            gsutil -q cp "${LOCAL_OUTPUT}" "${REMOTE_OUTPUT}" && \
                echo "[sync] uploaded $(wc -l < ${LOCAL_OUTPUT}) lines at $(date -u +%H:%M:%S)"
        fi
    done
}
_sync_loop &
SYNC_PID=$!
trap "kill ${SYNC_PID} 2>/dev/null || true" EXIT

# --- 5. Run inference (num_shards=1: the manifest is already this task's shard) ---
echo "[entrypoint] starting inference"
python3 /app/relevance_vllm_infer.py \
    --model_name_or_path     "${LOCAL_MODEL}" \
    --input                  "${LOCAL_INPUT}" \
    --output                 "${LOCAL_OUTPUT}" \
    --shard_index            0 \
    --num_shards             1 \
    --max_chars              "${MAX_CHARS:-4000}" \
    --max_length             "${MAX_LENGTH:-2048}" \
    --batch_size             "${BATCH_SIZE:-2000}" \
    --gcs_read_concurrency   "${GCS_READ_CONCURRENCY:-64}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTIL:-0.9}"

# --- 6. Final upload ---
kill "${SYNC_PID}" 2>/dev/null || true
echo "[entrypoint] uploading final output to ${REMOTE_OUTPUT}"
gsutil cp "${LOCAL_OUTPUT}" "${REMOTE_OUTPUT}"
echo "[entrypoint] task ${TASK_INDEX} done — $(wc -l < ${LOCAL_OUTPUT}) rows written"
