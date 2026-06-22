#!/usr/bin/env bash
# Cloud Batch entrypoint — one task = one GPU shard.
# Required env vars (set in the batch job spec):
#   INPUT_GCS          gs://bucket/path/input.jsonl
#   OUTPUT_GCS_PREFIX  gs://bucket/path/output/run-001
#   MODEL_GCS          gs://bucket/path/models/qwen3-4b-merged
#
# Optional env vars (all have defaults that match vllm_infer.py):
#   MAX_NEW_TOKENS     (default 4096)
#   MAX_MODEL_LEN      (default: not set → vLLM auto-detects)
#   MAX_CHARS          (default 3000)
#   MIN_CHARS          (default 200)
#   MAX_PARAS          (default 15)
#   OVERLAP_PARAS      (default 1)
#   BATCH_SIZE         (default 500)
#   TOP_K_CANDIDATES   (default: not set)
#   GPU_MEMORY_UTIL    (default 0.95)
#   TEMPERATURE        (default 0.0)
#   QUANTIZATION       (default: not set, e.g. fp8)
#
# Cloud Batch injects:
#   BATCH_TASK_INDEX   0-based task index
#   BATCH_TASK_COUNT   total number of tasks

set -euo pipefail

TASK_INDEX=${BATCH_TASK_INDEX:-0}
TASK_COUNT=${BATCH_TASK_COUNT:-1}

echo "[entrypoint] task ${TASK_INDEX}/${TASK_COUNT} starting"

LOCAL_MODEL=/local/model
LOCAL_INPUT=/local/input.jsonl
LOCAL_OUTPUT=/local/shard-${TASK_INDEX}.jsonl
REMOTE_OUTPUT="${OUTPUT_GCS_PREFIX}/shard-${TASK_INDEX}.jsonl"

# LOCAL_MODEL must exist as a directory before the multi-file `gsutil cp` below,
# otherwise gsutil treats it as a single-file destination and errors out.
mkdir -p "${LOCAL_MODEL}"

# --- 1. Pull model (parallel, ~8 GB, ~1-2 min) ---
echo "[entrypoint] downloading model from ${MODEL_GCS}"
gsutil -m cp -r "${MODEL_GCS}/*" "${LOCAL_MODEL}/"
echo "[entrypoint] model ready"

# --- 2. Pull input ---
echo "[entrypoint] downloading input from ${INPUT_GCS}"
gsutil cp "${INPUT_GCS}" "${LOCAL_INPUT}"
echo "[entrypoint] input ready ($(wc -l < ${LOCAL_INPUT}) lines)"

# --- 3. Resume: pull any partial output from a previous Spot retry ---
echo "[entrypoint] checking for partial output at ${REMOTE_OUTPUT}"
gsutil cp "${REMOTE_OUTPUT}" "${LOCAL_OUTPUT}" 2>/dev/null && \
    echo "[entrypoint] resuming from $(wc -l < ${LOCAL_OUTPUT}) already-processed articles" || \
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

# --- 5. Build extra args ---
EXTRA_ARGS=""
[[ -n "${MAX_MODEL_LEN:-}" ]]      && EXTRA_ARGS+=" --max_model_len ${MAX_MODEL_LEN}"
[[ -n "${TOP_K_CANDIDATES:-}" ]]   && EXTRA_ARGS+=" --top_k_candidates ${TOP_K_CANDIDATES}"
[[ -n "${QUANTIZATION:-}" ]]       && EXTRA_ARGS+=" --quantization ${QUANTIZATION}"

# --- 6. Run inference ---
echo "[entrypoint] starting inference"
python3 /app/scripts/train/inference/vllm_infer.py \
    --model_name_or_path "${LOCAL_MODEL}" \
    --input               "${LOCAL_INPUT}" \
    --output              "${LOCAL_OUTPUT}" \
    --shard_index         "${TASK_INDEX}" \
    --num_shards          "${TASK_COUNT}" \
    --max_new_tokens      "${MAX_NEW_TOKENS:-4096}" \
    --max_chars           "${MAX_CHARS:-3000}" \
    --min_chars           "${MIN_CHARS:-200}" \
    --max_paras           "${MAX_PARAS:-15}" \
    --overlap_paras       "${OVERLAP_PARAS:-1}" \
    --batch_size          "${BATCH_SIZE:-500}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTIL:-0.95}" \
    --temperature         "${TEMPERATURE:-0.0}" \
    ${EXTRA_ARGS}

# --- 7. Final upload ---
kill "${SYNC_PID}" 2>/dev/null || true
echo "[entrypoint] uploading final output to ${REMOTE_OUTPUT}"
gsutil cp "${LOCAL_OUTPUT}" "${REMOTE_OUTPUT}"
echo "[entrypoint] task ${TASK_INDEX} done — $(wc -l < ${LOCAL_OUTPUT}) articles written"
