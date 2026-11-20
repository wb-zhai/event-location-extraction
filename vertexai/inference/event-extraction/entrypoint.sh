#!/usr/bin/env bash
# Cloud Batch entrypoint — one task = one GPU shard = one manifest file.
# Required env vars (set in the batch job spec):
#   MANIFEST_GCS_PREFIX  gs://bucket/path/manifests   (contains manifest-000.jsonl ...)
#   OUTPUT_GCS_PREFIX    gs://bucket/path/output/run-001
#   MODEL_GCS            gs://bucket/path/models/qwen3-4b-merged
#
# Optional env vars (all have defaults that match vllm_infer.py):
#   MAX_NEW_TOKENS       (default 4096)
#   MAX_MODEL_LEN        (default: not set → vLLM auto-detects)
#   MAX_CHARS            (default 3000)
#   MIN_CHARS            (default 200)
#   MAX_PARAS            (default 15)
#   OVERLAP_PARAS        (default 1)
#   BATCH_SIZE           (default 500)
#   GCS_READ_CONCURRENCY (default 64)
#   TOP_K_CANDIDATES     (default: not set)
#   GPU_MEMORY_UTIL      (default 0.95)
#   TEMPERATURE          (default 0.0)
#   QUANTIZATION         (default: not set, e.g. fp8)
#   LEAN_OUTPUT          true/false — write only {id, predictions} (default true)
#   USE_GUIDED_DECODING  true/false — constrain output to the event schema (default false)
#   GUIDED_DECODING_BACKEND (default: vllm_infer.py's own default, xgrammar)
#   ONTOLOGY             (default: ontologies/zhai/bona.v4.json)
#   ONTOLOGY_DESCRIPTIONS  none|all (default none; 'all' for *-desc models)
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
LOCAL_OUTPUT=/local/shard-${TASK_INDEX}.jsonl
REMOTE_INPUT="${MANIFEST_GCS_PREFIX}/manifest-${SHARD}.jsonl"
REMOTE_OUTPUT="${OUTPUT_GCS_PREFIX}/shard-${TASK_INDEX}.jsonl"

# LOCAL_MODEL must exist as a directory before the multi-file `gsutil cp` below,
# otherwise gsutil treats it as a single-file destination and errors out.
mkdir -p "${LOCAL_MODEL}"

# --- 1. Pull model (parallel, ~8 GB, ~1-2 min) ---
echo "[entrypoint] downloading model from ${MODEL_GCS}"
gsutil -m cp -r "${MODEL_GCS}/*" "${LOCAL_MODEL}/"
echo "[entrypoint] model ready"

# --- 2. Pull this task's manifest shard only ---
# The manifest holds DB metadata only (id, gcs_path, publish_date, language); article
# bodies are streamed per-article from GCS during inference. So a task downloads ~1/N of
# a few GB here, never the whole corpus.
echo "[entrypoint] downloading manifest ${REMOTE_INPUT}"
gsutil cp "${REMOTE_INPUT}" "${LOCAL_INPUT}"
echo "[entrypoint] manifest ready ($(wc -l < ${LOCAL_INPUT}) rows)"

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
[[ -n "${MAX_MODEL_LEN:-}" ]]           && EXTRA_ARGS+=" --max_model_len ${MAX_MODEL_LEN}"
[[ -n "${TOP_K_CANDIDATES:-}" ]]        && EXTRA_ARGS+=" --top_k_candidates ${TOP_K_CANDIDATES}"
[[ -n "${QUANTIZATION:-}" ]]            && EXTRA_ARGS+=" --quantization ${QUANTIZATION}"
[[ -n "${RETRIEVER_MODEL_NAME:-}" ]]    && EXTRA_ARGS+=" --retriever_model_name ${RETRIEVER_MODEL_NAME}"
[[ -n "${RETRIEVER_INDEX:-}" ]]         && EXTRA_ARGS+=" --retriever_index ${RETRIEVER_INDEX}"
[[ -n "${RETRIEVER_GPU_MEM_UTIL:-}" ]]  && EXTRA_ARGS+=" --retriever_gpu_memory_utilization ${RETRIEVER_GPU_MEM_UTIL}"
[[ -n "${RETRIEVER_QUERY_MODE:-}" ]]    && EXTRA_ARGS+=" --retriever_query_mode ${RETRIEVER_QUERY_MODE}"
[[ -n "${RETRIEVER_MAX_MODEL_LEN:-}" ]] && EXTRA_ARGS+=" --retriever_max_model_len ${RETRIEVER_MAX_MODEL_LEN}"
[[ -n "${ONTOLOGY:-}" ]]                && EXTRA_ARGS+=" --ontology ${ONTOLOGY}"
[[ -n "${ONTOLOGY_DESCRIPTIONS:-}" ]]   && EXTRA_ARGS+=" --ontology_descriptions ${ONTOLOGY_DESCRIPTIONS}"
[[ -n "${GUIDED_DECODING_BACKEND:-}" ]] && EXTRA_ARGS+=" --guided_decoding_backend ${GUIDED_DECODING_BACKEND}"
[[ "${LEAN_OUTPUT:-true}" == "true" ]]  && EXTRA_ARGS+=" --lean_output"
# Constrains event_type to the ontology enum. Off by default: it costs throughput, but it is
# the only thing that stops the model inventing labels outside bona.v4.
[[ "${USE_GUIDED_DECODING:-false}" == "true" ]] && EXTRA_ARGS+=" --use_guided_decoding"

# --- 6. Run inference (num_shards=1: the manifest is already this task's shard) ---
echo "[entrypoint] starting inference"
python3 /app/scripts/event_extraction/inference/vllm_infer.py \
    --model_name_or_path "${LOCAL_MODEL}" \
    --input               "${LOCAL_INPUT}" \
    --output              "${LOCAL_OUTPUT}" \
    --gcs_input \
    --shard_index         0 \
    --num_shards          1 \
    --gcs_read_concurrency "${GCS_READ_CONCURRENCY:-64}" \
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
