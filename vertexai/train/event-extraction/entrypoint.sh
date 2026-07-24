#!/bin/bash
set -euo pipefail

# Required env vars (passed by Vertex AI job)
: "${GCS_BUCKET:?GCS_BUCKET must be set}"
: "${GCS_DATA_PREFIX:?GCS_DATA_PREFIX must be set}"
: "${GCS_SAVE_PREFIX:?GCS_SAVE_PREFIX must be set}"
: "${GCS_TRAIN_CONFIG:?GCS_TRAIN_CONFIG must be set}"
: "${JOB_NAME:?JOB_NAME must be set}"

SAVE_DIR="/app/saves/${JOB_NAME}"
GCS_SAVE_PATH="gs://${GCS_BUCKET}/${GCS_SAVE_PREFIX}/${JOB_NAME}"
GCS_TRAIN_CONFIG_URI="gs://${GCS_BUCKET}/${GCS_TRAIN_CONFIG}"
TRAIN_CONFIG="/tmp/train_config.yaml"

# Export HF token if provided
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

echo "=== Downloading data from GCS ==="
gsutil -m rsync -r "gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}" /app/data/

echo "=== Downloading train config from GCS ==="
gsutil cp "${GCS_TRAIN_CONFIG_URI}" "${TRAIN_CONFIG}"

echo "=== Starting background checkpoint sync (every 10 min) ==="
(
    while true; do
        sleep 600
        echo "[sync] Syncing saves to GCS..."
        gsutil -m rsync -r "${SAVE_DIR}" "${GCS_SAVE_PATH}" 2>/dev/null || true
    done
) &
SYNC_PID=$!

# Final sync on exit (success or failure)
cleanup() {
    echo "=== Final sync to GCS ==="
    kill "${SYNC_PID}" 2>/dev/null || true
    gsutil -m rsync -r "${SAVE_DIR}" "${GCS_SAVE_PATH}" || true
    echo "=== Done. Output at ${GCS_SAVE_PATH} ==="
}
trap cleanup EXIT

# MAX_STEPS overrides epoch-based training for cheap debug runs (e.g. MAX_STEPS=10)
MAX_STEPS_ARG=""
if [[ -n "${MAX_STEPS:-}" && "${MAX_STEPS}" != "0" ]]; then
    MAX_STEPS_ARG="max_steps=${MAX_STEPS}"
    echo "  [debug] max_steps=${MAX_STEPS}"
fi

echo "=== Starting training: ${JOB_NAME} ==="
# dataset_dir and output_dir are dynamic; everything else comes from the YAML config.
# With a YAML config, LlamaFactory merges overrides via OmegaConf, which expects
# key=value syntax (not argparse --key value). CLI overrides win over YAML values,
# so MAX_STEPS_ARG safely overrides num_train_epochs.
llamafactory-cli train "${TRAIN_CONFIG}" \
    dataset_dir=/app/data \
    output_dir="${SAVE_DIR}" \
    ${MAX_STEPS_ARG}

FINETUNING_TYPE=$(python3 -c "import yaml; print(yaml.safe_load(open('${TRAIN_CONFIG}')).get('finetuning_type', ''))")
if [[ "${FINETUNING_TYPE}" == "lora" ]]; then
    echo "=== Merging LoRA adapter into ${SAVE_DIR}/merged ==="
    MERGE_CONFIG="/tmp/merge_config.yaml"
    python3 <<PYEOF
import yaml
cfg = yaml.safe_load(open("${TRAIN_CONFIG}"))
merge_cfg = {
    "model_name_or_path": cfg["model_name_or_path"],
    "adapter_name_or_path": "${SAVE_DIR}",
    "template": cfg["template"],
    "export_dir": "${SAVE_DIR}/merged",
    "export_size": 5,
    "export_device": "cpu",
    "export_legacy_format": False,
}
if "trust_remote_code" in cfg:
    merge_cfg["trust_remote_code"] = cfg["trust_remote_code"]
with open("${MERGE_CONFIG}", "w") as f:
    yaml.dump(merge_cfg, f)
PYEOF
    llamafactory-cli export "${MERGE_CONFIG}"
fi
