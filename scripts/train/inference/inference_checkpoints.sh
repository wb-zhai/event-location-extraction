#!/bin/bash

CHECKPOINT_FOLDER="models/new_schema/qwen3_5-lora-a100-40gb-20260617-151840"

echo "Running inference for all checkpoints in folder: $CHECKPOINT_FOLDER"
# list all checkpoint directories in the folder before the loop
echo "Found the following checkpoints:"
ls -d "$CHECKPOINT_FOLDER"/checkpoint-*

# let's get the base name of the model from the adapter_config.json file in the first checkpoint directory
first_checkpoint=$(ls -d "$CHECKPOINT_FOLDER"/checkpoint-* | head -n 1)
if [ -d "$first_checkpoint" ]; then
    adapter_config_file="$first_checkpoint/adapter_config.json"
    if [ -f "$adapter_config_file" ]; then
        model_name=$(jq -r '.base_model_name_or_path' "$adapter_config_file")
        echo "Base model name extracted from adapter_config.json: $model_name"
    else
        echo "adapter_config.json not found in $first_checkpoint. Exiting."
        exit 1
    fi
else
    echo "No checkpoint directories found in $CHECKPOINT_FOLDER. Exiting."
    exit 1
fi

for checkpoint_path in "$CHECKPOINT_FOLDER"/checkpoint-*; do
    if [ -d "$checkpoint_path" ]; then
        echo "Running inference for checkpoint: $checkpoint_path"
        python scripts/train/inference/vllm_infer.py \
            --model_name_or_path "$model_name" \
            --adapter_name_or_path "$checkpoint_path" \
            --input dataset/zhai/v3/science/dev.jsonl \
            --output dataset/risk-factor/run-15062025/predictions/qwen3_5-lora-a100-40gb-20260617-151840/$(basename "$checkpoint_path").dev.jsonl \
            --temperature 0 \
            --top_k_candidates 70 \
            --max_model_len 8192 \
            --max_chars 3000 \
            --max_new_tokens 2048
    fi
done
