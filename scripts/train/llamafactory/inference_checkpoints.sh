#!/bin/bash

CHECKPOINT_FOLDER="saves/Qwen3-4B-Instruct-2507/lora/train_2026-06-09-11-00-no_desc_700_args_dropped"

echo "Running inference for all checkpoints in folder: $CHECKPOINT_FOLDER"
# list all checkpoint directories in the folder before the loop
echo "Found the following checkpoints:"
ls -d "$CHECKPOINT_FOLDER"/checkpoint-*

for checkpoint_path in "$CHECKPOINT_FOLDER"/checkpoint-*; do
    if [ -d "$checkpoint_path" ]; then
        echo "Running inference for checkpoint: $checkpoint_path"
        python scripts/vllm_infer.py \
            --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
            --adapter_name_or_path "$checkpoint_path" \
            --dataset risk_v2_dev_no_desc_700_args_dropped \
            --template qwen3_nothink \
            --cutoff_len 16134 \
            --save_name "predictions/qwen3-4b_no_desc_700_args_dropped/risk_v2_dev_no_desc_700_args_dropped_inference_results_$(basename "$checkpoint_path").json" \
            --temperature 0 \
            --max_new_tokens 8192 \
            --enable_thinking False \
            --batch_size 4096
            # --top_p 0.95 \
            # --top_k 20 \
            # --min_p 0.0 \
            # --presence_penalty 1.5 \
            # --repetition_penalty 1.0
    fi
done

# python scripts/vllm_infer.py \
#     --model_name_or_path Qwen/Qwen3.5-4B \
#     --dataset risk_dev_no_desc_700 \
#     --template qwen3_5_nothink \
#     --cutoff_len 8192 \
#     --save_name predictions/qwen3_5-4b/risk_dev_no_desc_700_inference_results_no_thinking.json \
#     --temperature 1.0 \
#     --max_new_tokens 8192 \
#     --enable_thinking False \
#     --batch_size 4096 \
#     --top_p 0.95 \
#     --top_k 20 \
#     --min_p 0.0 \
#     --presence_penalty 1.5 \
#     --repetition_penalty 1.0

#saves/Qwen3-4B-Instruct-2507/lora/train_2026-06-04-11-40-no_desc_700_dropped/merged \

# qwen 3.5
# Instruct (or non-thinking) mode for reasoning tasks: 
# temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Thinking mode for general tasks: 
# temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0