#!/bin/bash

python scripts/vllm_infer.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset risk_random_no_desc_700 \
    --template qwen3_nothink \
    --cutoff_len 8192 \
    --save_name predictions/qwen3-4b/risk_random_no_desc_700_inference_results_temp0.json \
    --temperature 0 \
    --max_new_tokens 4096 \
    --enable_thinking False \
    --batch_size 4096

#saves/Qwen3-4B-Instruct-2507/lora/train_2026-06-04-11-40-no_desc_700_dropped/merged \