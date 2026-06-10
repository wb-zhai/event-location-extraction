#!/bin/bash

# llamafactory-cli train \
#     --stage sft \
#     --do_train True \
#     --do_eval True \
#     --model_name_or_path Qwen/Qwen3.5-4B \
#     --template qwen3_5_nothink \
#     --finetuning_type lora \
#     --flash_attn auto \
#     --dataset_dir data \
#     --dataset risk_train_no_desc_700_dropped \
#     --eval_dataset risk_dev_no_desc_700_dropped \
#     --cutoff_len 8192 \
#     --learning_rate 0.0001 \
#     --num_train_epochs 3.0 \
#     --max_samples 1000000 \
#     --preprocessing_num_workers 8 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --weight_decay 0.01 \
#     --logging_steps 5 \
#     --save_steps 200 \
#     --eval_strategy steps \
#     --eval_steps 200 \
#     --per_device_eval_batch_size 4 \
#     --predict_with_generate True \
#     --compute_event_metrics True \
#     --do_sample False \
#     --temperature 0 \
#     --max_new_tokens 4096 \
#     --warmup_ratio 0.1 \
#     --packing False \
#     --enable_thinking False \
#     --report_to wandb \
#     --output_dir saves/Qwen3.5-4B/lora/train_2026-06-05-11-10-no_desc_700_dropped \
#     --bf16 True \
#     --plot_loss True \
#     --trust_remote_code True \
#     --ddp_timeout 180000000 \
#     --include_num_input_tokens_seen True \
#     --optim adamw_torch \
#     --lora_rank 16 \
#     --lora_alpha 32 \
#     --lora_dropout 0.05
# #     #--lora_target "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --do_eval True \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --template qwen3_nothink \
    --finetuning_type lora \
    --flash_attn auto \
    --dataset_dir data \
    --dataset risk_v2_train_no_desc_700_args_dropped \
    --eval_dataset risk_v2_dev_no_desc_700_args_dropped \
    --cutoff_len 16384 \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --max_samples 1000000 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --logging_steps 5 \
    --save_steps 200 \
    --eval_strategy steps \
    --eval_steps 200 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate False \
    --compute_event_metrics False \
    --do_sample False \
    --temperature 0 \
    --max_new_tokens 4096 \
    --warmup_ratio 0.1 \
    --packing False \
    --enable_thinking False \
    --report_to wandb \
    --output_dir saves/Qwen3-4B-Instruct-2507/lora/train_2026-06-09-11-45-no_desc_700_args_dropped \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
#     #--lora_target "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"