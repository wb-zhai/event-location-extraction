#!/bin/bash

# Qwen 3.5 0.8B
# PYTHONPATH=. python src/train/train_unsloth.py \
#   --model_name unsloth/Qwen3.5-0.8B \
#   --train_file dataset/risk-factor/run-15052025/sft/train-windows-1500.jsonl \
#   --eval_file dataset/risk-factor/run-15052025/sft/dev_200-windows-1500.jsonl \
#   --output_dir outputs/zhai/qwen3.5-0.8B-full-sft-window-1500 \
#   --ontology_file ontologies/risk-factors/risk.label.description.training.json \
#   --num_event_candidates -1 \
#   --num_relation_candidates -1 \
#   --train_candidate_shuffle_prob 0.5 \
#   --train_gold_candidate_dropout_prob 0.05 \
#   --candidate_sampling_seed 13 \
#   --max_seq_length 8192 \
#   --batch_size 4 \
#   --grad_accum 8 \
#   --lr 2e-5 \
#   --epochs 3 \
#   --filter_overlong_samples
  # --load_in_4bit \
  # --lora_r 16 \

  # PYTHONPATH=. python src/train/train_unsloth.py \
  # --model_name unsloth/Qwen3.5-0.8B \
  # --train_file dataset/risk-factor/run-15052025/sft/train-windows-1500.jsonl \
  # --eval_file dataset/risk-factor/run-15052025/sft/dev_200-windows-1500.jsonl \
  # --output_dir outputs/zhai/qwen3.5-0.8B-lora-sft-window-1500 \
  # --ontology_file ontologies/risk-factors/risk.label.description.training.json \
  # --num_event_candidates -1 \
  # --num_relation_candidates -1 \
  # --train_candidate_shuffle_prob 0.5 \
  # --train_gold_candidate_dropout_prob 0.05 \
  # --candidate_sampling_seed 13 \
  # --max_seq_length 8192 \
  # --batch_size 4 \
  # --grad_accum 8 \
  # --lr 2e-5 \
  # --epochs 3 \
  # --load_in_4bit \
  # --lora_r 16 \
  # --train_on_responses_only

  # PYTHONPATH=. python src/train/train_unsloth.py \
  # --model_name unsloth/Qwen3.5-0.8B \
  # --train_file dataset/risk-factor/run-15052025/sft/train-windows-1500.jsonl \
  # --eval_file dataset/risk-factor/run-15052025/sft/dev_200-windows-1500.jsonl \
  # --output_dir outputs/zhai/qwen3.5-0.8B-lora-sft-window-1500-fullprompt \
  # --ontology_file ontologies/risk-factors/risk.label.description.training.json \
  # --num_event_candidates -1 \
  # --num_relation_candidates -1 \
  # --train_candidate_shuffle_prob 0.5 \
  # --train_gold_candidate_dropout_prob 0.05 \
  # --candidate_sampling_seed 13 \
  # --max_seq_length 8192 \
  # --batch_size 4 \
  # --grad_accum 8 \
  # --lr 2e-5 \
  # --epochs 3 \
  # --load_in_4bit \
  # --lora_r 16


# LFM2.5 350M
PYTHONPATH=. python src/train/train_unsloth.py \
  --model_name LiquidAI/LFM2.5-350M \
  --train_file dataset/risk-factor/run-15052025/sft/train-windows-1500.jsonl \
  --eval_file dataset/risk-factor/run-15052025/sft/dev_200-windows-1500.jsonl \
  --output_dir outputs/zhai/lfm2.5-350M-full-sft-window-1500-v2 \
  --ontology_file ontologies/risk-factors/risk.label.description.training.json \
  --num_event_candidates -1 \
  --num_relation_candidates -1 \
  --train_candidate_shuffle_prob 0.5 \
  --train_gold_candidate_dropout_prob 0.05 \
  --candidate_sampling_seed 13 \
  --max_seq_length 8192 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 2e-5 \
  --epochs 3 \
  --filter_overlong_samples \
  --train_on_responses_only
  # --load_in_4bit \
  # --lora_r 16 \

# LFM2.5 350M
PYTHONPATH=. python src/train/train_unsloth.py \
  --model_name LiquidAI/LFM2.5-350M \
  --train_file dataset/risk-factor/run-15052025/sft/train-windows-1500.jsonl \
  --eval_file dataset/risk-factor/run-15052025/sft/dev_200-windows-1500.jsonl \
  --output_dir outputs/zhai/lfm2.5-350M-full-sft-window-1500-full-prompt \
  --ontology_file ontologies/risk-factors/risk.label.description.training.json \
  --num_event_candidates -1 \
  --num_relation_candidates -1 \
  --train_candidate_shuffle_prob 0.5 \
  --train_gold_candidate_dropout_prob 0.05 \
  --candidate_sampling_seed 13 \
  --max_seq_length 8192 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 2e-5 \
  --epochs 3 \
  --filter_overlong_samples