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
# PYTHONPATH=. python src/train/train_unsloth.py \
#   --model_name LiquidAI/LFM2.5-350M \
#   --train_file dataset/risk-factor/run-15052025/sft/train-windows-500.bona.v3_dropped.jsonl \
#   --eval_file dataset/risk-factor/run-15052025/sft/dev-windows-500.bona.v3_dropped.jsonl \
#   --output_dir outputs/zhai/lfm2.5-350M-full-sft-window-500-bonav3-full-prompt \
#   --ontology_file ontologies/zhai/ontology.json \
#   --num_event_candidates -1 \
#   --num_relation_candidates -1 \
#   --train_candidate_shuffle_prob 0.5 \
#   --train_gold_candidate_dropout_prob 0.05 \
#   --candidate_sampling_seed 13 \
#   --max_seq_length 8192 \
#   --batch_size 16 \
#   --grad_accum 2 \
#   --lr 2e-5 \
#   --epochs 10 \
#   --filter_overlong_samples \
#   --full_finetuning

# LFM2.5 350M
# PYTHONPATH=. python src/train/train_unsloth.py \
#   --model_name LiquidAI/LFM2.5-350M \
#   --train_file dataset/risk-factor/run-15052025/sft/train-windows-700.jsonl \
#   --eval_file dataset/risk-factor/run-15052025/sft/dev_200-windows-700.jsonl \
#   --output_dir outputs/zhai/lfm2.5-350M-full-sft-window-700 \
#   --ontology_file ontologies/risk-factors/risk.label.description.training.json \
#   --num_event_candidates -1 \
#   --num_relation_candidates -1 \
#   --train_candidate_shuffle_prob 0.5 \
#   --train_gold_candidate_dropout_prob 0.05 \
#   --candidate_sampling_seed 13 \
#   --max_seq_length 8192 \
#   --batch_size 4 \
#   --grad_accum 4 \
#   --lr 2e-5 \
#   --epochs 3 \
#   --filter_overlong_samples \
#   --train_on_responses_only \
#   --full_finetuning
  # --load_in_4bit \
  # --lora_r 16 \


# Recommended Qwen 3.5 0.8B recipe for the 500-window Zhai run.
# Changes vs the previous preset:
# - response-only masking to avoid learning to reproduce prompt scaffolding
# - ontology descriptions to better separate semantically similar labels
# - distinct output dir so the baseline run is preserved
# PYTHONPATH=. python src/train/train_unsloth.py \
#   --model_name unsloth/Qwen3.5-0.8B \
#   --train_file dataset/risk-factor/run-15052025/sft/train-windows-500.bona.v3.qwen3.5_0.8.jsonl \
#   --eval_file dataset/risk-factor/run-15052025/sft/dev-windows-500.bona.v3.qwen3.5_0.8.jsonl \
#   --output_dir outputs/zhai/qwen3.5-0.8B-lora-sft-window-500-bonav3-response-only-described \
#   --ontology_file ontologies/zhai/ontology.json \
#   --description \
#   --num_event_candidates -1 \
#   --num_relation_candidates -1 \
#   --train_candidate_shuffle_prob 0.5 \
#   --train_gold_candidate_dropout_prob 0.0 \
#   --candidate_sampling_seed 13 \
#   --max_seq_length 8192 \
#   --batch_size 4 \
#   --grad_accum 8 \
#   --lr 5e-5 \
#   --epochs 5 \
#   --filter_overlong_samples \
#   --load_in_4bit \
#   --lora_r 32 \
#   --train_on_responses_only

  # PYTHONPATH=. python src/train/train_unsloth.py \
  # --model_name unsloth/Qwen3.5-4B \
  # --train_file dataset/risk-factor/run-15052025/sft/train-windows-500.bona.v3.qwen3.5_4B.jsonl \
  # --eval_file dataset/risk-factor/run-15052025/sft/dev-windows-500.bona.v3.qwen3.5_4B.jsonl \
  # --output_dir outputs/zhai/qwen3.5-4B-lora-sft-window-500-bonav3-response-only-described \
  # --ontology_file ontologies/zhai/ontology.json \
  # --description \
  # --num_event_candidates -1 \
  # --num_relation_candidates -1 \
  # --train_candidate_shuffle_prob 0.5 \
  # --train_gold_candidate_dropout_prob 0.0 \
  # --candidate_sampling_seed 13 \
  # --max_seq_length 8192 \
  # --batch_size 32 \
  # --grad_accum 1 \
  # --lr 5e-5 \
  # --epochs 3 \
  # --filter_overlong_samples \
  # --load_in_8bit \
  # --lora_r 32 \
  # --train_on_responses_only

# PYTHONPATH=. python src/train/train_unsloth.py \
#   --model_name unsloth/Qwen3.5-0.8B \
#   --train_file dataset/risk-factor/run-15052025/sft/train.v4.sft.context.events.384.jsonl \
#   --eval_file dataset/risk-factor/run-15052025/sft/dev.v4.sft.context.events.384.jsonl \
#   --output_dir outputs/zhai/qwen3.5-0.8B-lora-sft-window-500-v4-response-events-only-omit_offsets-described \
#   --ontology_file ontologies/zhai/ontology.json \
#   --description \
#   --num_event_candidates -1 \
#   --num_relation_candidates -1 \
#   --train_candidate_shuffle_prob 1.0 \
#   --train_gold_candidate_dropout_prob 0.0 \
#   --candidate_sampling_seed 13 \
#   --max_seq_length 8192 \
#   --batch_size 4 \
#   --grad_accum 8 \
#   --lr 5e-5 \
#   --epochs 5 \
#   --filter_overlong_samples \
#   --load_in_4bit \
#   --lora_r 32 \
#   --train_on_responses_only \
#   --max_empty_event_ratio 0.2 \
#   --events_only \
#   --omit_offsets

# PYTHONPATH=. python src/train/train_unsloth.py \
#   --model_name unsloth/Qwen3.5-4B \
#   --train_file dataset/risk-factor/run-15052025/sft/train.v4.sft.context.events.384.jsonl \
#   --eval_file dataset/risk-factor/run-15052025/sft/dev.v4.sft.context.events.384.jsonl \
#   --output_dir outputs/zhai/qwen3.5-4B-lora-sft-window-500-v4-response-events-only-omit_offsets-v3 \
#   --ontology_file ontologies/zhai/ontology.json \
#   --num_event_candidates -1 \
#   --num_relation_candidates -1 \
#   --train_candidate_shuffle_prob 1.0 \
#   --train_gold_candidate_dropout_prob 0.0 \
#   --candidate_sampling_seed 13 \
#   --max_seq_length 8192 \
#   --batch_size 4 \
#   --grad_accum 8 \
#   --lr 2e-4 \
#   --epochs 3 \
#   --filter_overlong_samples \
#   --load_in_4bit \
#   --lora_r 32 \
#   --train_on_responses_only \
#   --max_empty_event_ratio 0.2 \
#   --events_only \
#   --omit_offsets

PYTHONPATH=. python src/train/train_unsloth.py \
  --model_name unsloth/Qwen3.5-4B \
  --train_file dataset/risk-factor/run-15052025/sft/train.v4.sft.context.384.jsonl \
  --eval_file dataset/risk-factor/run-15052025/sft/dev.v4.sft.context.384.jsonl \
  --output_dir outputs/zhai/qwen3.5-4B-lora-sft-window-500-v4-response-omit_offsets \
  --ontology_file ontologies/zhai/ontology.json \
  --num_event_candidates -1 \
  --num_relation_candidates -1 \
  --train_candidate_shuffle_prob 1.0 \
  --train_gold_candidate_dropout_prob 0.0 \
  --candidate_sampling_seed 13 \
  --max_seq_length 8192 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 2e-4 \
  --epochs 3 \
  --filter_overlong_samples \
  --load_in_4bit \
  --lora_r 32 \
  --train_on_responses_only \
  --max_empty_event_ratio 0.2 \
  --omit_offsets

PYTHONPATH=. python src/train/train_unsloth.py \
  --model_name unsloth/Qwen3.5-0.8B \
  --train_file dataset/risk-factor/run-15052025/sft/train.v4.sft.context.events.384.jsonl \
  --eval_file dataset/risk-factor/run-15052025/sft/dev.v4.sft.context.events.384.jsonl \
  --output_dir outputs/zhai/qwen3.5-0.8B-lora-sft-window-500-v4-response-events-only-omit_offsets \
  --ontology_file ontologies/zhai/ontology.json \
  --num_event_candidates -1 \
  --num_relation_candidates -1 \
  --train_candidate_shuffle_prob 1.0 \
  --train_gold_candidate_dropout_prob 0.0 \
  --candidate_sampling_seed 13 \
  --max_seq_length 8192 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 2e-4 \
  --epochs 3 \
  --filter_overlong_samples \
  --load_in_4bit \
  --lora_r 32 \
  --train_on_responses_only \
  --max_empty_event_ratio 0.2 \
  --events_only \
  --omit_offsets

PYTHONPATH=. python src/train/train_unsloth.py \
  --model_name unsloth/Qwen3.5-7B \
  --train_file dataset/risk-factor/run-15052025/sft/train.v4.sft.context.384.jsonl \
  --eval_file dataset/risk-factor/run-15052025/sft/dev.v4.sft.context.384.jsonl \
  --output_dir outputs/zhai/qwen3.5-7B-lora-sft-window-500-v4-response-omit_offsets \
  --ontology_file ontologies/zhai/ontology.json \
  --num_event_candidates -1 \
  --num_relation_candidates -1 \
  --train_candidate_shuffle_prob 1.0 \
  --train_gold_candidate_dropout_prob 0.0 \
  --candidate_sampling_seed 13 \
  --max_seq_length 8192 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 2e-4 \
  --epochs 3 \
  --filter_overlong_samples \
  --load_in_4bit \
  --lora_r 32 \
  --train_on_responses_only \
  --max_empty_event_ratio 0.2 \
  --omit_offsets
