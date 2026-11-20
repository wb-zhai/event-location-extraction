#!/bin/bash

# Qwen 3.5 0.8B
# PYTHONPATH=. python src/train/train_unsloth.py \
#   --model_name unsloth/Qwen3.5-0.8B \
#   --train_file dataset/maven-arg/processed/sft/train-window-120-events.jsonl \
#   --eval_file dataset/maven-arg/processed/sft/valid-window-120-events.jsonl \
#   --output_dir outputs/qwen3.5-0.8B-sft-window-120-events \
#   --ontology_file ontologies/maven-arg/ontology.json \
#   --num_event_candidates 50 \
#   --num_relation_candidates 50 \
#   --train_candidate_shuffle_prob 0.5 \
#   --train_gold_candidate_dropout_prob 0.05 \
#   --candidate_sampling_seed 13 \
#   --max_seq_length 2048 \
#   --batch_size 2 \
#   --grad_accum 8 \
#   --lr 2e-4 \
#   --epochs 3 \
#   --load_in_4bit \
#   --lora_r 16 \
#   --max_train_samples 1000 \
#   --filter_overlong_samples

# LFM2.5 350M
PYTHONPATH=. python src/train/train_unsloth.py \
  --model_name LiquidAI/LFM2.5-350M \
  --train_file dataset/maven-arg/processed/sft/train-window-120-events.jsonl \
  --eval_file dataset/maven-arg/processed/sft/valid-window-120-events.jsonl \
  --output_dir outputs/lfm2.5-350M-full-sft-window-120-events \
  --ontology_file ontologies/maven-arg/ontology.json \
  --num_event_candidates 50 \
  --num_relation_candidates 50 \
  --train_candidate_shuffle_prob 0.5 \
  --train_gold_candidate_dropout_prob 0.05 \
  --candidate_sampling_seed 13 \
  --max_seq_length 4096 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 2e-4 \
  --epochs 3 \
  --max_train_samples 5000 \
  --filter_overlong_samples \
  --train_on_responses_only
  # --load_in_4bit \
  # --lora_r 16 \