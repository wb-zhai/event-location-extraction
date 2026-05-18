#!/bin/bash

PYTHONPATH=. python src/train/train.py \
  --model_name "answerdotai/ModernBERT-base" \
  --train_file dataset/risk-factor/run-15052025/reader/train-windows-256-modernbert.jsonl \
  --eval_file dataset/risk-factor/run-15052025/reader/dev_200-windows-256-modernbert.jsonl \
  --output_dir outputs/zhai/reader/modernbert-only-events-run-1 \
  --ontology_file ontologies/risk-factors/risk.label.description.training.json \
  --num_event_candidates -1 \
  --num_relation_candidates -1 \
  --train_candidate_shuffle_prob 0.5 \
  --candidate_sampling_seed 13 \
  --max_length 8192 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --num_epochs 10 \
  --only-event
#   --train_gold_candidate_dropout_prob 0.05 \