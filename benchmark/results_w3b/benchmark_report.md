# Event-Location Extraction Benchmark Report

**Date**: 2026-03-29 06:04
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-122B-A10B | 334 | 251 | 61 | 0.571 | 0.846 | **0.682** | 0.611 | 0.771 | 0.517 | 16.89s |
| Ministral3-14B | 304 | 496 | 91 | 0.380 | 0.770 | **0.509** | 0.423 | 0.639 | 0.341 | 23.50s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-122B-A10B | 28 | 89 | 29 | 0.239 | 0.491 | **0.322** | 0.267 | 0.406 | 0.192 | 16.89s |
| Ministral3-14B | 28 | 176 | 29 | 0.137 | 0.491 | **0.215** | 0.160 | 0.324 | 0.120 | 23.50s |

## Summary

- **Best entity extraction**: Qwen3.5-122B-A10B (F1=0.682)
- **Best event-location linking**: Qwen3.5-122B-A10B (F1=0.322)
- **Fastest inference**: Qwen3.5-122B-A10B (16.89s/article)
- **Best efficiency** (F1/log2(time)): Qwen3.5-122B-A10B (score=0.0759)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| Qwen3.5-122B-A10B | 506.5s | 16.89s | 3.6 |
| Ministral3-14B | 704.9s | 23.50s | 2.6 |