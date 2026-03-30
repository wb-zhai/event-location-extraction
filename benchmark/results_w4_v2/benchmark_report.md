# Event-Location Extraction Benchmark Report

**Date**: 2026-03-29 11:29
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B-Claude-Distill-v2 | 302 | 255 | 93 | 0.542 | 0.765 | **0.634** | 0.576 | 0.707 | 0.465 | 8.02s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B-Claude-Distill-v2 | 26 | 157 | 31 | 0.142 | 0.456 | **0.217** | 0.165 | 0.316 | 0.121 | 8.02s |

## Summary

- **Best entity extraction**: Qwen3.5-4B-Claude-Distill-v2 (F1=0.634)
- **Best event-location linking**: Qwen3.5-4B-Claude-Distill-v2 (F1=0.217)
- **Fastest inference**: Qwen3.5-4B-Claude-Distill-v2 (8.02s/article)
- **Best efficiency** (F1/log2(time)): Qwen3.5-4B-Claude-Distill-v2 (score=0.0652)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| Qwen3.5-4B-Claude-Distill-v2 | 240.6s | 8.02s | 7.5 |