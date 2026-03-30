# Event-Location Extraction Benchmark Report

**Date**: 2026-03-28 11:01
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B-4bit | 144 | 121 | 251 | 0.543 | 0.365 | **0.436** | 0.495 | 0.390 | 0.279 | 5.71s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B-4bit | 8 | 58 | 49 | 0.121 | 0.140 | **0.130** | 0.125 | 0.136 | 0.070 | 5.71s |

## Summary

- **Best entity extraction**: Qwen3.5-2B-4bit (F1=0.436)
- **Best event-location linking**: Qwen3.5-2B-4bit (F1=0.130)
- **Fastest inference**: Qwen3.5-2B-4bit (5.71s/article)
- **Best efficiency** (F1/log2(time)): Qwen3.5-2B-4bit (score=0.0442)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| Qwen3.5-2B-4bit | 171.3s | 5.71s | 10.5 |