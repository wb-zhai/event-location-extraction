# Event-Location Extraction Benchmark Report

**Date**: 2026-03-29 06:12
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B-Claude-Distill-v1 | 304 | 186 | 91 | 0.620 | 0.770 | **0.687** | 0.645 | 0.734 | 0.523 | 19.35s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B-Claude-Distill-v1 | 28 | 95 | 29 | 0.228 | 0.491 | **0.311** | 0.255 | 0.399 | 0.184 | 19.35s |

## Summary

- **Best entity extraction**: Qwen3.5-4B-Claude-Distill-v1 (F1=0.687)
- **Best event-location linking**: Qwen3.5-4B-Claude-Distill-v1 (F1=0.311)
- **Fastest inference**: Qwen3.5-4B-Claude-Distill-v1 (19.35s/article)
- **Best efficiency** (F1/log2(time)): Qwen3.5-4B-Claude-Distill-v1 (score=0.0704)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| Qwen3.5-4B-Claude-Distill-v1 | 580.5s | 19.35s | 3.1 |