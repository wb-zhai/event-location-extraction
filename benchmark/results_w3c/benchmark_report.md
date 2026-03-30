# Event-Location Extraction Benchmark Report

**Date**: 2026-03-29 06:05
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-80B-A3B | 249 | 142 | 146 | 0.637 | 0.630 | **0.634** | 0.635 | 0.632 | 0.464 | 18.09s |
| JoyAI-Flash-48B-A3B | 255 | 210 | 140 | 0.548 | 0.646 | **0.593** | 0.565 | 0.624 | 0.421 | 10.92s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| JoyAI-Flash-48B-A3B | 30 | 140 | 27 | 0.176 | 0.526 | **0.264** | 0.203 | 0.377 | 0.152 | 10.92s |
| Qwen3-Coder-80B-A3B | 15 | 77 | 42 | 0.163 | 0.263 | **0.201** | 0.176 | 0.234 | 0.112 | 18.09s |

## Summary

- **Best entity extraction**: Qwen3-Coder-80B-A3B (F1=0.634)
- **Best event-location linking**: JoyAI-Flash-48B-A3B (F1=0.264)
- **Fastest inference**: JoyAI-Flash-48B-A3B (10.92s/article)
- **Best efficiency** (F1/log2(time)): JoyAI-Flash-48B-A3B (score=0.0716)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| JoyAI-Flash-48B-A3B | 327.6s | 10.92s | 5.5 |
| Qwen3-Coder-80B-A3B | 542.8s | 18.09s | 3.3 |