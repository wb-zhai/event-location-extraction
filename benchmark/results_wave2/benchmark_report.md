# Event-Location Extraction Benchmark Report

**Date**: 2026-03-28 21:50
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Phi4-14B | 259 | 195 | 136 | 0.571 | 0.656 | **0.610** | 0.586 | 0.637 | 0.439 | 11.23s |
| Gemma3-12B | 263 | 211 | 132 | 0.555 | 0.666 | **0.605** | 0.574 | 0.640 | 0.434 | 20.65s |
| Qwen3-14B | 263 | 223 | 132 | 0.541 | 0.666 | **0.597** | 0.562 | 0.636 | 0.426 | 10.82s |
| Phi4-Mini-3.8B | 110 | 128 | 285 | 0.462 | 0.279 | **0.348** | 0.408 | 0.302 | 0.210 | 16.37s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma3-12B | 23 | 99 | 34 | 0.189 | 0.404 | **0.257** | 0.211 | 0.329 | 0.147 | 20.65s |
| Phi4-14B | 21 | 92 | 36 | 0.186 | 0.368 | **0.247** | 0.206 | 0.308 | 0.141 | 11.23s |
| Qwen3-14B | 15 | 61 | 42 | 0.197 | 0.263 | **0.226** | 0.208 | 0.247 | 0.127 | 10.82s |
| Phi4-Mini-3.8B | 11 | 91 | 46 | 0.108 | 0.193 | **0.138** | 0.118 | 0.167 | 0.074 | 16.37s |

## Summary

- **Best entity extraction**: Phi4-14B (F1=0.610)
- **Best event-location linking**: Gemma3-12B (F1=0.257)
- **Fastest inference**: Qwen3-14B (10.82s/article)
- **Best efficiency** (F1/log2(time)): Phi4-14B (score=0.0663)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| Qwen3-14B | 324.5s | 10.82s | 5.5 |
| Phi4-14B | 337.0s | 11.23s | 5.3 |
| Phi4-Mini-3.8B | 491.1s | 16.37s | 3.7 |
| Gemma3-12B | 619.5s | 20.65s | 2.9 |