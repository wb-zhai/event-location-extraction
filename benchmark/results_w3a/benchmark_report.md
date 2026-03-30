# Event-Location Extraction Benchmark Report

**Date**: 2026-03-29 05:38
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Jan-v3-4B | 251 | 245 | 144 | 0.506 | 0.635 | **0.563** | 0.527 | 0.605 | 0.392 | 8.02s |
| TinyAya-3.35B | 149 | 124 | 246 | 0.546 | 0.377 | **0.446** | 0.501 | 0.402 | 0.287 | 5.25s |
| Nanbeige4.1-3B | 0 | 1 | 395 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 38.65s |
| Falcon-H1R-7B | 0 | 0 | 395 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 33.06s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Jan-v3-4B | 18 | 122 | 39 | 0.129 | 0.316 | **0.183** | 0.146 | 0.245 | 0.101 | 8.02s |
| Nanbeige4.1-3B | 0 | 1 | 57 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 38.65s |
| TinyAya-3.35B | 0 | 0 | 57 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 5.25s |
| Falcon-H1R-7B | 0 | 0 | 57 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 33.06s |

## Summary

- **Best entity extraction**: Jan-v3-4B (F1=0.563)
- **Best event-location linking**: Jan-v3-4B (F1=0.183)
- **Fastest inference**: TinyAya-3.35B (5.25s/article)
- **Best efficiency** (F1/log2(time)): Jan-v3-4B (score=0.0550)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| TinyAya-3.35B | 157.4s | 5.25s | 11.4 |
| Jan-v3-4B | 240.4s | 8.02s | 7.5 |
| Falcon-H1R-7B | 991.8s | 33.06s | 1.8 |
| Nanbeige4.1-3B | 1159.6s | 38.65s | 1.6 |