# Event-Location Extraction Benchmark Report

**Date**: 2026-03-29 12:21
**Match mode**: semantic
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B-Claude-Distill-v1 | 307 | 183 | 88 | 0.626 | 0.777 | **0.694** | 0.652 | 0.742 | 0.531 | 10.25s |
| Qwen3.5-27B-4bit | 357 | 286 | 38 | 0.555 | 0.904 | **0.688** | 0.602 | 0.803 | 0.524 | 26.91s |
| Qwen3.5-35B-A3B-4bit | 294 | 226 | 101 | 0.565 | 0.744 | **0.643** | 0.594 | 0.700 | 0.473 | 5.28s |
| Qwen3.5-9B-4bit | 289 | 250 | 106 | 0.536 | 0.732 | **0.619** | 0.566 | 0.682 | 0.448 | 10.80s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-4bit | 34 | 87 | 23 | 0.281 | 0.597 | **0.382** | 0.314 | 0.487 | 0.236 | 5.28s |
| Qwen3.5-27B-4bit | 43 | 149 | 14 | 0.224 | 0.754 | **0.345** | 0.261 | 0.512 | 0.209 | 26.91s |
| Qwen3.5-4B-Claude-Distill-v1 | 31 | 92 | 26 | 0.252 | 0.544 | **0.344** | 0.282 | 0.442 | 0.208 | 10.25s |
| Qwen3.5-9B-4bit | 37 | 146 | 20 | 0.202 | 0.649 | **0.308** | 0.234 | 0.450 | 0.182 | 10.80s |

## Summary

- **Best entity extraction**: Qwen3.5-4B-Claude-Distill-v1 (F1=0.694)
- **Best event-location linking**: Qwen3.5-35B-A3B-4bit (F1=0.382)
- **Fastest inference**: Qwen3.5-35B-A3B-4bit (5.28s/article)
- **Best efficiency** (F1/log2(time)): Qwen3.5-35B-A3B-4bit (score=0.1334)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| Qwen3.5-35B-A3B-4bit | 158.2s | 5.28s | 11.4 |
| Qwen3.5-4B-Claude-Distill-v1 | 307.4s | 10.25s | 5.9 |
| Qwen3.5-9B-4bit | 324.1s | 10.80s | 5.6 |
| Qwen3.5-27B-4bit | 807.4s | 26.91s | 2.2 |