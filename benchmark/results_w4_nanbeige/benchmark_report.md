# Event-Location Extraction Benchmark Report

**Date**: 2026-03-29 06:24
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Nanbeige4.1-3B-retry | 0 | 0 | 395 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 59.05s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Nanbeige4.1-3B-retry | 0 | 0 | 57 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 59.05s |

## Summary

- **Best entity extraction**: Nanbeige4.1-3B-retry (F1=0.000)
- **Best event-location linking**: Nanbeige4.1-3B-retry (F1=0.000)
- **Fastest inference**: Nanbeige4.1-3B-retry (59.05s/article)
- **Best efficiency** (F1/log2(time)): Nanbeige4.1-3B-retry (score=0.0000)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| Nanbeige4.1-3B-retry | 1771.4s | 59.05s | 1.0 |