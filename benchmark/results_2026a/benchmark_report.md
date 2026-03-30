# Event-Location Extraction Benchmark Report

**Date**: 2026-03-28 22:18
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| LFM2.5-1.2B-Think | 0 | 0 | 395 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 9.14s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| LFM2.5-1.2B-Think | 0 | 0 | 57 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 9.14s |

## Summary

- **Best entity extraction**: LFM2.5-1.2B-Think (F1=0.000)
- **Best event-location linking**: LFM2.5-1.2B-Think (F1=0.000)
- **Fastest inference**: LFM2.5-1.2B-Think (9.14s/article)
- **Best efficiency** (F1/log2(time)): LFM2.5-1.2B-Think (score=0.0000)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| LFM2.5-1.2B-Think | 274.2s | 9.14s | 6.6 |