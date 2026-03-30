# Event-Location Extraction Benchmark Report

**Date**: 2026-03-28 21:44
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma3-27B-QAT | 309 | 245 | 86 | 0.558 | 0.782 | **0.651** | 0.592 | 0.724 | 0.483 | 34.42s |
| Mistral-Small-3.1-24B | 0 | 0 | 0 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 0.00s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma3-27B-QAT | 32 | 94 | 25 | 0.254 | 0.561 | **0.350** | 0.285 | 0.452 | 0.212 | 34.42s |
| Mistral-Small-3.1-24B | 0 | 0 | 0 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 0.00s |

## Summary

- **Best entity extraction**: Gemma3-27B-QAT (F1=0.651)
- **Best event-location linking**: Gemma3-27B-QAT (F1=0.350)
- **Fastest inference**: Mistral-Small-3.1-24B (0.00s/article)
- **Best efficiency** (F1/log2(time)): Gemma3-27B-QAT (score=0.0674)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| Mistral-Small-3.1-24B | 0.0s | 0.00s | inf |
| Gemma3-27B-QAT | 1032.6s | 34.42s | 1.7 |