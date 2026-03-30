# Event-Location Extraction Benchmark Report

**Date**: 2026-03-28 22:36
**Match mode**: relaxed
**Articles evaluated**: 30

## Entity Extraction (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| LFM2-24B-A2B | 157 | 156 | 238 | 0.502 | 0.398 | **0.444** | 0.477 | 0.415 | 0.285 | 3.36s |
| GLM4.7-Flash-31B | 69 | 83 | 326 | 0.454 | 0.175 | **0.252** | 0.344 | 0.199 | 0.144 | 22.26s |
| NemotronCascade-30B-A3B | 0 | 0 | 395 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 2.95s |

## Event-Location Linking (micro-averaged)

| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |
|---|---|---|---|---|---|---|---|---|---|---|
| LFM2-24B-A2B | 13 | 103 | 44 | 0.112 | 0.228 | **0.150** | 0.125 | 0.189 | 0.081 | 3.36s |
| GLM4.7-Flash-31B | 5 | 42 | 52 | 0.106 | 0.088 | **0.096** | 0.102 | 0.091 | 0.051 | 22.26s |
| NemotronCascade-30B-A3B | 0 | 0 | 57 | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 0.000 | 2.95s |

## Summary

- **Best entity extraction**: LFM2-24B-A2B (F1=0.444)
- **Best event-location linking**: LFM2-24B-A2B (F1=0.150)
- **Fastest inference**: NemotronCascade-30B-A3B (2.95s/article)
- **Best efficiency** (F1/log2(time)): LFM2-24B-A2B (score=0.0620)

## Inference Speed

| Model | Total Time | Avg/Article | Articles/Min |
|---|---|---|---|
| NemotronCascade-30B-A3B | 88.6s | 2.95s | 20.3 |
| LFM2-24B-A2B | 100.9s | 3.36s | 17.8 |
| GLM4.7-Flash-31B | 667.9s | 22.26s | 2.7 |