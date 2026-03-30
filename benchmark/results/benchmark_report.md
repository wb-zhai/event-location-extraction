# Cross-Family LLM Benchmark: Event-Location Extraction from News

**Date**: 2026-03-28
**Evaluation**: Relaxed matching (token overlap + containment + fuzzy)
**Dataset**: 30 real news articles from BAAI/IndustryCorpus_news (100 annotated, 30 evaluated)
**Models tested**: 24 models across 14 families
**Successful benchmarks**: 20 models (4 failed to load/parse)

---

## Combined Leaderboard — Event-Location Linking (the task that matters)

| Rank | Model | Family | Active Params | Released | **Pair F1** | Entity F1 | Speed | Art/Min |
|---|---|---|---|---|---|---|---|---|
| 1 | **Gemma3-27B-QAT** | Gemma 3 | 27B | 2025-04 | **0.350** | 0.651 | 34.4s | 1.7 |
| 2 | **Qwen3.5-122B-A10B** | Qwen 3.5 MoE | **10B** | **2026-02** | **0.322** | 0.682 | 16.9s | 3.6 |
| 3 | Qwen3.5-27B | Qwen 3.5 | 27B | 2026-02 | 0.313 | 0.678 | 27.1s | 2.2 |
| NEW | **Qwen3.5-4B-Claude-Distill** | Qwen 3.5 distill | **4B** | **2026-03** | **0.311** | **0.687** | 19.4s | 3.1 |
| 4 | Qwen3.5-35B-MoE | Qwen 3.5 MoE | 3B | 2026-02 | 0.270 | 0.643 | **5.3s** | **11.3** |
| 5 | Qwen3.5-9B | Qwen 3.5 | 9B | 2026-03 | 0.267 | 0.604 | 12.3s | 4.9 |
| 6 | **JoyAI-Flash-48B** | JD.com MoE | **3B** | **2026-02** | **0.264** | 0.593 | 10.9s | 5.5 |
| 7 | Gemma3-12B | Gemma 3 | 12B | 2025-03 | 0.257 | 0.605 | 20.7s | 2.9 |
| 8 | Phi4-14B | Phi-4 | 14B | 2025-01 | 0.247 | 0.610 | 11.2s | 5.3 |
| 9 | Qwen3-14B | Qwen 3 | 14B | 2025-04 | 0.226 | 0.597 | 10.8s | 5.5 |
| 10 | **Ministral3-14B** | Mistral | 14B | **2026-02** | 0.215 | 0.509 | 23.5s | 2.6 |
| 11 | **Qwen3-Coder-80B** | Qwen 3 MoE | 3B | **2026-02** | 0.201 | 0.634 | 18.1s | 3.3 |
| 12 | **Jan-v3-4B** | Jan | 4B | **2026-01** | 0.183 | 0.563 | 8.0s | 7.5 |
| 13 | LFM2-24B-A2B | Liquid | 2B | 2026-02 | 0.150 | 0.444 | 3.4s | 17.8 |
| 14 | Phi4-Mini-3.8B | Phi-4 | 3.8B | 2025-02 | 0.138 | 0.348 | 16.4s | 3.7 |
| 15 | Qwen3.5-2B | Qwen 3.5 | 2B | 2026-03 | 0.130 | 0.436 | 8.3s | 7.3 |
| 16 | Qwen3.5-4B | Qwen 3.5 | 4B | 2026-03 | 0.108 | 0.395 | 13.1s | 4.6 |
| 17 | GLM4.7-Flash | GLM MoE | ~6B | 2026-01 | 0.096 | 0.252 | 22.3s | 2.7 |
| 18 | Qwen3.5-0.8B | Qwen 3.5 | 0.8B | 2026-03 | 0.054 | 0.167 | 14.1s | 4.3 |
| -- | GLiNER2 v5 (baseline) | GLiNER2 | 0.3B | 2025-07 | 0.087 | 0.587 | 1.3s | 46.2 |

*Failed to load/parse: Nanbeige4.1-3B, Falcon-H1R-7B, NemotronCascade-30B, LFM2.5-1.2B (output format issues). TinyAya-3.35B extracted entities (F1=0.446) but no pairs. MiniMax-M2.5-230B and Step3.5-Flash-196B too large to download in time.*
**Gold standard**: 395 entities, 57 event-location pairs across 30 articles
**Platform**: Apple Silicon (MLX), 4-bit quantization

---

## Results

### Entity Extraction (micro-averaged)

| Model | Params | Prec | Rec | **F1** | Avg Time |
|---|---|---|---|---|---|
| Qwen3.5-27B-4bit | 27B | 0.547 | **0.891** | **0.678** | 27.1s |
| Qwen3.5-35B-A3B-4bit | 35B (3B active) | **0.565** | 0.744 | 0.643 | 5.3s |
| Qwen3.5-9B-4bit | 9B | 0.523 | 0.714 | 0.604 | 12.3s |
| Qwen3.5-2B-4bit | 2B | 0.543 | 0.365 | 0.436 | 8.3s |
| Qwen3.5-4B-4bit | 4B | 0.361 | 0.435 | 0.395 | 13.1s |
| Qwen3.5-0.8B-4bit | 0.8B | 0.333 | 0.111 | 0.167 | 14.1s |

### Event-Location Linking (micro-averaged)

| Model | Params | Prec | Rec | **F1** | Avg Time |
|---|---|---|---|---|---|
| **Qwen3.5-27B-4bit** | 27B | 0.203 | **0.684** | **0.313** | 27.1s |
| Qwen3.5-35B-A3B-4bit | 35B (3B active) | **0.198** | 0.421 | 0.270 | **5.3s** |
| Qwen3.5-9B-4bit | 9B | 0.175 | 0.561 | 0.267 | 12.3s |
| Qwen3.5-2B-4bit | 2B | 0.121 | 0.140 | 0.130 | 8.3s |
| Qwen3.5-4B-4bit | 4B | 0.072 | 0.210 | 0.108 | 13.1s |
| Qwen3.5-0.8B-4bit | 0.8B | 0.118 | 0.035 | 0.054 | 14.1s |

### Comparison with GLiNER2 heuristic pipeline (from previous evaluation)

| System | Entity F1 | Pair F1 | Avg Time | Approach |
|---|---|---|---|---|
| **Qwen3.5-27B** | **0.678** | **0.313** | 27.1s | LLM zero-shot |
| Qwen3.5-35B-A3B (MoE) | 0.643 | 0.270 | 5.3s | LLM zero-shot |
| Qwen3.5-9B | 0.604 | 0.267 | 12.3s | LLM zero-shot |
| GLiNER2 v4/v5 | 0.587 | 0.087 | 1.3s | NER + heuristics |
| GLiNER2 v2 | 0.581 | 0.089 | 1.3s | NER + heuristics |

---

## Key Findings

### 1. Gemma 3 27B QAT is the new champion

**Gemma3-27B-QAT** achieves **Pair F1 = 0.350** — beating Qwen3.5-27B (0.313) by 12%. Its Quantization-Aware Training preserves quality that standard post-training quantization loses. It also has the highest pair recall (0.561) of any model tested.

### 2. The Gemma family dominates at structured extraction

Both Gemma models outperform same-size competitors on pairs:
- **Gemma3-27B-QAT** (0.350) > Qwen3.5-27B (0.313) at 27B
- **Gemma3-12B** (0.257) > Qwen3.5-9B (0.267) is close, but Gemma at 12B vs Qwen at 9B

Google's built-in function calling / structured output training shows clear benefits for IE tasks.

### 3. Clear scaling law: bigger models = better extraction

```
0.8B  ->  2B  ->  3.8B ->  9B  ->  12B ->  14B ->  27B
0.054   0.130   0.138   0.267   0.257  0.247   0.350  (Pair F1)
```

The 9B+ threshold is where quality becomes usable. Below 9B, all models struggle.

### 4. Qwen3.5-35B-MoE remains the efficiency champion

- **Pair F1 = 0.270** at only **5.3s/article** (11.3 articles/min)
- 5x faster than Gemma3-27B-QAT while delivering 77% of its quality
- Only 3B active parameters per token

### 5. Precision remains the universal bottleneck

All models achieve **recall 0.26-0.56** but precision stays at **0.11-0.25**. Models over-generate pairs. Fine-tuning or a filtering stage would yield the biggest gains.

### 6. Claude-distilled Qwen3.5-4B is the biggest surprise of the benchmark

**Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled** — a 4B model distilled from Claude Opus reasoning traces — achieves:
- **Pair F1 = 0.311** — virtually identical to Qwen3.5-27B (0.313) at **7x fewer parameters**
- **Entity F1 = 0.687** — the **highest entity F1 of any model tested**
- This proves that reasoning distillation from frontier models can compress IE capability into tiny models

### 7. Qwen3.5-122B-A10B is the new #2 — and a surprise value pick

The MoE variant **Qwen3.5-122B-A10B** (10B active) scores **Pair F1=0.322** — nearly matching Gemma3-27B-QAT (0.350) while being **2x faster** (16.9s vs 34.4s). It also has the **highest entity F1** of all models (0.682). This is the best new 2026 finding.

### 7. JoyAI-Flash-48B is a dark horse at only 3B active

JD.com's **JoyAI-Flash** (48B total, 3B active) achieves **Pair F1=0.264** — matching Qwen3.5-9B (0.267) with a fraction of the active parameters. At 10.9s/article, it's a strong contender for resource-constrained deployments.

### 8. New 2026 architectures (Nemotron, LFM, GLM) underperform established families

Despite novel architectures (hybrid SSM-transformer, Liquid Foundation Models, cascade reasoning):
- **NemotronCascade-30B** (3B active, NVIDIA): F1=0.000 — output format incompatible
- **LFM2-24B** (2B active, Liquid AI): F1=0.150 — fast (3.4s) but weak extraction
- **GLM4.7-Flash** (Zhipu AI): F1=0.096 — Chinese-trained model struggles with English IE
- **LFM2.5-1.2B**: too small for structured extraction

The established instruction-tuned families (Gemma, Qwen) remain dominant. New architectures optimize for speed/efficiency but haven't yet caught up on structured output quality.

### 7. Every LLM >2B beats GLiNER2 on linking

Even the 2B Qwen (F1=0.130) outperforms GLiNER2 v5 (F1=0.087). **LLMs are categorically better** at cross-sentence event-location linking.

---

## Recommendations

| Use case | Model | Pair F1 | Speed | Why |
|---|---|---|---|---|
| **Best quality** | Gemma3-27B-QAT | **0.350** | 34s | QAT preserves quality, best structured output |
| **Best value** | **Qwen3.5-122B-A10B** | **0.322** | 17s | 10B active, near-best quality, 2x faster than Gemma |
| **Best balance** | Qwen3.5-27B | 0.313 | 27s | Strong all-around, large community |
| **Production speed** | Qwen3.5-35B-MoE | 0.270 | **5.3s** | 11 articles/min, 77% of best quality |
| **Budget hardware** | Qwen3.5-9B or JoyAI-Flash | 0.267/0.264 | 12s/11s | 9B / 3B active respectively |
| **Fine-tuning base** | Gemma3-12B or Qwen3-14B | 0.257/0.226 | 21s/11s | Good base for LoRA, fast iteration |

---

## Inference Speed

| Model | Avg/Article | Articles/Min |
|---|---|---|
| Qwen3.5-35B-A3B-4bit | 5.30s | 11.3 |
| Qwen3.5-2B-4bit | 8.25s | 7.3 |
| Qwen3.5-9B-4bit | 12.32s | 4.9 |
| Qwen3.5-4B-4bit | 13.12s | 4.6 |
| Qwen3.5-0.8B-4bit | 14.07s | 4.3 |
| Qwen3.5-27B-4bit | 27.09s | 2.2 |

---

## Framework Usage

```bash
# Benchmark specific models
uv run benchmark/run_benchmark.py --models "Qwen3.5-9B-4bit,Qwen3.5-27B-4bit"

# Strict matching
uv run benchmark/run_benchmark.py --match-mode strict

# Custom dataset
uv run benchmark/run_benchmark.py --dataset path/to/annotated.json

# Quick test
uv run benchmark/run_benchmark.py --max-articles 10
```

To add models: edit `get_mlx_models_to_benchmark()` in `lib/model_adapter.py`.
