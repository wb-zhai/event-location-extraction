# event-location-extraction
A test repo for event-location-extraction with GLiNER2.

Included risk factors defined after https://www.science.org/doi/10.1126/sciadv.abm3449. Gliner2 extracts one single event. 

Test data in `event-location-extraction/benchmark/data`. Building on https://huggingface.co/datasets/BAAI/IndustryCorpus_news, LEMONADE and RAMS datasets.

Run benchmark with `uv run run_evaluation.py`:

```zsh 
(base) ➜  event-location-extraction git:(main) uv run run_evaluation.py 
Reading inline script metadata from `run_evaluation.py`
Installed 66 packages in 902ms
Loaded 50 articles, 50 annotations
You are using a model of type extractor to instantiate a model of type . This is not supported for all configurations of models and can yield errors.
============================================================
🧠 Model Configuration
============================================================
Encoder model      : microsoft/deberta-v3-large
Counting layer     : count_lstm
Token pooling      : first
============================================================
  [50/50] article_049 (2456 chars, 20 gold ents, 4 gold pairs)

===============================================================================================
ENTITY EXTRACTION  (micro-averaged over 50 articles, 671 gold entities)
===============================================================================================
Ver            |   TP    FP    FN |   Prec    Rec     F1   F0.5     F2   Jacc |   Time
--------------------------------------------------------------------------------------
v2             |  461   455   210 |  0.503  0.687  0.581  0.532  0.640  0.409 |  58.5s
v4             |  390   269   281 |  0.592  0.581  0.587  0.590  0.583  0.415 |  58.6s
v5_no_coref    |  390   270   281 |  0.591  0.581  0.586  0.589  0.583  0.414 |  60.8s

===============================================================================================
EVENT-LOCATION LINKING  (micro-averaged over 50 articles, 140 gold pairs)
===============================================================================================
Ver            |   TP    FP    FN |   Prec    Rec     F1   F0.5     F2   Jacc |   Time
--------------------------------------------------------------------------------------
v2             |   13   139   127 |  0.086  0.093  0.089  0.087  0.091  0.047 |  58.5s
v4             |    9    58   131 |  0.134  0.064  0.087  0.110  0.072  0.045 |  58.6s
v5_no_coref    |    9    58   131 |  0.134  0.064  0.087  0.110  0.072  0.045 |  60.8s

[SAVED] evaluation_results.json
```

