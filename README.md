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

## Dataset

[IEPile](https://github.com/zjunlp/IEPile): A Large-Scale Information Extraction Corpus. In IEPile, the instruction format of IEPile adopts a JSON-like string structure, which is essentially a dictionary-type string composed of the following three main components: (1) 'instruction': Task description, which outlines the task to be performed by the instruction (one of `NER`, `RE`, `EE`, `EET`, `EEA`). (2) 'schema': A list of schemas to be extracted (entity types, relation types, event types). (3) 'input': The text from which information is to be extracted.

Below is a data example:

```json
{
    "task": "NER", 
    "source": "CoNLL2003", 
    "instruction": "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\", \"schema\": [\"person\", \"organization\", \"else\", \"location\"], \"input\": \"284 Robert Allenby ( Australia ) 69 71 71 73 , Miguel Angel Martin ( Spain ) 75 70 71 68 ( Allenby won at first play-off hole )\"}", 
    "output": "{\"person\": [\"Robert Allenby\", \"Allenby\", \"Miguel Angel Martin\"], \"organization\": [], \"else\": [], \"location\": [\"Australia\", \"Spain\"]}"
}

{
  "task": "EE", 
  "source": "PHEE", 
  "instruction": "{\"instruction\": \"You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.\", \"schema\": [{\"event_type\": \"potential therapeutic event\", \"trigger\": true, \"arguments\": [\"Treatment.Time_elapsed\", \"Treatment.Route\", \"Treatment.Freq\", \"Treatment\", \"Subject.Race\", \"Treatment.Disorder\", \"Effect\", \"Subject.Age\", \"Combination.Drug\", \"Treatment.Duration\", \"Subject.Population\", \"Subject.Disorder\", \"Treatment.Dosage\", \"Treatment.Drug\"]}, {\"event_type\": \"adverse event\", \"trigger\": true, \"arguments\": [\"Subject.Population\", \"Subject.Age\", \"Effect\", \"Treatment.Drug\", \"Treatment.Dosage\", \"Treatment.Freq\", \"Subject.Gender\", \"Treatment.Disorder\", \"Subject\", \"Treatment\", \"Treatment.Time_elapsed\", \"Treatment.Duration\", \"Subject.Disorder\", \"Subject.Race\", \"Combination.Drug\"]}], \"input\": \"Our findings reveal that even in patients without a history of seizures, pregabalin can cause a cortical negative myoclonus.\"}", 
  "output": "{\"potential therapeutic event\": [], \"adverse event\": [{\"trigger\": \"cause \", \"arguments\": {\"Subject.Population\": \"NAN\", \"Subject.Age\": \"NAN\", \"Effect\": \"cortical negative myoclonus\", \"Treatment.Drug\": \"pregabalin\", \"Treatment.Dosage\": \"NAN\", \"Treatment.Freq\": \"NAN\", \"Subject.Gender\": \"NAN\", \"Treatment.Disorder\": \"NAN\", \"Subject\": \"patients without a history of seizures\", \"Treatment\": \"pregabalin\", \"Treatment.Time_elapsed\": \"NAN\", \"Treatment.Duration\": \"NAN\", \"Subject.Disorder\": \"NAN\", \"Subject.Race\": \"NAN\", \"Combination.Drug\": \"NAN\"}}]}"
}

{
  "task": "RE", 
  "source": "NYT11", 
  "instruction": "{\"instruction\": \"You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.\", \"schema\": [\"neighborhood of\", \"nationality\", \"children\", \"place of death\"], \"input\": \" In the way New Jersey students know that Thomas Edison 's laboratory is in West Orange , the people of Colma know that Wyatt Earp 's ashes are buried at Hills of Eternity , a Jewish cemetery he was n't ; his wife was , and that Joe DiMaggio is at Holy Cross Cemetery , where visitors often lean bats against his gravestone . \"}", 
  "output": "{\"neighborhood of\": [], \"nationality\": [], \"children\": [], \"place of death\": [{\"subject\": \"Thomas Edison\", \"object\": \"West Orange\"}]}"
}
```

