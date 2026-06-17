# eval_v3_sft.py — Evaluation Guide

Evaluates a distilled event-location extraction model against annotated gold data.
Each line of the input JSONL contains both the ground-truth annotation and the model
predictions for one document.

## Usage

```bash
python scripts/llms/inference/eval_v3_sft.py \
  --pred-jsonl dataset/risk-factor/run-15062025/predictions/quick_dev.jsonl \
  [--report-json /tmp/report.json] \
  [--errors-jsonl /tmp/errors.jsonl]
```

`--report-json` dumps the full nested metrics dict.  
`--errors-jsonl` dumps one row per document that had unmatched events, useful for error analysis.

---

## Output format

The report prints two tables:

- **MICRO** — pooled TP/FP/FN across all documents (standard NLP micro-average).
- **MACRO** — P/R/F1 computed per document then averaged; gives equal weight to each document regardless of how many events it contains.

Each table has two column groups:

| Group | Meaning |
|-------|---------|
| **RELAXED** | Lenient matching (headline number) |
| **EXACT** | Strict matching; shows how much the model deviates from verbatim gold |

---

## The six metric families

### 1. Event extraction (overall)

Measures whether the model finds the right events at all.

Each model prediction is matched one-to-one to a gold event using:
- `event_type` must agree (case-insensitive).
- `grounding_quote` token overlap (SequenceMatcher ratio) ≥ 0.5.

Matched pairs = **TP**; unmatched gold events = **FN**; unmatched predictions = **FP**.

> **Why quote overlap and not exact match?** The model frequently appends location
> context to the grounding quote (e.g. gold: *"5,000 IDP households"* → pred:
> *"5,000 IDP households in Maiduguri"*). Requiring verbatim equality would
> misclassify these as wrong events.

Exact tier additionally requires identical normalized quotes (ratio = 1.0).

---

### 2. Event type (multiset)

Measures whether the model identifies the correct categories of events, independent
of how well it grounds them.

Computed as a **multiset** comparison of `event_type` values per document.
If gold has two "displaced" events and the model predicts one, that counts as
TP=1, FN=1.

Event type uses the same exact/relaxed labels as other families, but the scores
are identical in both tiers because type matching is always a simple string
equality (no fuzzy component).

---

### 3. Event type (doc-level set)

Same as family 2 but **deduplicates** event types per document before comparing.
If gold has two "displaced" events and the model predicts one, that counts as
TP=1, FN=0 (the type was covered). This answers the question *"did the model
identify all distinct event categories present in this document?"* without
penalizing it for not finding every individual instance.

As with family 2, exact and relaxed scores are identical (type matching is always
string equality).

---

### 4. Location extraction

Measures whether the model extracts the correct place names, independent of which
event they are attached to.

Per document, compares the **sets** of `event_location` values (ignoring
`"not_stated"`). A predicted location matches a gold location if:

| Tier | Matching rule |
|------|--------------|
| **Relaxed** | After lowercasing and stripping directional prefixes (*northern*, *southern*, *eastern*, *western*, *central*): exact equality **or** one is a substring of the other **or** SequenceMatcher ratio ≥ 0.85 |
| **Exact** | Normalized string equality only |

> **Why strip directional prefixes?** Annotations often normalize *"northern
> Bangladesh"* to *"Bangladesh"*. The model may or may not strip the prefix; both
> are correct.

---

### 5. Event-location pairing

Measures whether the model correctly associates a location **with the right event**.
This is harder than location extraction alone.

**Conditioned on Step 1**: among event pairs that were already matched (TP events),
counts how many also have a correct location (same rules as family 3). Events that
were not matched at all still contribute to FP/FN, so a missed event is not
double-penalized — it already appears in the event extraction score.

---

### 6. Event-time pairing

Measures whether the model correctly associates a time expression **with the right
event**. Same conditioning as event-location.

`event_time` values are ISO 8601 strings (`"2018-01/2018-10"`, `"2015"`, `"not_stated"`).

| Tier | Match rule |
|------|-----------|
| **Exact** | Normalized string equality |
| **Relaxed** | Exact equality **or** the two values share at least one year in common (e.g. `"2018-01"` and `"2018"` both start with `"2018"` → match) |

> The relaxed rule avoids penalizing the model for predicting the correct year but
> getting the month/day granularity wrong, which is often ambiguous from the source text.

---

## Interpreting the numbers

A few things to keep in mind when reading the scorecard:

**Relaxed ≫ exact on event extraction** is expected. The model often extends
the grounding quote to include location or time context, which is semantically
correct but doesn't match verbatim.

**Event type F1 > event extraction F1** is expected. The model may identify
the right category of event but anchor it to a slightly different span, or
merge two gold events into one.

**Event type (set) ≥ event type (multiset)** is expected. The set metric gives
credit for covering a category at least once, so documents with repeated event
types (e.g. multiple displacement events) will score higher on the set metric
even if the model underestimates the count.

**Location > event-location pairing** is expected. The model might extract
the correct locations overall but associate them with the wrong events (or miss
some events entirely).

**Macro vs micro divergence** tells you about consistency. If macro F1 is much
higher than micro, the model is strong on short documents but struggles on
long ones with many events. The reverse suggests the model handles dense
documents better than sparse ones.
