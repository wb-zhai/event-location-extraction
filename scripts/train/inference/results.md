# Event Extraction Results

## New Schema Setup

| | |
|---|---|
| **Model** | Qwen3.5-4B |
| **Method** | LoRA SFT |
| **Training data** | ~2,600 news articles |
| **Epochs** | 3 |
| **Hardware** | A100 40GB |
| **Training time** | ~3 hours |

### Windowing

Articles are split into overlapping windows on paragraph and sentence boundaries, with each window capped at 3,000 characters.

With the Qwen3.5 tokenizer:

| | Tokens |
|---|---|
| avg | 1,179 |
| min | 709 |
| max | 2,177 |

### Output Schema

```json
{
  "events": [
    {
      "event_type": "string",
      "grounding_quote": "string",
      "event_location": "string | not_stated",
      "event_time": "string | not_stated",
      "time_status": "past | ongoing | forecast | not_stated",
      "modality": "asserted | projected"
    }
  ]
}
```

All scores use **relaxed** matching, with field-specific rules:

- **Event type** — exact string equality after case and whitespace normalization. There is no fuzzy matching: `drought` ≠ `failed rains`.
- **Grounding quote** — character-level overlap ≥ 50%. Events are paired greedily by descending overlap score, one-to-one.
- **Location** — directional prefixes are stripped (*northern*, *southern*, etc.). A match is counted if either string contains the other, or if character-level overlap is ≥ 85%. For `;`-separated multi-location strings, every part on both sides must match.
- **Time** — exact after normalization, or same year sufficient for ISO date strings.

---

## Results (dev set, relaxed matching)

Evaluation is done at the window level. Each ~1,000–2,000 token window is scored independently, then results are micro-averaged.

### Fine-grained event type (exact label match)

| Metric | What counts as correct | P | R | F1 |
|---|---|---|---|---|
| **Event span match** | Predicted event type AND grounding quote both match a gold event | 54.7 | 56.2 | **55.4** |
| **Event type match** | Predicted event type matches; grounding quote ignored; each occurrence counted separately | 66.3 | 68.2 | **67.2** |
| **Event type coverage** | Predicted event type appears anywhere in the window's gold; duplicates collapsed to one per type | 76.1 | 70.6 | **73.3** |

The ~12-point F1 gap between the first two rows mainly comes from grounding quote errors: the model often identifies the right event type but anchors it to the wrong span. The remaining gap to coverage reflects count errors, where the model predicts too many or too few instances of a type that is present.

### Cluster-level (synonym-grouped label match)

Before scoring, event types are mapped to semantic clusters. Near-synonyms such as `drought`, `failed rains`, and `inadequate rainfall` therefore count as the same label. **This is currently the more useful view** because it reduces noise from ontology ambiguity.

| Metric | What counts as correct | P | R | F1 |
|---|---|---|---|---|
| **Event span match** | Predicted event cluster AND grounding quote both match a gold event | 58.7 | 60.3 | **59.5** |
| **Event type match** | Predicted event cluster matches; each occurrence counted separately | 74.4 | 76.4 | **75.4** |
| **Event type coverage** | Predicted event cluster appears anywhere in the window's gold; duplicates collapsed | 88.3 | 83.8 | **86.0** |

### Location & Time

End-to-end F1 only gives credit when the event is found, grounded to the right quote, and the linked field also matches. These scores are intentionally strict and include upstream event errors.

Conditional accuracy looks only at events that were already matched by event type and grounding quote, then checks whether the linked location or time is correct. This is the better view of pure location/time linking quality, so location is reported this way below.

#### Location

For location, the most useful number is conditional accuracy on matched events. The evaluator first aligns predicted events to gold events using event type and grounding quote. It then checks whether the location attached to each matched event is correct. This separates location-linking quality from upstream event detection and quote-grounding errors.

Relaxed location matching strips directional prefixes such as *northern* or *central*, allows substring matches, and accepts high character overlap. `Gold stated` excludes cases where the gold location is `not_stated`, so it focuses only on events with a concrete place to extract.

| Metric | Relaxed Acc. | Correct / total |
|---|---:|---:|
| **Location on matched events** | 82.4 | 1458 / 1769 |
| **Location on matched events (gold stated)** | 86.1 | 1189 / 1381 |
| **Cluster location on matched events** | 82.5 | 1567 / 1899 |
| **Cluster location on matched events (gold stated)** | 86.3 | 1283 / 1487 |

#### Time

For time, the most useful number is also conditional accuracy on matched events. The evaluator first aligns predicted events to gold events using event type and grounding quote. It then checks whether the `event_time` attached to each matched event is correct.

Relaxed time matching accepts exact normalized matches, and for ISO-style dates also accepts sharing the same year. `Gold stated` excludes cases where the gold time is `not_stated`, so it focuses only on events with a concrete time expression to extract.

| Metric | Relaxed Acc. | Correct / total |
|---|---:|---:|
| **Time on matched events** | 89.4 | 1581 / 1769 |
| **Time on matched events (gold stated)** | 83.6 | 588 / 703 |
| **Cluster time on matched events** | 89.3 | 1695 / 1899 |
| **Cluster time on matched events (gold stated)** | 84.1 | 639 / 760 |

Metric details:

- **Cluster event-location/time pairing** uses the same logic, but the initial event match uses synonym clusters instead of exact fine-grained labels. This explains why cluster pair scores are usually slightly higher than the fine-grained pair scores.
- **Gold stated** rows exclude cases where the gold field is `not_stated`. They focus on examples where there was a concrete location or time to extract, rather than giving credit for correctly predicting absence.

The main takeaway is that location and time linking are much stronger once upstream event matching is factored out. Linked location accuracy is **82.4** overall and **86.1** when a gold location is stated. Time linking is similarly strong overall (**89.4**), and remains high (**83.6**) when only events with a stated gold time are considered.

---

## Interpreting the scores

### Why cluster-level matters

Fine-grained type matching requires the model to produce the exact ontology label. The ontology contains many near-synonym labels with subtle boundaries, so a semantically valid prediction can still be scored as a false positive. Cluster-level matching groups those labels before scoring and gives a cleaner view of what the model captured. The ~4-point lift on span match (55.4 → 59.5) and ~13-point lift on type coverage (73.3 → 86.0) are best read as reduced annotation/evaluation noise, not as a change in model behavior.

The model is still trained on fine-grained labels, so the same ambiguity also affects the training signal. If similar contexts are labeled inconsistently across near-synonyms, the model can learn conflicting associations. That may hurt performance beyond what the evaluation gap alone shows.

### Why fine-grained scores underestimate quality

The event type ontology contains many near-synonym clusters where the "right" label is genuinely ambiguous. The rainfall-deficit family alone has eight overlapping entries:

> `drought` · `failed rains` · `inadequate rainfall` · `abnormally low rainfall` · `shortage of rains` · `lack of rains` · `prolonged dry spell` · `scanty rainfall`

Each entry has a precise intended use, but in practice distinctions such as *"statistical anomaly"* versus *"functional shortfall against needs"* are easy to blur. A human annotator and the model can choose different labels while making essentially the same semantic call. Some false positives and false negatives are therefore likely to be valid synonym choices rather than true extraction errors. The same issue appears in the hunger (`mass hunger` / `acute hunger` / `life-threatening hunger` / `massive starvation`) and conflict families. Until evaluation uses a synonym-aware ontology, fine-grained type scores should be treated as a lower bound.

### Summary

- The model is good at identifying **which event types are present** in a window (cluster type coverage F1 = **86.0**).
- Grounding quotes are the main weakness, costing ~12 F1 points compared with type-only scoring.
- Location linking on already matched events is strong (**82.4** relaxed; **86.1** when gold location is stated).
- Time linking is also strong on matched events (**89.4** relaxed overall; **83.6** when gold time is stated).
- Fine-grained type scores are pulled down by ontology ambiguity and should improve with synonym-aware evaluation or ontology consolidation.
