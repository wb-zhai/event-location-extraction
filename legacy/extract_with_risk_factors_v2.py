# /// script
# dependencies = [
#   "gliner2", "urllib3", "requests"
# ]
# ///

"""
GLiNER2 Event-Location Extraction with Risk Factors
====================================================
Extracts event/risk-factor entities and geographic locations from news text,
then links them via sentence co-occurrence.

Strategy:
  - NER extraction with `include_spans=True` gives reliable entity spans
    with character offsets and confidence scores.
  - GLiNER2's relation extraction and structured JSON are too sparse for
    multi-event texts, so we build event-location links by finding which
    events and locations share the same sentence.
"""

import re
import json
import uuid
from gliner2 import GLiNER2

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

# ─────────────────────────────────────────────────────────────────────────────
# RISK-FACTOR ENTITY SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

RISK_FACTOR_ENTITIES = {
    "agricultural production issues": (
        "Failed crops, bad harvests, lack of cultivation, livestock or cattle death, "
        "agricultural infrastructure damage."
    ),
    "conflicts and violence": (
        "Warfare, fighting, civil strife, rebel insurgencies, militia groups, warlords, "
        "banditry, piracy, air attacks, sieges, bombings, terrorism, violent suppression."
    ),
    "economic issues": (
        "Economic crises, collapsing economy, disrupted trade, external debt, brain drain, "
        "rising inflation, drastic price rises, widespread poverty."
    ),
    "environmental issues": (
        "Ecological crises, climate change, greenhouse gases, environmental degradation, "
        "natural disasters."
    ),
    "food crisis": (
        "Food insecurity, mass hunger, acute hunger, starvation, malnutrition, dehydration, "
        "gastrointestinal diseases, infant mortality."
    ),
    "forced displacement": (
        "Forced migration, displaced individuals, vulnerable populations fleeing homes, "
        "asylum seekers, refugees, makeshift camps."
    ),
    "humanitarian aid": (
        "Humanitarian disasters, aid appeals, food assistance, international embargoes, "
        "restricted humanitarian access, withheld or stolen relief, self-reliance."
    ),
    "land-related issues": (
        "Land invasions, hostile land grabs, systemic land seizures, burning houses, "
        "failed land reform, land degradation, poor soil quality, destroyed forests."
    ),
    "pests and diseases": (
        "Swarms of locusts, agricultural pests, potato blight, epidemics, cholera outbreaks, "
        "animal diseases like rinderpest or cattle plague."
    ),
    "political instability": (
        "Collapse of government, lack of authority, corruption, mismanagement, power struggles, "
        "secession, coup d'etat, violent overthrows, dictatorships, military juntas."
    ),
    "weather shocks": (
        "Climatic hazards, weather extremes, destructive floods, cyclones, "
        "prolonged dry spells, harsh droughts, abnormally low rainfall."
    ),
    "other": (
        "Uncategorized catastrophes, man-made disasters, severe population crises."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# NER SCHEMA (entities only — this is what GLiNER2 does best)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ner_schema():
    entity_types = {
        "event": "Any event, incident, crisis, disaster, or phenomenon.",
        "location": "Any geographic location, city, country, or region.",
        "date": "A temporal expression including specific dates or relative times.",
        "organization": "An organization, government body, agency, or institution.",
    }
    entity_types.update(RISK_FACTOR_ENTITIES)
    return extractor.create_schema().entities(entity_types)


NER_SCHEMA = _build_ner_schema()

# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE BOUNDARY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n\n+')

def _sentence_spans(text):
    """Return list of (start, end) character ranges for each sentence."""
    spans = []
    prev = 0
    for m in _SENTENCE_RE.finditer(text):
        spans.append((prev, m.start()))
        prev = m.end()
    if prev < len(text):
        spans.append((prev, len(text)))
    return spans


def _sentence_index_for(char_start, sentence_spans):
    """Return the index of the sentence containing char_start."""
    for i, (s, e) in enumerate(sentence_spans):
        if s <= char_start < e:
            return i
    return -1

# ─────────────────────────────────────────────────────────────────────────────
# CORE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

# Labels that count as "event-like" for linking purposes
_EVENT_LABELS = {"event"} | set(RISK_FACTOR_ENTITIES.keys())


def extract_entities(text, threshold=0.3):
    """Run NER with spans and confidence, return raw result dict."""
    return extractor.extract(
        text, NER_SCHEMA,
        threshold=threshold,
        include_spans=True,
        include_confidence=True,
    )


def build_event_location_pairs(raw_entities, text):
    """
    Link events/risk-factors to locations by sentence co-occurrence.

    For every sentence that contains at least one event-like entity AND
    at least one location entity, emit a pair.
    """
    sentences = _sentence_spans(text)

    # Bucket entities by sentence
    events_by_sentence = {}   # sent_idx -> list of span dicts
    locations_by_sentence = {}

    for label, spans in raw_entities.get("entities", {}).items():
        is_event = label in _EVENT_LABELS
        is_location = label == "location"
        if not (is_event or is_location):
            continue
        for span in spans:
            si = _sentence_index_for(span["start"], sentences)
            if si < 0:
                continue
            entry = {**span, "label": label}
            if is_event:
                events_by_sentence.setdefault(si, []).append(entry)
            else:
                locations_by_sentence.setdefault(si, []).append(entry)

    pairs = []
    seen = set()
    for si in sorted(events_by_sentence):
        locs = locations_by_sentence.get(si, [])
        if not locs:
            # Fall back: check neighboring sentences (si-1, si+1)
            for neighbor in (si - 1, si + 1):
                locs = locations_by_sentence.get(neighbor, [])
                if locs:
                    break
        for ev in events_by_sentence[si]:
            for loc in locs:
                key = (ev["text"].lower(), loc["text"].lower(), ev["label"])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "event_text": ev["text"],
                    "event_label": ev["label"],
                    "event_confidence": round(ev["confidence"], 4),
                    "location_text": loc["text"],
                    "location_confidence": round(loc["confidence"], 4),
                    "event_start": ev["start"],
                    "event_end": ev["end"],
                    "location_start": loc["start"],
                    "location_end": loc["end"],
                })
    return pairs


def flatten_entities(raw_entities):
    """Deduplicate and flatten the per-label entity lists into a single list."""
    seen = set()
    flat = []
    for label, spans in raw_entities.get("entities", {}).items():
        for span in spans:
            key = (span["text"].lower(), label)
            if key in seen:
                continue
            seen.add(key)
            flat.append({
                "text": span["text"],
                "label": label,
                "confidence": round(span["confidence"], 4),
                "start": span["start"],
                "end": span["end"],
            })
    flat.sort(key=lambda e: e["start"])
    return flat

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKED EXTRACTION FOR LONG ARTICLES
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_text(text, max_chars=1500, overlap=200):
    """Split text at sentence boundaries into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            last_period = text.rfind('. ', start, end)
            if last_period > start + max_chars // 2:
                end = last_period + 2
        chunks.append((text[start:end], start))
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_from_article(text, threshold=0.3, document_id=None,
                         source_url=None, publication_date=None):
    """Full pipeline: chunk -> NER -> merge -> link by co-occurrence."""
    chunks = _chunk_text(text)
    merged = {}  # label -> {text_lower: span_dict}

    for chunk_str, offset in chunks:
        raw = extract_entities(chunk_str, threshold=threshold)
        for label, spans in raw.get("entities", {}).items():
            if label not in merged:
                merged[label] = {}
            for span in spans:
                key = span["text"].lower()
                adjusted = {
                    **span,
                    "start": span["start"] + offset,
                    "end": span["end"] + offset,
                }
                if key not in merged[label] or adjusted["confidence"] > merged[label][key]["confidence"]:
                    merged[label][key] = adjusted

    # Rebuild into the same shape extract_entities returns
    merged_raw = {"entities": {label: list(spans.values()) for label, spans in merged.items()}}

    entities = flatten_entities(merged_raw)
    pairs = build_event_location_pairs(merged_raw, text)

    return {
        "document": {
            "id": document_id or str(uuid.uuid4()),
            "source_url": source_url,
            "publication_date": publication_date,
            "text_length": len(text),
        },
        "entities": entities,
        "event_location_pairs": pairs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    article = (
        "A catastrophic sequence of events has unfolded across the eastern province of Valoria "
        "over the past six months. Following an unprecedented prolonged dry spell and an overall "
        "lack of rains, the region is now experiencing one of the most severe weather shocks in "
        "its history. The abnormally low rainfall has devastated the local economy and triggered "
        "severe agricultural production issues. Farmers in the rural outskirts of Oakhaven are "
        "facing bad harvests and failed crops, while water distribution shortages have led to "
        "widespread cattle death, severely impacting the livelihoods of thousands.\n\n"
        "As a direct consequence of the disruption to farming, a profound food crisis has gripped "
        "the nation. Local health organizations are reporting rampant food insecurity and mass hunger, "
        "with hospitals in the capital city of Kingsbridge overwhelmed by a largely malnourished and "
        "dehydrated population suffering from gastrointestinal diseases. Infant mortality rates have "
        "sharply increased over the past quarter.\n\n"
        "The escalating tragedy has spurred a massive wave of forced displacement. Civilians uprooted "
        "from their homes by the life-threatening hunger are fleeing the rural collapse, seeking "
        "shelter in makeshift camps along the border of neighboring Aethelgard. International aid "
        "organizations have issued an urgent aid appeal, citing a deteriorating humanitarian situation. "
        "However, efforts for international intervention remain hindered by restricted humanitarian "
        "access and recent incidents of stolen food aid, leaving many displaced individuals to rely "
        "entirely on self reliance in the face of this immense tragedy."
    )

    print("=" * 60)
    print("NER EXTRACTION (with spans + confidence)")
    print("=" * 60)
    raw = extract_entities(article, threshold=0.3)
    entities = flatten_entities(raw)
    pairs = build_event_location_pairs(raw, article)

    print(f"\nFound {len(entities)} entities:")
    for e in entities:
        print(f"  [{e['label']:>35}]  {e['text']:<40}  conf={e['confidence']:.3f}  ({e['start']}-{e['end']})")

    print(f"\nFound {len(pairs)} event-location pairs:")
    for p in pairs:
        print(f"  {p['event_text']:<40} -> {p['location_text']:<15}  (label={p['event_label']})")

    print("\n" + "=" * 60)
    print("FULL PIPELINE (chunked + merged)")
    print("=" * 60)
    result = extract_from_article(
        article, threshold=0.3,
        document_id="article-2026-multi-crisis",
        source_url="https://example.com/multi-crisis",
        publication_date="2026-03-05",
    )
    print(json.dumps(result, indent=2))

    with open("gliner2_extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Saved to 'gliner2_extraction_results.json'")
