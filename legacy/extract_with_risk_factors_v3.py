# /// script
# dependencies = [
#   "gliner2", "urllib3", "requests", "pysbd"
# ]
# ///

"""
GLiNER2 Event-Location Extraction with Risk Factors — v3
=========================================================
Extracts event/risk-factor entities and geographic locations from news text,
then links them via sentence co-occurrence with waterfall anchoring.

Improvements over v2:
  A) Overlap deduplication: prefer specific risk-factor labels over generic
     "event" when both match the same span.
  B) Robust sentence splitting via pysbd (handles abbreviations, decimals).
  C) Waterfall anchoring: events without a same-sentence location inherit
     the most recently mentioned location (news-article pattern).
  D) Per-label confidence thresholds to reduce noise.
"""

import json
import uuid
import pysbd
from gliner2 import GLiNER2

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

# ─────────────────────────────────────────────────────────────────────────────
# RISK-FACTOR ENTITY DEFINITIONS
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

# Labels that are "event-like" for linking purposes
_EVENT_LABELS = {"event"} | set(RISK_FACTOR_ENTITIES.keys())

# ─────────────────────────────────────────────────────────────────────────────
# D) PER-LABEL CONFIDENCE THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
# Locations need high confidence (prevent false geo-tags).
# Risk factors can be slightly more lenient.
# Generic "event" needs a higher bar to avoid vague noun phrases.

LABEL_THRESHOLDS = {
    "location":     0.80,
    "event":        0.50,
    "date":         0.50,
    "organization": 0.60,
    # Risk factors — moderate threshold
    **{k: 0.40 for k in RISK_FACTOR_ENTITIES},
}

# Global floor: anything below this is never kept
GLOBAL_THRESHOLD = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# NER SCHEMA
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
# B) ROBUST SENTENCE SPLITTING (pysbd)
# ─────────────────────────────────────────────────────────────────────────────

_segmenter = pysbd.Segmenter(language="en", clean=False)

def _sentence_spans(text):
    """Return (start, end) character ranges for each sentence using pysbd."""
    segments = _segmenter.segment(text)
    spans = []
    pos = 0
    for seg in segments:
        idx = text.find(seg, pos)
        if idx == -1:
            idx = pos
        spans.append((idx, idx + len(seg)))
        pos = idx + len(seg)
    return spans


def _sentence_index_for(char_start, sentence_spans):
    for i, (s, e) in enumerate(sentence_spans):
        if s <= char_start < e:
            return i
    return -1

# ─────────────────────────────────────────────────────────────────────────────
# CORE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities(text):
    """Run NER with spans and confidence. Uses GLOBAL_THRESHOLD for the model
    call, then applies per-label filtering afterwards."""
    return extractor.extract(
        text, NER_SCHEMA,
        threshold=GLOBAL_THRESHOLD,
        include_spans=True,
        include_confidence=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A) OVERLAP DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _spans_overlap(a, b):
    """True if two span dicts share any characters."""
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def flatten_entities(raw_entities):
    """Deduplicate spans: if a specific risk-factor and generic 'event' cover
    the same text, keep only the risk-factor. Then apply per-label thresholds."""
    all_spans = []
    for label, span_list in raw_entities.get("entities", {}).items():
        for span in span_list:
            all_spans.append({**span, "label": label})

    # Sort by start position, then prefer specific labels over "event"
    all_spans.sort(key=lambda s: (s["start"], s["label"] == "event"))

    filtered = []
    for span in all_spans:
        # Apply per-label threshold
        threshold = LABEL_THRESHOLDS.get(span["label"], GLOBAL_THRESHOLD)
        if span["confidence"] < threshold:
            continue

        # Check overlap with already-accepted spans
        merged = False
        for existing in filtered:
            if _spans_overlap(span, existing):
                # Overlap found: prefer specific risk-factor over generic "event"
                if span["label"] != "event" and existing["label"] == "event":
                    existing["label"] = span["label"]
                    existing["confidence"] = span["confidence"]
                merged = True
                break
        if not merged:
            filtered.append(span)

    for e in filtered:
        e["confidence"] = round(e["confidence"], 4)
    return sorted(filtered, key=lambda e: e["start"])


# ─────────────────────────────────────────────────────────────────────────────
# C) WATERFALL ANCHORING
# ─────────────────────────────────────────────────────────────────────────────

def build_event_location_pairs(entities, text):
    """Link events/risk-factors to locations using waterfall anchoring.

    Strategy (in order of priority):
      1. If the event's sentence contains location(s), link to those.
      2. Otherwise, inherit the most recently mentioned location
         ("waterfall" / "state machine" pattern common in news writing).
    """
    sentences = _sentence_spans(text)

    # Bucket entities by sentence
    events_by_sentence = {}   # si -> [entity]
    locations_by_sentence = {}

    for ent in entities:
        si = _sentence_index_for(ent["start"], sentences)
        if si < 0:
            continue
        if ent["label"] == "location":
            locations_by_sentence.setdefault(si, []).append(ent)
        elif ent["label"] in _EVENT_LABELS:
            events_by_sentence.setdefault(si, []).append(ent)

    # Walk sentences in order, maintaining the "active location" state
    active_location = None
    pairs = []
    seen = set()

    for si in range(len(sentences)):
        # Update active location if this sentence introduces one
        locs_here = locations_by_sentence.get(si, [])
        if locs_here:
            # Pick the highest-confidence location as the new anchor
            active_location = max(locs_here, key=lambda l: l["confidence"])

        events_here = events_by_sentence.get(si)
        if not events_here:
            continue

        # Determine which locations to link to
        if locs_here:
            # Same-sentence locations: link to all of them
            link_targets = locs_here
        elif active_location:
            # Waterfall: inherit the last-mentioned location
            link_targets = [active_location]
        else:
            continue

        for ev in events_here:
            for loc in link_targets:
                key = (ev["text"].lower(), loc["text"].lower(), ev["label"])
                if key in seen:
                    continue
                seen.add(key)
                pair = {
                    "event_text": ev["text"],
                    "event_label": ev["label"],
                    "event_confidence": ev["confidence"],
                    "location_text": loc["text"],
                    "location_confidence": loc["confidence"],
                    "event_start": ev["start"],
                    "event_end": ev["end"],
                    "location_start": loc["start"],
                    "location_end": loc["end"],
                }
                if loc not in locs_here:
                    pair["cross_sentence"] = True
                pairs.append(pair)

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKED EXTRACTION FOR LONG ARTICLES
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_text(text, max_chars=1500, overlap=200):
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


def extract_from_article(text, document_id=None,
                         source_url=None, publication_date=None):
    """Full pipeline: chunk -> NER -> merge -> deduplicate -> waterfall link."""
    chunks = _chunk_text(text)
    merged = {}  # label -> {text_lower: span_dict}

    for chunk_str, offset in chunks:
        raw = extract_entities(chunk_str)
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
                # Keep the highest-confidence occurrence
                if key not in merged[label] or adjusted["confidence"] > merged[label][key]["confidence"]:
                    merged[label][key] = adjusted

    merged_raw = {"entities": {label: list(spans.values()) for label, spans in merged.items()}}

    entities = flatten_entities(merged_raw)
    pairs = build_event_location_pairs(entities, text)

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

    print("=" * 70)
    print("NER EXTRACTION (deduplicated, per-label thresholds)")
    print("=" * 70)
    raw = extract_entities(article)
    entities = flatten_entities(raw)
    pairs = build_event_location_pairs(entities, article)

    print(f"\nFound {len(entities)} entities (after dedup + threshold filtering):")
    for e in entities:
        print(f"  [{e['label']:>35}]  {e['text']:<40}  conf={e['confidence']:.3f}  ({e['start']}-{e['end']})")

    print(f"\nFound {len(pairs)} event-location pairs:")
    for p in pairs:
        xsent = "  [cross-sentence]" if p.get("cross_sentence") else ""
        print(f"  {p['event_text']:<40} -> {p['location_text']:<15}  (label={p['event_label']}){xsent}")

    print("\n" + "=" * 70)
    print("FULL PIPELINE (chunked + merged)")
    print("=" * 70)
    result = extract_from_article(
        article,
        document_id="article-2026-multi-crisis",
        source_url="https://example.com/multi-crisis",
        publication_date="2026-03-05",
    )
    print(json.dumps(result, indent=2))

    with open("gliner2_extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Saved to 'gliner2_extraction_results.json'")
