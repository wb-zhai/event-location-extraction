# /// script
# dependencies = [
#   "gliner2", "urllib3", "requests", "semantic-text-splitter"
# ]
# ///

"""
GLiNER2 Event-Location Extraction with Risk Factors — v4
=========================================================
Production-ready pipeline that extracts event/risk-factor entities and
geographic locations from news text, then links them via waterfall
anchoring with character-distance disambiguation.

Architecture:
  - GLiNER2 for zero-shot NER (what it does best).
  - semantic-text-splitter (Rust) for sentence splitting + GLiNER chunking.
  - Python logic for entity linking (what GLiNER is bad at).

Improvements over v3:
  - Replaced pysbd with semantic-text-splitter (zero deps, Rust-fast,
    native character offsets via chunk_indices()).
  - Character-distance anchoring: when a sentence has multiple locations,
    each event links to the closest location, not all of them.
  - Scope/scale awareness: phrases like "the nation" or "the country"
    trigger a fallback to the last province/country-level location
    instead of the last micro-location.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

from gliner2 import GLiNER2
from semantic_text_splitter import TextSplitter

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

# ─────────────────────────────────────────────────────────────────────────────
# RISK-FACTOR ENTITY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

RISK_FACTOR_ENTITIES: dict[str, str] = {
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

_EVENT_LABELS = {"event"} | set(RISK_FACTOR_ENTITIES)

# ─────────────────────────────────────────────────────────────────────────────
# PER-LABEL CONFIDENCE THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

LABEL_THRESHOLDS: dict[str, float] = {
    "location":     0.80,
    "event":        0.50,
    "date":         0.50,
    "organization": 0.60,
    **{k: 0.40 for k in RISK_FACTOR_ENTITIES},
}

_GLOBAL_THRESHOLD = 0.30  # floor passed to GLiNER; per-label filtering is post-hoc

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

_NER_SCHEMA = _build_ner_schema()

# ─────────────────────────────────────────────────────────────────────────────
# TEXT SPLITTING  (semantic-text-splitter, Rust)
# ─────────────────────────────────────────────────────────────────────────────
# Two splitters:
#   _sentence_splitter — for co-occurrence linking (sentence-level)
#   _chunk_splitter    — for GLiNER input batching (≤ 1 500 chars)

_sentence_splitter = TextSplitter((1, 350), trim=False)
_chunk_splitter    = TextSplitter((200, 1500), trim=False)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) for each sentence. Skips empty segments."""
    spans = []
    for offset, chunk in _sentence_splitter.chunk_indices(text):
        if chunk.strip():
            spans.append((offset, offset + len(chunk)))
    return spans


def chunk_spans(text: str) -> list[tuple[str, int]]:
    """Return (chunk_text, offset) tuples for GLiNER batching."""
    return [
        (chunk, offset)
        for offset, chunk in _chunk_splitter.chunk_indices(text)
        if chunk.strip()
    ]


def _sentence_index_for(char_start: int, spans: list[tuple[int, int]]) -> int:
    for i, (s, e) in enumerate(spans):
        if s <= char_start < e:
            return i
    return -1

# ─────────────────────────────────────────────────────────────────────────────
# SCOPE / SCALE AWARENESS
# ─────────────────────────────────────────────────────────────────────────────
# If a sentence contains a macro-scope phrase ("the nation", "the country",
# "the province") but no named location, we should anchor to the last
# *major* location rather than the last micro-location (city/village).
#
# We keep a simple two-tier location state: "major" (province/country/region
# mentioned first) and "local" (most recent of any kind).

_MACRO_SCOPE_RE = re.compile(
    r"\bthe\s+(?:nation|country|countries|province|provinces|region|regions"
    r"|republic|territory|territories|state|homeland)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    text: str
    label: str
    confidence: float
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EventLocationPair:
    event_text: str
    event_label: str
    event_confidence: float
    location_text: str
    location_confidence: float
    event_start: int
    event_end: int
    location_start: int
    location_end: int
    cross_sentence: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["cross_sentence"]:
            del d["cross_sentence"]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# CORE NER
# ─────────────────────────────────────────────────────────────────────────────

def _extract_raw(text: str) -> dict:
    """Run GLiNER2 NER with spans + confidence."""
    return extractor.extract(
        text, _NER_SCHEMA,
        threshold=_GLOBAL_THRESHOLD,
        include_spans=True,
        include_confidence=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION  (Improvement A from v3, carried forward)
# ─────────────────────────────────────────────────────────────────────────────

def _spans_overlap(a: dict, b: dict) -> bool:
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def _flatten_and_dedup(raw: dict) -> list[Entity]:
    """Deduplicate overlapping spans: prefer risk-factor over generic 'event'.
    Apply per-label confidence thresholds."""
    all_spans: list[dict] = []
    for label, span_list in raw.get("entities", {}).items():
        for span in span_list:
            all_spans.append({**span, "label": label})

    # Sort: by position, then specific labels before generic "event"
    all_spans.sort(key=lambda s: (s["start"], s["label"] == "event"))

    filtered: list[dict] = []
    for span in all_spans:
        threshold = LABEL_THRESHOLDS.get(span["label"], _GLOBAL_THRESHOLD)
        if span["confidence"] < threshold:
            continue

        merged = False
        for existing in filtered:
            if _spans_overlap(span, existing):
                if span["label"] != "event" and existing["label"] == "event":
                    existing["label"] = span["label"]
                    existing["confidence"] = span["confidence"]
                merged = True
                break
        if not merged:
            filtered.append(span)

    return [
        Entity(
            text=s["text"],
            label=s["label"],
            confidence=round(s["confidence"], 4),
            start=s["start"],
            end=s["end"],
        )
        for s in sorted(filtered, key=lambda s: s["start"])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# EVENT-LOCATION LINKING  (waterfall + closest-distance + scope awareness)
# ─────────────────────────────────────────────────────────────────────────────

def _build_pairs(entities: list[Entity], text: str) -> list[EventLocationPair]:
    """Link events to locations.

    Priority:
      1. Same-sentence locations — pick the closest one by character distance.
      2. Macro-scope fallback — if the sentence contains "the nation" etc.,
         use the last major (first-mentioned) location.
      3. Waterfall — inherit the most recently mentioned location.
    """
    sentences = sentence_spans(text)

    # Bucket entities by sentence
    events_by_sent: dict[int, list[Entity]] = {}
    locations_by_sent: dict[int, list[Entity]] = {}

    for ent in entities:
        si = _sentence_index_for(ent.start, sentences)
        if si < 0:
            continue
        if ent.label == "location":
            locations_by_sent.setdefault(si, []).append(ent)
        elif ent.label in _EVENT_LABELS:
            events_by_sent.setdefault(si, []).append(ent)

    # Track two tiers of location state
    major_location: Optional[Entity] = None   # first location seen (usually province/country)
    active_location: Optional[Entity] = None  # most recently mentioned

    pairs: list[EventLocationPair] = []
    seen: set[tuple[str, str, str]] = set()

    for si in range(len(sentences)):
        locs_here = locations_by_sent.get(si, [])

        # Update location state
        if locs_here:
            best = max(locs_here, key=lambda l: l.confidence)
            if major_location is None:
                major_location = best
            active_location = best

        events_here = events_by_sent.get(si)
        if not events_here:
            continue

        # Determine link targets
        sent_start, sent_end = sentences[si]
        sent_text = text[sent_start:sent_end]

        if locs_here:
            # Same-sentence: closest location per event (character distance)
            for ev in events_here:
                closest = min(locs_here, key=lambda loc: abs(loc.start - ev.start))
                _emit(pairs, seen, ev, closest, cross_sentence=False)
        elif _MACRO_SCOPE_RE.search(sent_text) and major_location:
            # Scope fallback: "the nation" → first-mentioned major location
            for ev in events_here:
                _emit(pairs, seen, ev, major_location, cross_sentence=True)
        elif active_location:
            # Waterfall: inherit last-mentioned location
            for ev in events_here:
                _emit(pairs, seen, ev, active_location, cross_sentence=True)

    return pairs


def _emit(
    pairs: list[EventLocationPair],
    seen: set,
    ev: Entity,
    loc: Entity,
    cross_sentence: bool,
) -> None:
    key = (ev.text.lower(), loc.text.lower(), ev.label)
    if key in seen:
        return
    seen.add(key)
    pairs.append(EventLocationPair(
        event_text=ev.text,
        event_label=ev.label,
        event_confidence=ev.confidence,
        location_text=loc.text,
        location_confidence=loc.confidence,
        event_start=ev.start,
        event_end=ev.end,
        location_start=loc.start,
        location_end=loc.end,
        cross_sentence=cross_sentence,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKED EXTRACTION  (for long articles)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_chunks(text: str) -> dict:
    """Run NER on each chunk, stitch spans back to document-level offsets,
    keeping the highest-confidence occurrence of each entity."""
    merged: dict[str, dict[str, dict]] = {}  # label -> {text_lower: span}

    for chunk_str, offset in chunk_spans(text):
        raw = _extract_raw(chunk_str)
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

    return {"entities": {label: list(spans.values()) for label, spans in merged.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def extract(
    text: str,
    *,
    document_id: str | None = None,
    source_url: str | None = None,
    publication_date: str | None = None,
) -> dict:
    """Full extraction pipeline.

    Returns a dict with:
      - document   : metadata
      - entities   : deduplicated entity list with offsets + confidence
      - event_location_pairs : linked pairs with cross_sentence flag
    """
    merged_raw = _merge_chunks(text)
    entities = _flatten_and_dedup(merged_raw)
    pairs = _build_pairs(entities, text)

    return {
        "document": {
            "id": document_id or str(uuid.uuid4()),
            "source_url": source_url,
            "publication_date": publication_date,
            "text_length": len(text),
        },
        "entities": [e.to_dict() for e in entities],
        "event_location_pairs": [p.to_dict() for p in pairs],
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

    # ── Pretty-print entities and pairs ──
    result = extract(
        article,
        document_id="article-2026-multi-crisis",
        source_url="https://example.com/multi-crisis",
        publication_date="2026-03-05",
    )

    entities = result["entities"]
    pairs = result["event_location_pairs"]

    print("=" * 70)
    print(f"ENTITIES  ({len(entities)} total)")
    print("=" * 70)
    for e in entities:
        print(f"  [{e['label']:>35}]  {e['text']:<40}  conf={e['confidence']:.3f}  ({e['start']}-{e['end']})")

    print(f"\n{'=' * 70}")
    print(f"EVENT-LOCATION PAIRS  ({len(pairs)} total)")
    print("=" * 70)
    for p in pairs:
        xsent = "  [cross-sentence]" if p.get("cross_sentence") else ""
        print(f"  {p['event_text']:<40} -> {p['location_text']:<15}  (label={p['event_label']}){xsent}")

    # ── Sentence debug ──
    print(f"\n{'=' * 70}")
    print("SENTENCE BOUNDARIES")
    print("=" * 70)
    for i, (s, e) in enumerate(sentence_spans(article)):
        print(f"  S{i:2d}  [{s:4d}-{e:4d}]  {article[s:e].strip()[:90]!r}")

    # ── Save JSON ──
    print(f"\n{'=' * 70}")
    print("FULL JSON OUTPUT")
    print("=" * 70)
    print(json.dumps(result, indent=2))

    with open("gliner2_extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Saved to 'gliner2_extraction_results.json'")
