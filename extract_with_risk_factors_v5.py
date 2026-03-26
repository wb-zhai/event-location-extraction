# /// script
# dependencies = [
#   "gliner2", "urllib3", "requests", "semantic-text-splitter",
#   "spacy",
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
# ]
# ///

"""
GLiNER2 Event-Location Extraction with Risk Factors — v5
=========================================================
Production-grade pipeline: Neural NER + Coreference Resolution +
Dependency Parsing + Gazetteer + Deterministic Linking.

Architecture:
  GLiNER2                  — zero-shot NER for entities and risk factors
  fastcoref (subprocess)   — coreference resolution via _coref_worker.py
                              (isolated env: transformers<4.40)
  spaCy en_core_web_sm     — dep parsing (source/dest) + PhraseMatcher
  semantic-text-splitter   — Rust sentence splitting + GLiNER chunking

Improvements over v4:
  1. Coreference resolution: "the region", "its", "the nation" resolved
     to antecedent locations BEFORE linking. Replaces fragile regex hack.
  2. Dep parsing: migration sentences ("fleeing FROM X TO Y") are parsed
     for source/destination, preventing the co-occurrence trap.
  3. Frequency-based major location: most-mentioned location, not first-seen.
  4. Gazetteer hybrid recall: PhraseMatcher catches known crisis terms that
     GLiNER misses at higher thresholds.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
import warnings
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import spacy
from spacy.matcher import PhraseMatcher
from gliner2 import GLiNER2
from semantic_text_splitter import TextSplitter

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

_gliner = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
_nlp = spacy.load("en_core_web_sm")

_COREF_WORKER = Path(__file__).parent / "_coref_worker.py"

# ─────────────────────────────────────────────────────────────────────────────
# RISK-FACTOR DEFINITIONS
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

_GLOBAL_THRESHOLD = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# GAZETTEER  (Improvement 4: hybrid recall via PhraseMatcher)
# ─────────────────────────────────────────────────────────────────────────────

CRISIS_GAZETTEER: dict[str, list[str]] = {
    "agricultural production issues": [
        "crop failure", "harvest failure", "failed crops", "bad harvests",
        "cattle death", "livestock death", "lack of cultivation",
    ],
    "conflicts and violence": [
        "civil war", "armed conflict", "bombing", "insurgency", "terrorism",
        "violent clashes", "fighting", "warfare", "siege", "airstrike",
    ],
    "economic issues": [
        "economic crisis", "hyperinflation", "poverty", "unemployment",
        "economic collapse", "price surge",
    ],
    "environmental issues": [
        "climate change", "deforestation", "environmental degradation",
    ],
    "food crisis": [
        "famine", "food crisis", "food insecurity", "mass hunger",
        "starvation", "malnutrition", "acute hunger",
    ],
    "forced displacement": [
        "displacement", "refugee crisis", "forced migration", "internally displaced",
        "refugee camp", "makeshift camp",
    ],
    "humanitarian aid": [
        "aid appeal", "humanitarian crisis", "food aid", "relief effort",
        "humanitarian access",
    ],
    "pests and diseases": [
        "cholera", "epidemic", "pandemic", "locust swarm", "plague",
        "disease outbreak", "rinderpest",
    ],
    "political instability": [
        "coup", "government collapse", "regime change", "political crisis",
        "martial law",
    ],
    "weather shocks": [
        "drought", "flood", "cyclone", "hurricane", "typhoon", "tornado",
        "heatwave", "frost", "rural collapse",
    ],
    "other": [
        "man-made disaster",
    ],
}

_phrase_matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
for _label, _terms in CRISIS_GAZETTEER.items():
    _patterns = [_nlp.make_doc(t) for t in _terms]
    _phrase_matcher.add(_label, _patterns)

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
    return _gliner.create_schema().entities(entity_types)

_NER_SCHEMA = _build_ner_schema()

# ─────────────────────────────────────────────────────────────────────────────
# TEXT SPLITTING  (semantic-text-splitter, Rust)
# ─────────────────────────────────────────────────────────────────────────────

_sentence_splitter = TextSplitter((1, 350), trim=False)
_chunk_splitter    = TextSplitter((200, 1500), trim=False)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    return [
        (offset, offset + len(chunk))
        for offset, chunk in _sentence_splitter.chunk_indices(text)
        if chunk.strip()
    ]


def chunk_spans(text: str) -> list[tuple[str, int]]:
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
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    text: str
    label: str
    confidence: float
    start: int
    end: int
    source: str = "gliner"

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
    link_method: str = "same_sentence"

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["cross_sentence"]:
            del d["cross_sentence"]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# 1. COREFERENCE RESOLUTION  (subprocess, isolated env)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_coreferences(text: str) -> dict[tuple[int, int], tuple[int, int, str]]:
    """Run fastcoref via subprocess. Returns mention -> antecedent map."""
    if not _COREF_WORKER.exists():
        return {}

    try:
        proc = subprocess.run(
            ["uv", "run", str(_COREF_WORKER)],
            input=json.dumps([text]),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            print(f"  [coref] worker failed: {proc.stderr[:200]}", file=sys.stderr)
            return {}

        results = json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"  [coref] error: {exc}", file=sys.stderr)
        return {}

    mention_to_antecedent: dict[tuple[int, int], tuple[int, int, str]] = {}
    for cluster in results[0]:
        offsets = cluster["offsets"]
        strings = cluster["strings"]
        if len(offsets) < 2:
            continue
        ant_start, ant_end = offsets[0]
        ant_text = strings[0]
        for (m_start, m_end), m_text in zip(offsets[1:], strings[1:]):
            mention_to_antecedent[(m_start, m_end)] = (ant_start, ant_end, ant_text)

    return mention_to_antecedent


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEPENDENCY PARSING: SOURCE vs DESTINATION
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_VERBS = frozenset({
    "flee", "escape", "uproot", "displace", "evacuate", "leave", "abandon",
    "expel", "deport", "drive", "force",
})
_DEST_VERBS = frozenset({
    "arrive", "seek", "shelter", "reach", "resettle", "move", "migrate",
    "relocate", "settle",
})


def _classify_location_role(
    loc_start: int, loc_end: int, sent_start: int, sent_end: int, text: str,
) -> str:
    """Use spaCy dep parsing to classify a location as SOURCE, DESTINATION, or LOCATION."""
    sent_text = text[sent_start:sent_end]
    doc = _nlp(sent_text)
    rel_start = loc_start - sent_start
    rel_end = loc_end - sent_start

    for token in doc:
        if not (token.idx >= rel_start and token.idx < rel_end):
            continue
        head = token
        while head.dep_ in ("compound", "flat"):
            head = head.head
        if head.dep_ == "pobj" and head.head.dep_ == "prep":
            prep = head.head
            prep_text = prep.text.lower()
            verb = prep.head
            while verb.pos_ != "VERB" and verb.head != verb:
                verb = verb.head
            lemma = verb.lemma_.lower()
            if prep_text == "from":
                return "SOURCE"
            elif prep_text in ("to", "into"):
                return "DESTINATION"
            elif prep_text == "in":
                if lemma in _SOURCE_VERBS:
                    return "SOURCE"
                elif lemma in _DEST_VERBS:
                    return "DESTINATION"
        break
    return "LOCATION"


# ─────────────────────────────────────────────────────────────────────────────
# CORE NER + GAZETTEER
# ─────────────────────────────────────────────────────────────────────────────

def _extract_raw(text: str) -> dict:
    return _gliner.extract(
        text, _NER_SCHEMA,
        threshold=_GLOBAL_THRESHOLD,
        include_spans=True,
        include_confidence=True,
    )


def _gazetteer_entities(text: str) -> list[dict]:
    doc = _nlp(text)
    matches = _phrase_matcher(doc)
    results = []
    for match_id, start, end in matches:
        span = doc[start:end]
        label = _nlp.vocab.strings[match_id]
        results.append({
            "text": span.text,
            "label": label,
            "confidence": 1.0,
            "start": span.start_char,
            "end": span.end_char,
            "source": "gazetteer",
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _spans_overlap(a: dict, b: dict) -> bool:
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def _flatten_and_dedup(raw: dict, gazetteer_spans: list[dict]) -> list[Entity]:
    all_spans: list[dict] = []
    for label, span_list in raw.get("entities", {}).items():
        for span in span_list:
            all_spans.append({**span, "label": label, "source": "gliner"})
    all_spans.extend(gazetteer_spans)

    all_spans.sort(key=lambda s: (s["start"], s["label"] == "event"))

    filtered: list[dict] = []
    for span in all_spans:
        if span.get("source") != "gazetteer":
            threshold = LABEL_THRESHOLDS.get(span["label"], _GLOBAL_THRESHOLD)
            if span["confidence"] < threshold:
                continue

        merged = False
        for existing in filtered:
            if _spans_overlap(span, existing):
                if span["label"] != "event" and existing["label"] == "event":
                    existing["label"] = span["label"]
                    existing["confidence"] = span["confidence"]
                    existing["source"] = span.get("source", "gliner")
                merged = True
                break
        if not merged:
            filtered.append(span)

    return [
        Entity(
            text=s["text"], label=s["label"],
            confidence=round(s["confidence"], 4),
            start=s["start"], end=s["end"],
            source=s.get("source", "gliner"),
        )
        for s in sorted(filtered, key=lambda s: s["start"])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# EVENT-LOCATION LINKING
# ─────────────────────────────────────────────────────────────────────────────

def _build_pairs(
    entities: list[Entity],
    text: str,
    coref_map: dict[tuple[int, int], tuple[int, int, str]],
) -> list[EventLocationPair]:
    """Link events to locations.

    Priority:
      1. Same-sentence: closest location by char distance, with dep-parse
         source/destination check.
      2. Coreference: if the sentence has a mention that resolves to a
         location via coref, use the antecedent.
      3. Macro-scope fallback: "the nation" etc. -> frequency-based major location.
      4. Waterfall: inherit the most recently mentioned location.
    """
    sentences = sentence_spans(text)

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

    # Frequency-based major location
    loc_counts: Counter[str] = Counter()
    for locs in locations_by_sent.values():
        for loc in locs:
            loc_counts[loc.text.lower()] += 1

    loc_by_text: dict[str, Entity] = {}
    for locs in locations_by_sent.values():
        for loc in locs:
            key = loc.text.lower()
            if key not in loc_by_text or loc.confidence > loc_by_text[key].confidence:
                loc_by_text[key] = loc

    major_location: Optional[Entity] = None
    if loc_counts:
        major_text = loc_counts.most_common(1)[0][0]
        major_location = loc_by_text.get(major_text)

    # Coref-resolved location index
    coref_locations_by_sent: dict[int, list[Entity]] = {}
    for (m_start, m_end), (ant_start, ant_end, ant_text) in coref_map.items():
        for loc_ent in loc_by_text.values():
            if (loc_ent.start >= ant_start and loc_ent.end <= ant_end) or \
               (ant_start >= loc_ent.start and ant_end <= loc_ent.end):
                si = _sentence_index_for(m_start, sentences)
                if si >= 0:
                    coref_locations_by_sent.setdefault(si, []).append(loc_ent)
                break

    # Walk sentences
    active_location: Optional[Entity] = None
    pairs: list[EventLocationPair] = []
    seen: set[tuple[str, str, str]] = set()

    for si in range(len(sentences)):
        locs_here = locations_by_sent.get(si, [])
        coref_locs = coref_locations_by_sent.get(si, [])

        if locs_here:
            active_location = max(locs_here, key=lambda l: l.confidence)

        events_here = events_by_sent.get(si)
        if not events_here:
            continue

        sent_start, sent_end = sentences[si]

        if locs_here:
            for ev in events_here:
                sorted_locs = sorted(locs_here, key=lambda loc: abs(loc.start - ev.start))
                for loc in sorted_locs:
                    role = _classify_location_role(
                        loc.start, loc.end, sent_start, sent_end, text,
                    )
                    if role == "DESTINATION" and ev.label in _EVENT_LABELS:
                        ev_role = _classify_location_role(
                            ev.start, ev.end, sent_start, sent_end, text,
                        )
                        if ev_role == "SOURCE":
                            continue
                    _emit(pairs, seen, ev, loc, False, "same_sentence")
                    break
        elif coref_locs:
            best_coref = max(coref_locs, key=lambda l: l.confidence)
            for ev in events_here:
                _emit(pairs, seen, ev, best_coref, True, "coref")
        elif _has_macro_scope(text[sent_start:sent_end]) and major_location:
            for ev in events_here:
                _emit(pairs, seen, ev, major_location, True, "coref")
        elif active_location:
            for ev in events_here:
                _emit(pairs, seen, ev, active_location, True, "waterfall")

    return pairs


_MACRO_SCOPE_RE = re.compile(
    r"\bthe\s+(?:nation|country|countries|province|provinces|region|regions"
    r"|republic|territory|territories|state|homeland)\b",
    re.IGNORECASE,
)


def _has_macro_scope(sent_text: str) -> bool:
    return bool(_MACRO_SCOPE_RE.search(sent_text))


def _emit(
    pairs: list[EventLocationPair], seen: set,
    ev: Entity, loc: Entity, cross_sentence: bool, link_method: str,
) -> None:
    key = (ev.text.lower(), loc.text.lower(), ev.label)
    if key in seen:
        return
    seen.add(key)
    pairs.append(EventLocationPair(
        event_text=ev.text, event_label=ev.label,
        event_confidence=ev.confidence,
        location_text=loc.text, location_confidence=loc.confidence,
        event_start=ev.start, event_end=ev.end,
        location_start=loc.start, location_end=loc.end,
        cross_sentence=cross_sentence, link_method=link_method,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKED EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _merge_chunks(text: str) -> dict:
    merged: dict[str, dict[str, dict]] = {}
    for chunk_str, offset in chunk_spans(text):
        raw = _extract_raw(chunk_str)
        for label, spans in raw.get("entities", {}).items():
            if label not in merged:
                merged[label] = {}
            for span in spans:
                key = span["text"].lower()
                adjusted = {**span, "start": span["start"] + offset, "end": span["end"] + offset}
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

    1. Coreference resolution (fastcoref, subprocess)
    2. Chunked GLiNER2 NER
    3. Gazetteer phrase matching (spaCy PhraseMatcher)
    4. Entity deduplication + per-label threshold filtering
    5. Event-location linking (dep-parse + coref + waterfall)
    """
    coref_map = _resolve_coreferences(text)
    merged_raw = _merge_chunks(text)
    gaz_spans = _gazetteer_entities(text)
    entities = _flatten_and_dedup(merged_raw, gaz_spans)
    pairs = _build_pairs(entities, text, coref_map)

    return {
        "document": {
            "id": document_id or str(uuid.uuid4()),
            "source_url": source_url,
            "publication_date": publication_date,
            "text_length": len(text),
        },
        "entities": [e.to_dict() for e in entities],
        "event_location_pairs": [p.to_dict() for p in pairs],
        "coreference_clusters": [
            {"mention": text[ms:me], "antecedent": ant_text, "mention_offsets": [ms, me]}
            for (ms, me), (_, _, ant_text) in coref_map.items()
        ],
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

    result = extract(
        article,
        document_id="article-2026-multi-crisis",
        source_url="https://example.com/multi-crisis",
        publication_date="2026-03-05",
    )

    entities = result["entities"]
    pairs = result["event_location_pairs"]
    coref = result["coreference_clusters"]

    print("=" * 74)
    print(f"COREFERENCE CLUSTERS  ({len(coref)} resolved mentions)")
    print("=" * 74)
    for c in coref:
        print(f"  '{c['mention']}'  ->  '{c['antecedent']}'")

    print(f"\n{'=' * 74}")
    print(f"ENTITIES  ({len(entities)} total)")
    print("=" * 74)
    for e in entities:
        src = f"  [{e['source']}]" if e["source"] != "gliner" else ""
        print(f"  [{e['label']:>35}]  {e['text']:<42} conf={e['confidence']:.3f}  ({e['start']}-{e['end']}){src}")

    print(f"\n{'=' * 74}")
    print(f"EVENT-LOCATION PAIRS  ({len(pairs)} total)")
    print("=" * 74)
    for p in pairs:
        flags = []
        if p.get("cross_sentence"):
            flags.append("cross-sent")
        flags.append(p["link_method"])
        flag_str = f"  [{', '.join(flags)}]"
        print(f"  {p['event_text']:<42} -> {p['location_text']:<14} ({p['event_label']}){flag_str}")

    print(f"\n{'=' * 74}")
    print("SENTENCE BOUNDARIES")
    print("=" * 74)
    for i, (s, e) in enumerate(sentence_spans(article)):
        print(f"  S{i:2d}  [{s:4d}-{e:4d}]  {article[s:e].strip()[:90]!r}")

    with open("gliner2_extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Saved to 'gliner2_extraction_results.json'")
