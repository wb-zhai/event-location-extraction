# /// script
# dependencies = [
#   "gliner2", "urllib3", "requests"
# ]
# ///

"""
GLiNER2 Event-Location Extraction Schema for News Articles
===========================================================
Production-ready schema definitions for extracting linked event→location
pairs from news articles. Fixed and optimized for zero-shot semantic mapping.
"""

import re
import uuid
import json
from typing import Optional
from gliner2 import GLiNER2

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 1: COMBINED MULTI-TASK SCHEMA
# ═════════════════════════════════════════════════════════════════════════════
# Descriptions drastically condensed to strong keywords to prevent 
# embedding dilution and drastically increase GLiNER2's precision.

combined_schema = (
    extractor.create_schema()

    # ── Named Entity Recognition ──
    .entities({
        "event": "Any discrete event, sudden incident, or specific occurrence (e.g., earthquake, attack, election).",
        "location": "Any geographic location, place, city, country, region, facility, or venue.",
        "date": "A temporal expression including specific dates or relative times.",
        "organization": "An organization, government body, agency, or institution.",
        
        # ── Risk Factor Clusters (Optimized for Embeddings) ──
        "agricultural production issues": "Agricultural impairment, failed crops, bad harvests, lack of cultivation, livestock or cattle death, and agricultural infrastructure damage.",
        "conflicts and violence": "Warfare, fighting, civil strife, rebel insurgencies, militia groups, warlords, banditry, piracy, air attacks, sieges, bombings, terrorism, and violent suppression.",
        "economic issues": "Economic crises, collapsing economy, disrupted trade, external debt, brain drain, rising inflation, drastic price rises, and widespread poverty.",
        "environmental issues": "Ecological crises, climate change, greenhouse gases, environmental degradation, and natural disasters.",
        "food crisis": "Food insecurity, mass hunger, acute hunger, starvation, malnutrition, dehydration, and related gastrointestinal diseases or infant mortality.",
        "forced displacement": "Forced migration, displaced individuals, vulnerable populations fleeing homes, asylum seekers, refugees, and makeshift camps.",
        "humanitarian aid": "Humanitarian disasters, aid appeals, food assistance, international embargoes, restricted humanitarian access, withheld or stolen relief, and reliance on self-reliance.",
        "land-related issues": "Land invasions, hostile land grabs, systemic land seizures, burning houses, failed land reform, land degradation, poor soil quality, and destroyed forests.",
        "pests and diseases": "Swarms of locusts, agricultural pests, potato blight, epidemics, cholera outbreaks, and animal diseases like rinderpest or cattle plague.",
        "political instability": "Collapse of government, lack of authority, corruption, mismanagement, power struggles, secession, coup d'etat, violent overthrows, dictatorships, and strict military juntas.",
        "weather shocks": "Climatic hazards, weather extremes, destructive floods, cyclones, prolonged dry spells, harsh droughts, and abnormally low rainfall.",
        "other": "Uncategorized catastrophes, man-made disasters, severe population crises, and overarching tragedies."
    })

    # ── Relation Extraction ──
    # Expanded prompt so the model knows it can link ANY risk factor
    .relations({
        "located_in": "Connects ANY event, disaster, crisis, issue, or risk factor to the specific geographic location (city, region, country) where it occurred.",
        "originated_from": "Connects an event or crisis to the location where it started.",
        "impacted": "Connects an event or crisis to the infrastructure, economy, or population it affected.",
    })

    # ── Structured JSON Extraction ──
    .structure("events_list")
    .field("specific_event", dtype="str", description="Name of the specific incident (e.g., 'cattle death', 'gastrointestinal diseases')")
    .field("event_category", dtype="str", choices=["natural_disaster", "armed_conflict", "political", "economic", "social_unrest", "health", "agriculture", "other"])
    .field("exact_location", dtype="str", description="The EXACT city, region, or country where THIS specific event occurred")
    .field("severity", dtype="str", choices=["low", "medium", "high", "critical"])
    
    # ── Article-Level Classification ──
    .classification("news_section", ["world", "politics", "business", "science", "sports", "environment"])
)

def extract_combined(text: str, threshold: float = 0.3) -> dict:
    return extractor.extract(text, combined_schema, threshold=threshold)


# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 2: STRUCTURED JSON ONLY (BEST FOR EXACT LINKING)
# ═════════════════════════════════════════════════════════════════════════════
# FIXED syntax: GLiNER2 extract_json absolutely requires the double-colon string 
# syntax to parse field specifications.

STRUCTURED_JSON_SCHEMA = {
    "events":[
        "event_description::str::Name or description of the specific incident (e.g., 'cattle death', 'food insecurity')",
        "event_category::str::Category of the event (e.g. natural_disaster, health, agriculture, conflict)",
        "exact_location::str::The EXACT city, region, or country where THIS event happened",
        "date::str::When the event occurred",
        "severity::str::Severity level (low, medium, high, critical)"
    ]
}

def extract_structured(text: str, threshold: float = 0.3) -> dict:
    return extractor.extract_json(text, STRUCTURED_JSON_SCHEMA, threshold=threshold)


# ═════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING: NORMALIZE OUTPUT TO STANDARD EVENT-LOCATION SCHEMA
# ═════════════════════════════════════════════════════════════════════════════

def find_span_offsets(text: str, mention: str) -> Optional[tuple[int, int]]:
    match = re.search(re.escape(mention), text, re.IGNORECASE)
    if match:
        return (match.start(), match.end())
    return None

def _fuzzy_match_entity(mention: str, entity_text_to_id: dict) -> Optional[str]:
    """Handles slight span mismatches between NER and Relation Extraction."""
    m_lower = mention.lower()
    # 1. Exact match
    if m_lower in entity_text_to_id:
        return entity_text_to_id[m_lower]
    # 2. Substring fallback
    for ent, eid in entity_text_to_id.items():
        if m_lower in ent or ent in m_lower:
            return eid
    return None

def normalize_combined_output(
    raw_output: dict, source_text: str, document_id: str = None, 
    source_url: str = None, publication_date: str = None,
) -> dict:
    doc_id = document_id or str(uuid.uuid4())
    entities =[]
    entity_text_to_id = {}

    for label, mentions in raw_output.get("entities", {}).items():
        for mention in mentions:
            if mention.lower() in entity_text_to_id: 
                continue 
            eid = f"e{len(entities) + 1}"
            offsets = find_span_offsets(source_text, mention)
            entities.append({
                "id": eid,
                "text": mention,
                "label": label.upper(),
                "start": offsets[0] if offsets else None,
                "end": offsets[1] if offsets else None,
            })
            entity_text_to_id[mention.lower()] = eid

    event_location_pairs =[]
    for rel_type, pairs in raw_output.get("relation_extraction", {}).items():
        for head, tail in pairs:
            head_id = _fuzzy_match_entity(head, entity_text_to_id)
            tail_id = _fuzzy_match_entity(tail, entity_text_to_id)
            if head_id and tail_id:
                event_location_pairs.append({
                    "event_id": head_id,
                    "location_id": tail_id,
                    "relation_type": rel_type,
                })

    return {
        "document": {"id": doc_id, "source_url": source_url, "publication_date": publication_date, "text_length": len(source_text)},
        "entities": entities,
        "event_location_pairs": event_location_pairs,
        "event_records": raw_output.get("events_list",[]),
        "article_classification": raw_output.get("news_section"),
    }

# ═════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING FOR LONG ARTICLES
# ═════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, max_chars: int = 1500, overlap_chars: int = 200) -> list[tuple[str, int]]:
    chunks =[]
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            last_period = text.rfind('. ', start, end)
            if last_period > start + max_chars // 2:
                end = last_period + 2
        chunks.append((text[start:end], start))
        if end >= len(text): break
        start = end - overlap_chars
    return chunks

def extract_from_long_article(
    text: str, threshold: float = 0.3, document_id: str = None, 
    source_url: str = None, publication_date: str = None,
) -> dict:
    chunks = chunk_text(text)
    all_entities = {}
    all_relations = {}
    all_records =[]
    article_section = None

    for chunk_text_str, offset in chunks:
        result = extract_combined(chunk_text_str, threshold=threshold)
        for label, mentions in result.get("entities", {}).items():
            if label not in all_entities: all_entities[label] = set()
            all_entities[label].update(mentions)

        for rel_type, pairs in result.get("relation_extraction", {}).items():
            if rel_type not in all_relations: all_relations[rel_type] = set()
            all_relations[rel_type].update((h, t) for h, t in pairs)

        all_records.extend(result.get("events_list",[]))
        if not article_section: article_section = result.get("news_section")

    merged = {
        "entities": {k: list(v) for k, v in all_entities.items()},
        "relation_extraction": {k: list(v) for k, v in all_relations.items()},
        "events_list": all_records,
        "news_section": article_section,
    }

    return normalize_combined_output(merged, text, document_id, source_url, publication_date)

# ═════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
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

    all_outputs = {}

    print("=" * 60)
    print("APPROACH 2: Structured JSON Extraction (BEST FOR EXACT MAPPING)")
    print("=" * 60)
    result_structured = extract_structured(article, threshold=0.3)
    print(json.dumps(result_structured, indent=2))
    all_outputs["approach_2_structured"] = result_structured

    print("\n" + "=" * 60)
    print("FULL PIPELINE: Normalized Output (Graphs & JSON)")
    print("=" * 60)
    normalized = extract_from_long_article(
        article, threshold=0.3, document_id="article-2026",
        source_url="https://example.com/multi-crisis", publication_date="2026-03-05",
    )
    print(json.dumps(normalized, indent=2))
    all_outputs["full_pipeline_normalized"] = normalized

    with open("gliner2_extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Outputs securely saved to 'gliner2_extraction_results.json'")