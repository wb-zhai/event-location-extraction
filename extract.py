# /// script
# dependencies =[
#   "gliner2","urllib3","requests"
# ]
# ///

"""
GLiNER2 Event-Location Extraction Schema for News Articles
===========================================================

Production-ready schema definitions for extracting linked event→location
pairs from news articles using GLiNER2's schema-driven API.

Based on:
  - GLiNER2 paper (Zaratiana et al., EMNLP 2025 Demos, arXiv:2507.18546)
  - GLiNER2 library v1.2.3+ (pip install gliner2)
  - Model: fastino/gliner2-large-v1 (340M params, Apache 2.0)

Three approaches provided (in order of recommendation):
  1. COMBINED MULTI-TASK SCHEMA — entities + relations + structured JSON in one pass
  2. STRUCTURED JSON SCHEMA — extract_json for inherently linked event-location records
  3. RELATION-ONLY SCHEMA — extract_relations for graph-style output
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
# APPROACH 1: COMBINED MULTI-TASK SCHEMA (RECOMMENDED)
# ═════════════════════════════════════════════════════════════════════════════

combined_schema = (
    extractor.create_schema()

    # ── Named Entity Recognition ──
    .entities({
       "event": (
       "Any event, incident, process, or phenomenon mentioned in the news article — including both discrete occurrences and ongoing or long-term developments. Extract ALL events: sudden incidents (e.g. earthquakes, attacks, sport events, elections) as well as structural or gradual phenomena (e.g. climate change, rising temperatures, economic recession, demographic shifts). "
            ),
        "location": (
            "Any geographic location, place, city, country, region, facility, venue, "
            "or address where an event occurs. Extract ALL locations."
        ),
        "date": "A temporal expression including specific dates or relative times.",
        "organization": "An organization, government body, agency, or institution.",
    })

    # ── Relation Extraction ──
    .relations({
        "occurred_at": "An event took place at or happened in a specific geographic location.",
        "originated_from": "An event started from or originated in a specific location.",
        "impacted": "An event affected, damaged, or had consequences for a specific location.",
    })

    # ── Structured JSON Extraction ──
    .structure("events_list")
    .field("event_name", dtype="str", description="Name or short description of the event")
    .field("event_type", dtype="str", choices=[
        "natural_disaster", "armed_conflict", "political", "economic", 
        "social_unrest", "accident", "ceremony", "health", "legal", "other"
    ])
    .field("location_name", dtype="str", description="City, region, or country where THIS event occurred")
    .field("location_type", dtype="str", choices=["city", "country", "region", "facility", "venue", "other"])
    .field("severity", dtype="str", choices=["low", "medium", "high", "critical"])
    
    # ── Article-Level Classification ──
    .classification("news_section",["world", "politics", "business", "science", "sports", "environment"])
)

def extract_combined(text: str, threshold: float = 0.3) -> dict:
    """Run the full combined extraction pipeline. Lowered threshold to 0.3 to ensure multiple captures."""
    return extractor.extract(text, combined_schema, threshold=threshold)


# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 2: STRUCTURED JSON ONLY (SIMPLEST FOR LINKED PAIRS)
# ═════════════════════════════════════════════════════════════════════════════

# The schema MUST be a dictionary. GLiNER2 will automatically extract 
# an array of these objects if multiple events are found in the text.
STRUCTURED_JSON_SCHEMA = {
    "events":[
        "event_name::str::Name or brief description of the news event",
        "event_type::[natural_disaster|armed_conflict|political|social_unrest|accident|other]::str::Category",
        "location::str::Geographic location (city/country) where THIS specific event occurred",
        "date::str::When the event occurred",
        "severity::[low|medium|high|critical]::str::Significance level"
    ]
}

def extract_structured(text: str, threshold: float = 0.3) -> dict:
    """Extract event-location pairs as structured JSON records."""
    return extractor.extract_json(text, STRUCTURED_JSON_SCHEMA, threshold=threshold)

# ═════════════════════════════════════════════════════════════════════════════
# APPROACH 3: RELATION EXTRACTION ONLY (FOR KNOWLEDGE GRAPHS)
# ═════════════════════════════════════════════════════════════════════════════

RELATION_SCHEMA_WITH_DESCRIPTIONS = {
    "occurred_at": "An event took place at, happened in, or struck a specific geographic location",
    "originated_from": "An event started from, originated in, or was launched from a specific location",
    "impacted": "An event affected, damaged, or had consequences for a specific location",
    "spread_to": "An event expanded to, spread to, or reached an additional geographic area",
    "evacuated_to": "People or resources were moved to a destination location as a result of an event",
}

def extract_relations_only(text: str) -> dict:
    """Extract event-location relation tuples for graph construction."""
    schema = extractor.create_schema().relations(RELATION_SCHEMA_WITH_DESCRIPTIONS)
    return extractor.extract(text, schema)

def extract_relations_simple(text: str) -> dict:
    """Quick relation extraction with label names only."""
    return extractor.extract_relations(
        text,["occurred_at", "originated_from", "impacted", "spread_to"]
    )


# ═════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING: NORMALIZE OUTPUT TO STANDARD EVENT-LOCATION SCHEMA
# ═════════════════════════════════════════════════════════════════════════════

def find_span_offsets(text: str, mention: str) -> Optional[tuple[int, int]]:
    """Find character start/end offsets for a mention in the source text."""
    match = re.search(re.escape(mention), text, re.IGNORECASE)
    if match:
        return (match.start(), match.end())
    return None

def normalize_combined_output(
    raw_output: dict,
    source_text: str,
    document_id: str = None,
    source_url: str = None,
    publication_date: str = None,
) -> dict:
    """
    Transform GLiNER2 combined extraction output into the canonical
    event-location JSON schema with character offsets and IDs.
    """
    doc_id = document_id or str(uuid.uuid4())

    entities =[]
    entity_text_to_id = {}

    for label, mentions in raw_output.get("entities", {}).items():
        for mention in mentions:
            if mention.lower() in entity_text_to_id: 
                continue # prevent duplicates
            eid = f"e{len(entities) + 1}"
            offsets = find_span_offsets(source_text, mention)
            entity = {
                "id": eid,
                "text": mention,
                "label": label.upper(),
                "start": offsets[0] if offsets else None,
                "end": offsets[1] if offsets else None,
            }
            entities.append(entity)
            entity_text_to_id[mention.lower()] = eid

    event_location_pairs =[]
    for rel_type, pairs in raw_output.get("relation_extraction", {}).items():
        for head, tail in pairs:
            head_id = entity_text_to_id.get(head.lower())
            tail_id = entity_text_to_id.get(tail.lower())
            if head_id and tail_id:
                event_location_pairs.append({
                    "event_id": head_id,
                    "location_id": tail_id,
                    "relation_type": rel_type,
                })

    # Fetch "events_list" (matches the approach 1 schema name)
    event_records = raw_output.get("events_list",[])

    return {
        "document": {
            "id": doc_id,
            "source_url": source_url,
            "publication_date": publication_date,
            "text_length": len(source_text),
        },
        "entities": entities,
        "event_location_pairs": event_location_pairs,
        "event_records": event_records,
        "article_classification": raw_output.get("news_section"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING FOR LONG ARTICLES
# ═════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    max_chars: int = 1500,   
    overlap_chars: int = 200  
) -> list[tuple[str, int]]:
    """Split text into overlapping chunks for GLiNER2 processing."""
    chunks =[]
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
        start = end - overlap_chars
    return chunks

def extract_from_long_article(
    text: str,
    threshold: float = 0.3,
    document_id: str = None,
    source_url: str = None,
    publication_date: str = None,
) -> dict:
    """Full pipeline for long news articles: chunk → extract → merge → normalize."""
    chunks = chunk_text(text)

    all_entities = {}
    all_relations = {}
    all_records =[]
    article_section = None

    for chunk_text_str, offset in chunks:
        result = extract_combined(chunk_text_str, threshold=threshold)

        for label, mentions in result.get("entities", {}).items():
            if label not in all_entities:
                all_entities[label] = set()
            all_entities[label].update(mentions)

        for rel_type, pairs in result.get("relation_extraction", {}).items():
            if rel_type not in all_relations:
                all_relations[rel_type] = set()
            all_relations[rel_type].update((h, t) for h, t in pairs)

        # Merge structural records
        all_records.extend(result.get("events_list",[]))

        if not article_section:
            article_section = result.get("news_section")

    merged = {
        "entities": {k: list(v) for k, v in all_entities.items()},
        "relation_extraction": {k: list(v) for k, v in all_relations.items()},
        "events_list": all_records,
        "news_section": article_section,
    }

    return normalize_combined_output(
        merged, text,
        document_id=document_id,
        source_url=source_url,
        publication_date=publication_date,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CANONICAL OUTPUT JSON SCHEMA (for validation / documentation)
# ═════════════════════════════════════════════════════════════════════════════
CANONICAL_OUTPUT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GLiNER2 Event-Location Extraction Output",
    "type": "object",
    "required": ["document", "entities", "event_location_pairs"],
    "properties": {
        "document": {"type": "object", "properties": {"id": {"type": "string"}}},
        "entities": {"type": "array", "items": {"type": "object"}},
        "event_location_pairs": {"type": "array", "items": {"type": "object"}},
        "event_records": {"type": "array", "items": {"type": "object"}},
    }
}


# ═════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Text containing multiple distinct events and locations
    article = (
        "A magnitude 6.2 earthquake struck central Turkey early Tuesday "
        "morning, causing widespread damage. Meanwhile, in neighboring Syria, "
        "a massive rescue operation commenced in Aleppo following the collapse "
        "of several residential buildings. On Wednesday, a violent protest erupted "
        "in Istanbul over the government's response time, leading to fierce clashes "
        "between demonstrators and local police."
    )

    all_outputs = {}

    print("=" * 60)
    print("APPROACH 1: Combined Multi-Task Extraction")
    print("=" * 60)
    result_combined = extract_combined(article, threshold=0.3)
    print(json.dumps(result_combined, indent=2))
    all_outputs["approach_1_combined"] = result_combined

    print("\n" + "=" * 60)
    print("APPROACH 2: Structured JSON Extraction")
    print("=" * 60)
    result_structured = extract_structured(article, threshold=0.3)
    print(json.dumps(result_structured, indent=2))
    all_outputs["approach_2_structured"] = result_structured

    print("\n" + "=" * 60)
    print("APPROACH 3: Relation Extraction Only")
    print("=" * 60)
    result_relations = extract_relations_only(article)
    print(json.dumps(result_relations, indent=2))
    all_outputs["approach_3_relations_only"] = result_relations

    print("\n" + "=" * 60)
    print("FULL PIPELINE: Normalized Output (with batching)")
    print("=" * 60)
    normalized = extract_from_long_article(
        article,
        threshold=0.3,
        document_id="article-2026-multi-event",
        source_url="https://example.com/multi-crisis",
        publication_date="2026-03-05",
    )
    print(json.dumps(normalized, indent=2))
    all_outputs["full_pipeline_normalized"] = normalized

    # Save to File properly using json.dump
    output_filename = "gliner2_extraction_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] All outputs have been properly saved to '{output_filename}'")
