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
        
        # ── Risk Factor Clusters ──
        "agricultural production issues": "A state of severe agricultural impairment where harvests are devastated by failed crops, bad harvests, and an overall harvest decline due to water distribution shortages and a lack of cultivation leaving farmers unable to sow; this disruption to farming is compounded by a massive toll on livestock where many livestock had died leading to widespread cattle death, as well as severe infrastructure damage and a lack of agricultural infrastructure causing a transport bottleneck and lack of roads.",
        "conflicts and violence": "A widespread destructive pattern of mayhem and continued strife marked by years of warfare, prolonged fighting, and civil strife, where a conflict or internal strife escalates into clan warfare and a specific clan battle between rival clans; this environment features a rebel insurgency, militia groups, rival warlords, or a single dominant warlord, as well as gangs of bandits engaging in looting and pirates hijacking convoys; the violence is often exacerbated by foreign troops launching a major offensive or the offensive involving an air attack, a siege, or a blockade, alongside acts of terrorism by a terrorist, international terrorists, or jihadist groups unleashing a bombing campaign, all while a brutal government engages in violent suppression, repression, police torture, and other severe human rights abuses.",
        "economic issues": "A severe economic crisis that has devastated the economy resulting in a collapsing economy characterized by slashed export, reduced imports, disrupted trade, and reduced national output, fueling increased external debt and a brain drain, while a steep rise in rising inflation causes a drastic price rise, specifically impacting the price of food through rising food prices, ultimately trapping the population in deep economic impoverishment and an inescapable cycle of poverty.",
        "environmental issues": "An overarching ecological crisis driven by climate change, where an excess of carbon and greenhouse gases exacerbates environmental degradation and increases the frequency of a devastating natural disaster.",
        "food crisis": "A profound systemic dysfunction triggering a severe food crisis and multiple hunger crises defined by rampant food insecurity, mass hunger, and acute hunger, which devolves into widespread apathy and massive starvation where a largely malnourished and dehydrated population suffers from life-threatening hunger and gastrointestinal diseases, ultimately driving up rates of infant mortality.",
        "forced displacement": "A crisis of forced migration where vulnerable populations, primarily civilians uprooted from their homes, are forced to flee and become displaced individuals, asylum seekers, or refugees seeking shelter in makeshift camps.",
        "humanitarian aid": "A dire humanitarian disaster causing global international alarm over the deteriorating humanitarian situation, prompting an urgent aid appeal and a call for donations for foreign aid and food assistance; however, efforts for international intervention are hindered by an international embargo, restricted humanitarian access, restricted relief flights, and incidents of withheld relief or stolen food aid, tragically resulting in situations where aid workers died, forcing the affected populace to survive entirely without international aid and rely strictly on self reliance.",
        "land-related issues": "A complex crisis involving vital farmland characterized by aggressive land invasions, a hostile land grab, and systemic land seizures involving the horrific tactic of burning houses and pushing peasants off their properties, alongside failed attempts at land reform, rampant land degradation leading to poor soil quality, and extensive damage leaving vast forests destroyed.",
        "pests and diseases": "A severe biological hazard where destructive swarms of locusts and other agricultural pests cause events like potato blight, while simultaneously triggering deadly epidemics, including a sudden cholera outbreak among humans, and devastating animal diseases such as rinderpest and the infamous cattle plague.",
        "political instability": "A volatile environment characterized by a total collapse of government and an overarching lack of authority driven by deep-rooted corruption, severe mismanagement, and a continuous power struggle or push for secession, often resulting in a politically engineered coup d'etat where a corrupt government faces a violent overthrow and established regimes were toppled, leading to the rise of authoritarian or totalitarian dictators, a harsh military dictatorship, or a strict military junta governing as oppressive regimes frequently promoting anti-western policies.",
        "weather shocks": "A sequence of unpredictable climatic hazards and weather extremes, ranging from destructive floods triggered by severe rains and a powerful cyclone, to critical issues with water availability stemming from a prolonged dry spell and a harsh drought caused by abnormally low rainfall, inadequate rainfall, scanty rainfall, failed rains, a general shortage of rains, and an overall lack of rains.",
        "other": "An uncategorized catastrophe and immense tragedy that has wreaked havoc on society, often manifesting as a man-made disaster or a severe population crisis reminiscent of the historical slave trade, leading to an alarming level of continued deterioration exacerbated by a profound lack of alternatives."
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

    # Long text containing multiple distinct events, locations, and risk factors
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