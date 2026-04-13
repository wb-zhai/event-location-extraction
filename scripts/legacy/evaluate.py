# /// script
# dependencies = []
# ///

"""
Evaluate all script versions against a human-annotated ground truth.
Computes Precision, Recall, F1, Accuracy, and other metrics for both
entity extraction and event-location linking.
"""

import json

# ═════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH  (human-annotated for the test article)
# ═════════════════════════════════════════════════════════════════════════════

# --- Entities: (text_lower, expected_label_category) ---
# label_category groups: "location", "event_or_risk", "date", "organization"
# For event_or_risk we accept either generic "event" or the specific risk factor.

GOLD_ENTITIES = {
    # Locations
    ("valoria", "location"),
    ("oakhaven", "location"),
    ("kingsbridge", "location"),
    ("aethelgard", "location"),
    # Dates
    ("past six months", "date"),
    ("past quarter", "date"),
    # Organizations
    ("local health organizations", "organization"),
    ("international aid organizations", "organization"),
    # Weather shocks
    ("prolonged dry spell", "event_or_risk"),
    ("lack of rains", "event_or_risk"),
    ("weather shocks", "event_or_risk"),
    ("abnormally low rainfall", "event_or_risk"),
    # Agricultural
    ("agricultural production issues", "event_or_risk"),
    ("bad harvests", "event_or_risk"),
    ("failed crops", "event_or_risk"),
    ("cattle death", "event_or_risk"),
    ("water distribution shortages", "event_or_risk"),
    # Economic
    ("local economy", "event_or_risk"),
    # Food crisis
    ("food crisis", "event_or_risk"),
    ("food insecurity", "event_or_risk"),
    ("mass hunger", "event_or_risk"),
    ("malnourished and dehydrated population", "event_or_risk"),
    ("gastrointestinal diseases", "event_or_risk"),
    ("infant mortality rates", "event_or_risk"),
    ("life-threatening hunger", "event_or_risk"),
    # Forced displacement
    ("forced displacement", "event_or_risk"),
    ("makeshift camps", "event_or_risk"),
    ("displaced individuals", "event_or_risk"),
    ("rural collapse", "event_or_risk"),
    # Humanitarian
    ("aid appeal", "event_or_risk"),
    ("restricted humanitarian access", "event_or_risk"),
    ("stolen food aid", "event_or_risk"),
    ("deteriorating humanitarian situation", "event_or_risk"),
}

# --- Event-Location pairs: (event_text_lower, location_text_lower) ---
# Ground truth: which location each event/risk factor actually belongs to
# based on careful reading of the article.

GOLD_PAIRS = {
    # Weather shocks -> Valoria (province-level, stated in sentence 1)
    ("prolonged dry spell", "valoria"),
    ("lack of rains", "valoria"),
    ("weather shocks", "valoria"),
    ("abnormally low rainfall", "valoria"),
    # Agricultural -> Oakhaven (explicitly "rural outskirts of Oakhaven")
    ("agricultural production issues", "oakhaven"),
    ("bad harvests", "oakhaven"),
    ("failed crops", "oakhaven"),
    ("cattle death", "oakhaven"),
    ("water distribution shortages", "oakhaven"),
    # Economic -> Valoria (province-level, "devastated the local economy")
    ("local economy", "valoria"),
    # Food crisis -> Valoria (national level: "gripped the nation")
    ("food crisis", "valoria"),
    # Food crisis details -> Kingsbridge (explicitly "capital city of Kingsbridge")
    ("food insecurity", "kingsbridge"),
    ("mass hunger", "kingsbridge"),
    ("malnourished and dehydrated population", "kingsbridge"),
    ("gastrointestinal diseases", "kingsbridge"),
    ("infant mortality rates", "kingsbridge"),
    ("life-threatening hunger", "valoria"),  # "uprooted BY the hunger" = source
    # Forced displacement -> Aethelgard (destination: "camps along...Aethelgard")
    ("forced displacement", "aethelgard"),
    ("makeshift camps", "aethelgard"),
    ("displaced individuals", "aethelgard"),
    ("rural collapse", "aethelgard"),
    # Humanitarian -> Aethelgard (context: aid situation around the border)
    ("aid appeal", "aethelgard"),
    ("restricted humanitarian access", "aethelgard"),
    ("stolen food aid", "aethelgard"),
    ("deteriorating humanitarian situation", "aethelgard"),
}

# ═════════════════════════════════════════════════════════════════════════════
# VERSION OUTPUTS  (extracted from actual runs)
# ═════════════════════════════════════════════════════════════════════════════

# For each version, we store the set of (text_lower, label_category) for entities
# and (event_lower, location_lower) for pairs.

def _categorize_label(label: str) -> str:
    label = label.lower()
    if label == "location":
        return "location"
    if label == "date":
        return "date"
    if label == "organization":
        return "organization"
    return "event_or_risk"

# --- v1 (extract_with_risk_factors.py) ---
V1_ENTITIES = {
    ("forced displacement", "event_or_risk"), ("weather shocks", "event_or_risk"),
    ("food crisis", "event_or_risk"), ("stolen food aid", "event_or_risk"),
    ("infant mortality rates", "event_or_risk"), ("agricultural production issues", "event_or_risk"),
    ("aid appeal", "event_or_risk"), ("water distribution shortages", "event_or_risk"),
    ("cattle death", "event_or_risk"), ("restricted humanitarian access", "event_or_risk"),
    ("food insecurity", "event_or_risk"),
    ("aethelgard", "location"), ("kingsbridge", "location"), ("oakhaven", "location"), ("valoria", "location"),
    ("past six months", "date"), ("past quarter", "date"),
    ("international aid organizations", "organization"), ("local health organizations", "organization"),
    ("gastrointestinal diseases", "event_or_risk"), ("displaced individuals", "event_or_risk"),
}

V1_PAIRS = {
    ("weather shocks", "valoria"),
}

# --- v2 (extract_with_risk_factors_v2.py) ---
V2_ENTITIES = V1_ENTITIES | {
    ("prolonged dry spell", "event_or_risk"), ("lack of rains", "event_or_risk"),
    ("abnormally low rainfall", "event_or_risk"), ("bad harvests", "event_or_risk"),
    ("failed crops", "event_or_risk"), ("mass hunger", "event_or_risk"),
    ("food crisis", "event_or_risk"), ("makeshift camps", "event_or_risk"),
    ("life-threatening hunger", "event_or_risk"), ("rural collapse", "event_or_risk"),
    ("deteriorating humanitarian situation", "event_or_risk"),
    ("malnourished and dehydrated population", "event_or_risk"),
    ("self reliance", "event_or_risk"),
}

V2_PAIRS = {
    ("weather shocks", "valoria"), ("lack of rains", "valoria"), ("prolonged dry spell", "valoria"),
    ("abnormally low rainfall", "oakhaven"), ("agricultural production issues", "oakhaven"),
    ("local economy", "oakhaven"),  # WRONG: should be valoria
    ("water distribution shortages", "oakhaven"), ("cattle death", "oakhaven"),
    ("bad harvests", "oakhaven"), ("failed crops", "oakhaven"),
    ("food crisis", "oakhaven"),  # WRONG: should be valoria
    ("mass hunger", "kingsbridge"), ("food insecurity", "kingsbridge"),
    ("malnourished and dehydrated population", "kingsbridge"),
    ("gastrointestinal diseases", "kingsbridge"), ("infant mortality rates", "kingsbridge"),
    ("forced displacement", "aethelgard"), ("makeshift camps", "aethelgard"),
    ("rural collapse", "aethelgard"), ("life-threatening hunger", "aethelgard"),  # WRONG: source is valoria
    ("deteriorating humanitarian situation", "aethelgard"), ("aid appeal", "aethelgard"),
    ("restricted humanitarian access", "aethelgard"), ("stolen food aid", "aethelgard"),
    ("displaced individuals", "aethelgard"),
}

# --- v3 ---
V3_ENTITIES = {
    ("valoria", "location"), ("oakhaven", "location"), ("kingsbridge", "location"), ("aethelgard", "location"),
    ("past six months", "date"), ("past quarter", "date"),
    ("local health organizations", "organization"), ("international aid organizations", "organization"),
    ("prolonged dry spell", "event_or_risk"), ("lack of rains", "event_or_risk"),
    ("weather shocks", "event_or_risk"), ("abnormally low rainfall", "event_or_risk"),
    ("local economy", "event_or_risk"), ("agricultural production issues", "event_or_risk"),
    ("bad harvests", "event_or_risk"), ("failed crops", "event_or_risk"),
    ("water distribution shortages", "event_or_risk"),
    ("food crisis", "event_or_risk"), ("food insecurity", "event_or_risk"), ("mass hunger", "event_or_risk"),
    ("gastrointestinal diseases", "event_or_risk"), ("infant mortality rates", "event_or_risk"),
    ("forced displacement", "event_or_risk"), ("makeshift camps", "event_or_risk"),
    ("displaced individuals", "event_or_risk"),
    ("restricted humanitarian access", "event_or_risk"), ("stolen food aid", "event_or_risk"),
}

V3_PAIRS = {
    ("prolonged dry spell", "valoria"), ("lack of rains", "valoria"),
    ("weather shocks", "valoria"), ("abnormally low rainfall", "oakhaven"),
    ("local economy", "valoria"),  # FIXED from v2
    ("agricultural production issues", "valoria"),  # slightly off, should be oakhaven
    ("bad harvests", "oakhaven"), ("failed crops", "oakhaven"),
    ("water distribution shortages", "oakhaven"),
    ("food crisis", "oakhaven"),  # WRONG: should be valoria
    ("food insecurity", "kingsbridge"), ("mass hunger", "kingsbridge"),
    ("gastrointestinal diseases", "kingsbridge"), ("infant mortality rates", "kingsbridge"),
    ("forced displacement", "kingsbridge"),  # WRONG: should be aethelgard
    ("makeshift camps", "aethelgard"),
    ("restricted humanitarian access", "aethelgard"), ("stolen food aid", "aethelgard"),
    ("displaced individuals", "aethelgard"),
}

# --- v4 ---
V4_ENTITIES = V3_ENTITIES | {
    ("cattle death", "event_or_risk"), ("farming", "event_or_risk"),
    ("escalating tragedy", "event_or_risk"), ("civilians", "event_or_risk"),
    ("life-threatening hunger", "event_or_risk"), ("malnourished and dehydrated population", "event_or_risk"),
    ("deteriorating humanitarian situation", "event_or_risk"),
    ("self reliance", "event_or_risk"), ("immense tragedy", "event_or_risk"),
    ("hospitals", "organization"),
}

V4_PAIRS = {
    ("prolonged dry spell", "valoria"), ("lack of rains", "valoria"),
    ("weather shocks", "valoria"), ("abnormally low rainfall", "oakhaven"),
    ("local economy", "oakhaven"),  # regressed from v3
    ("agricultural production issues", "oakhaven"),
    ("bad harvests", "oakhaven"), ("failed crops", "oakhaven"),
    ("water distribution shortages", "oakhaven"), ("cattle death", "oakhaven"),
    ("farming", "valoria"), ("food crisis", "valoria"),  # FIXED
    ("food insecurity", "kingsbridge"), ("mass hunger", "kingsbridge"),
    ("malnourished and dehydrated population", "kingsbridge"),
    ("gastrointestinal diseases", "kingsbridge"), ("infant mortality rates", "kingsbridge"),
    ("forced displacement", "kingsbridge"),  # WRONG
    ("makeshift camps", "aethelgard"), ("life-threatening hunger", "aethelgard"),  # WRONG for hunger
    ("deteriorating humanitarian situation", "aethelgard"),
    ("restricted humanitarian access", "aethelgard"), ("stolen food aid", "aethelgard"),
    ("displaced individuals", "aethelgard"), ("aid appeal", "aethelgard"),
    ("self reliance", "aethelgard"), ("immense tragedy", "aethelgard"),
    ("rural collapse", "aethelgard"),
}

# --- v5 ---
V5_ENTITIES = V4_ENTITIES | {
    ("rural collapse", "event_or_risk"),
    ("aid appeal", "event_or_risk"),
    ("displacement", "event_or_risk"),
}

V5_PAIRS = {
    ("prolonged dry spell", "valoria"), ("lack of rains", "valoria"),
    ("weather shocks", "valoria"), ("abnormally low rainfall", "oakhaven"),
    ("local economy", "oakhaven"),
    ("agricultural production issues", "oakhaven"),
    ("bad harvests", "oakhaven"), ("failed crops", "oakhaven"),
    ("water distribution shortages", "oakhaven"), ("cattle death", "oakhaven"),
    ("farming", "valoria"), ("food crisis", "valoria"),  # FIXED via coref
    ("food insecurity", "kingsbridge"), ("mass hunger", "kingsbridge"),
    ("malnourished and dehydrated population", "kingsbridge"),
    ("gastrointestinal diseases", "kingsbridge"), ("infant mortality rates", "kingsbridge"),
    ("escalating tragedy", "kingsbridge"),
    ("displacement", "kingsbridge"),  # waterfall, slightly off
    ("life-threatening hunger", "aethelgard"),  # still same-sentence trap
    ("makeshift camps", "aethelgard"), ("rural collapse", "aethelgard"),
    ("deteriorating humanitarian situation", "aethelgard"),
    ("restricted humanitarian access", "aethelgard"), ("stolen food aid", "aethelgard"),
    ("displaced individuals", "aethelgard"), ("aid appeal", "aethelgard"),
    ("self reliance", "aethelgard"), ("immense tragedy", "aethelgard"),
    ("civilians", "aethelgard"),
    ("forced displacement", "kingsbridge"),  # waterfall carried kingsbridge
}


# ═════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(predicted: set, gold: set) -> dict:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    tn = 0  # not applicable for open-set extraction

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0  # Jaccard/IoU
    f05 = (1 + 0.25) * precision * recall / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
    f2 = (1 + 4) * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "F0.5": round(f05, 4),
        "F2": round(f2, 4),
        "Jaccard": round(accuracy, 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    versions = {
        "v1": (V1_ENTITIES, V1_PAIRS),
        "v2": (V2_ENTITIES, V2_PAIRS),
        "v3": (V3_ENTITIES, V3_PAIRS),
        "v4": (V4_ENTITIES, V4_PAIRS),
        "v5": (V5_ENTITIES, V5_PAIRS),
    }

    # --- Entity Extraction Metrics ---
    print("=" * 90)
    print("ENTITY EXTRACTION METRICS")
    print(f"Gold standard: {len(GOLD_ENTITIES)} entities")
    print("=" * 90)
    header = f"{'Ver':>4} | {'TP':>3} {'FP':>3} {'FN':>3} | {'Prec':>6} {'Rec':>6} {'F1':>6} {'F0.5':>6} {'F2':>6} {'Jacc':>6}"
    print(header)
    print("-" * len(header))

    ent_results = {}
    for ver, (ents, _) in versions.items():
        m = compute_metrics(ents, GOLD_ENTITIES)
        ent_results[ver] = m
        print(f"{ver:>4} | {m['TP']:>3} {m['FP']:>3} {m['FN']:>3} | {m['Precision']:>6.3f} {m['Recall']:>6.3f} {m['F1']:>6.3f} {m['F0.5']:>6.3f} {m['F2']:>6.3f} {m['Jaccard']:>6.3f}")

    # --- Event-Location Pair Metrics ---
    print(f"\n{'=' * 90}")
    print("EVENT-LOCATION LINKING METRICS")
    print(f"Gold standard: {len(GOLD_PAIRS)} pairs")
    print("=" * 90)
    print(header)
    print("-" * len(header))

    pair_results = {}
    for ver, (_, pairs) in versions.items():
        m = compute_metrics(pairs, GOLD_PAIRS)
        pair_results[ver] = m
        print(f"{ver:>4} | {m['TP']:>3} {m['FP']:>3} {m['FN']:>3} | {m['Precision']:>6.3f} {m['Recall']:>6.3f} {m['F1']:>6.3f} {m['F0.5']:>6.3f} {m['F2']:>6.3f} {m['Jaccard']:>6.3f}")

    # --- Detailed Error Analysis ---
    print(f"\n{'=' * 90}")
    print("ERROR ANALYSIS (v5)")
    print("=" * 90)

    v5_ents, v5_pairs = versions["v5"]

    missed_ents = GOLD_ENTITIES - v5_ents
    extra_ents = v5_ents - GOLD_ENTITIES
    if missed_ents:
        print("\n  Missed entities (FN):")
        for text, label in sorted(missed_ents):
            print(f"    - '{text}' [{label}]")
    if extra_ents:
        print("\n  Extra entities (FP):")
        for text, label in sorted(extra_ents):
            print(f"    + '{text}' [{label}]")

    missed_pairs = GOLD_PAIRS - v5_pairs
    wrong_pairs = v5_pairs - GOLD_PAIRS
    if missed_pairs:
        print("\n  Missed pairs (FN):")
        for ev, loc in sorted(missed_pairs):
            # Check if we linked it somewhere else
            actual = [l for e, l in v5_pairs if e == ev]
            if actual:
                print(f"    - '{ev}' -> '{loc}'  (linked to '{actual[0]}' instead)")
            else:
                print(f"    - '{ev}' -> '{loc}'  (not linked at all)")
    if wrong_pairs:
        print("\n  Wrong pairs (FP):")
        for ev, loc in sorted(wrong_pairs):
            expected = [l for e, l in GOLD_PAIRS if e == ev]
            if expected:
                print(f"    + '{ev}' -> '{loc}'  (should be '{expected[0]}')")
            else:
                print(f"    + '{ev}' -> '{loc}'  (event not in gold)")

    # Save as JSON
    output = {
        "entity_metrics": {v: m for v, m in ent_results.items()},
        "pair_metrics": {v: m for v, m in pair_results.items()},
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] evaluation_results.json")
