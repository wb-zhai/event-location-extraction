"""Fix invalid event_type annotations by mapping them to valid ontology labels.

Reads prediction JSONL shards from input_dir, remaps any event_type not present
in the ontology to the closest valid label using string similarity, and writes
fixed shards to output_dir. Events with no plausible mapping are dropped.

Usage:
    python fix_annotations.py \
        --input_dir  dataset/db/predictions/south_sudan_articles_a100_geo_ranking_wbadmin \
        --output_dir dataset/db/predictions/south_sudan_articles_a100_geo_ranking_wbadmin_fixed \
        [--ontology  ontologies/zhai/science.json] \
        [--threshold 0.65] \
        [--dry_run]
"""
from __future__ import annotations

import concurrent.futures
import difflib
import functools
import json
import os
import pathlib
from collections import Counter
from typing import Optional

import fire
from tqdm import tqdm


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_ONTOLOGY = REPO_ROOT / "ontologies" / "zhai" / "science.json"

# Hand-coded overrides for cases where string similarity alone is insufficient.
# Keys are invalid labels (lowercase); values are valid ontology labels or None (drop).
MANUAL_OVERRIDES: dict[str, Optional[str]] = {
    # # plural / typo variants
    # "coups": "coup",
    # "coups d'etat": "coup",
    # "coup attempt": "coup",
    # "military coup": "coup",
    # "topple": "regimes were toppled",
    # "toppling": "regimes were toppled",
    # "regimes were topple": "regimes were toppled",
    # "regimes were topped": "regimes were toppled",
    # "regime were toppled": "regimes were toppled",
    # "junta": "military junta",
    # "flood": "floods",
    # "flooding": "floods",
    # "flood crisis": "floods",
    # "heavy rains": "severe rains",
    # "storms": "cyclone",
    # "storm": "cyclone",
    # "hurricane": "cyclone",
    # "heatwaves": "weather extremes",
    # "hail": "weather extremes",
    # "tornado": "weather extremes",
    # "tornadoes": "weather extremes",
    # "landslides": "floods",
    # "climate extremes": "weather extremes",
    # "climate hazards": "climatic hazards",
    # "climate crisis": "climate change",
    # "climate catastrophe": "climate change",
    # "climate change": "climate change",  # capitalisation mismatch
    # "failed harvests": "harvests are devastated",
    # "disrupted farming": "disruption to farming",
    # "crop damage": "failed crops",
    # "forest fires": "forests destroyed",
    # "forest fire": "forests destroyed",
    # "forest destroyed": "forests destroyed",
    # "illegal logging": "forests destroyed",
    # "soil degradation": "land degradation",
    # "ecological degradation": "environmental degradation",
    # "environmental disaster": "environmental degradation",
    # "drought relief": "drought",
    # "reduced rainfall": "inadequate rainfall",
    # "lack of water": "water availability",
    # "lack of water availability": "water availability",
    # "water crisis": "water availability",
    # "water contamination": "water availability",
    # "water quality": "water availability",
    # "freshwater availability": "water availability",
    # "food prices": "price of food",
    # "food price rise": "rising food prices",
    # "food price inflation": "rising food prices",
    # "food prices rise": "rising food prices",
    # "food price spikes": "rising food prices",
    # "food price crisis": "food crisis",
    # "food catastrophe": "food crisis",
    # "food crises": "food crisis",
    # "hunger crisis": "food crisis",
    # "hunger crises": "hunger crises",
    # "hunger": "mass hunger",
    # "famine": "massive starvation",
    # "forced starvation": "massive starvation",
    # "undernourished": "malnourished",
    # "under-nourished": "malnourished",
    # "dehydration": "dehydrated",
    # "infant mortality": "infant mortality",
    # "maternal mortality": None,
    # "economic collapse": "collapsing economy",
    # "economic catastrophe": "collapsing economy",
    # "economic disaster": "collapsing economy",
    # "economic decline": "reduced national output",
    # "economic damage": "reduced national output",
    # "economic mismanagement": "mismanagement",
    # "economic instability": "economic crisis",
    # "economic inequality": "economic impoverishment",
    # "financial crisis": "economic crisis",
    # "recession": "economic crisis",
    # "hyperinflation": "rising inflation",
    # "inflation": "rising inflation",
    # "rising prices": "price rise",
    # "prices rise": "price rise",
    # "high food prices": "rising food prices",
    # "price collapse": None,
    # "price drop": None,
    # "currency collapse": "collapsing economy",
    # "fuel crisis": "economic crisis",
    # "fuel scarcity": "economic crisis",
    # "energy crisis": "economic crisis",
    # "unemployment": "economic impoverishment",
    # "high unemployment": "economic impoverishment",
    # "job losses": "economic impoverishment",
    # "increase in external debt": "increased external debt",
    # "reduced international aid": "without international aid",
    # "lack of international aid": "without international aid",
    # "reduced foreign aid": "foreign aid",
    # "reduced food assistance": "food assistance",
    # "aid workers killed": "aid workers died",
    # "aid workers kidnapped": "restricted humanitarian access",
    # "aid workers detained": "restricted humanitarian access",
    # "aid workers displaced": "restricted humanitarian access",
    # "aid workers expelled": "restricted humanitarian access",
    # "aid workers disappeared": "restricted humanitarian access",
    # "foreign aid workers": "foreign aid",
    # "disrupted humanitarian access": "restricted humanitarian access",
    # "reduced humanitarian access": "restricted humanitarian access",
    # "reduced humanitarian assistance": "restricted humanitarian access",
    # "blocked humanitarian access": "restricted humanitarian access",
    # "lack of humanitarian access": "restricted humanitarian access",
    # "limited humanitarian access": "restricted humanitarian access",
    # "aid restricted humanitarian access": "restricted humanitarian access",
    # "aid blockade": "blockade",
    # "aid convoys": "convoys",
    # "convoy": "convoys",
    # "convoy attack": "convoys",
    # "lost food aid": "stolen food aid",
    # "stolen food": "stolen food aid",
    # "international terrorism": "terrorism",
    # "terrorist attack": "terrorism",
    # "terrorist attacks": "terrorism",
    # "terrorist bombing": "bombing campaign",
    # "terrorist organizations": "terrorist groups",
    # "terrorist groups": "terrorist",
    # "terrorists": "terrorist",
    # "foreign terrorists": "international terrorists",
    # "foreign fighters": "international terrorists",
    # "international troops": "foreign troops",
    # "foreign intervention": "international intervention",
    # "military intervention": "international intervention",
    # "international military intervention": "international intervention",
    # "humanitarian intervention": "international intervention",
    # "jihadist groups": "jihadist groups",
    # "insurgency": "rebel insurgency",
    # "rebels": "rebel insurgency",
    # "rebel groups": "rebel insurgency",
    # "militia groups": "militia groups",
    # "militants": "militia groups",
    # "gang violence": "conflict",
    # "gang warfare": "clan warfare",
    # "gang activity": "conflict",
    # "political violence": "conflict",
    # "criminal violence": "conflict",
    # "civil war": "prolonged fighting",
    # "civil conflict": "civil strife",
    # "internal conflict": "internal strife",
    # "political strife": "civil strife",
    # "civels strife": "civil strife",
    # "continued fighting": "prolonged fighting",
    # "offensive": "the offensive",
    # "failed offensive": "the offensive",
    # "the offensive": "the offensive",
    # "air attack": "air attack",
    # "bombing": "bombing campaign",
    # "chemical attack": "air attack",
    # "chemical weapon": "terrorism",
    # "chemical weapon attack": "terrorism",
    # "chemical weapons": "terrorism",
    # "missile launch": "terrorism",
    # "missile test": "terrorism",
    # "IED": "terrorism",
    # "surgical strike": "air attack",
    # "landmine explosion": "terrorism",
    # "terrorist bombing": "bombing campaign",
    # "bombing campaign": "bombing campaign",
    # "propagating violence": "conflict",
    # "spreading violence": "conflict",
    # "local conflicts": "conflict",
    # "political crisis": "power struggle",
    # "political turmoil": "power struggle",
    # "political upheaval": "power struggle",
    # "political instability": "power struggle",
    # "political unrest": "civil strife",
    # "political chaos": "power struggle",
    # "political impasse": "power struggle",
    # "political repression": "repression",
    # "political engineering": "politically engineered",
    # "political engineered": "politically engineered",
    # "political situation": None,
    # "political regime change": "coup",
    # "instability": "civil strife",
    # "unrest": "civil strife",
    # "insecurity": "conflict",
    # "rising insecurity": "conflict",
    # "security issues": "conflict",
    # "security concerns": "conflict",
    # "security situation": None,
    # "crisis": None,
    # "crises": None,
    # "disaster": "natural disaster",
    # "catastrophe": "catastrophe",
    # "emergency": None,
    # "accident": None,
    # "accidents": None,
    # "chaos": "dysfunction",
    # "destruction": "infrastructure damage",
    # "death": None,
    # "excess deaths": None,
    # "repressive regimes": "oppressive regimes",
    # "authoritarian regimes": "oppressive regimes",
    # "brutal regimes": "brutal government",
    # "dictator": "military dictatorship",
    # "warlords": "rival warlords",
    # "bands": "gangs of bandits",
    # "bands of bandits": "gangs of bandits",
    # "theft": "looting",
    # "burglary": "looting",
    # "robbery": "looting",
    # "cattle theft": "looting",
    # "poaching": "looting",
    # "illegal hunting": "looting",
    # "overhunting": "looting",
    # "overfishing": "looting",
    # "piracy": "pirates",
    # "genocide": "human rights abuses",
    # "genocides": "human rights abuses",
    # "genocidal efforts": "human rights abuses",
    # "war crimes": "human rights abuses",
    # "police brutality": "police torture",
    # "torture": "police torture",
    # "suppression": "violent suppression",
    # "persecution": "repression",
    # "sexual abuse": "human rights abuses",
    # "apartheid": "oppressive regimes",
    # "child labor": "human rights abuses",
    # "child labour": "human rights abuses",
    # "child soldiers": "human rights abuses",
    # "child soldier": "human rights abuses",
    # "child abduction": "kidnapping",
    # "kidnapping": "restricted humanitarian access",
    # "hostage": "restricted humanitarian access",
    # "rising kidnapping incidents": "restricted humanitarian access",
    # "human trafficking": "slave trade",
    # "forced displacement": "displaced",
    # "internally displaced": "displaced",
    # "displacement": "displaced",
    # "uprooted": "civilians uprooted",
    # "civilians uprooting": "civilians uprooted",
    # "forced eviction": "civilians uprooted",
    # "forced migration": "migration",
    # "migration crisis": "migration",
    # "refugee": "refugees",
    # "refugee crisis": "refugees",
    # "refugee crises": "refugees",
    # "refugee camp": "makeshift camps",
    # "refugee camps": "makeshift camps",
    # "asylum seekers": "asylum seekers",
    # "expulsion": "civilians uprooted",
    # "exile": "migration",
    # "immigration": "migration",
    # "flight": "flee",
    # "evacuation": "flee",
    # "separation of families": "civilians uprooted",
    # "land grabbing": "land grab",
    # "land grabs": "land grab",
    # "land grabbers": "land grab",
    # "land seizures": "land seizures",
    # "confiscation of land": "land seizures",
    # "confiscation of farmland": "land seizures",
    # "confiscation": "land seizures",
    # "land loss": "land seizures",
    # "land invasions": "land invasions",
    # "land conflict": "land invasions",
    # "land disputes": "land invasions",
    # "pushing peasants off": "pushing peasants off",
    # "land policy": "land reform",
    # "devolution": None,
    # "social issue": None,
    # "social crisis": None,
    # "social instability": None,
    # "housing crisis": None,
    # "housing insecurity": None,
    # "rising rent": None,
    # "rising cost of living": "price rise",
    # "cyberattack": None,
    # "cyber attack": None,
    # "cyber-attack": None,
    # "cybercrimes": None,
    # "cybersecurity incident": None,
    # "internet outage": None,
    # "internet shutdown": None,
    # "data theft": None,
    # "school shooting": None,
    # "school closures": None,
    # "education disruption": None,
    # "supply chain disruption": "transport bottleneck",
    # "supply chain crisis": "transport bottleneck",
    # "transport bottleneck": "transport bottleneck",
    # "infrastructure damage": "infrastructure damage",
    # "public infrastructure damage": "infrastructure damage",
    # "lack of roads": "lack of roads",
    # "submerged": "floods",
    # "storm surge": "cyclone",
    # "exceptional rainfall": "severe rains",
    # "slashed import": "reduced imports",
    # "restricted trade": "disrupted trade",
    # "disruption to trade": "disrupted trade",
    # "toll on trade": "disrupted trade",
    # "embargo": "international embargo",
    # "economic embargo": "international embargo",
    # "international embargo": "international embargo",
    # "blockade": "blockade",
    # "seige": "siege",
    # "power outage": None,
    # "martial law": "military dictatorship",
    # "lockdown": None,
    # "pandemic": "epidemics",
    # "pandemics": "epidemics",
    # "coronavirus": "epidemics",
    # "dengue outbreak": "epidemics",
    # "measles outbreak": "epidemics",
    # "diphtheria": "epidemics",
    # "polio": "epidemics",
    # "diarrhoea": "epidemics",
    # "health crisis": "epidemics",
    # "public health risk": "epidemics",
    # "air pollution": "environmental degradation",
    # "earthquake": None,
    # "animal attack": None,
    # "animal aggression": None,
    # "wildlife incident": None,
    # "die-offs": "livestock had died",
    # "livestock death": "livestock had died",
    # "brain drain": "brain drain",
    # "migrant": "migration",
    # "corporate fraud": "corruption",
    # "drug crisis": None,
    # "drug problem": None,
    # "foreign aid": "foreign aid",
    # "international aid": "foreign aid",
    # "emergency response": None,
    # "failed states": "collapse of government",
    # "regional instability": "civil strife",
    # "criminality": "conflict",
    # "crime": "conflict",
    # "high levels of crime": "conflict",
    # "devastate the economy": "devastated the economy",
    # "country in crisis": None,
    # "humanitarian crisis": "humanitarian disaster",
    # "humanitarian situation": "humanitarian situation",
    # "famine crisis": "massive starvation",
    # "food catastrophe": "food crisis",
    # "agricultural crisis": "food crisis",
    # "food insecurity": "food insecurity",
    # "": None,
    # "political engineered": "politically engineered",
    # "rising water prices": "price rise",
    # "rebel insurgency": "rebel insurgency",
    # "internl strife": "internal strife",
    "coups": "coup",

    "political crisis": "power struggle",

    "accident": None,

    "gang violence": "gangs of bandits",

    "economic collapse": "collapsing economy",

    "regimes were topple": "regimes were toppled",

    "climate extremes": "weather extremes",

    "flooding": "floods",

    "political instability": "civil strife",

    "foreign terrorists": "international terrorists",

    "kidnapping": None,

    "crisis": None,

    "cyberattack": None,

    "climate hazards": "climatic hazards",

    "genocide": "human rights abuses",

    "hunger": "mass hunger",

    "": None,

    "economic catastrophe": "economic crisis",

    "humanitarian crisis": "humanitarian disaster",

    "water contamination": "water availability",

    "ecological degradation": "environmental degradation",

    "political turmoil": "civil strife",

    "dehydration": "dehydrated",

    "food prices": "price of food",

    "lack of water availability": "water availability",

    "aid workers kidnapped": None,

    "rising prices": "price rise",

    "climate crisis": "climate change",

    "aid workers detained": "human rights abuses",

    "political engineered": "politically engineered",

    "convoy": "convoys",

    "repressive regimes": "oppressive regimes",

    "terrorist groups": "terrorist",

    "aid workers killed": "aid workers died",

    "disrupted humanitarian access": "restricted humanitarian access",

    "instability": None,

    "reduced humanitarian access": "restricted humanitarian access",

    "environmental disaster": "environmental degradation",

    "slashed import": "reduced imports",

    "fire": None,

    "soil degradation": "poor soil quality",

    "political violence": "civil strife",

    "heavy rains": "severe rains",

    "health crisis": "epidemics",

    "police brutality": "violent suppression",

    "propagating violence": "continued strife",

    "food price rise": "rising food prices",

    "devolution": None,

    "maternal mortality": None,

    "theft": "looting",

    "civil war": "conflict",

    "earthquake": "natural disaster",

    "crime": None,

    "flood": "floods",

    "internally displaced": "displaced",

    "junta": "military junta",

    "chemical attack": "human rights abuses",

    "genocides": "human rights abuses",

    "housing crisis": None,

    "expulsion": "displaced",

    "high food prices": "price of food",

    "cyber attack": None,

    "martial law": "military dictatorship",

    "unemployment": "economic crisis",

    "financial crisis": "economic crisis",

    "climate catastrophe": "climate change",

    "burglary": "looting",

    "forced displacement": "displaced",

    "land grabbing": "land grab",

    "displacement": "displaced",

    "price collapse": None,

    "chemical weapons": "human rights abuses",

    "reduced foreign aid": "foreign aid",

    "storm surge": "cyclone",

    "missile launch": None,

    "undernourished": "malnourished",

    "crises": None,

    "accidents": None,

    "school shooting": None,

    "hostage": None,

    "authoritarian regimes": "authoritarian",

    "aid workers displaced": "displaced",

    "international terrorism": "international terrorists",

    "continued fighting": "prolonged fighting",

    "uprooted": "civilians uprooted",

    "political engineering": "politically engineered",

    "child labor": None,

    "terrorist bombing": "terrorism",

    "hyperinflation": "rising inflation",

    "offensive": "the offensive",

    "land loss": "land degradation",

    "suppression": "repression",

    "famine": "massive starvation",

    "recession": "economic crisis",

    "disrupted farming": "disruption to farming",

    "disaster": "catastrophe",

    "failed harvests": "harvests are devastated",

    "death": None,

    "forest fires": "forests destroyed",

    "overfishing": "environmental degradation",

    "evacuation": "flee",

    "foreign intervention": "international intervention",

    "water crisis": "water availability",

    "cyber-attack": None,

    "civil conflict": "internal strife",

    "human-made disaster": "man-made disaster",

    "chemical weapon": "human rights abuses",

    "security situation": None,

    "cattle theft": "looting",

    "lack of humanitarian access": "restricted humanitarian access",

    "land disputes": "land grab",

    "chaos": "mayhem",

    "price fall": None,

    "social issue": None,

    "tornado": "weather extremes",

    "lost food aid": "stolen food aid",

    "insurgency": "rebel insurgency",

    "disruption to trade": "disrupted trade",

    "economic decline": "economic crisis",

    "political upheaval": "civil strife",

    "energy crisis": None,

    "military intervention": "international intervention",

    "civels strife": "civil strife",

    "economic mismanagement": "mismanagement",

    "forest destroyed": "forests destroyed",

    "terrorists": "terrorist",

    "exile": "flee",

    "political strife": "civil strife",

    "gang warfare": "gangs of bandits",

    "bombing": "air attack",

    "reduced food assistance": "food assistance",

    "child soldiers": None,

    "storms": "weather extremes",

    "economic instability": "economic crisis",

    "livestock death": "livestock had died",

    "aid restricted humanitarian access": "restricted humanitarian access",

    "prices rise": "price rise",

    "economic damage": "devastated the economy",

    "topple": "overthrow",

    "refugee crisis": "refugees",

    "coup attempt": "coup",

    "price drop": None,

    "under-nourished": "malnourished",

    "corporate fraud": "corruption",

    "migrant": "migration",

    "human trafficking": "slave trade",

    "insecurity": None,

    "land conflict": "land grab",

    "dictator": "dictators",

    "rising water prices": "price rise",

    "refugee camps": "makeshift camps",

    "hunger crisis": "food crisis",

    "destruction": "catastrophe",

    "seige": "siege",

    "terrorist attack": "terrorism",

    "reduced humanitarian assistance": "foreign aid",

    "food crises": "hunger crises",

    "aid workers expelled": "displaced",

    "coups d'etat": "d'etat",

    "animal attack": None,

    "forced eviction": "displaced",

    "refugee": "refugees",

    "political repression": "repression",

    "blocked humanitarian access": "restricted humanitarian access",

    "water quality": "water availability",

    "persecution": "human rights abuses",

    "cycles of poverty": "cycle of poverty",

    "heatwaves": "weather extremes",

    "forest fire": "forests destroyed",

    "embargo": "international embargo",

    "inflation": "rising inflation",

    "increase in external debt": "increased external debt",

    "storm": "weather extremes",

    "political unrest": "civil strife",

    "poaching": "environmental degradation",

    "lack of international aid": "without international aid",

    "drug crisis": None,

    "foreign aid workers": None,

    "political situation": None,

    "civilians uprooting": "civilians uprooted",

    "rising kidnapping incidents": None,

    "animal aggression": None,

    "power outage": None,

    "warlords": "warlord",

    "cybercrimes": None,

    "flight": "flee",

    "submerged": "floods",

    "supply chain disruption": "disrupted trade",

    "high unemployment": "economic crisis",

    "social instability": "civil strife",

    "coronavirus": "epidemics",

    "rebels": "rebel insurgency",

    "political regime change": "overthrow",

    "confiscation of land": "land seizures",

    "confiscation of farmland": "land seizures",

    "exceptional rainfall": "severe rains",

    "terrorist organizations": "terrorist",

    "wildlife incident": None,

    "tornadoes": "weather extremes",

    "child soldier": None,

    "Climate Change": "climate change",

    "aid workers disappeared": "human rights abuses",

    "local conflicts": "conflict",

    "foreign fighters": "foreign troops",

    "economic embargo": "international embargo",

    "torture": "human rights abuses",

    "emergency response": None,

    "chemical weapon attack": "human rights abuses",

    "IED": None,

    "humanitarian intervention": "international intervention",

    "bands": "gangs of bandits",

    "rebel groups": "rebel insurgency",

    "dengue outbreak": "epidemics",

    "measles outbreak": "epidemics",

    "air pollution": "environmental degradation",

    "public infrastructure damage": "infrastructure damage",

    "immigration": "migration",

    "land grabs": "land grab",

    "regimes were topped": "regimes were toppled",

    "apartheid": "oppressive regimes",

    "limited humanitarian access": "restricted humanitarian access",

    "polio": "epidemics",

    "freshwater availability": "water availability",

    "fuel crisis": None,

    "flood crisis": "floods",

    "hurricane": "cyclone",

    "military involvement": "international intervention",

    "school closures": None,

    "terrorist attacks": "terrorism",

    "illegal hunting": "environmental degradation",

    "quality has deteriorated": "continued deterioration",

    "regime were toppled": "regimes were toppled",

    "supply chain crisis": "disrupted trade",

    "devastating natural disaster": "natural disaster",

    "failed states": "collapse of government",

    "social crisis": "civil strife",

    "political impasse": "power struggle",

    "agricultural crisis": "disruption to farming",

    "reduced international aid": "foreign aid",

    "robbery": "looting",

    "internet outage": None,

    "forced starvation": "massive starvation",

    "crop damage": "harvests are devastated",

    "cybersecurity incident": None,

    "lack of water": "water availability",

    "food prices rise": "rising food prices",

    "reduced rainfall": "inadequate rainfall",

    "rising rent": "price rise",

    "confiscation": "land seizures",

    "drought relief": "drought",

    "lockdown": None,

    "public health risk": "epidemics",

    "pandemic": "epidemics",

    "education disruption": None,

    "military coup": "coup",

    "data theft": None,

    "convoy attack": "convoys",

    "gang activity": "gangs of bandits",

    "regional instability": "civil strife",

    "criminal violence": "mayhem",

    "international troops": "foreign troops",

    "devastate the economy": "devastated the economy",

    "hail": "weather extremes",

    "forced migration": "migration",

    "food price inflation": "rising food prices",

    "criminality": None,

    "economic disaster": "economic crisis",

    "brutal regimes": "brutal government",

    "refugee crises": "refugees",

    "diphtheria": "epidemics",

    "sexual abuse": "human rights abuses",

    "bands of bandits": "gangs of bandits",

    "stolen food": "looting",

    "toppling": "overthrow",

    "landmine explosion": None,

    "rising cost of living": "price rise",

    "rising insecurity": None,

    "security issues": None,

    "fuel scarcity": None,

    "restricted trade": "disrupted trade",

    "Internet shutdown": None,

    "refugee camp": "makeshift camps",

    "high levels of crime": None,

    "land grabbers": "land grab",

    "die-offs": "livestock had died",

    "internal conflict": "internal strife",

    "aid convoys": "convoys",

    "currency collapse": "economic crisis",

    "political chaos": "mayhem",

    "job losses": "economic crisis",

    "overhunting": "environmental degradation",

    "excess deaths": None,

    "international aid": "foreign aid",

    "unrest": "civil strife",

    "emergency": None,

    "security concerns": None,

    "militants": "militia groups",

    "child labour": None,

    "land policy": "land reform",

    "genocidal efforts": "human rights abuses",

    "housing insecurity": None,

    "migration crisis": "migration",

    "failed offensive": "the offensive",

    "economic": None,

    "international military intervention": "international intervention",

    "aid blockade": "blockade",

    "war crimes": "human rights abuses",

    "illegal logging": "forests destroyed",

    "separation of families": "displaced",

    "landslides": "natural disaster",

    "toll on trade": "disrupted trade",

    "pandemics": "epidemics",

    "diarrhoea": "gastrointestinal",

    "food price crisis": "rising food prices",

    "food price spikes": "rising food prices",

    "food catastrophe": "food crisis",

    "spreading violence": "continued strife",

    "child abduction": None,

    "drug problem": None,

    "economic inequality": "economic impoverishment",

    "missile test": None,

    "surgical strike": "air attack",
}


def _load_ontology(ontology_path: pathlib.Path) -> tuple[list[str], dict[str, str]]:
    with open(ontology_path) as f:
        data = json.load(f)
    labels = sorted(data["events"].keys())
    descriptions = data["events"]
    return labels, descriptions


def _fuzzy_match(invalid: str, valid_labels: list[str], threshold: float) -> Optional[str]:
    """Return the best fuzzy match above threshold, or None."""
    if not invalid:
        return None
    matches = difflib.get_close_matches(
        invalid.lower(),
        [v.lower() for v in valid_labels],
        n=1,
        cutoff=threshold,
    )
    if not matches:
        return None
    matched_lower = matches[0]
    return next(v for v in valid_labels if v.lower() == matched_lower)


def _build_mapping(
    invalid_types: list[str],
    valid_labels: list[str],
    valid_set: set[str],
    threshold: float,
) -> dict[str, Optional[str]]:
    mapping: dict[str, Optional[str]] = {}
    overrides_lower = {k.lower(): v for k, v in MANUAL_OVERRIDES.items()}

    for invalid in invalid_types:
        override = overrides_lower.get(invalid.lower())
        if override is not None or invalid.lower() in overrides_lower:
            if override is None or override in valid_set:
                mapping[invalid] = override
                continue
        fuzzy = _fuzzy_match(invalid, valid_labels, threshold)
        mapping[invalid] = fuzzy

    return mapping


def _scan_shard(shard: pathlib.Path, valid_set: frozenset) -> tuple[int, Counter]:
    counter: Counter = Counter()
    total = 0
    with open(shard) as f:
        for line in f:
            rec = json.loads(line)
            for wp in rec.get("window_predictions", []):
                for ev in wp.get("prediction", {}).get("events", []):
                    total += 1
                    et = ev.get("event_type", "")
                    if et not in valid_set:
                        counter[et] += 1
    return total, counter


def _remap_events(events: list[dict], valid_set: frozenset, mapping: dict) -> tuple[list[dict], Counter]:
    out = []
    stats: Counter = Counter()
    for ev in events:
        et = ev.get("event_type", "")
        if et in valid_set:
            out.append(ev)
            stats["kept"] += 1
        elif mapping.get(et) is not None:
            out.append({**ev, "event_type": mapping[et]})
            stats["fixed"] += 1
        else:
            stats["dropped"] += 1
    return out, stats


def _fix_shard(
    shard: pathlib.Path,
    output_path: pathlib.Path,
    valid_set: frozenset,
    mapping: dict,
) -> Counter:
    out_file = output_path / shard.name
    stats: Counter = Counter()
    with open(shard) as fin, open(out_file, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            for wp in rec.get("window_predictions", []):
                pred = wp.get("prediction", {})
                pred["events"], s = _remap_events(pred.get("events", []), valid_set, mapping)
                stats += s
            if "predictions" in rec:
                rec["predictions"], s = _remap_events(rec["predictions"], valid_set, mapping)
                stats += s
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return stats


def fix_annotations(
    input_dir: str,
    output_dir: str = "",
    ontology: str = str(DEFAULT_ONTOLOGY),
    threshold: float = 0.65,
    dry_run: bool = False,
    report: bool = False,
    workers: int = 0,
) -> None:
    """Fix invalid event_type labels in prediction JSONL shards.

    Args:
        input_dir:  Directory containing *.geo.jsonl (or *.jsonl) shards.
        output_dir: Destination directory for fixed shards.
        ontology:   Path to the ontology JSON file (science.json format).
        threshold:  Minimum difflib similarity score to accept a fuzzy match (0–1).
        dry_run:    Print the proposed mapping without writing any files.
        report:     Only print unique invalid event types and their counts; no mapping, no output.
        workers:    Number of parallel worker processes (0 = cpu_count).
    """
    if not output_dir and not dry_run and not report:
        raise ValueError("output_dir is required unless --dry_run or --report is set")

    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir) if output_dir else pathlib.Path(input_dir)
    ontology_path = pathlib.Path(ontology)
    n_workers = workers or os.cpu_count() or 1

    valid_labels, _ = _load_ontology(ontology_path)
    valid_set: frozenset = frozenset(valid_labels)

    shards = sorted(input_path.glob("*.geo.jsonl")) or sorted(input_path.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No *.geo.jsonl or *.jsonl files found in {input_path}")

    # --- Pass 1: collect all unique invalid event types (parallel) ---
    invalid_counter: Counter = Counter()
    total_events = 0
    scan_fn = functools.partial(_scan_shard, valid_set=valid_set)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(scan_fn, shard): shard for shard in shards}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(shards), desc="Scanning"):
            t, c = fut.result()
            total_events += t
            invalid_counter += c

    unique_invalid = list(invalid_counter.keys())
    total_invalid = sum(invalid_counter.values())
    print(f"\nFound {total_invalid} invalid event annotations ({len(unique_invalid)} unique types) out of {total_events:,} total")

    if not unique_invalid:
        print("Nothing to fix.")
        return

    if report:
        print(f"\n{'INVALID LABEL':<45} {'COUNT':>6}")
        print("-" * 53)
        for et, count in sorted(invalid_counter.items(), key=lambda x: -x[1]):
            print(f"  {repr(et):<43} {count:>6}")
        return

    # --- Build mapping ---
    mapping = _build_mapping(unique_invalid, valid_labels, set(valid_set), threshold)

    fixed = {k: v for k, v in mapping.items() if v is not None}
    dropped = {k for k, v in mapping.items() if v is None}

    print(f"Will fix:  {sum(invalid_counter[k] for k in fixed)} events ({len(fixed)} unique types)")
    print(f"Will drop: {sum(invalid_counter[k] for k in dropped)} events ({len(dropped)} unique types)")
    print(f"\n{'INVALID LABEL':<45} {'COUNT':>6}  {'ACTION'}")
    print("-" * 75)
    for invalid, count in sorted(invalid_counter.items(), key=lambda x: -x[1]):
        action = f"→ {mapping[invalid]}" if mapping[invalid] else "DROP"
        print(f"  {repr(invalid):<43} {count:>6}  {action}")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    # --- Pass 2: apply mapping and write output (parallel) ---
    output_path.mkdir(parents=True, exist_ok=True)
    total_stats: Counter = Counter()
    fix_fn = functools.partial(_fix_shard, output_path=output_path, valid_set=valid_set, mapping=mapping)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fix_fn, shard): shard for shard in shards}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(shards), desc="Fixing"):
            total_stats += fut.result()

    print(
        f"\nDone — kept: {total_stats['kept']:,}  fixed: {total_stats['fixed']:,}  dropped: {total_stats['dropped']:,}"
    )
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    fire.Fire(fix_annotations)
