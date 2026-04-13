# /// script
# dependencies = [
#   "gliner2", "urllib3", "requests", "semantic-text-splitter",
#   "spacy",
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
# ]
# ///

"""
Run v2, v4, v5 extraction on all 50 real news articles and evaluate
against gold annotations.
"""

import json
import sys
import os
import time
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent

with open(BASE / "news_articles_50.json") as f:
    articles = json.load(f)

with open(BASE / "gold_annotations.json") as f:
    gold_annotations = json.load(f)

gold_by_id = {a["id"]: a for a in gold_annotations}

print(f"Loaded {len(articles)} articles, {len(gold_annotations)} annotations")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT EXTRACTION PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

# We import the v5 pipeline (which includes GLiNER2 + spaCy)
# and also replicate v2's simpler logic for comparison.

from gliner2 import GLiNER2
from semantic_text_splitter import TextSplitter
import re

extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

# ── Risk factor definitions (shared) ──
RISK_FACTOR_ENTITIES = {
    "agricultural production issues": "Failed crops, bad harvests, lack of cultivation, livestock or cattle death, agricultural infrastructure damage.",
    "conflicts and violence": "Warfare, fighting, civil strife, rebel insurgencies, militia groups, warlords, banditry, piracy, air attacks, sieges, bombings, terrorism, violent suppression.",
    "economic issues": "Economic crises, collapsing economy, disrupted trade, external debt, brain drain, rising inflation, drastic price rises, widespread poverty.",
    "environmental issues": "Ecological crises, climate change, greenhouse gases, environmental degradation, natural disasters.",
    "food crisis": "Food insecurity, mass hunger, acute hunger, starvation, malnutrition, dehydration, gastrointestinal diseases, infant mortality.",
    "forced displacement": "Forced migration, displaced individuals, vulnerable populations fleeing homes, asylum seekers, refugees, makeshift camps.",
    "humanitarian aid": "Humanitarian disasters, aid appeals, food assistance, international embargoes, restricted humanitarian access, withheld or stolen relief, self-reliance.",
    "land-related issues": "Land invasions, hostile land grabs, systemic land seizures, burning houses, failed land reform, land degradation, poor soil quality, destroyed forests.",
    "pests and diseases": "Swarms of locusts, agricultural pests, potato blight, epidemics, cholera outbreaks, animal diseases like rinderpest or cattle plague.",
    "political instability": "Collapse of government, lack of authority, corruption, mismanagement, power struggles, secession, coup d'etat, violent overthrows, dictatorships, military juntas.",
    "weather shocks": "Climatic hazards, weather extremes, destructive floods, cyclones, prolonged dry spells, harsh droughts, abnormally low rainfall.",
    "other": "Uncategorized catastrophes, man-made disasters, severe population crises.",
}

_EVENT_LABELS = {"event"} | set(RISK_FACTOR_ENTITIES)

ENTITY_TYPES = {
    "event": "Any event, incident, crisis, disaster, or phenomenon.",
    "location": "Any geographic location, city, country, or region.",
    "date": "A temporal expression including specific dates or relative times.",
    "organization": "An organization, government body, agency, or institution.",
    **RISK_FACTOR_ENTITIES,
}

NER_SCHEMA = extractor.create_schema().entities(ENTITY_TYPES)

# ─────────────────────────────────────────────────────────────────────────────
# V2 LOGIC: simple sentence regex + neighbor fallback
# ─────────────────────────────────────────────────────────────────────────────

_V2_SENT_RE = re.compile(r'(?<=[.!?])\s+|\n\n+')

def _v2_sentence_spans(text):
    spans = []
    prev = 0
    for m in _V2_SENT_RE.finditer(text):
        spans.append((prev, m.start()))
        prev = m.end()
    if prev < len(text):
        spans.append((prev, len(text)))
    return spans

def _v2_sent_idx(char_start, spans):
    for i, (s, e) in enumerate(spans):
        if s <= char_start < e:
            return i
    return -1

def run_v2(text):
    raw = extractor.extract(text, NER_SCHEMA, threshold=0.3, include_spans=True, include_confidence=True)
    # Flatten + dedup (v2 style: keep all, no overlap filtering)
    entities = []
    seen = set()
    for label, spans in raw.get("entities", {}).items():
        for span in spans:
            key = (span["text"].lower(), label)
            if key in seen:
                continue
            seen.add(key)
            entities.append({**span, "label": label})

    sentences = _v2_sentence_spans(text)
    events_by_sent = {}
    locs_by_sent = {}
    for ent in entities:
        si = _v2_sent_idx(ent["start"], sentences)
        if si < 0:
            continue
        if ent["label"] == "location":
            locs_by_sent.setdefault(si, []).append(ent)
        elif ent["label"] in _EVENT_LABELS:
            events_by_sent.setdefault(si, []).append(ent)

    pairs = []
    pair_seen = set()
    for si in sorted(events_by_sent):
        locs = locs_by_sent.get(si, [])
        if not locs:
            for neighbor in (si - 1, si + 1):
                locs = locs_by_sent.get(neighbor, [])
                if locs:
                    break
        for ev in events_by_sent[si]:
            for loc in locs:
                key = (ev["text"].lower(), loc["text"].lower(), ev["label"])
                if key in pair_seen:
                    continue
                pair_seen.add(key)
                pairs.append((ev["text"].lower(), loc["text"].lower()))

    return entities, pairs


# ─────────────────────────────────────────────────────────────────────────────
# V4 LOGIC: waterfall + closest distance + macro-scope
# ─────────────────────────────────────────────────────────────────────────────

_sentence_splitter = TextSplitter((1, 350), trim=False)

LABEL_THRESHOLDS = {
    "location": 0.80, "event": 0.50, "date": 0.50, "organization": 0.60,
    **{k: 0.40 for k in RISK_FACTOR_ENTITIES},
}

_MACRO_RE = re.compile(
    r"\bthe\s+(?:nation|country|countries|province|provinces|region|regions"
    r"|republic|territory|territories|state|homeland)\b", re.IGNORECASE,
)

def _sts_sentence_spans(text):
    return [(o, o + len(c)) for o, c in _sentence_splitter.chunk_indices(text) if c.strip()]

def _sts_sent_idx(char_start, spans):
    for i, (s, e) in enumerate(spans):
        if s <= char_start < e:
            return i
    return -1

def _v4_dedup(raw):
    all_spans = []
    for label, span_list in raw.get("entities", {}).items():
        for span in span_list:
            all_spans.append({**span, "label": label})
    all_spans.sort(key=lambda s: (s["start"], s["label"] == "event"))
    filtered = []
    for span in all_spans:
        thresh = LABEL_THRESHOLDS.get(span["label"], 0.30)
        if span["confidence"] < thresh:
            continue
        merged = False
        for ex in filtered:
            if max(span["start"], ex["start"]) < min(span["end"], ex["end"]):
                if span["label"] != "event" and ex["label"] == "event":
                    ex["label"] = span["label"]
                    ex["confidence"] = span["confidence"]
                merged = True
                break
        if not merged:
            filtered.append(span)
    return sorted(filtered, key=lambda s: s["start"])

def run_v4(text):
    raw = extractor.extract(text, NER_SCHEMA, threshold=0.30, include_spans=True, include_confidence=True)
    entities = _v4_dedup(raw)
    sentences = _sts_sentence_spans(text)

    events_by_sent = {}
    locs_by_sent = {}
    for ent in entities:
        si = _sts_sent_idx(ent["start"], sentences)
        if si < 0:
            continue
        if ent["label"] == "location":
            locs_by_sent.setdefault(si, []).append(ent)
        elif ent["label"] in _EVENT_LABELS:
            events_by_sent.setdefault(si, []).append(ent)

    major_loc = None
    active_loc = None
    pairs = []
    pair_seen = set()

    for si in range(len(sentences)):
        locs_here = locs_by_sent.get(si, [])
        if locs_here:
            best = max(locs_here, key=lambda l: l["confidence"])
            if major_loc is None:
                major_loc = best
            active_loc = best

        events_here = events_by_sent.get(si)
        if not events_here:
            continue

        ss, se = sentences[si]
        sent_text = text[ss:se]

        if locs_here:
            for ev in events_here:
                closest = min(locs_here, key=lambda loc: abs(loc["start"] - ev["start"]))
                key = (ev["text"].lower(), closest["text"].lower(), ev["label"])
                if key not in pair_seen:
                    pair_seen.add(key)
                    pairs.append((ev["text"].lower(), closest["text"].lower()))
        elif _MACRO_RE.search(sent_text) and major_loc:
            for ev in events_here:
                key = (ev["text"].lower(), major_loc["text"].lower(), ev["label"])
                if key not in pair_seen:
                    pair_seen.add(key)
                    pairs.append((ev["text"].lower(), major_loc["text"].lower()))
        elif active_loc:
            for ev in events_here:
                key = (ev["text"].lower(), active_loc["text"].lower(), ev["label"])
                if key not in pair_seen:
                    pair_seen.add(key)
                    pairs.append((ev["text"].lower(), active_loc["text"].lower()))

    return entities, pairs


# ─────────────────────────────────────────────────────────────────────────────
# V5 LOGIC: v4 + coref + gazetteer + dep parse
# ─────────────────────────────────────────────────────────────────────────────

import spacy
from spacy.matcher import PhraseMatcher

_nlp = spacy.load("en_core_web_sm")

CRISIS_GAZETTEER = {
    "agricultural production issues": ["crop failure", "harvest failure", "failed crops", "bad harvests", "cattle death", "livestock death"],
    "conflicts and violence": ["civil war", "armed conflict", "bombing", "insurgency", "terrorism", "violent clashes", "fighting", "warfare", "siege", "airstrike"],
    "economic issues": ["economic crisis", "hyperinflation", "poverty", "unemployment", "economic collapse"],
    "food crisis": ["famine", "food crisis", "food insecurity", "mass hunger", "starvation", "malnutrition"],
    "forced displacement": ["displacement", "refugee crisis", "forced migration", "internally displaced", "refugee camp"],
    "humanitarian aid": ["aid appeal", "humanitarian crisis", "food aid", "relief effort"],
    "pests and diseases": ["cholera", "epidemic", "pandemic", "locust swarm", "plague", "disease outbreak"],
    "political instability": ["coup", "government collapse", "regime change", "political crisis"],
    "weather shocks": ["drought", "flood", "cyclone", "hurricane", "typhoon", "tornado", "heatwave", "rural collapse"],
}

_pm = PhraseMatcher(_nlp.vocab, attr="LOWER")
for _lbl, _terms in CRISIS_GAZETTEER.items():
    _pm.add(_lbl, [_nlp.make_doc(t) for t in _terms])

_SOURCE_VERBS = frozenset({"flee", "escape", "uproot", "displace", "evacuate", "leave", "abandon"})
_DEST_VERBS = frozenset({"arrive", "seek", "shelter", "reach", "resettle", "move", "migrate"})

def _classify_loc_role(loc_s, loc_e, sent_s, sent_e, text):
    doc = _nlp(text[sent_s:sent_e])
    rel_s = loc_s - sent_s
    for token in doc:
        if not (token.idx >= rel_s and token.idx < loc_e - sent_s):
            continue
        head = token
        while head.dep_ in ("compound", "flat"):
            head = head.head
        if head.dep_ == "pobj" and head.head.dep_ == "prep":
            prep = head.head
            verb = prep.head
            while verb.pos_ != "VERB" and verb.head != verb:
                verb = verb.head
            if prep.text.lower() == "from":
                return "SOURCE"
            elif prep.text.lower() in ("to", "into"):
                return "DESTINATION"
            elif prep.text.lower() == "in":
                if verb.lemma_.lower() in _SOURCE_VERBS:
                    return "SOURCE"
                elif verb.lemma_.lower() in _DEST_VERBS:
                    return "DESTINATION"
        break
    return "LOCATION"

def _coref_map_for(text):
    """Try to run coref subprocess; return empty dict on failure."""
    worker = BASE / "_coref_worker.py"
    if not worker.exists():
        return {}
    import subprocess
    try:
        proc = subprocess.run(
            ["uv", "run", str(worker)],
            input=json.dumps([text]), capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            return {}
        results = json.loads(proc.stdout)
        m = {}
        for cluster in results[0]:
            offs = cluster["offsets"]
            strs = cluster["strings"]
            if len(offs) < 2:
                continue
            for (ms, me), mt in zip(offs[1:], strs[1:]):
                m[(ms, me)] = (offs[0][0], offs[0][1], strs[0])
        return m
    except Exception:
        return {}

def run_v5(text, use_coref=True):
    raw = extractor.extract(text, NER_SCHEMA, threshold=0.30, include_spans=True, include_confidence=True)
    entities = _v4_dedup(raw)

    # Gazetteer
    doc = _nlp(text)
    for mid, start, end in _pm(doc):
        span = doc[start:end]
        label = _nlp.vocab.strings[mid]
        gaz_ent = {"text": span.text, "label": label, "confidence": 1.0,
                    "start": span.start_char, "end": span.end_char, "source": "gazetteer"}
        overlap = False
        for ex in entities:
            if max(gaz_ent["start"], ex["start"]) < min(gaz_ent["end"], ex["end"]):
                if gaz_ent["label"] != "event" and ex.get("label") == "event":
                    ex["label"] = gaz_ent["label"]
                    ex["confidence"] = 1.0
                overlap = True
                break
        if not overlap:
            entities.append(gaz_ent)
    entities.sort(key=lambda e: e["start"])

    sentences = _sts_sentence_spans(text)
    events_by_sent = {}
    locs_by_sent = {}
    for ent in entities:
        si = _sts_sent_idx(ent["start"], sentences)
        if si < 0:
            continue
        if ent["label"] == "location":
            locs_by_sent.setdefault(si, []).append(ent)
        elif ent["label"] in _EVENT_LABELS:
            events_by_sent.setdefault(si, []).append(ent)

    # Frequency-based major location
    loc_counts = Counter()
    loc_by_text = {}
    for locs in locs_by_sent.values():
        for loc in locs:
            k = loc["text"].lower()
            loc_counts[k] += 1
            if k not in loc_by_text or loc["confidence"] > loc_by_text[k]["confidence"]:
                loc_by_text[k] = loc
    major_loc = loc_by_text.get(loc_counts.most_common(1)[0][0]) if loc_counts else None

    # Coref
    coref_locs_by_sent = {}
    if use_coref:
        cmap = _coref_map_for(text)
        for (ms, me), (as_, ae, at) in cmap.items():
            for le in loc_by_text.values():
                if (le["start"] >= as_ and le["end"] <= ae) or (as_ >= le["start"] and ae <= le["end"]):
                    si = _sts_sent_idx(ms, sentences)
                    if si >= 0:
                        coref_locs_by_sent.setdefault(si, []).append(le)
                    break

    active_loc = None
    pairs = []
    pair_seen = set()

    for si in range(len(sentences)):
        locs_here = locs_by_sent.get(si, [])
        coref_locs = coref_locs_by_sent.get(si, [])
        if locs_here:
            active_loc = max(locs_here, key=lambda l: l["confidence"])
        events_here = events_by_sent.get(si)
        if not events_here:
            continue
        ss, se = sentences[si]

        if locs_here:
            for ev in events_here:
                sorted_locs = sorted(locs_here, key=lambda l: abs(l["start"] - ev["start"]))
                for loc in sorted_locs:
                    role = _classify_loc_role(loc["start"], loc["end"], ss, se, text)
                    if role == "DESTINATION" and ev["label"] in _EVENT_LABELS:
                        continue
                    key = (ev["text"].lower(), loc["text"].lower(), ev["label"])
                    if key not in pair_seen:
                        pair_seen.add(key)
                        pairs.append((ev["text"].lower(), loc["text"].lower()))
                    break
        elif coref_locs:
            best = max(coref_locs, key=lambda l: l["confidence"])
            for ev in events_here:
                key = (ev["text"].lower(), best["text"].lower(), ev["label"])
                if key not in pair_seen:
                    pair_seen.add(key)
                    pairs.append((ev["text"].lower(), best["text"].lower()))
        elif _MACRO_RE.search(text[ss:se]) and major_loc:
            for ev in events_here:
                key = (ev["text"].lower(), major_loc["text"].lower(), ev["label"])
                if key not in pair_seen:
                    pair_seen.add(key)
                    pairs.append((ev["text"].lower(), major_loc["text"].lower()))
        elif active_loc:
            for ev in events_here:
                key = (ev["text"].lower(), active_loc["text"].lower(), ev["label"])
                if key not in pair_seen:
                    pair_seen.add(key)
                    pairs.append((ev["text"].lower(), active_loc["text"].lower()))

    return entities, pairs


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(predicted_set, gold_set):
    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) > 0 else 0.0
    jacc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "P": round(p, 4), "R": round(r, 4),
            "F1": round(f1, 4), "F0.5": round(f05, 4), "F2": round(f2, 4), "Jacc": round(jacc, 4)}


def categorize_label(label):
    label = label.lower()
    if label == "location":
        return "location"
    if label == "date":
        return "date"
    if label == "organization":
        return "organization"
    return "event_or_risk"


# ─────────────────────────────────────────────────────────────────────────────
# RUN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    versions = {"v2": run_v2, "v4": run_v4, "v5_no_coref": lambda t: run_v5(t, use_coref=False)}

    # Accumulators
    all_ent_metrics = {v: {"TP": 0, "FP": 0, "FN": 0} for v in versions}
    all_pair_metrics = {v: {"TP": 0, "FP": 0, "FN": 0} for v in versions}
    timings = {v: 0.0 for v in versions}

    for idx, article in enumerate(articles):
        aid = article["id"]
        text = article["text"]
        gold = gold_by_id.get(aid, {"entities": [], "event_location_pairs": []})

        gold_ent_set = set()
        for ent in gold["entities"]:
            gold_ent_set.add((ent[0].lower(), ent[1]))
        gold_pair_set = set()
        for pair in gold["event_location_pairs"]:
            gold_pair_set.add((pair[0].lower(), pair[1].lower()))

        print(f"\r  [{idx+1:2d}/50] {aid} ({len(text)} chars, {len(gold_ent_set)} gold ents, {len(gold_pair_set)} gold pairs)", end="", flush=True)

        for vname, vfunc in versions.items():
            t0 = time.time()
            try:
                ents, pairs = vfunc(text)
            except Exception as e:
                print(f"\n    ERROR {vname} on {aid}: {e}")
                ents, pairs = [], []
            elapsed = time.time() - t0
            timings[vname] += elapsed

            pred_ent_set = set()
            for ent in ents:
                lbl = ent.get("label", "event")
                pred_ent_set.add((ent["text"].lower(), categorize_label(lbl)))

            pred_pair_set = set()
            for p in pairs:
                pred_pair_set.add((p[0].lower(), p[1].lower()))

            em = compute_metrics(pred_ent_set, gold_ent_set)
            pm = compute_metrics(pred_pair_set, gold_pair_set)

            all_ent_metrics[vname]["TP"] += em["TP"]
            all_ent_metrics[vname]["FP"] += em["FP"]
            all_ent_metrics[vname]["FN"] += em["FN"]
            all_pair_metrics[vname]["TP"] += pm["TP"]
            all_pair_metrics[vname]["FP"] += pm["FP"]
            all_pair_metrics[vname]["FN"] += pm["FN"]

    print("\n")

    # Compute aggregate metrics
    def agg(acc):
        tp, fp, fn = acc["TP"], acc["FP"], acc["FN"]
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        f05 = 1.25*p*r/(0.25*p+r) if (0.25*p+r) else 0
        f2 = 5*p*r/(4*p+r) if (4*p+r) else 0
        jacc = tp/(tp+fp+fn) if (tp+fp+fn) else 0
        return {**acc, "P": round(p,4), "R": round(r,4), "F1": round(f1,4),
                "F0.5": round(f05,4), "F2": round(f2,4), "Jacc": round(jacc,4)}

    print("=" * 95)
    print("ENTITY EXTRACTION  (micro-averaged over 50 articles, 671 gold entities)")
    print("=" * 95)
    hdr = f"{'Ver':<14} | {'TP':>4} {'FP':>5} {'FN':>5} | {'Prec':>6} {'Rec':>6} {'F1':>6} {'F0.5':>6} {'F2':>6} {'Jacc':>6} | {'Time':>6}"
    print(hdr)
    print("-" * len(hdr))
    for v in versions:
        m = agg(all_ent_metrics[v])
        t = timings[v]
        print(f"{v:<14} | {m['TP']:>4} {m['FP']:>5} {m['FN']:>5} | {m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} {m['F0.5']:>6.3f} {m['F2']:>6.3f} {m['Jacc']:>6.3f} | {t:>5.1f}s")

    print(f"\n{'=' * 95}")
    print("EVENT-LOCATION LINKING  (micro-averaged over 50 articles, 140 gold pairs)")
    print("=" * 95)
    print(hdr)
    print("-" * len(hdr))
    for v in versions:
        m = agg(all_pair_metrics[v])
        t = timings[v]
        print(f"{v:<14} | {m['TP']:>4} {m['FP']:>5} {m['FN']:>5} | {m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f} {m['F0.5']:>6.3f} {m['F2']:>6.3f} {m['Jacc']:>6.3f} | {t:>5.1f}s")

    # Save results
    results = {
        "entity_metrics": {v: agg(all_ent_metrics[v]) for v in versions},
        "pair_metrics": {v: agg(all_pair_metrics[v]) for v in versions},
        "timings_seconds": {v: round(timings[v], 2) for v in versions},
        "dataset": {"articles": 50, "gold_entities": 671, "gold_pairs": 140},
    }
    with open(BASE / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] evaluation_results.json")
