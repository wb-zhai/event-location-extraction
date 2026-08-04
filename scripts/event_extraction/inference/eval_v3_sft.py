from __future__ import annotations

import argparse
import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _gold_events(rec: dict[str, Any]) -> list[dict[str, Any]]:
    events = rec.get("annotation", {}).get("events", [])
    return [e for e in events if isinstance(e, dict)]


def _pred_events(rec: dict[str, Any]) -> list[dict[str, Any]]:
    events = rec.get("predictions", [])
    return [e for e in events if isinstance(e, dict)]


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

_DIRECTION_PREFIX = re.compile(
    r"^(northern?|southern?|eastern?|western?|central)\s+",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _norm_loc(s: str) -> str:
    s = _norm(s)
    s = _DIRECTION_PREFIX.sub("", s)
    return s.strip()


def _split_locs(s: str) -> list[str]:
    """Split a ';'-separated location string into individual normalised parts."""
    parts = []
    for part in s.split(";"):
        n = _norm_loc(part.strip())
        if n and n != "not_stated":
            parts.append(n)
    return parts


def _quote_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# ---------------------------------------------------------------------------
# Cluster map helpers
# ---------------------------------------------------------------------------

def _load_cluster_map(path: Path) -> dict[str, str]:
    """Load {normalized_event_name -> cluster} from a Studio results JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {_norm(entry["name"]): entry["cluster"] for entry in data}


def _to_cluster(event_type: str, cluster_map: dict[str, str]) -> str:
    """Return the cluster for an event type, falling back to the type itself."""
    return cluster_map.get(_norm(event_type), _norm(event_type))


# ---------------------------------------------------------------------------
# Field-level comparisons
# ---------------------------------------------------------------------------

_LOC_FUZZY_THRESHOLD = 0.85


def _loc_match(gold: str, pred: str, tier: str) -> bool:
    """Compare two location strings, supporting ';'-separated multiple locations."""
    gn = _norm_loc(gold)
    pn = _norm_loc(pred)
    if gn == "not_stated" or pn == "not_stated":
        return gn == pn
    g_parts = _split_locs(gold)
    p_parts = _split_locs(pred)
    if not g_parts and not p_parts:
        return True
    if not g_parts or not p_parts:
        return False
    if tier == "exact":
        return set(g_parts) == set(p_parts)
    # relaxed: every part in each side must fuzzy-match some part on the other side
    def _single(a: str, b: str) -> bool:
        if a == b or a in b or b in a:
            return True
        return SequenceMatcher(None, a, b).ratio() >= _LOC_FUZZY_THRESHOLD
    return (
        all(any(_single(g, p) for p in p_parts) for g in g_parts)
        and all(any(_single(p, g) for g in g_parts) for p in p_parts)
    )


def _parse_iso_parts(s: str) -> list[str]:
    """Split an ISO 8601 date or range into year prefixes."""
    parts = s.split("/")
    years = []
    for p in parts:
        if p and p != "not_stated":
            years.append(p[:4])  # first 4 chars = year
    return years


def _admin_level_match(gold: str, pred: str, tier: str) -> bool:
    """Compare two ';'-separated event_location_admin_level strings."""
    gn = _norm(gold)
    pn = _norm(pred)
    if gn == pn:
        return True
    if tier == "exact":
        return False
    # relaxed: same set of segments, order-insensitive
    g_parts = {p.strip() for p in gn.split(";") if p.strip()}
    p_parts = {p.strip() for p in pn.split(";") if p.strip()}
    return g_parts == p_parts


def _time_match(gold: str, pred: str, tier: str) -> bool:
    """Compare two event_time strings. tier='exact' or 'relaxed'."""
    gn = _norm(gold)
    pn = _norm(pred)
    if gn == pn:
        return True
    if tier == "exact":
        return False
    # relaxed: same-year prefix counts
    g_years = _parse_iso_parts(gn)
    p_years = _parse_iso_parts(pn)
    if not g_years and not p_years:
        return gn == pn  # both not_stated handled above already
    return bool(set(g_years) & set(p_years))


# ---------------------------------------------------------------------------
# Event matching (alignment)
# ---------------------------------------------------------------------------

_QUOTE_MATCH_THRESHOLD = 0.5


def _event_sim(
    gold: dict, pred: dict,
    cluster_map: dict[str, str] | None = None,
) -> float:
    """Similarity score for matching a predicted event to a gold event."""
    if cluster_map is not None:
        gt = _to_cluster(gold.get("event_type", ""), cluster_map)
        pt = _to_cluster(pred.get("event_type", ""), cluster_map)
    else:
        gt = _norm(gold.get("event_type", ""))
        pt = _norm(pred.get("event_type", ""))
    if gt != pt:
        return 0.0
    return _quote_sim(
        gold.get("grounding_quote", ""),
        pred.get("grounding_quote", ""),
    )


def _match_events(
    gold: list[dict], pred: list[dict], tier: str,
    cluster_map: dict[str, str] | None = None,
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """
    Greedy one-to-one bipartite matching by descending similarity.
    Returns (matched_pairs, unmatched_gold, unmatched_pred).
    tier='exact' requires quote_sim==1.0; 'relaxed' requires >=0.5.
    """
    threshold = 1.0 if tier == "exact" else _QUOTE_MATCH_THRESHOLD
    # build all (sim, gi, pi) triples
    sims: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            s = _event_sim(g, p, cluster_map=cluster_map)
            if s >= threshold:
                sims.append((s, gi, pi))
    sims.sort(key=lambda x: -x[0])

    matched_g: set[int] = set()
    matched_p: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    for s, gi, pi in sims:
        if gi in matched_g or pi in matched_p:
            continue
        pairs.append((gold[gi], pred[pi]))
        matched_g.add(gi)
        matched_p.add(pi)

    unmatched_gold = [g for i, g in enumerate(gold) if i not in matched_g]
    unmatched_pred = [p for i, p in enumerate(pred) if i not in matched_p]
    return pairs, unmatched_gold, unmatched_pred


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _accuracy(correct: int, total: int) -> dict[str, float]:
    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc, "correct": correct, "total": total}


def _macro_average(per_doc: list[dict[str, float]]) -> dict[str, float]:
    if not per_doc:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": statistics.mean(d["precision"] for d in per_doc),
        "recall": statistics.mean(d["recall"] for d in per_doc),
        "f1": statistics.mean(d["f1"] for d in per_doc),
    }


def _macro_accuracy(per_doc: list[dict[str, float]]) -> dict[str, float]:
    if not per_doc:
        return {"accuracy": 0.0}
    return {"accuracy": statistics.mean(d["accuracy"] for d in per_doc)}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    records: list[dict[str, Any]],
    cluster_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    # accumulators keyed by (family, tier)
    acc: dict[tuple[str, str], dict[str, int]] = {}
    macro_acc: dict[tuple[str, str], list[dict]] = {}
    accuracy_acc: dict[tuple[str, str], dict[str, int]] = {}
    macro_accuracy_acc: dict[tuple[str, str], list[dict]] = {}

    families = [
        "event", "event_type", "event_type_set",
        "location", "event_location", "event_time", "event_location_admin_level",
    ]
    accuracy_families = [
        "location_on_matched_events",
        "location_on_matched_events_gold_stated",
        "time_on_matched_events",
        "time_on_matched_events_gold_stated",
        "admin_level_on_matched_events",
        "admin_level_on_matched_events_gold_stated",
    ]
    if cluster_map is not None:
        families = [
            "event", "event_type", "event_type_set",
            "cluster_event", "cluster_type", "cluster_type_set",
            "location", "event_location", "cluster_event_location",
            "event_time", "cluster_event_time",
            "event_location_admin_level", "cluster_event_location_admin_level",
        ]
        accuracy_families.extend(
            [
                "cluster_location_on_matched_events",
                "cluster_location_on_matched_events_gold_stated",
                "cluster_time_on_matched_events",
                "cluster_time_on_matched_events_gold_stated",
                "cluster_admin_level_on_matched_events",
                "cluster_admin_level_on_matched_events_gold_stated",
            ]
        )
    tiers = ["exact", "relaxed"]
    for fam in families:
        for tier in tiers:
            acc[(fam, tier)] = {"tp": 0, "fp": 0, "fn": 0}
            macro_acc[(fam, tier)] = []
    for fam in accuracy_families:
        for tier in tiers:
            accuracy_acc[(fam, tier)] = {"correct": 0, "total": 0}
            macro_accuracy_acc[(fam, tier)] = []

    # per-event-type breakdown: keyed by (event_type, tier)
    type_acc: dict[tuple[str, str], dict[str, int]] = {}
    # per-event-type conditional accuracy: keyed by (event_type, tier, field)
    type_cond_acc: dict[tuple[str, str, str], dict[str, int]] = {}

    error_rows: list[dict] = []

    for rec in records:
        gold = _gold_events(rec)
        pred = _pred_events(rec)
        doc_id = rec.get("id", "")

        for tier in tiers:
            # ---------------------------------------------------------------
            # 1. Event extraction overall
            # ---------------------------------------------------------------
            pairs, unmatched_g, unmatched_p = _match_events(gold, pred, tier)
            tp = len(pairs)
            fp = len(unmatched_p)
            fn = len(unmatched_g)
            acc[("event", tier)]["tp"] += tp
            acc[("event", tier)]["fp"] += fp
            acc[("event", tier)]["fn"] += fn
            macro_acc[("event", tier)].append(_prf(tp, fp, fn))

            # ---------------------------------------------------------------
            # 1a. Per-event-type breakdown (event detection)
            # ---------------------------------------------------------------
            for g, p in pairs:
                et = _norm(g.get("event_type", ""))
                d = type_acc.setdefault((et, tier), {"tp": 0, "fp": 0, "fn": 0})
                d["tp"] += 1
            for g in unmatched_g:
                et = _norm(g.get("event_type", ""))
                d = type_acc.setdefault((et, tier), {"tp": 0, "fp": 0, "fn": 0})
                d["fn"] += 1
            for p in unmatched_p:
                et = _norm(p.get("event_type", ""))
                d = type_acc.setdefault((et, tier), {"tp": 0, "fp": 0, "fn": 0})
                d["fp"] += 1

            # ---------------------------------------------------------------
            # 1b. Cluster-level event extraction & event-location pairing
            # ---------------------------------------------------------------
            if cluster_map is not None:
                c_pairs, c_unmatched_g, c_unmatched_p = _match_events(
                    gold, pred, tier, cluster_map=cluster_map
                )
                c_tp = len(c_pairs)
                c_fp = len(c_unmatched_p)
                c_fn = len(c_unmatched_g)
                acc[("cluster_event", tier)]["tp"] += c_tp
                acc[("cluster_event", tier)]["fp"] += c_fp
                acc[("cluster_event", tier)]["fn"] += c_fn
                macro_acc[("cluster_event", tier)].append(_prf(c_tp, c_fp, c_fn))

                cel_tp = sum(
                    1
                    for g, p in c_pairs
                    if _loc_match(
                        g.get("event_location", ""), p.get("event_location", ""), tier
                    )
                )
                cel_fp = c_tp - cel_tp + c_fp
                cel_fn = c_tp - cel_tp + c_fn
                acc[("cluster_event_location", tier)]["tp"] += cel_tp
                acc[("cluster_event_location", tier)]["fp"] += cel_fp
                acc[("cluster_event_location", tier)]["fn"] += cel_fn
                macro_acc[("cluster_event_location", tier)].append(
                    _prf(cel_tp, cel_fp, cel_fn)
                )

                c_stated_pairs = [
                    (g, p)
                    for g, p in c_pairs
                    if _split_locs(g.get("event_location", ""))
                ]
                c_stated_correct = sum(
                    1
                    for g, p in c_stated_pairs
                    if _loc_match(
                        g.get("event_location", ""), p.get("event_location", ""), tier
                    )
                )
                for fam, correct, total in [
                    ("cluster_location_on_matched_events", cel_tp, len(c_pairs)),
                    (
                        "cluster_location_on_matched_events_gold_stated",
                        c_stated_correct,
                        len(c_stated_pairs),
                    ),
                ]:
                    accuracy_acc[(fam, tier)]["correct"] += correct
                    accuracy_acc[(fam, tier)]["total"] += total
                    macro_accuracy_acc[(fam, tier)].append(_accuracy(correct, total))

                cet_tp = sum(
                    1
                    for g, p in c_pairs
                    if _time_match(
                        g.get("event_time", "not_stated"),
                        p.get("event_time", "not_stated"),
                        tier,
                    )
                )
                cet_fp = c_tp - cet_tp + c_fp
                cet_fn = c_tp - cet_tp + c_fn
                acc[("cluster_event_time", tier)]["tp"] += cet_tp
                acc[("cluster_event_time", tier)]["fp"] += cet_fp
                acc[("cluster_event_time", tier)]["fn"] += cet_fn
                macro_acc[("cluster_event_time", tier)].append(
                    _prf(cet_tp, cet_fp, cet_fn)
                )

                c_time_stated_pairs = [
                    (g, p)
                    for g, p in c_pairs
                    if _norm(g.get("event_time", "not_stated")) != "not_stated"
                ]
                c_time_stated_correct = sum(
                    1
                    for g, p in c_time_stated_pairs
                    if _time_match(
                        g.get("event_time", "not_stated"),
                        p.get("event_time", "not_stated"),
                        tier,
                    )
                )
                for fam, correct, total in [
                    ("cluster_time_on_matched_events", cet_tp, len(c_pairs)),
                    (
                        "cluster_time_on_matched_events_gold_stated",
                        c_time_stated_correct,
                        len(c_time_stated_pairs),
                    ),
                ]:
                    accuracy_acc[(fam, tier)]["correct"] += correct
                    accuracy_acc[(fam, tier)]["total"] += total
                    macro_accuracy_acc[(fam, tier)].append(_accuracy(correct, total))

                cal_tp = sum(
                    1
                    for g, p in c_pairs
                    if _admin_level_match(
                        g.get("event_location_admin_level", "not_stated"),
                        p.get("event_location_admin_level", "not_stated"),
                        tier,
                    )
                )
                cal_fp = c_tp - cal_tp + c_fp
                cal_fn = c_tp - cal_tp + c_fn
                acc[("cluster_event_location_admin_level", tier)]["tp"] += cal_tp
                acc[("cluster_event_location_admin_level", tier)]["fp"] += cal_fp
                acc[("cluster_event_location_admin_level", tier)]["fn"] += cal_fn
                macro_acc[("cluster_event_location_admin_level", tier)].append(
                    _prf(cal_tp, cal_fp, cal_fn)
                )

                c_admin_level_stated_pairs = [
                    (g, p)
                    for g, p in c_pairs
                    if _norm(g.get("event_location_admin_level", "not_stated")) != "not_stated"
                ]
                c_admin_level_stated_correct = sum(
                    1
                    for g, p in c_admin_level_stated_pairs
                    if _admin_level_match(
                        g.get("event_location_admin_level", "not_stated"),
                        p.get("event_location_admin_level", "not_stated"),
                        tier,
                    )
                )
                for fam, correct, total in [
                    ("cluster_admin_level_on_matched_events", cal_tp, len(c_pairs)),
                    (
                        "cluster_admin_level_on_matched_events_gold_stated",
                        c_admin_level_stated_correct,
                        len(c_admin_level_stated_pairs),
                    ),
                ]:
                    accuracy_acc[(fam, tier)]["correct"] += correct
                    accuracy_acc[(fam, tier)]["total"] += total
                    macro_accuracy_acc[(fam, tier)].append(_accuracy(correct, total))

            # ---------------------------------------------------------------
            # 2. Event type (multiset per doc)
            # ---------------------------------------------------------------
            gold_types = [_norm(e.get("event_type", "")) for e in gold]
            pred_types = [_norm(e.get("event_type", "")) for e in pred]

            gt_bag = list(gold_types)
            pt_bag = list(pred_types)
            type_tp = 0
            for t in list(pt_bag):
                if t in gt_bag:
                    type_tp += 1
                    gt_bag.remove(t)
                    pt_bag.remove(t)
            type_fp = len(pt_bag)
            type_fn = len(gt_bag)
            acc[("event_type", tier)]["tp"] += type_tp
            acc[("event_type", tier)]["fp"] += type_fp
            acc[("event_type", tier)]["fn"] += type_fn
            macro_acc[("event_type", tier)].append(_prf(type_tp, type_fp, type_fn))

            # ---------------------------------------------------------------
            # 2b. Event type set (doc-level, ignoring duplicate types)
            # ---------------------------------------------------------------
            gold_type_set = {_norm(e.get("event_type", "")) for e in gold}
            pred_type_set = {_norm(e.get("event_type", "")) for e in pred}
            set_tp = len(gold_type_set & pred_type_set)
            set_fp = len(pred_type_set - gold_type_set)
            set_fn = len(gold_type_set - pred_type_set)
            acc[("event_type_set", tier)]["tp"] += set_tp
            acc[("event_type_set", tier)]["fp"] += set_fp
            acc[("event_type_set", tier)]["fn"] += set_fn
            macro_acc[("event_type_set", tier)].append(_prf(set_tp, set_fp, set_fn))

            # ---------------------------------------------------------------
            # 2c-2d. Cluster-level type metrics (only when cluster_map given)
            # ---------------------------------------------------------------
            if cluster_map is not None:
                gold_clusters = [_to_cluster(e.get("event_type", ""), cluster_map) for e in gold]
                pred_clusters = [_to_cluster(e.get("event_type", ""), cluster_map) for e in pred]

                gc_bag = list(gold_clusters)
                pc_bag = list(pred_clusters)
                ct_tp = 0
                for t in list(pc_bag):
                    if t in gc_bag:
                        ct_tp += 1
                        gc_bag.remove(t)
                        pc_bag.remove(t)
                ct_fp = len(pc_bag)
                ct_fn = len(gc_bag)
                acc[("cluster_type", tier)]["tp"] += ct_tp
                acc[("cluster_type", tier)]["fp"] += ct_fp
                acc[("cluster_type", tier)]["fn"] += ct_fn
                macro_acc[("cluster_type", tier)].append(_prf(ct_tp, ct_fp, ct_fn))

                gold_cluster_set = {_to_cluster(e.get("event_type", ""), cluster_map) for e in gold}
                pred_cluster_set = {_to_cluster(e.get("event_type", ""), cluster_map) for e in pred}
                cs_tp = len(gold_cluster_set & pred_cluster_set)
                cs_fp = len(pred_cluster_set - gold_cluster_set)
                cs_fn = len(gold_cluster_set - pred_cluster_set)
                acc[("cluster_type_set", tier)]["tp"] += cs_tp
                acc[("cluster_type_set", tier)]["fp"] += cs_fp
                acc[("cluster_type_set", tier)]["fn"] += cs_fn
                macro_acc[("cluster_type_set", tier)].append(_prf(cs_tp, cs_fp, cs_fn))

            # ---------------------------------------------------------------
            # 3. Location extraction (per-doc set, ignoring not_stated)
            # ---------------------------------------------------------------
            gold_locs = [
                loc
                for e in gold
                for loc in _split_locs(e.get("event_location", ""))
            ]
            pred_locs = [
                loc
                for e in pred
                for loc in _split_locs(e.get("event_location", ""))
            ]

            gl_remaining = list(gold_locs)
            loc_tp = 0
            for pl in pred_locs:
                for i, gl in enumerate(gl_remaining):
                    if _loc_match(gl, pl, tier):
                        loc_tp += 1
                        gl_remaining.pop(i)
                        break
            loc_fp = len(pred_locs) - loc_tp
            loc_fn = len(gl_remaining)
            acc[("location", tier)]["tp"] += loc_tp
            acc[("location", tier)]["fp"] += loc_fp
            acc[("location", tier)]["fn"] += loc_fn
            macro_acc[("location", tier)].append(_prf(loc_tp, loc_fp, loc_fn))

            # ---------------------------------------------------------------
            # 4. Event-location pairing (conditioned on matched events)
            # ---------------------------------------------------------------
            el_tp = sum(
                1
                for g, p in pairs
                if _loc_match(
                    g.get("event_location", ""), p.get("event_location", ""), tier
                )
            )
            el_fp = tp - el_tp + fp  # wrong-location matched + fully unmatched pred
            el_fn = tp - el_tp + fn  # wrong-location matched + fully unmatched gold
            acc[("event_location", tier)]["tp"] += el_tp
            acc[("event_location", tier)]["fp"] += el_fp
            acc[("event_location", tier)]["fn"] += el_fn
            macro_acc[("event_location", tier)].append(_prf(el_tp, el_fp, el_fn))

            stated_pairs = [
                (g, p)
                for g, p in pairs
                if _split_locs(g.get("event_location", ""))
            ]
            stated_correct = sum(
                1
                for g, p in stated_pairs
                if _loc_match(
                    g.get("event_location", ""), p.get("event_location", ""), tier
                )
            )
            for fam, correct, total in [
                ("location_on_matched_events", el_tp, len(pairs)),
                (
                    "location_on_matched_events_gold_stated",
                    stated_correct,
                    len(stated_pairs),
                ),
            ]:
                accuracy_acc[(fam, tier)]["correct"] += correct
                accuracy_acc[(fam, tier)]["total"] += total
                macro_accuracy_acc[(fam, tier)].append(_accuracy(correct, total))

            for g, p in pairs:
                et = _norm(g.get("event_type", ""))
                d = type_cond_acc.setdefault((et, tier, "location"), {"correct": 0, "total": 0})
                d["total"] += 1
                if _loc_match(g.get("event_location", ""), p.get("event_location", ""), tier):
                    d["correct"] += 1

            # ---------------------------------------------------------------
            # 5. Event-time pairing (conditioned on matched events)
            # ---------------------------------------------------------------
            et_tp = sum(
                1
                for g, p in pairs
                if _time_match(
                    g.get("event_time", "not_stated"),
                    p.get("event_time", "not_stated"),
                    tier,
                )
            )
            et_fp = tp - et_tp + fp
            et_fn = tp - et_tp + fn
            acc[("event_time", tier)]["tp"] += et_tp
            acc[("event_time", tier)]["fp"] += et_fp
            acc[("event_time", tier)]["fn"] += et_fn
            macro_acc[("event_time", tier)].append(_prf(et_tp, et_fp, et_fn))

            time_stated_pairs = [
                (g, p)
                for g, p in pairs
                if _norm(g.get("event_time", "not_stated")) != "not_stated"
            ]
            time_stated_correct = sum(
                1
                for g, p in time_stated_pairs
                if _time_match(
                    g.get("event_time", "not_stated"),
                    p.get("event_time", "not_stated"),
                    tier,
                )
            )
            for fam, correct, total in [
                ("time_on_matched_events", et_tp, len(pairs)),
                (
                    "time_on_matched_events_gold_stated",
                    time_stated_correct,
                    len(time_stated_pairs),
                ),
            ]:
                accuracy_acc[(fam, tier)]["correct"] += correct
                accuracy_acc[(fam, tier)]["total"] += total
                macro_accuracy_acc[(fam, tier)].append(_accuracy(correct, total))

            for g, p in pairs:
                et = _norm(g.get("event_type", ""))
                d = type_cond_acc.setdefault((et, tier, "time"), {"correct": 0, "total": 0})
                d["total"] += 1
                if _time_match(
                    g.get("event_time", "not_stated"), p.get("event_time", "not_stated"), tier
                ):
                    d["correct"] += 1

            # ---------------------------------------------------------------
            # 6. Event-location-admin-level pairing (conditioned on matched events)
            # ---------------------------------------------------------------
            al_tp = sum(
                1
                for g, p in pairs
                if _admin_level_match(
                    g.get("event_location_admin_level", "not_stated"),
                    p.get("event_location_admin_level", "not_stated"),
                    tier,
                )
            )
            al_fp = tp - al_tp + fp
            al_fn = tp - al_tp + fn
            acc[("event_location_admin_level", tier)]["tp"] += al_tp
            acc[("event_location_admin_level", tier)]["fp"] += al_fp
            acc[("event_location_admin_level", tier)]["fn"] += al_fn
            macro_acc[("event_location_admin_level", tier)].append(_prf(al_tp, al_fp, al_fn))

            admin_level_stated_pairs = [
                (g, p)
                for g, p in pairs
                if _norm(g.get("event_location_admin_level", "not_stated")) != "not_stated"
            ]
            admin_level_stated_correct = sum(
                1
                for g, p in admin_level_stated_pairs
                if _admin_level_match(
                    g.get("event_location_admin_level", "not_stated"),
                    p.get("event_location_admin_level", "not_stated"),
                    tier,
                )
            )
            for fam, correct, total in [
                ("admin_level_on_matched_events", al_tp, len(pairs)),
                (
                    "admin_level_on_matched_events_gold_stated",
                    admin_level_stated_correct,
                    len(admin_level_stated_pairs),
                ),
            ]:
                accuracy_acc[(fam, tier)]["correct"] += correct
                accuracy_acc[(fam, tier)]["total"] += total
                macro_accuracy_acc[(fam, tier)].append(_accuracy(correct, total))

            for g, p in pairs:
                et = _norm(g.get("event_type", ""))
                d = type_cond_acc.setdefault((et, tier, "admin_level"), {"correct": 0, "total": 0})
                d["total"] += 1
                if _admin_level_match(
                    g.get("event_location_admin_level", "not_stated"),
                    p.get("event_location_admin_level", "not_stated"),
                    tier,
                ):
                    d["correct"] += 1

            # ---------------------------------------------------------------
            # Error logging (relaxed tier only, keep it once)
            # ---------------------------------------------------------------
            if tier == "relaxed" and (unmatched_g or unmatched_p):
                error_rows.append(
                    {
                        "doc_id": doc_id,
                        "unmatched_gold": unmatched_g,
                        "unmatched_pred": unmatched_p,
                    }
                )

    # Build final metrics dict
    metrics: dict[str, Any] = {}
    for fam in families:
        metrics[fam] = {}
        for tier in tiers:
            a = acc[(fam, tier)]
            micro = _prf(a["tp"], a["fp"], a["fn"])
            macro = _macro_average(macro_acc[(fam, tier)])
            metrics[fam][tier] = {"micro": micro, "macro": macro}

    for fam in accuracy_families:
        metrics[fam] = {}
        for tier in tiers:
            a = accuracy_acc[(fam, tier)]
            micro = _accuracy(a["correct"], a["total"])
            macro = _macro_accuracy(macro_accuracy_acc[(fam, tier)])
            metrics[fam][tier] = {"micro": micro, "macro": macro}

    all_types = sorted({et for (et, _tier) in type_acc.keys()})
    by_type: dict[str, Any] = {}
    for et in all_types:
        by_type[et] = {}
        for tier in tiers:
            a = type_acc.get((et, tier), {"tp": 0, "fp": 0, "fn": 0})
            by_type[et][tier] = {"event": _prf(a["tp"], a["fp"], a["fn"])}
            for field in ("location", "time", "admin_level"):
                d = type_cond_acc.get((et, tier, field), {"correct": 0, "total": 0})
                by_type[et][tier][field] = _accuracy(d["correct"], d["total"])
    metrics["by_event_type"] = by_type

    metrics["_errors"] = error_rows
    return metrics


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_FAMILY_LABELS = {
    "event": "Event extraction",
    "event_type": "Event type (multiset)",
    "event_type_set": "Event type (doc-level set)",
    "cluster_event": "Cluster event extraction",
    "cluster_type": "Cluster type (multiset)",
    "cluster_type_set": "Cluster type (doc-level set)",
    "location": "Location extraction",
    "event_location": "End-to-end event-location",
    "cluster_event_location": "Cluster end-to-end event-location",
    "event_time": "End-to-end event-time",
    "cluster_event_time": "Cluster end-to-end event-time",
    "event_location_admin_level": "End-to-end event-admin-level",
    "cluster_event_location_admin_level": "Cluster end-to-end event-admin-level",
}

_ACCURACY_LABELS = {
    "location_on_matched_events": "Location on matched events",
    "location_on_matched_events_gold_stated": "Location on matched events (gold stated)",
    "cluster_location_on_matched_events": "Cluster location on matched events",
    "cluster_location_on_matched_events_gold_stated": (
        "Cluster location on matched events (gold stated)"
    ),
    "time_on_matched_events": "Time on matched events",
    "time_on_matched_events_gold_stated": "Time on matched events (gold stated)",
    "cluster_time_on_matched_events": "Cluster time on matched events",
    "cluster_time_on_matched_events_gold_stated": (
        "Cluster time on matched events (gold stated)"
    ),
    "admin_level_on_matched_events": "Admin level on matched events",
    "admin_level_on_matched_events_gold_stated": (
        "Admin level on matched events (gold stated)"
    ),
    "cluster_admin_level_on_matched_events": "Cluster admin level on matched events",
    "cluster_admin_level_on_matched_events_gold_stated": (
        "Cluster admin level on matched events (gold stated)"
    ),
}

_COL_W = 52  # width of metric name column
_NUM_W = 7   # width of each number column


def _row(label: str, r_p: float, r_r: float, r_f1: float,
         e_p: float, e_r: float, e_f1: float, suffix: str = "") -> str:
    def pct(v: float) -> str:
        return f"{v * 100:5.1f}"
    return (
        f"  {label:<{_COL_W}}"
        f"  {pct(r_p)}  {pct(r_r)}  {pct(r_f1)}"
        f"    {pct(e_p)}  {pct(e_r)}  {pct(e_f1)}"
        + (f"   {suffix}" if suffix else "")
    )


def _header() -> str:
    h1 = f"  {'Metric':<{_COL_W}}  {'── RELAXED ──────────':21}    {'── EXACT ────────────':21}"
    h2 = f"  {'':{'<'}{_COL_W}}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}    {'Prec':>5}  {'Rec':>5}  {'F1':>5}"
    sep = "  " + "-" * (_COL_W + 48)
    return "\n".join([h1, h2, sep])


def _accuracy_row(
    label: str,
    r_acc: float,
    e_acc: float,
    suffix: str = "",
) -> str:
    def pct(v: float) -> str:
        return f"{v * 100:5.1f}"
    return (
        f"  {label:<{_COL_W}}"
        f"  {pct(r_acc)}"
        f"    {pct(e_acc)}"
        + (f"   {suffix}" if suffix else "")
    )


def _accuracy_header() -> str:
    h1 = f"  {'Metric':<{_COL_W}}  {'RELAXED':>5}    {'EXACT':>5}"
    sep = "  " + "-" * (_COL_W + 21)
    return "\n".join([h1, sep])


_TYPE_COL_W = 32


def _type_header() -> str:
    h1 = (
        f"  {'Event type':<{_TYPE_COL_W}}  {'N':>5}   "
        f"{'── RELAXED ──':15}    {'── EXACT ────':15}   "
        f"{'── RELAXED ACC ─':17}"
    )
    h2 = (
        f"  {'':<{_TYPE_COL_W}}  {'':>5}   "
        f"{'Prec':>5} {'Rec':>5} {'F1':>5}    {'Prec':>5} {'Rec':>5} {'F1':>5}   "
        f"{'Loc':>5} {'Time':>5} {'AdmLvl':>7}"
    )
    sep = "  " + "-" * (_TYPE_COL_W + 78)
    return "\n".join([h1, h2, sep])


def _type_row(
    et: str, n: int, r: dict[str, float], e: dict[str, float],
    loc_acc: float, time_acc: float, adm_acc: float,
) -> str:
    def pct(v: float, w: int = 5) -> str:
        return f"{v * 100:{w}.1f}"
    return (
        f"  {et:<{_TYPE_COL_W}}  {n:>5}   "
        f"{pct(r['precision'])} {pct(r['recall'])} {pct(r['f1'])}    "
        f"{pct(e['precision'])} {pct(e['recall'])} {pct(e['f1'])}   "
        f"{pct(loc_acc)} {pct(time_acc)} {pct(adm_acc, 7)}"
    )


def _format_by_type_report(metrics: dict[str, Any]) -> list[str]:
    by_type = metrics.get("by_event_type")
    if not by_type:
        return []
    lines: list[str] = []
    lines.append("\n\n  PER EVENT TYPE  (micro, sorted by gold support, relaxed matching)\n")
    lines.append(_type_header())

    def support(et: str) -> int:
        r = by_type[et]["relaxed"]["event"]
        return r["tp"] + r["fn"]

    for et in sorted(by_type.keys(), key=support, reverse=True):
        r = by_type[et]["relaxed"]["event"]
        e = by_type[et]["exact"]["event"]
        n = r["tp"] + r["fn"]
        loc_acc = by_type[et]["relaxed"]["location"]["accuracy"]
        time_acc = by_type[et]["relaxed"]["time"]["accuracy"]
        adm_acc = by_type[et]["relaxed"]["admin_level"]["accuracy"]
        lines.append(_type_row(et, n, r, e, loc_acc, time_acc, adm_acc))
    return lines


def _format_report(metrics: dict[str, Any], n_records: int) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"EVALUATION REPORT  ({n_records} records)")
    lines.append("=" * 72)

    # ── micro table ────────────────────────────────────────────────────────
    lines.append("\n  MICRO  (pooled counts)\n")
    lines.append(_header())
    for fam, label in _FAMILY_LABELS.items():
        if fam not in metrics:
            continue
        mu_r = metrics[fam]["relaxed"]["micro"]
        mu_e = metrics[fam]["exact"]["micro"]
        counts = f"TP={mu_r['tp']:4d}  FP={mu_r['fp']:4d}  FN={mu_r['fn']:4d}"
        lines.append(_row(label,
                          mu_r["precision"], mu_r["recall"], mu_r["f1"],
                          mu_e["precision"], mu_e["recall"], mu_e["f1"],
                          suffix=counts))

    # ── macro table ────────────────────────────────────────────────────────
    lines.append("\n\n  MACRO  (per-document average)\n")
    lines.append(_header())
    for fam, label in _FAMILY_LABELS.items():
        if fam not in metrics:
            continue
        ma_r = metrics[fam]["relaxed"]["macro"]
        ma_e = metrics[fam]["exact"]["macro"]
        lines.append(_row(label,
                          ma_r["precision"], ma_r["recall"], ma_r["f1"],
                          ma_e["precision"], ma_e["recall"], ma_e["f1"]))

    accuracy_labels = {
        fam: label
        for fam, label in _ACCURACY_LABELS.items()
        if fam in metrics
    }
    if accuracy_labels:
        lines.append("\n\n  MICRO CONDITIONAL ACCURACY  (pooled matched events)\n")
        lines.append(_accuracy_header())
        for fam, label in accuracy_labels.items():
            mu_r = metrics[fam]["relaxed"]["micro"]
            mu_e = metrics[fam]["exact"]["micro"]
            counts = f"correct={mu_r['correct']:4d}  total={mu_r['total']:4d}"
            lines.append(
                _accuracy_row(
                    label,
                    mu_r["accuracy"],
                    mu_e["accuracy"],
                    suffix=counts,
                )
            )

        lines.append("\n\n  MACRO CONDITIONAL ACCURACY  (per-document average)\n")
        lines.append(_accuracy_header())
        for fam, label in accuracy_labels.items():
            ma_r = metrics[fam]["relaxed"]["macro"]
            ma_e = metrics[fam]["exact"]["macro"]
            lines.append(_accuracy_row(label, ma_r["accuracy"], ma_e["accuracy"]))

    lines.extend(_format_by_type_report(metrics))

    n_err = len(metrics["_errors"])
    lines.append(f"\n{'=' * 72}")
    lines.append(f"  Docs with unmatched events (relaxed): {n_err} / {n_records}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate distilled event-location extraction model predictions."
    )
    parser.add_argument(
        "--pred-jsonl",
        required=True,
        type=Path,
        help="Predictions JSONL file (contains both annotation and predictions fields).",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write the full metrics dict as JSON.",
    )
    parser.add_argument(
        "--errors-jsonl",
        type=Path,
        default=None,
        help="Optional path to write unmatched events per document.",
    )
    parser.add_argument(
        "--cluster",
        type=Path,
        default=None,
        metavar="CLUSTER_JSON",
        help=(
            "Optional Studio results JSON mapping event names to clusters. "
            "When provided, adds cluster-level type metrics to the report."
        ),
    )
    args = parser.parse_args()

    cluster_map: dict[str, str] | None = None
    if args.cluster:
        cluster_map = _load_cluster_map(args.cluster)
        print(f"Loaded cluster map with {len(cluster_map)} entries from {args.cluster}")

    records = _load_jsonl(args.pred_jsonl)
    print(f"Loaded {len(records)} records from {args.pred_jsonl}")

    metrics = evaluate(records, cluster_map=cluster_map)

    print(_format_report(metrics, len(records)))

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        out = {k: v for k, v in metrics.items() if k != "_errors"}
        args.report_json.write_text(json.dumps(out, indent=2))
        print(f"Metrics written to {args.report_json}")

    if args.errors_jsonl:
        args.errors_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.errors_jsonl, "w", encoding="utf-8") as f:
            for row in metrics["_errors"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Error rows written to {args.errors_jsonl}")


if __name__ == "__main__":
    main()
