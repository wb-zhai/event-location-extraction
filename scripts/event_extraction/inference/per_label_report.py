from __future__ import annotations

"""Per-label (event-type / cluster) micro breakdown for three metric families:

  - Event extraction        (alignment-based: same event_type + quote overlap)
  - Event type (multiset)   (per-doc bag-of-labels)
  - Event type (doc-level set) (per-doc set-of-labels)

...and their cluster-level counterparts (using a Studio cluster map to fold
event types into coarser clusters). Reuses the matching/normalisation helpers
from eval_v3_sft.py so results stay consistent with the main report.

Usage:
    python per_label_report.py --pred-jsonl PRED.jsonl --cluster CLUSTER.json [--tier relaxed|exact]
"""

import argparse
import collections
from pathlib import Path
from typing import Any

from eval_v3_sft import (
    _gold_events,
    _load_cluster_map,
    _load_jsonl,
    _match_events,
    _norm,
    _prf,
    _split_locs,
    _to_cluster,
)


def _pred_events(rec: dict[str, Any]) -> list[dict[str, Any]]:
    events = rec.get("predictions", [])
    return [e for e in events if isinstance(e, dict)]


def _label_fn(cluster_map: dict[str, str] | None):
    if cluster_map is None:
        return lambda et: _norm(et)
    return lambda et: _to_cluster(et, cluster_map)


def per_label_breakdown(
    records: list[dict[str, Any]],
    tier: str,
    cluster_map: dict[str, str] | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    """Returns {family: {label: {tp, fp, fn}}} for the three families above."""
    label_of = _label_fn(cluster_map)

    event_acc: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    multiset_acc: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    set_acc: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for rec in records:
        gold = _gold_events(rec)
        pred = _pred_events(rec)

        # -------------------------------------------------------------
        # 1. Event extraction (alignment-based) per label
        # -------------------------------------------------------------
        pairs, unmatched_g, unmatched_p = _match_events(
            gold, pred, tier, cluster_map=cluster_map
        )
        for g, p in pairs:
            event_acc[label_of(g.get("event_type", ""))]["tp"] += 1
        for g in unmatched_g:
            event_acc[label_of(g.get("event_type", ""))]["fn"] += 1
        for p in unmatched_p:
            event_acc[label_of(p.get("event_type", ""))]["fp"] += 1

        # -------------------------------------------------------------
        # 2. Event type (multiset) per label
        # -------------------------------------------------------------
        gold_counts = collections.Counter(
            label_of(e.get("event_type", "")) for e in gold
        )
        pred_counts = collections.Counter(
            label_of(e.get("event_type", "")) for e in pred
        )
        for lbl in set(gold_counts) | set(pred_counts):
            g_n, p_n = gold_counts[lbl], pred_counts[lbl]
            tp = min(g_n, p_n)
            multiset_acc[lbl]["tp"] += tp
            multiset_acc[lbl]["fp"] += p_n - tp
            multiset_acc[lbl]["fn"] += g_n - tp

        # -------------------------------------------------------------
        # 3. Event type (doc-level set) per label
        # -------------------------------------------------------------
        gold_set = {label_of(e.get("event_type", "")) for e in gold}
        pred_set = {label_of(e.get("event_type", "")) for e in pred}
        for lbl in gold_set & pred_set:
            set_acc[lbl]["tp"] += 1
        for lbl in pred_set - gold_set:
            set_acc[lbl]["fp"] += 1
        for lbl in gold_set - pred_set:
            set_acc[lbl]["fn"] += 1

    return {
        "event": dict(event_acc),
        "event_type_multiset": dict(multiset_acc),
        "event_type_set": dict(set_acc),
    }


def _text_locs(ev: dict[str, Any]) -> list[str]:
    """Location keys from the raw ';'-separated event_location string."""
    parts = _split_locs(ev.get("event_location", ""))
    return parts or ["not_stated"]


def _geo_locs(field: str):
    """Location keys from the resolved Photon `geotaxonomy` list on an event.

    `geotaxonomy` is a list of geocoder hits (one per ';'-separated location
    piece); `field` picks which resolved attribute to key on, e.g. "country",
    "countrycode", "resolved_name", "photon_type". Events with no geotaxonomy
    (Photon found nothing) or a missing field fall back to "not_geocoded".
    """
    def extractor(ev: dict[str, Any]) -> list[str]:
        geo = ev.get("geotaxonomy") or []
        keys = []
        for g in geo:
            v = g.get(field)
            if v:
                keys.append(_norm(str(v)))
        return keys or ["not_geocoded"]
    return extractor


def per_location_breakdown(
    records: list[dict[str, Any]],
    tier: str,
    locs_of=_text_locs,
) -> dict[str, dict[str, int]]:
    """Event-type extraction performance (event_type + quote alignment)
    bucketed by *gold* event_location (falls back to the predicted location
    for false positives, since those have no gold location to attribute to).
    """
    acc: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for rec in records:
        gold = _gold_events(rec)
        pred = _pred_events(rec)
        pairs, unmatched_g, unmatched_p = _match_events(gold, pred, tier)

        for g, p in pairs:
            for loc in locs_of(g):
                acc[loc]["tp"] += 1
        for g in unmatched_g:
            for loc in locs_of(g):
                acc[loc]["fn"] += 1
        for p in unmatched_p:
            for loc in locs_of(p):
                acc[loc]["fp"] += 1

    return dict(acc)


def per_location_doc_level_set_breakdown(
    records: list[dict[str, Any]],
    locs_of=_text_locs,
) -> dict[str, dict[str, int]]:
    """Event type (doc-level set) performance, scoped to (doc, location) pairs.

    For each document and each location mentioned in it, compares the set of
    gold event_types occurring at that location against the set of predicted
    event_types occurring at that location (no quote alignment required -
    same "did this type get reported anywhere" semantics as the doc-level
    event_type_set family, just restricted to one location's events at a time).
    """
    acc: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    def types_by_location(events: list[dict[str, Any]]) -> dict[str, set[str]]:
        by_loc: dict[str, set[str]] = collections.defaultdict(set)
        for e in events:
            et = _norm(e.get("event_type", ""))
            for loc in locs_of(e):
                by_loc[loc].add(et)
        return by_loc

    for rec in records:
        gold_by_loc = types_by_location(_gold_events(rec))
        pred_by_loc = types_by_location(_pred_events(rec))

        for loc in set(gold_by_loc) | set(pred_by_loc):
            g_types = gold_by_loc.get(loc, set())
            p_types = pred_by_loc.get(loc, set())
            acc[loc]["tp"] += len(g_types & p_types)
            acc[loc]["fp"] += len(p_types - g_types)
            acc[loc]["fn"] += len(g_types - p_types)

    return dict(acc)


_COL_W = 34


def _fmt_table(
    title: str, acc: dict[str, dict[str, int]], top_n: int | None = None
) -> list[str]:
    lines = [f"\n  {title}\n"]
    header = (
        f"  {'Label':<{_COL_W}}  {'N':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
        f"    {'TP':>4}  {'FP':>4}  {'FN':>4}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    def support(lbl: str) -> int:
        a = acc[lbl]
        return a["tp"] + a["fn"]

    ranked = sorted(acc.keys(), key=support, reverse=True)
    shown, rest = (ranked[:top_n], ranked[top_n:]) if top_n else (ranked, [])

    for lbl in shown:
        a = acc[lbl]
        m = _prf(a["tp"], a["fp"], a["fn"])
        n = a["tp"] + a["fn"]
        lines.append(
            f"  {lbl:<{_COL_W}}  {n:>5}  {m['precision']*100:6.1f}  "
            f"{m['recall']*100:6.1f}  {m['f1']*100:6.1f}"
            f"    {a['tp']:>4}  {a['fp']:>4}  {a['fn']:>4}"
        )

    if rest:
        rtp = sum(acc[l]["tp"] for l in rest)
        rfp = sum(acc[l]["fp"] for l in rest)
        rfn = sum(acc[l]["fn"] for l in rest)
        rm = _prf(rtp, rfp, rfn)
        rn = rtp + rfn
        lines.append(
            f"  {f'... other ({len(rest)} labels)':<{_COL_W}}  {rn:>5}  "
            f"{rm['precision']*100:6.1f}  {rm['recall']*100:6.1f}  {rm['f1']*100:6.1f}"
            f"    {rtp:>4}  {rfp:>4}  {rfn:>4}"
        )

    # overall micro row
    tp = sum(a["tp"] for a in acc.values())
    fp = sum(a["fp"] for a in acc.values())
    fn = sum(a["fn"] for a in acc.values())
    m = _prf(tp, fp, fn)
    lines.append("  " + "-" * (len(header) - 2))
    lines.append(
        f"  {'MICRO (all labels)':<{_COL_W}}  {tp+fn:>5}  {m['precision']*100:6.1f}  "
        f"{m['recall']*100:6.1f}  {m['f1']*100:6.1f}"
        f"    {tp:>4}  {fp:>4}  {fn:>4}"
    )
    return lines


def _fmt_bottom_table(
    title: str,
    acc: dict[str, dict[str, int]],
    bottom_n: int = 20,
    min_support: int = 5,
) -> list[str]:
    """Worst-performing labels by F1 (ascending), filtered to a minimum
    support so single-occurrence labels (trivially 0% or 100%) don't drown
    out the genuinely weak, well-observed ones."""
    lines = [f"\n  {title}  (min support={min_support})\n"]
    header = (
        f"  {'Label':<{_COL_W}}  {'N':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
        f"    {'TP':>4}  {'FP':>4}  {'FN':>4}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    def support(lbl: str) -> int:
        a = acc[lbl]
        return a["tp"] + a["fn"]

    eligible = [lbl for lbl in acc if support(lbl) >= min_support]
    ranked = sorted(
        eligible, key=lambda lbl: (_prf(**acc[lbl])["f1"], -support(lbl))
    )

    if not eligible:
        lines.append(f"  (no labels with support >= {min_support})")
        return lines

    for lbl in ranked[:bottom_n]:
        a = acc[lbl]
        m = _prf(a["tp"], a["fp"], a["fn"])
        n = support(lbl)
        lines.append(
            f"  {lbl:<{_COL_W}}  {n:>5}  {m['precision']*100:6.1f}  "
            f"{m['recall']*100:6.1f}  {m['f1']*100:6.1f}"
            f"    {a['tp']:>4}  {a['fp']:>4}  {a['fn']:>4}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Per-label micro breakdown for Event extraction, Event type "
            "(multiset), Event type (doc-level set), and their cluster variants."
        )
    )
    parser.add_argument("--pred-jsonl", required=True, type=Path)
    parser.add_argument(
        "--cluster",
        type=Path,
        default=None,
        help="Studio cluster JSON mapping event names to clusters.",
    )
    parser.add_argument(
        "--tier", choices=["exact", "relaxed"], default="relaxed",
        help="Matching tier for the alignment-based 'event' family.",
    )
    parser.add_argument(
        "--by-location", action="store_true",
        help=(
            "Also report event-type extraction performance (event_type + "
            "quote alignment) bucketed by gold event_location."
        ),
    )
    parser.add_argument(
        "--top-n", type=int, default=40,
        help="Truncate the by-location table to the top-N locations by support "
             "(default 40; the remainder is pooled into an 'other' row).",
    )
    parser.add_argument(
        "--geo-field",
        choices=["country", "countrycode", "resolved_name", "photon_type", "adm_code"],
        default=None,
        help=(
            "If set, group --by-location using the resolved Photon `geotaxonomy` "
            "list on each event (keyed on this field) instead of the raw "
            "event_location text. Requires a predictions file with a "
            "'geotaxonomy' key on events (the '*.geo.jsonl' files)."
        ),
    )
    parser.add_argument(
        "--bottom-n", type=int, default=20,
        help="Also list the N worst locations by F1 (ascending), for both "
             "by-location tables (default 20). Set to 0 to disable.",
    )
    parser.add_argument(
        "--min-support", type=int, default=5,
        help="Minimum support (TP+FN) a location needs to qualify for the "
             "worst-N table, so single-occurrence locations don't dominate "
             "(default 5).",
    )
    args = parser.parse_args()

    records = _load_jsonl(args.pred_jsonl)
    print(f"Loaded {len(records)} records from {args.pred_jsonl}")

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(f"PER-LABEL REPORT  ({len(records)} records, tier={args.tier})")
    lines.append("=" * 90)

    fine = per_label_breakdown(records, tier=args.tier, cluster_map=None)
    lines.extend(_fmt_table("EVENT EXTRACTION  (per event type)", fine["event"]))
    lines.extend(_fmt_table("EVENT TYPE MULTISET  (per event type)", fine["event_type_multiset"]))
    lines.extend(_fmt_table("EVENT TYPE SET (doc-level)  (per event type)", fine["event_type_set"]))

    if args.cluster:
        cluster_map = _load_cluster_map(args.cluster)
        print(f"Loaded cluster map with {len(cluster_map)} entries from {args.cluster}")
        coarse = per_label_breakdown(records, tier=args.tier, cluster_map=cluster_map)
        lines.extend(_fmt_table("CLUSTER EVENT EXTRACTION  (per cluster)", coarse["event"]))
        lines.extend(_fmt_table("CLUSTER TYPE MULTISET  (per cluster)", coarse["event_type_multiset"]))
        lines.extend(_fmt_table("CLUSTER TYPE SET (doc-level)  (per cluster)", coarse["event_type_set"]))

    if args.by_location:
        locs_of = _geo_locs(args.geo_field) if args.geo_field else _text_locs
        loc_label = f"geotaxonomy.{args.geo_field}" if args.geo_field else "raw event_location"

        by_loc = per_location_breakdown(records, tier=args.tier, locs_of=locs_of)
        lines.extend(
            _fmt_table(
                f"EVENT EXTRACTION  (per {loc_label}, top {args.top_n})",
                by_loc,
                top_n=args.top_n,
            )
        )
        by_loc_set = per_location_doc_level_set_breakdown(records, locs_of=locs_of)
        lines.extend(
            _fmt_table(
                f"EVENT TYPE SET (doc-level)  (per {loc_label}, top {args.top_n})",
                by_loc_set,
                top_n=args.top_n,
            )
        )

        if args.bottom_n:
            lines.extend(
                _fmt_bottom_table(
                    f"WORST EVENT EXTRACTION  (per {loc_label})",
                    by_loc,
                    bottom_n=args.bottom_n,
                    min_support=args.min_support,
                )
            )
            lines.extend(
                _fmt_bottom_table(
                    f"WORST EVENT TYPE SET (doc-level)  (per {loc_label})",
                    by_loc_set,
                    bottom_n=args.bottom_n,
                    min_support=args.min_support,
                )
            )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
