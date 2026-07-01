"""Set-based location matching and micro/macro P/R/F1.

Normalisation (_norm, _norm_loc, direction-prefix stripping) and the fuzzy
threshold are ported from `scripts/train/inference/eval_v3_sft.py`
(`_norm_loc` / `_LOC_FUZZY_THRESHOLD`), which already solves the same
"compare two location strings" problem for the SFT event evaluator.
"""
from __future__ import annotations

import statistics
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_DIRECTION_PREFIX = re.compile(
    r"^(northern?|southern?|eastern?|western?|central)\s+",
    re.IGNORECASE,
)

_FUZZY_THRESHOLD = 0.85

MATCH_TIERS = ("exact", "normalized", "fuzzy")


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _norm_loc(s: str) -> str:
    return _DIRECTION_PREFIX.sub("", _norm(s)).strip()


def _matches(pred: str, gold: str, tier: str) -> bool:
    if tier == "exact":
        return pred == gold
    pn, gn = _norm_loc(pred), _norm_loc(gold)
    if tier == "normalized":
        return pn == gn
    if tier == "fuzzy":
        if pn == gn or pn in gn or gn in pn:
            return True
        return SequenceMatcher(None, pn, gn).ratio() >= _FUZZY_THRESHOLD
    raise ValueError(f"Unknown match tier {tier!r}. Choose from {MATCH_TIERS}.")


def dedupe_locations(strings: list[str]) -> list[str]:
    """Collapse to one surface form per normalised location, first-seen wins."""
    seen: set[str] = set()
    out: list[str] = []
    for s in strings:
        key = _norm_loc(s)
        if not s or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def score_sets(pred: list[str], gold: list[str], tier: str = "normalized") -> tuple[int, int, int]:
    """Greedy one-to-one matching between two (already deduped) location sets.

    Returns (tp, fp, fn). Greedy is sufficient here because both inputs are
    deduped per-document sets, typically a handful of items.
    """
    gold_remaining: list[str | None] = list(gold)
    matched_pred = 0
    for p in pred:
        match_idx = next(
            (j for j, g in enumerate(gold_remaining) if g is not None and _matches(p, g, tier)),
            None,
        )
        if match_idx is not None:
            matched_pred += 1
            gold_remaining[match_idx] = None
    tp = matched_pred
    fp = len(pred) - matched_pred
    fn = sum(1 for g in gold_remaining if g is not None)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


@dataclass
class Accumulator:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_doc: list[dict[str, float]] | None = None

    def __post_init__(self) -> None:
        if self.per_doc is None:
            self.per_doc = []

    def add(self, tp: int, fp: int, fn: int) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn
        # Docs with neither a gold nor a predicted location are trivial (no
        # signal either way) and would otherwise be scored F1=0 by the 0/0
        # convention below, unfairly punishing macro averages on datasets
        # where most documents mention no location (e.g. OntoNotes sentences).
        if tp or fp or fn:
            self.per_doc.append(prf(tp, fp, fn))

    def micro(self) -> dict[str, float]:
        return prf(self.tp, self.fp, self.fn)

    def macro(self) -> dict[str, float]:
        if not self.per_doc:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {
            "precision": statistics.mean(d["precision"] for d in self.per_doc),
            "recall": statistics.mean(d["recall"] for d in self.per_doc),
            "f1": statistics.mean(d["f1"] for d in self.per_doc),
        }
