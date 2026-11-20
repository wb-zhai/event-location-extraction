"""Common gold-annotation representation shared by every dataset loader.

Every dataset (in-domain zhai data or a standard NER benchmark) is reduced to
a list of GoldDoc — a document's text plus the set of location surface
strings it contains. This is the single shape that scoring (metrics.py)
and every loader (datasets.py) agree on.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GoldDoc:
    id: str
    text: str
    locations: list[str]
    # Present only when the dataset's native label schema is directly
    # comparable to a backend's fine-grained labels (currently: OntoNotes 5's
    # GPE/LOC/FAC vs. spaCy's GPE/LOC/FAC). None otherwise.
    fine: dict[str, list[str]] | None
    fine_schema: str | None
