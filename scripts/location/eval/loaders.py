"""Dataset loaders -> list[GoldDoc].

Each loader normalises a different gold source into the common GoldDoc shape
(see gold.py). Add a new benchmark by writing one loader and registering it
in DATASET_REGISTRY.

Standard benchmarks are pulled from the `tner` project's script-free Parquet
mirrors on the Hugging Face Hub (avoids `trust_remote_code`, which the
official `conll2003`/`wnut_17` loading scripts require and which recent
`datasets` releases restrict). Tag id -> name maps are the datasets' published
`dataset/label.json` (stable, versioned release artifacts), hardcoded below to
avoid a second network round-trip at eval time:
  https://huggingface.co/datasets/tner/conll2003/raw/main/dataset/label.json
  https://huggingface.co/datasets/tner/wnut2017/raw/main/dataset/label.json
  https://huggingface.co/datasets/tner/ontonotes5/raw/main/dataset/label.json
"""

from __future__ import annotations

import pathlib
import sys
from typing import Callable

from .gold import GoldDoc
from .metrics import dedupe_locations

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.event_extraction.generation.io_utils import iter_jsonl, resolve_path  # noqa: E402

DEFAULT_ZHAI_PATH = "dataset/zhai/v3/science/train.jsonl"


def load_zhai(
    path: str | None = None, limit: int | None = None, **_: object
) -> list[GoldDoc]:
    resolved = resolve_path(path or DEFAULT_ZHAI_PATH)
    docs: list[GoldDoc] = []
    for row in iter_jsonl(resolved):
        text = (row.get("source") or {}).get("text") or ""
        events = (row.get("annotation") or {}).get("events") or []
        raw_locations: list[str] = []
        for event in events:
            loc = event.get("event_location") or ""
            for part in loc.split(";"):
                part = part.strip()
                if part and part != "not_stated":
                    raw_locations.append(part)
        docs.append(
            GoldDoc(
                id=str(row.get("id")),
                text=text,
                locations=dedupe_locations(raw_locations),
                fine=None,
                fine_schema=None,
            )
        )
        if limit is not None and len(docs) >= limit:
            break
    return docs


def _iob_to_spans(tokens: list[str], tag_names: list[str]) -> dict[str, list[str]]:
    """Reconstruct entity surface strings per type from IOB2 tags."""
    spans: dict[str, list[str]] = {}
    current_type: str | None = None
    current_tokens: list[str] = []

    def flush() -> None:
        nonlocal current_type, current_tokens
        if current_type is not None and current_tokens:
            spans.setdefault(current_type, []).append(" ".join(current_tokens))
        current_type = None
        current_tokens = []

    for tok, tag in zip(tokens, tag_names):
        if tag == "O":
            flush()
            continue
        prefix, _, ent_type = tag.partition("-")
        if prefix == "B" or ent_type != current_type:
            flush()
            current_type = ent_type
            current_tokens = [tok]
        else:
            current_tokens.append(tok)
    flush()
    return spans


def _load_tner(
    repo_id: str,
    config: str,
    id2label: dict[int, str],
    location_types: list[str],
    fine_schema: str | None,
    split: str = "test",
    limit: int | None = None,
) -> list[GoldDoc]:
    from datasets import load_dataset

    # tner/* repos ship a loading script, which recent `datasets` releases
    # refuse to execute. Load the Hub's auto-converted Parquet mirror instead
    # (script-free, same data): refs/convert/parquet branch.
    url = (
        f"https://huggingface.co/datasets/{repo_id}/resolve/"
        f"refs%2Fconvert%2Fparquet/{config}/{split}/0000.parquet"
    )
    ds = load_dataset("parquet", data_files=url, split="train")
    docs: list[GoldDoc] = []
    for i, row in enumerate(ds):
        tokens = row["tokens"]
        tag_names = [id2label[t] for t in row["tags"]]
        spans = _iob_to_spans(tokens, tag_names)

        fine: dict[str, list[str]] = {}
        raw_locations: list[str] = []
        for ent_type in location_types:
            type_spans = spans.get(ent_type, [])
            if type_spans:
                fine[ent_type] = type_spans
                raw_locations.extend(type_spans)

        docs.append(
            GoldDoc(
                id=f"{repo_id}:{split}:{i}",
                text=" ".join(tokens),
                locations=dedupe_locations(raw_locations),
                fine=fine if fine_schema else None,
                fine_schema=fine_schema,
            )
        )
        if limit is not None and len(docs) >= limit:
            break
    return docs


_CONLL2003_ID2LABEL = {
    0: "O",
    1: "B-ORG",
    2: "B-MISC",
    3: "B-PER",
    4: "I-PER",
    5: "B-LOC",
    6: "I-ORG",
    7: "I-MISC",
    8: "I-LOC",
}

_WNUT17_ID2LABEL = {
    0: "B-corporation",
    1: "B-creative-work",
    2: "B-group",
    3: "B-location",
    4: "B-person",
    5: "B-product",
    6: "I-corporation",
    7: "I-creative-work",
    8: "I-group",
    9: "I-location",
    10: "I-person",
    11: "I-product",
    12: "O",
}

_ONTONOTES5_ID2LABEL = {
    0: "O",
    1: "B-CARDINAL",
    2: "B-DATE",
    3: "I-DATE",
    4: "B-PERSON",
    5: "I-PERSON",
    6: "B-NORP",
    7: "B-GPE",
    8: "I-GPE",
    9: "B-LAW",
    10: "I-LAW",
    11: "B-ORG",
    12: "I-ORG",
    13: "B-PERCENT",
    14: "I-PERCENT",
    15: "B-ORDINAL",
    16: "B-MONEY",
    17: "I-MONEY",
    18: "B-WORK_OF_ART",
    19: "I-WORK_OF_ART",
    20: "B-FAC",
    21: "B-TIME",
    22: "I-CARDINAL",
    23: "B-LOC",
    24: "B-QUANTITY",
    25: "I-QUANTITY",
    26: "I-NORP",
    27: "I-LOC",
    28: "B-PRODUCT",
    29: "I-TIME",
    30: "B-EVENT",
    31: "I-EVENT",
    32: "I-FAC",
    33: "B-LANGUAGE",
    34: "I-PRODUCT",
    35: "I-ORDINAL",
    36: "I-LANGUAGE",
}


def load_conll2003(
    split: str = "test", limit: int | None = None, **_: object
) -> list[GoldDoc]:
    return _load_tner(
        "tner/conll2003",
        "conll2003",
        _CONLL2003_ID2LABEL,
        location_types=["LOC"],
        fine_schema=None,
        split=split,
        limit=limit,
    )


def load_wnut17(
    split: str = "test", limit: int | None = None, **_: object
) -> list[GoldDoc]:
    return _load_tner(
        "tner/wnut2017",
        "wnut2017",
        _WNUT17_ID2LABEL,
        location_types=["location"],
        fine_schema=None,
        split=split,
        limit=limit,
    )


def load_ontonotes5(
    split: str = "test", limit: int | None = None, **_: object
) -> list[GoldDoc]:
    return _load_tner(
        "tner/ontonotes5",
        "ontonotes5",
        _ONTONOTES5_ID2LABEL,
        location_types=["GPE", "LOC", "FAC"],
        fine_schema="ontonotes-gpe-loc-fac",
        split=split,
        limit=limit,
    )


DATASET_REGISTRY: dict[str, Callable[..., list[GoldDoc]]] = {
    "zhai": load_zhai,
    "conll2003": load_conll2003,
    "wnut17": load_wnut17,
    "ontonotes5": load_ontonotes5,
}

# Backend whose native fine-grained label set matches a dataset's fine_schema.
# Only spaCy's GPE/LOC/FAC lines up with a standard benchmark (OntoNotes 5);
# GLiNER2's city/country/region/facility/landmark schema has no standard-
# benchmark equivalent, so it is never fine-grained-evaluated here.
BACKEND_FINE_SCHEMA: dict[str, str] = {
    "spacy": "ontonotes-gpe-loc-fac",
}
