"""Location mention extractors for news article text.

Two backends share the LocationExtractor interface:
  - Gliner2Extractor: zero-shot encoder NER with a custom location schema.
    Flexible label set; handles long texts via sliding windows.
  - SpacyExtractor: fast CNN NER (en_core_web_lg by default). No GPU needed;
    very high throughput via nlp.pipe batching. spaCy provides no per-entity
    confidence score, so confidence is fixed at 1.0.

Use build_extractor(backend, **kwargs) to instantiate.
Pass coarse=True (or call coarsen()) to collapse all labels to "location".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class LocationMention:
    text: str
    label: str       # fine-grained: GPE/LOC/FAC (spaCy) or city/country/… (gliner2)
    start: int
    end: int
    confidence: float
    source: str      # "gliner2" | "spacy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coarsen(mentions: list[LocationMention]) -> list[LocationMention]:
    """Return a new list with every label replaced by 'location'."""
    return [LocationMention(**{**asdict(m), "label": "location"}) for m in mentions]


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class LocationExtractor(ABC):
    name: str

    @abstractmethod
    def extract(self, text: str) -> list[LocationMention]: ...

    def extract_batch(self, texts: list[str]) -> list[list[LocationMention]]:
        return [self.extract(t) for t in texts]


# ---------------------------------------------------------------------------
# GLiNER2 backend
# ---------------------------------------------------------------------------

LOCATION_TYPES: dict[str, str] = {
    "city": "A city, town, village, or other human settlement.",
    "country": "A sovereign state or territory.",
    "region": "A geographic or administrative region, province, state, or area.",
    "facility": "A building, airport, port, hospital, stadium, or named infrastructure.",
    "landmark": "A named natural feature: river, mountain, lake, sea, forest, etc.",
}

_GLINER2_DEFAULT_MODEL = "fastino/gliner2-large-v1"
_GLINER2_WINDOW_CHARS = 1500   # ~400 tokens; conservative to stay under model limit
_GLINER2_OVERLAP_CHARS = 150   # overlap to avoid missing spans at window boundaries


class Gliner2Extractor(LocationExtractor):
    name = "gliner2"

    def __init__(
        self,
        model: str = _GLINER2_DEFAULT_MODEL,
        threshold: float = 0.5,
        location_types: dict[str, str] | None = None,
    ) -> None:
        from gliner2 import GLiNER2
        from semantic_text_splitter import TextSplitter

        self._model = GLiNER2.from_pretrained(model)
        self._schema = self._model.create_schema().entities(location_types or LOCATION_TYPES)
        self._threshold = threshold
        self._splitter = TextSplitter(_GLINER2_WINDOW_CHARS)

    def extract(self, text: str) -> list[LocationMention]:
        if not text:
            return []

        windows = self._splitter.chunks(text)
        mentions: list[LocationMention] = []
        seen: set[tuple[str, str, int]] = set()
        offset = 0

        for window in windows:
            raw = self._model.extract(
                window,
                self._schema,
                threshold=self._threshold,
                include_spans=True,
                include_confidence=True,
            )
            for label, spans in raw.get("entities", {}).items():
                for span in spans:
                    abs_start = offset + span["start"]
                    abs_end = offset + span["end"]
                    key = (span["text"].lower(), label, abs_start)
                    if key in seen:
                        continue
                    seen.add(key)
                    mentions.append(LocationMention(
                        text=span["text"],
                        label=label,
                        start=abs_start,
                        end=abs_end,
                        confidence=span["confidence"],
                        source="gliner2",
                    ))
            # Advance offset by window length minus overlap so next window's
            # spans are correctly re-based. Use actual char position in original.
            offset = text.find(window, offset) + len(window) - _GLINER2_OVERLAP_CHARS
            offset = max(offset, 0)

        return mentions


# ---------------------------------------------------------------------------
# spaCy backend
# ---------------------------------------------------------------------------

_SPACY_DEFAULT_MODEL = "en_core_web_lg"
_SPACY_LOCATION_LABELS = {"GPE", "LOC", "FAC"}


class SpacyExtractor(LocationExtractor):
    name = "spacy"

    def __init__(
        self,
        model: str = _SPACY_DEFAULT_MODEL,
        location_labels: set[str] | None = None,
        batch_size: int = 32,
    ) -> None:
        import spacy

        # Disable components we don't need — keeps only tok2vec + ner.
        self._nlp = spacy.load(model, disable=["parser", "lemmatizer", "attribute_ruler"])
        self._nlp.max_length = 2_000_000
        self._location_labels = location_labels or _SPACY_LOCATION_LABELS
        self._batch_size = batch_size

    def extract(self, text: str) -> list[LocationMention]:
        doc = self._nlp(text)
        return self._doc_to_mentions(doc)

    def extract_batch(self, texts: list[str]) -> list[list[LocationMention]]:
        docs = self._nlp.pipe(texts, batch_size=self._batch_size)
        return [self._doc_to_mentions(doc) for doc in docs]

    def _doc_to_mentions(self, doc: Any) -> list[LocationMention]:
        mentions = []
        for ent in doc.ents:
            if ent.label_ not in self._location_labels:
                continue
            mentions.append(LocationMention(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,  # spaCy CNN does not expose per-entity scores
                source="spacy",
            ))
        return mentions


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_extractor(backend: str, **kwargs: Any) -> LocationExtractor:
    """Instantiate a LocationExtractor by name ('gliner2' or 'spacy')."""
    if backend == "gliner2":
        return Gliner2Extractor(**kwargs)
    if backend == "spacy":
        return SpacyExtractor(**kwargs)
    raise ValueError(f"Unknown backend '{backend}'. Choose 'gliner2' or 'spacy'.")
