"""Deterministic lexical retrieval helpers for event generation."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class RetrievedItem:
    index: int
    score: float
    payload: Any


class BM25Index:
    """Tiny in-repo BM25 implementation to avoid heavy retrieval deps."""

    def __init__(
        self,
        documents: list[str],
        payloads: list[Any],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(documents) != len(payloads):
            raise ValueError("documents and payloads must have equal length")
        self.payloads = payloads
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freqs.update(set(tokens))
        doc_count = len(self.doc_tokens)
        self.idf = {
            term: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in self.doc_freqs.items()
        }

    def search(self, query: str, *, top_k: int) -> list[RetrievedItem]:
        if top_k < 1 or not self.payloads:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        unique_terms = set(query_terms)
        results: list[RetrievedItem] = []
        for index, term_freq in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[index]
            score = 0.0
            for term in unique_terms:
                freq = term_freq.get(term, 0)
                if freq <= 0:
                    continue
                idf = self.idf.get(term)
                if idf is None:
                    continue
                norm = 1.0 - self.b + self.b * (
                    doc_len / self.avg_doc_len if self.avg_doc_len else 0.0
                )
                denom = freq + self.k1 * norm
                score += idf * (freq * (self.k1 + 1.0)) / denom
            if score > 0.0:
                results.append(
                    RetrievedItem(index=index, score=score, payload=self.payloads[index])
                )
        results.sort(key=lambda item: (-item.score, item.index))
        return results[:top_k]
