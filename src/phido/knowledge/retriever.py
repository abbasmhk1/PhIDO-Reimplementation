"""Retriever interface + a dependency-free BM25 keyword implementation.

Stages depend on the `Retriever` protocol only, so a vector index can replace this
later (the paper notes full-context PDK selection stops scaling) without touching stages.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Protocol

from .text import tokenize


@dataclass(frozen=True)
class Hit:
    id: str
    score: float  # BM25
    coverage: float  # fraction of focus tokens present in the document (0..1)


class Retriever(Protocol):
    def search(self, query: str, k: int = 5, focus: Optional[str] = None) -> list[Hit]: ...


class KeywordRetriever:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tf = {i: Counter(tokenize(t)) for i, t in docs.items()}
        self._len = {i: sum(tf.values()) for i, tf in self._tf.items()}
        n = max(len(docs), 1)
        self._avg = (sum(self._len.values()) / n) or 1.0
        df: Counter = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, k: int = 5, focus: Optional[str] = None) -> list[Hit]:
        q = set(tokenize(query))
        if not q:
            return []
        f = set(tokenize(focus)) if focus is not None else q
        hits = []
        for doc_id, tf in self._tf.items():
            score = 0.0
            for w in q:
                freq = tf.get(w, 0)
                if freq:
                    norm = freq + self.k1 * (1 - self.b + self.b * self._len[doc_id] / self._avg)
                    score += self._idf[w] * freq * (self.k1 + 1) / norm
            if score > 0:
                cov = len(f & set(tf)) / len(f) if f else 0.0
                hits.append(Hit(doc_id, score, cov))
        hits.sort(key=lambda h: (-h.score, h.id))
        return hits[:k]
