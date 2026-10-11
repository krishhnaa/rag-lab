"""Retrieval: dense, BM25 sparse, and RRF hybrid fusion.

Why hybrid: dense embeddings capture meaning ("time off" ~ "vacation") but
miss exact tokens ("PTO-2024"); BM25 catches exact tokens but misses meaning.
Reciprocal Rank Fusion combines both ranked lists without tuning weights.

IMPORTANT on scores: dense scores are cosine similarities (-1..1), BM25
scores are unbounded positive numbers, and RRF scores top out around
2/(rrf_k+1). Never compare scores across methods - compare RANKS instead.
Every Hit carries its per-method ranks for exactly this reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk
from .embeddings import EmbeddingProvider
from .vectorstore import VectorStore

try:
    from rank_bm25 import BM25Okapi

    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Hit:
    """One retrieved chunk.

    score: the primary score for the method used (RRF fusion score for
        hybrid hits - NOT comparable to cosine/BM25 scores).
    ranks: 1-based rank in each method's list, e.g. {"dense": 1, "bm25": 3}.
    method_scores: raw score each method gave, e.g. {"dense": 0.589}.
    """

    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    method_scores: dict[str, float] = field(default_factory=dict)

    def rank_str(self) -> str:
        if not self.ranks:
            return ""
        return " | ".join(f"{m} #{r}" for m, r in self.ranks.items())


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        store: VectorStore,
        embedder: EmbeddingProvider,
    ):
        self.chunks = chunks
        self.store = store
        self.embedder = embedder
        self._bm25 = None
        if _HAS_BM25:
            self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def dense(self, query: str, k: int) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        return [
            Hit(c, s, {"dense": r}, {"dense": s})
            for r, (c, s) in enumerate(self.store.search(q, k), start=1)
        ]

    def bm25(self, query: str, k: int) -> list[Hit]:
        if self._bm25 is None:
            raise RuntimeError("rank_bm25 not installed: pip install rank-bm25")
        scores = self._bm25.get_scores(_tokenize(query))
        hits: list[Hit] = []
        for r, i in enumerate(np.argsort(-scores)[:k], start=1):
            if scores[i] <= 0:
                continue
            s = float(scores[i])
            hits.append(Hit(self.chunks[i], s, {"bm25": r}, {"bm25": s}))
        return hits

    def hybrid(
        self, query: str, k: int, bm25_k: int = 20, rrf_k: int = 60
    ) -> list[Hit]:
        """RRF fusion: score(d) = sum 1/(rrf_k + rank) over each ranked list."""
        runs: dict[str, list[Hit]] = {"dense": self.dense(query, bm25_k)}
        try:
            runs["bm25"] = self.bm25(query, bm25_k)
        except RuntimeError:
            pass  # dense-only fallback
        fused: dict[str, float] = {}
        info: dict[str, dict] = {}
        for method, hits in runs.items():
            for h in hits:
                r = h.ranks[method]
                cid = h.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + r)
                e = info.setdefault(
                    cid,
                    {
                        "chunk": h.chunk,
                        "ranks": {},
                        "method_scores": {},
                    },
                )
                e["ranks"][method] = r
                e["method_scores"][method] = h.method_scores[method]
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            Hit(
                info[cid]["chunk"],
                score,
                info[cid]["ranks"],
                info[cid]["method_scores"],
            )
            for cid, score in ranked
        ]

    def retrieve(
        self, query: str, k: int, hybrid: bool = True, **kwargs
    ) -> list[Hit]:
        if hybrid:
            return self.hybrid(query, k, **kwargs)
        return self.dense(query, k)
