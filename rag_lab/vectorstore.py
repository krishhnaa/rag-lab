"""A minimal persisted vector store: cosine search over numpy arrays.

Deliberately simple (no server, no heavy deps) so the retrieval math stays
visible. Swap for Chroma/pgvector later and note the tradeoffs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None  # (n, dim), normalized

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        assert embeddings.shape[0] == len(chunks)
        self.chunks.extend(chunks)
        self.embeddings = (
            embeddings
            if self.embeddings is None
            else np.vstack([self.embeddings, embeddings])
        )

    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if self.embeddings is None or not self.chunks:
            return []
        q = query_emb / max(np.linalg.norm(query_emb), 1e-12)
        scores = self.embeddings @ q  # cosine, since rows are normalized
        idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idx]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        (index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=1)
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        store = cls()
        store.embeddings = np.load(index_dir / "embeddings.npy")
        chunks = json.loads((index_dir / "chunks.json").read_text())
        store.chunks = [Chunk.from_dict(c) for c in chunks]
        return store
