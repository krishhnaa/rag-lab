"""Embedding providers. Swap implementations without touching the pipeline."""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np
import requests


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (n, dim) float32 array, L2-normalized rows."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int: ...


def _normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return (a / np.maximum(norms, 1e-12)).astype(np.float32)


class OllamaEmbeddings(EmbeddingProvider):
    """Embeddings via a local Ollama server (e.g. nomic-embed-text)."""

    def __init__(self, base_url: str, model: str, batch_size: int = 64):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self._dim: int | None = None

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embeddings"], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        # Batched so large corpora don't hit the timeout or one giant payload.
        batches = [
            self._embed_batch(texts[i:i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        vecs = np.vstack(batches)
        self._dim = vecs.shape[1]
        return _normalize(vecs)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed(["probe"])
        return self._dim  # type: ignore[return-value]


class StubEmbeddings(EmbeddingProvider):
    """Deterministic pseudo-random embeddings — for offline tests only.

    Each text is seeded by its own hash, so the same text always gets the
    same vector (a query identical to a chunk retrieves that chunk).
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        return np.random.default_rng(seed).standard_normal(self._dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array([self._vec(t) for t in texts], dtype=np.float32))

    @property
    def dim(self) -> int:
        return self._dim
