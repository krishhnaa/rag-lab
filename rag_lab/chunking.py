"""Chunking strategies. Chunking is a hyperparameter — measure it, don't guess.

- fixed:     sliding character window (simple baseline)
- recursive: split on paragraphs, then sentences, then characters, packing
             each chunk up to chunk_size (structure-aware, usually better)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .documents import Document


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str          # f"{doc_id}#{index}"
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(d["doc_id"], d["chunk_id"], d["text"], d.get("metadata", {}))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pack(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack pieces into chunks of <= size chars."""
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # start next chunk with overlap from the tail of the previous
            current = (current[-overlap:] + " " + piece).strip() if current else piece
            if len(current) > size:  # single huge piece: hard cut
                while len(current) > size:
                    chunks.append(current[:size])
                    current = current[size - overlap:]
                current = current.strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    out, step = [], max(1, size - overlap)
    for i in range(0, len(text), step):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        if i + size >= len(text):
            break
    return out


def chunk_recursive(text: str, size: int, overlap: int) -> list[str]:
    # 1. paragraphs 2. sentences 3. characters — finest split that fits
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            pieces.append(para)
        else:
            for sent in _split_sentences(para):
                if len(sent) <= size:
                    pieces.append(sent)
                else:  # pathological long sentence: hard cut
                    pieces.extend(chunk_fixed(sent, size, overlap))
    return _pack(pieces, size, overlap)


STRATEGIES = {"fixed": chunk_fixed, "recursive": chunk_recursive}


def chunk_documents(
    docs: list[Document],
    strategy: str = "recursive",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose {list(STRATEGIES)}")
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        # overlap >= size would make _pack's hard-cut loop never advance
        raise ValueError(
            f"need chunk_size > 0 and 0 <= chunk_overlap < chunk_size "
            f"(got size={chunk_size}, overlap={chunk_overlap})"
        )
    fn = STRATEGIES[strategy]
    chunks: list[Chunk] = []
    for doc in docs:
        for i, text in enumerate(fn(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=f"{doc.doc_id}#{i}",
                    text=text,
                    metadata={"title": doc.title, "strategy": strategy},
                )
            )
    return chunks
