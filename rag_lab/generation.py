"""Generation: citation-grounded prompting against retrieved chunks."""
from __future__ import annotations

from abc import ABC, abstractmethod

import requests

from .chunking import Chunk


class LLM(ABC):
    @abstractmethod
    def chat(self, system: str, user: str, temperature: float = 0.1) -> str: ...


class OllamaLLM(LLM):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": temperature},
                "stream": False,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


class StubLLM(LLM):
    """For offline tests: echoes chunk citations."""

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        return "Stub answer based on the provided context."


SYSTEM_PROMPT = """You answer questions using ONLY the context below.
- Every factual claim must cite its source as [1], [2], etc.
- If the context doesn't contain the answer, say "I don't know based on the provided documents."
- Be concise."""


def build_prompt(question: str, chunks: list[Chunk], max_chars: int) -> tuple[str, int]:
    """Return (prompt, n_used): chunks[:n_used] are the ones in the context.

    Chunks are added in rank order until max_chars is reached. The top chunk
    is always included (truncated if needed) so the model never gets an empty
    context just because one chunk is oversized.
    """
    parts = []
    total = 0
    for i, c in enumerate(chunks, start=1):
        block = f"[{i}] ({c.metadata.get('title', c.doc_id)})\n{c.text}\n"
        if total + len(block) > max_chars:
            if not parts:
                parts.append(block[:max_chars])
            break
        parts.append(block)
        total += len(block)
    context = "\n".join(parts)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer with citations:"
    return prompt, len(parts)
