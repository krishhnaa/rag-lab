"""The end-to-end RAG pipeline: retrieve -> generate with citations."""
from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import Chunk
from .generation import LLM, SYSTEM_PROMPT, build_prompt
from .retrieval import Hit, Retriever


@dataclass
class Answer:
    """chunks/scores/hits are what the model actually saw (citations [1]..[n]
    refer to these). retrieved is the full top-k list, used for metrics."""

    text: str
    question: str
    chunks: list[Chunk] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        top_k: int = 5,
        hybrid: bool = True,
        rrf_k: int = 60,
        bm25_k: int = 20,
        max_context_chars: int = 6000,
        temperature: float = 0.1,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.hybrid = hybrid
        self.rrf_k = rrf_k
        self.bm25_k = bm25_k
        self.max_context_chars = max_context_chars
        self.temperature = temperature

    def retrieve(self, question: str) -> list[Hit]:
        return self.retriever.retrieve(
            question,
            k=self.top_k,
            hybrid=self.hybrid,
            bm25_k=self.bm25_k,
            rrf_k=self.rrf_k,
        )

    def ask(self, question: str) -> Answer:
        results = self.retrieve(question)
        prompt, n_used = build_prompt(
            question, [h.chunk for h in results], self.max_context_chars
        )
        used = results[:n_used]
        text = self.llm.chat(SYSTEM_PROMPT, prompt, temperature=self.temperature)
        return Answer(
            text=text, question=question,
            chunks=[h.chunk for h in used], scores=[h.score for h in used],
            hits=used, retrieved=results,
        )
