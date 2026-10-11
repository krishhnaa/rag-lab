"""Experiment: compare chunking strategies on the eval set.

Rebuilds the index per strategy and reports retrieval metrics, so the
chunking choice is made with data, not taste.
"""
from __future__ import annotations

from pathlib import Path

from rag_lab.chunking import STRATEGIES, chunk_documents
from rag_lab.documents import load_docs
from rag_lab.embeddings import OllamaEmbeddings
from rag_lab.evaluate import load_eval_set, retrieval_metrics
from rag_lab.pipeline import RAGPipeline
from rag_lab.retrieval import Retriever
from rag_lab.vectorstore import VectorStore


def run(root: Path, conf: dict) -> None:
    docs = load_docs(root / conf["paths"]["docs"])
    items = load_eval_set(root / conf["paths"]["evals"])
    embedder = OllamaEmbeddings(
        conf["ollama"]["base_url"], conf["ollama"]["embed_model"]
    )
    r = conf["retrieval"]

    print(f"\n\U0001f52c Chunking experiment ({len(items)} eval questions)")
    print(f"{'strategy':<12} {'chunks':>7} {'hit@k':>7} {'mrr':>7}")
    print("-" * 36)
    for strategy in STRATEGIES:
        chunks = chunk_documents(
            docs, strategy,
            conf["chunking"]["chunk_size"], conf["chunking"]["chunk_overlap"],
        )
        store = VectorStore()
        store.add(chunks, embedder.embed([c.text for c in chunks]))
        retriever = Retriever(chunks, store, embedder)
        pipeline = RAGPipeline(
            retriever, None,  # type: ignore[arg-type]
            top_k=r["top_k"], hybrid=r["hybrid"],
            rrf_k=r["rrf_k"], bm25_k=r["bm25_k"],
        )
        m = retrieval_metrics(pipeline, items, r["top_k"])
        print(f"{strategy:<12} {len(chunks):>7} {m['hit@k']:>7.2f} {m['mrr']:>7.2f}")
    print("\nRe-ingest with the winner: python -m rag_lab ingest --strategy <name>")
