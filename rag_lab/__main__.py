"""CLI: python -m rag_lab <command>"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

from . import config as cfg
from .chunking import chunk_documents
from .documents import load_docs
from .embeddings import OllamaEmbeddings
from .evaluate import load_eval_set, print_report, run_evaluation
from .generation import OllamaLLM
from .index_meta import check_meta, write_meta
from .pipeline import RAGPipeline
from .retrieval import Retriever
from .vectorstore import VectorStore


def _paths(conf: dict, root: Path) -> dict[str, Path]:
    p = conf["paths"]
    return {
        "docs": root / p["docs"],
        "index": root / p["index"],
        "evals": root / p["evals"],
    }


def cmd_setup(root: Path, conf: dict) -> int:
    base = conf["ollama"]["base_url"]
    print(f"Checking Ollama at {base}…")
    try:
        models = requests.get(f"{base}/api/tags", timeout=10).json().get("models", [])
    except Exception as exc:
        print(f"\u274c Ollama not reachable: {exc}")
        print("   Install from https://ollama.com and run `ollama serve`.")
        return 1
    have = {m["name"].split(":")[0] for m in models}
    for key in ("embed_model", "llm_model"):
        want = conf["ollama"][key]
        name = want.split(":")[0]
        if name in have or want in have:
            print(f"  \u2713 {want} already pulled")
        else:
            print(f"  \u2b07 pulling {want} (one-time, may take a few minutes)…")
            requests.post(f"{base}/api/pull", json={"name": want}, timeout=1200)
            print(f"  \u2713 {want} ready")
    print("\n\u2705 Setup complete. Run: python -m rag_lab ingest")
    return 0


def cmd_ingest(root: Path, conf: dict, strategy: str | None) -> int:
    paths = _paths(conf, root)
    docs = load_docs(paths["docs"])
    print(f"Loaded {len(docs)} documents")
    strat = strategy or conf["chunking"]["strategy"]
    chunks = chunk_documents(
        docs, strat, conf["chunking"]["chunk_size"], conf["chunking"]["chunk_overlap"]
    )
    print(f"Chunked into {len(chunks)} chunks (strategy={strat})")
    embedder = OllamaEmbeddings(conf["ollama"]["base_url"], conf["ollama"]["embed_model"])
    print("Embedding…")
    store = VectorStore()
    vecs = embedder.embed([c.text for c in chunks])
    store.add(chunks, vecs)
    store.save(paths["index"])
    write_meta(paths["index"], conf, strat, len(chunks), int(vecs.shape[1]))
    print(f"\u2705 Index saved to {paths['index']} ({len(chunks)} chunks)")
    return 0


def judge_llm(conf: dict) -> OllamaLLM:
    """LLM judge. Defaults to the generator model; set evaluation.judge_model
    to a different one to avoid the model grading its own answers."""
    model = (conf.get("evaluation") or {}).get("judge_model") or conf["ollama"]["llm_model"]
    return OllamaLLM(conf["ollama"]["base_url"], model)


def load_pipeline_from_index(root: Path, conf: dict) -> RAGPipeline:
    paths = _paths(conf, root)
    if not (paths["index"] / "chunks.json").exists():
        print("No index found. Run: python -m rag_lab ingest", file=sys.stderr)
        sys.exit(2)
    errors, warnings = check_meta(paths["index"], conf)
    for w in warnings:
        print(f"\u26a0 {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"\u274c {e}", file=sys.stderr)
        sys.exit(2)
    store = VectorStore.load(paths["index"])
    embedder = OllamaEmbeddings(conf["ollama"]["base_url"], conf["ollama"]["embed_model"])
    retriever = Retriever(store.chunks, store, embedder)
    llm = OllamaLLM(conf["ollama"]["base_url"], conf["ollama"]["llm_model"])
    r = conf["retrieval"]
    return RAGPipeline(
        retriever, llm, top_k=r["top_k"], hybrid=r["hybrid"],
        rrf_k=r["rrf_k"], bm25_k=r["bm25_k"],
        max_context_chars=conf["generation"]["max_context_chars"],
        temperature=conf["generation"]["temperature"],
    )


def cmd_ask(root: Path, conf: dict, question: str, no_hybrid: bool) -> int:
    pipeline = load_pipeline_from_index(root, conf)
    if no_hybrid:
        pipeline.hybrid = False
    ans = pipeline.ask(question)
    print(f"\n\U0001f4ac {ans.text}\n")
    print("Sources:")
    for i, h in enumerate(ans.hits, 1):
        c = h.chunk
        print(f"  [{i}] {c.metadata.get('title', c.doc_id)} ({c.chunk_id}, score={h.score:.3f}, {h.rank_str()})")
    return 0


def cmd_evaluate(root: Path, conf: dict, judge: bool) -> int:
    paths = _paths(conf, root)
    pipeline = load_pipeline_from_index(root, conf)
    items = load_eval_set(paths["evals"])
    print(f"Evaluating on {len(items)} questions…")
    judge_model = judge_llm(conf) if judge else None
    result = run_evaluation(pipeline, items, judge=judge_model, verbose=True)
    print_report(result)
    return 0


def cmd_experiment(root: Path, conf: dict) -> int:
    from experiments.compare_chunking import run as run_exp

    run_exp(root, conf)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag_lab", description="RAG Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="Verify Ollama and pull models")
    p = sub.add_parser("ingest", help="Build the index from data/docs")
    p.add_argument("--strategy", choices=["fixed", "recursive"])

    p = sub.add_parser("ask", help="Ask a question")
    p.add_argument("question")
    p.add_argument("--no-hybrid", action="store_true", help="Dense-only retrieval")

    p = sub.add_parser("evaluate", help="Run the eval suite")
    p.add_argument("--judge", action="store_true", help="LLM-graded faithfulness")

    sub.add_parser("experiment", help="Compare chunking strategies")

    args = parser.parse_args(argv)
    root = cfg.project_root()
    conf = cfg.load_config(root)

    if args.command == "setup":
        return cmd_setup(root, conf)
    if args.command == "ingest":
        return cmd_ingest(root, conf, args.strategy)
    if args.command == "ask":
        return cmd_ask(root, conf, args.question, args.no_hybrid)
    if args.command == "evaluate":
        return cmd_evaluate(root, conf, args.judge)
    if args.command == "experiment":
        return cmd_experiment(root, conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
