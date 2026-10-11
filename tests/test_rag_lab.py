"""Offline tests: stub embeddings/LLM, no Ollama needed.

Run: python -m pytest
"""
from __future__ import annotations

import copy
from unittest import mock

import numpy as np
import pytest

from rag_lab.chunking import Chunk, chunk_documents
from rag_lab.documents import Document, load_docs
from rag_lab.embeddings import OllamaEmbeddings, StubEmbeddings
from rag_lab.evaluate import EvalItem, judge_faithfulness, run_evaluation
from rag_lab.generation import LLM, StubLLM, build_prompt
from rag_lab.index_meta import check_meta, write_meta
from rag_lab.pipeline import RAGPipeline
from rag_lab.retrieval import Retriever
from rag_lab.vectorstore import VectorStore

CONF = {
    "ollama": {"embed_model": "nomic-embed-text"},
    "chunking": {"strategy": "recursive", "chunk_size": 800, "chunk_overlap": 120},
}


def _chunk(i: int, text: str = "x") -> Chunk:
    return Chunk(f"doc{i}", f"doc{i}#0", text, {"title": f"Doc {i}"})


def _pipeline(texts: list[str], **kw) -> RAGPipeline:
    chunks = [_chunk(i, t) for i, t in enumerate(texts)]
    emb = StubEmbeddings()
    store = VectorStore()
    store.add(chunks, emb.embed([c.text for c in chunks]))
    return RAGPipeline(Retriever(chunks, store, emb), StubLLM(), **kw)


# ---------------- chunking ----------------

@pytest.mark.parametrize("size,overlap", [(100, 100), (100, 150), (0, 0), (100, -1)])
def test_invalid_overlap_rejected(size, overlap):
    doc = Document("d", "D", "hello world. " * 50, "d.md")
    with pytest.raises(ValueError):
        chunk_documents([doc], "recursive", size, overlap)


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_chunks_respect_size(strategy):
    text = ("Sentence number one is here. " * 40 + "\n\n") * 5 + "x" * 2000
    doc = Document("d", "D", text, "d.md")
    chunks = chunk_documents([doc], strategy, 200, 40)
    assert chunks and all(len(c.text) <= 200 for c in chunks)


# ---------------- documents ----------------

def test_load_docs_recurses_and_avoids_id_collisions(tmp_path):
    (tmp_path / "notes.md").write_text("# Notes\nmd body")
    (tmp_path / "notes.txt").write_text("txt body")
    (tmp_path / "hr").mkdir()
    (tmp_path / "hr" / "benefits.md").write_text("# Benefits\nbody")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.md").write_text("nope")

    ids = sorted(d.doc_id for d in load_docs(tmp_path))
    assert ids == ["hr/benefits", "notes", "notes.txt"]


# ---------------- embeddings ----------------

def test_stub_embeddings_deterministic_per_text():
    emb = StubEmbeddings()
    a = emb.embed(["alpha", "beta"])
    b = emb.embed(["beta", "alpha"])
    assert np.allclose(a[0], b[1]) and np.allclose(a[1], b[0])


def test_ollama_embeddings_batches_requests():
    emb = OllamaEmbeddings("http://x", "m", batch_size=3)

    def fake_post(url, json, timeout):
        resp = mock.Mock()
        resp.json.return_value = {"embeddings": [[1.0, 0.0]] * len(json["input"])}
        return resp

    with mock.patch("rag_lab.embeddings.requests.post", side_effect=fake_post) as post:
        out = emb.embed([str(i) for i in range(7)])
    assert post.call_count == 3  # 3 + 3 + 1
    assert out.shape == (7, 2) and emb.dim == 2
    assert emb.embed([]).shape == (0, 2)


# ---------------- prompt / pipeline ----------------

def test_build_prompt_reports_chunks_used():
    chunks = [_chunk(i, "a" * 100) for i in range(5)]
    prompt, n = build_prompt("q?", chunks, max_chars=250)
    assert n == 2 and "[2]" in prompt and "[3]" not in prompt


def test_build_prompt_keeps_oversized_top_chunk():
    prompt, n = build_prompt("q?", [_chunk(0, "a" * 1000)], max_chars=200)
    assert n == 1 and "[1]" in prompt


def test_answer_sources_match_prompt_context():
    p = _pipeline(["a" * 300] * 6, top_k=5, hybrid=False, max_context_chars=700)
    ans = p.ask("question")
    assert len(ans.retrieved) == 5
    assert len(ans.hits) == len(ans.chunks) == 2
    assert ans.hits == ans.retrieved[:2]


def test_shallow_copy_does_not_change_shared_pipeline():
    shared = _pipeline(["a", "b", "c"], top_k=5, hybrid=True)
    session = copy.copy(shared)
    session.top_k, session.hybrid = 1, False
    assert (shared.top_k, shared.hybrid) == (5, True)
    assert session.retriever is shared.retriever


# ---------------- evaluation ----------------

class _ScriptedLLM(LLM):
    def __init__(self, reply: str):
        self.reply = reply

    def chat(self, system, user, temperature=0.1):
        return self.reply


def test_judge_unparseable_returns_none():
    assert judge_faithfulness(_ScriptedLLM("looks fine"), "q", "c", "a") is None
    assert judge_faithfulness(_ScriptedLLM("SCORE: 4 REASON: ok"), "q", "c", "a") == 0.8


def test_run_evaluation_excludes_judge_failures():
    p = _pipeline(["vacation days are 20", "remote stipend is $750"], top_k=2, hybrid=False)
    items = [EvalItem("vacation days are 20", ["Stub"], "doc0")]
    result = run_evaluation(p, items, judge=_ScriptedLLM("no score here"))
    assert result["judge_failures"] == 1
    assert "faithfulness" not in result
    # Stub embeddings: identical text -> same vector -> rank 1.
    assert result["hit@k"] == 1.0 and result["mrr"] == 1.0


def test_run_evaluation_retrieves_once_per_question():
    p = _pipeline(["a", "b", "c"], top_k=2, hybrid=True)
    items = [EvalItem("q1", [], "doc0"), EvalItem("q2", [], "doc1")]
    with mock.patch.object(p.retriever, "retrieve", wraps=p.retriever.retrieve) as r:
        run_evaluation(p, items)
    assert r.call_count == len(items)


# ---------------- index metadata ----------------

def test_meta_matching_config_is_clean(tmp_path):
    write_meta(tmp_path, CONF, "recursive", 10, 768)
    assert check_meta(tmp_path, CONF) == ([], [])


def test_meta_embed_model_change_is_error(tmp_path):
    write_meta(tmp_path, CONF, "recursive", 10, 768)
    conf = copy.deepcopy(CONF)
    conf["ollama"]["embed_model"] = "mxbai-embed-large"
    errors, _ = check_meta(tmp_path, conf)
    assert len(errors) == 1 and "mxbai-embed-large" in errors[0]


def test_meta_chunking_change_is_warning(tmp_path):
    write_meta(tmp_path, CONF, "fixed", 10, 768)
    conf = copy.deepcopy(CONF)
    conf["chunking"]["chunk_size"] = 500
    errors, warnings = check_meta(tmp_path, conf)
    assert errors == [] and len(warnings) == 2  # chunk_size + strategy


def test_legacy_meta_warns(tmp_path):
    (tmp_path / "meta.json").write_text('{"strategy": "recursive", "n_chunks": 14}')
    errors, warnings = check_meta(tmp_path, CONF)
    assert errors == [] and len(warnings) == 1
