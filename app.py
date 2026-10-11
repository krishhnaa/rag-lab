"""RAG Lab - interactive demo UI.

Run from ~/rag-lab with the venv activated:
    streamlit run app.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from rag_lab import config as cfg
from rag_lab.__main__ import _paths, cmd_ingest, judge_llm, load_pipeline_from_index
from rag_lab.documents import load_docs
from rag_lab.evaluate import load_eval_set, run_evaluation
from rag_lab.index_meta import check_meta
from rag_lab.retrieval import Hit

st.set_page_config(page_title="RAG Lab", page_icon="search", layout="wide")

conf = cfg.load_config(ROOT)
paths = _paths(conf, ROOT)

# ---------------- sidebar ----------------
st.sidebar.title("Settings")
top_k = st.sidebar.slider("Top K chunks", 1, 10, conf["retrieval"]["top_k"])
hybrid = st.sidebar.checkbox(
    "Hybrid retrieval (RRF)", value=conf["retrieval"]["hybrid"]
)
st.sidebar.divider()
if st.sidebar.button("Rebuild index"):
    with st.spinner("Ingesting documents..."):
        cmd_ingest(ROOT, conf, None)
    st.cache_resource.clear()
    st.sidebar.success("Index rebuilt!")


@st.cache_resource
def get_pipeline():
    return load_pipeline_from_index(ROOT, conf)


if not (paths["index"] / "chunks.json").exists():
    st.warning(
        "No index found yet. Click **Rebuild index** in the sidebar "
        "(or run `python -m rag_lab ingest` in a terminal)."
    )
    st.stop()

meta_errors, meta_warnings = check_meta(paths["index"], conf)
for w in meta_warnings:
    st.sidebar.warning(w)
if meta_errors:
    for e in meta_errors:
        st.error(e)
    st.info("Click **Rebuild index** in the sidebar to fix this.")
    st.stop()

# The cached pipeline is shared by every browser session, so never mutate it.
# A shallow copy shares the heavy parts (index, BM25, clients) but gets its
# own top_k/hybrid, so one user's sidebar can't change another's results.
pipeline = copy.copy(get_pipeline())
pipeline.top_k = top_k
pipeline.hybrid = hybrid

# ---------------- helpers ----------------
def _short_title(hit: Hit) -> str:
    title = hit.chunk.metadata.get("title", hit.chunk.doc_id)
    return title[:32]


def _show_sources(hits: list[Hit]) -> None:
    with st.expander(f"Sources ({len(hits)})"):
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}] {_short_title(h)}** - {h.rank_str()}")
            text = h.chunk.text
            st.caption(text[:600] + ("..." if len(text) > 600 else ""))


def _show_score_chart(hits: list[Hit]) -> None:
    """Bar chart of scores + per-method agreement (hybrid mode)."""
    labels = [f"[{i + 1}] {_short_title(h)}" for i, h in enumerate(hits)]
    st.bar_chart(pd.DataFrame({"score": [h.score for h in hits]}, index=labels))
    methods = sorted({m for h in hits for m in h.method_scores})
    if len(methods) > 1:
        # Min-max normalize each method's raw scores so agreement is visible.
        # (Raw dense/BM25 scores live on different scales - see Hit docs.)
        norm: dict[str, list[float]] = {}
        for m in methods:
            vals = [h.method_scores.get(m, 0.0) for h in hits]
            lo, hi = min(vals), max(vals)
            norm[m] = (
                [(v - lo) / (hi - lo) for v in vals] if hi > lo else [1.0] * len(vals)
            )
        st.caption(
            "Per-method raw scores, min-max normalized - bars at equal height "
            "mean both methods agreed on that chunk."
        )
        st.bar_chart(pd.DataFrame(norm, index=labels))


# ---------------- tabs ----------------
tab_chat, tab_evals, tab_docs = st.tabs(["Chat", "Evals", "Documents"])

with tab_chat:
    st.subheader("Ask your documents")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("hits"):
                _show_sources(m["hits"])

    # Score chart for the most recent answer.
    last_hits = next(
        (m["hits"] for m in reversed(st.session_state.messages) if m.get("hits")),
        None,
    )
    if last_hits:
        with st.expander("Retrieval score chart", expanded=False):
            _show_score_chart(last_hits)

    if q := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving & generating..."):
                ans = pipeline.ask(q)
            st.markdown(ans.text)
            _show_sources(ans.hits)
        st.session_state.messages.append(
            {"role": "assistant", "content": ans.text, "hits": ans.hits}
        )
        st.rerun()

with tab_evals:
    st.subheader("Evaluation suite")
    items = load_eval_set(paths["evals"])
    st.write(f"{len(items)} questions in the eval set.")
    use_judge = st.checkbox("LLM-judge faithfulness (slower)", value=False)
    if st.button("Run evals"):
        judge = judge_llm(conf) if use_judge else None
        with st.spinner("Running evaluation..."):
            result = run_evaluation(pipeline, items, judge=judge)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Hit@{result['k']}", f"{result['hit@k']:.2f}")
        c2.metric("MRR", f"{result['mrr']:.2f}")
        c3.metric("Phrase recall", f"{result['phrase_recall']:.2f}")
        if "faithfulness" in result:
            st.metric("Faithfulness (judge)", f"{result['faithfulness']:.2f}")
        if result.get("judge_failures"):
            st.caption(
                f"{result['judge_failures']} judge replies couldn't be parsed "
                "and were excluded from faithfulness."
            )
        st.dataframe(result["rows"])

with tab_docs:
    st.subheader("Indexed documents")
    for d in load_docs(paths["docs"]):
        st.markdown(f"**{d.title}**")
        st.caption(f"`{d.doc_id}` - {len(d.text):,} chars")
