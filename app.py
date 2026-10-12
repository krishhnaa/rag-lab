"""RAG Lab - interactive demo UI.

Run from ~/rag-lab with the venv activated:
    streamlit run app.py
"""
from __future__ import annotations

import copy
import json
import re
import sys
from collections import Counter
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
from rag_lab.index_meta import META_FILE, check_meta
from rag_lab.retrieval import Hit

st.set_page_config(page_title="RAG Lab", page_icon="search", layout="wide")

conf = cfg.load_config(ROOT)
paths = _paths(conf, ROOT)

# Plain-language names for retrieval methods, used everywhere a rank is shown.
METHOD_LABELS = {"dense": "Meaning match", "bm25": "Keyword match"}

EXAMPLE_QUESTIONS = [
    "How many vacation days do employees with 4 years of tenure get?",
    "What is the home office stipend for remote employees?",
    "Is SMS allowed for multi-factor authentication?",
    "What is the CEO's favorite color?",  # not in the docs: shows a refusal
]

# ---------------- sidebar ----------------
st.sidebar.title("Settings")
st.sidebar.caption("These only affect your session.")
top_k = st.sidebar.slider(
    "Top K chunks", 1, 10, conf["retrieval"]["top_k"],
    help="How many document passages (chunks) are retrieved and handed to "
    "the model as context. More = more context but more noise.",
)
hybrid = st.sidebar.checkbox(
    "Hybrid retrieval (RRF)", value=conf["retrieval"]["hybrid"],
    help="On: combine meaning-based search (embeddings) with keyword search "
    "(BM25) using Reciprocal Rank Fusion. Off: meaning-based search only. "
    "Keyword search helps with exact terms like policy codes.",
)
st.sidebar.divider()
st.sidebar.subheader("Index")
if st.sidebar.button(
    "Rebuild index",
    help="Re-read everything in data/docs, split it into chunks and embed "
    "it. Do this after adding or changing documents.",
):
    with st.spinner("Ingesting documents..."):
        cmd_ingest(ROOT, conf, None)
    st.cache_resource.clear()
    st.sidebar.success("Index rebuilt!")


@st.cache_resource
def get_pipeline():
    return load_pipeline_from_index(ROOT, conf)


# ---------------- header ----------------
st.title("RAG Lab")
st.caption(
    "Ask questions about the documents in `data/docs`. Answers come only "
    "from those documents, cite their sources, and say \"I don't know\" "
    "when the documents don't cover the question."
)
with st.expander("How it works", icon=":material/info:"):
    c = conf["chunking"]
    st.markdown(
        f"""
1. **Load & chunk.** Each document is split into passages of about
   {c['chunk_size']} characters (overlapping by {c['chunk_overlap']}), using the
   `{c['strategy']}` strategy.
2. **Embed.** Each chunk is turned into a vector with
   `{conf['ollama']['embed_model']}`, so passages with similar *meaning* end up close together.
3. **Retrieve.** Your question is matched against the chunks by
   **meaning** (vector similarity) and, in hybrid mode, also by **keywords**
   (BM25). The two rankings are merged with Reciprocal Rank Fusion and the
   top **K** chunks are kept.
4. **Generate.** `{conf['ollama']['llm_model']}` answers using *only*
   those chunks and cites them as [1], [2]… Open **Sources** under an answer
   to read exactly what it was given.
5. **Evaluate.** The **Evals** tab scores the whole pipeline on a fixed set
   of questions, so every change is measured instead of guessed.

Everything runs locally through Ollama: no data leaves your machine.
"""
    )

# ---------------- index checks ----------------
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

chunk_counts = Counter(ch.doc_id for ch in pipeline.retriever.chunks)
try:
    meta = json.loads((paths["index"] / META_FILE).read_text())
except (OSError, ValueError):
    meta = {}
st.sidebar.markdown(
    f"**{len(chunk_counts)}** documents · **{sum(chunk_counts.values())}** chunks  \n"
    f"Embeddings: `{meta.get('embed_model', '?')}`  \n"
    f"Chunking: `{meta.get('strategy', '?')}`, {meta.get('chunk_size', '?')} chars"
)


# ---------------- helpers ----------------
def _short_title(hit: Hit) -> str:
    title = hit.chunk.metadata.get("title", hit.chunk.doc_id)
    return title[:32]


def _plain(text: str) -> str:
    """Escape Markdown so chunk text (which may contain '# Heading', '**',
    lists...) is shown literally instead of rendered as huge headings."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~<$])", r"\\\1", text)


def _rank_label(hit: Hit) -> str:
    return " · ".join(
        f"{METHOD_LABELS.get(m, m)} #{r}" for m, r in hit.ranks.items()
    )


def _settings_label(settings: dict) -> str:
    mode = "hybrid (meaning + keywords)" if settings["hybrid"] else "meaning only"
    return f"Top K {settings['top_k']} · {mode}"


def _show_sources(hits: list[Hit], settings: dict | None = None) -> None:
    with st.expander(f"Sources ({len(hits)})"):
        st.caption(
            "The passages the model was given, in rank order. **[n]** matches "
            "the citations in the answer. *Meaning match #2* means "
            "meaning-based search ranked it 2nd; *Keyword match #1* means "
            "keyword search ranked it 1st."
            + (f" Retrieved with: {_settings_label(settings)}." if settings else "")
        )
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}] {_short_title(h)}** · {_rank_label(h)}")
            text = h.chunk.text
            st.caption(_plain(text[:600]) + ("..." if len(text) > 600 else ""))


def _show_score_chart(hits: list[Hit]) -> None:
    """Bar chart of scores + per-method agreement (hybrid mode)."""
    labels = [f"[{i + 1}] {_short_title(h)}" for i, h in enumerate(hits)]
    st.caption(
        "Final score of each retrieved passage (higher = ranked higher). In "
        "hybrid mode this is the fused RRF score, so compare bars with each "
        "other, not with cosine or BM25 numbers."
    )
    st.bar_chart(pd.DataFrame({"score": [h.score for h in hits]}, index=labels))
    methods = sorted({m for h in hits for m in h.method_scores})
    if len(methods) > 1:
        # Min-max normalize each method's raw scores so agreement is visible.
        # (Raw dense/BM25 scores live on different scales - see Hit docs.)
        norm: dict[str, list[float]] = {}
        for m in methods:
            vals = [h.method_scores.get(m, 0.0) for h in hits]
            lo, hi = min(vals), max(vals)
            norm[METHOD_LABELS.get(m, m)] = (
                [(v - lo) / (hi - lo) for v in vals] if hi > lo else [1.0] * len(vals)
            )
        st.caption(
            "How each search method scored the same passages (scaled 0–1). "
            "Bars of equal height mean meaning and keyword search agreed."
        )
        st.bar_chart(pd.DataFrame(norm, index=labels))


# ---------------- tabs ----------------
tab_chat, tab_evals, tab_docs = st.tabs(["Chat", "Evals", "Documents"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("**Try an example question:**")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(ex, key=f"ex{i}", width="stretch"):
                st.session_state.pending_q = ex
        st.caption(
            "The last example isn't covered by the documents: the "
            "assistant should say it doesn't know instead of inventing an answer."
        )

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("hits"):
                _show_sources(m["hits"], m.get("settings"))

    # Score chart for the most recent answer.
    last_hits = next(
        (m["hits"] for m in reversed(st.session_state.messages) if m.get("hits")),
        None,
    )
    if last_hits:
        with st.expander("Retrieval score chart", expanded=False):
            _show_score_chart(last_hits)

    typed = st.chat_input("Ask a question about your documents...")
    if q := typed or st.session_state.pop("pending_q", None):
        settings = {"top_k": top_k, "hybrid": hybrid}
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents and writing an answer..."):
                ans = pipeline.ask(q)
            st.markdown(ans.text)
            _show_sources(ans.hits, settings)
        st.session_state.messages.append(
            {"role": "assistant", "content": ans.text, "hits": ans.hits,
             "settings": settings}
        )
        st.rerun()

with tab_evals:
    st.subheader("Evaluation suite")
    items = load_eval_set(paths["evals"])
    st.markdown(
        f"Runs **{len(items)} test questions** with known answers through the "
        "pipeline, using your current sidebar settings, and scores how well it "
        "did. Re-run this after changing documents, settings or code to see "
        "whether things got better or worse."
    )
    with st.expander(f"See the {len(items)} test questions"):
        st.dataframe(
            [{"question": it.question, "answer should mention": ", ".join(it.expected_phrases),
              "correct document": it.source_doc} for it in items],
            width="stretch",
        )
    use_judge = st.checkbox(
        "LLM-judge faithfulness (slower)", value=False,
        help="A model reads each answer next to the passages it was given and "
        "grades 1–5 whether every claim is supported by them, i.e. whether "
        "the answer stays faithful to the documents.",
    )
    if st.button("Run evals", type="primary"):
        judge = judge_llm(conf) if use_judge else None
        with st.spinner(f"Asking all {len(items)} questions..."):
            result = run_evaluation(pipeline, items, judge=judge)
        k = result["k"]
        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"Hit@{k}", f"{result['hit@k']:.2f}",
            help=f"Share of questions where the correct document appeared "
            f"anywhere in the top {k} retrieved passages. 1.00 = always found.",
        )
        c2.metric(
            "MRR", f"{result['mrr']:.2f}",
            help="Mean Reciprocal Rank: how high the correct document ranked. "
            "1st place scores 1, 2nd scores 0.5, 3rd 0.33… averaged over questions.",
        )
        c3.metric(
            "Phrase recall", f"{result['phrase_recall']:.2f}",
            help="Share of the expected key phrases (e.g. '20 days') that "
            "actually appear in the generated answers.",
        )
        if "faithfulness" in result:
            st.metric(
                "Faithfulness (judge)", f"{result['faithfulness']:.2f}",
                help="Average judge grade scaled to 0–1. 1.00 = every claim "
                "was supported by the retrieved passages.",
            )
        if result.get("judge_failures"):
            st.caption(
                f"{result['judge_failures']} judge replies couldn't be parsed "
                "and were excluded from faithfulness."
            )
        st.markdown("**Per question**")
        st.caption(
            "**rr** = reciprocal rank of the correct document (1 = ranked first, "
            "0 = not retrieved) · **recall** = phrase recall for that answer · "
            "**faith** = judge grade, if enabled."
        )
        st.dataframe(result["rows"], width="stretch")

with tab_docs:
    st.subheader("Indexed documents")
    st.caption(
        "Files in `data/docs` and how many chunks each contributed to the "
        "index. Add or edit files there, then click **Rebuild index**."
    )
    for d in load_docs(paths["docs"]):
        n = chunk_counts.get(d.doc_id, 0)
        status = f"{n} chunks" if n else "not indexed yet: click **Rebuild index**"
        st.markdown(f"**{d.title}**")
        st.caption(f"`{d.doc_id}` · {len(d.text):,} chars · {status}")
