# RAG Lab

A production-grade **Retrieval-Augmented Generation** system you can run
entirely on your laptop — no API keys, no cloud bills.

**What it demonstrates**

| Concern | How it's handled here |
|---|---|
| Ingestion | Markdown/txt/PDF loaders, pluggable chunking strategies |
| Embeddings | Ollama (`nomic-embed-text`) with a provider interface |
| Retrieval | Dense (cosine) + BM25 sparse, fused with RRF hybrid search |
| Generation | Ollama LLM with citation-grounded prompts |
| Evaluation | Retrieval metrics (Hit@k, MRR) + LLM-judge answer faithfulness |
| Experimentation | One command compares chunking strategies on the eval set |

## Quick start (on your Mac)

```bash
cd ~/rag-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Make sure Ollama is running and pull the models (one-time)
python -m rag_lab setup

# 2. Ingest the sample docs
python -m rag_lab ingest

# 3. Ask a question
python -m rag_lab ask "How many vacation days do new employees get?"

# 4. Run the eval suite
python -m rag_lab evaluate --judge

# 5. Compare chunking strategies head-to-head
python -m rag_lab experiment

# 6. Offline unit tests (stub embeddings/LLM, no Ollama needed)
pip install -r requirements-dev.txt
pytest
```

`ingest` records the embedding model and chunk settings in
`data/index/meta.json`. If you change `embed_model` in `config.yaml`, the CLI
and UI refuse the stale index and ask you to re-ingest. Changes to the
chunking settings only produce a warning.

For `--judge`, set `evaluation.judge_model` to a different model than
`llm_model` so the generator isn't grading its own answers.

## Demo UI

A Streamlit app (`app.py`) with chat, evals, and document tabs:

```bash
streamlit run app.py
```

- **💬 Chat** — ask questions, expand 📚 Sources to see retrieved chunks
  and their scores; toggle hybrid retrieval in the sidebar to compare.
- **📊 Evals** — run the eval suite from the browser, with metric cards.
- **📄 Documents** — what's currently indexed.

## Project layout

```
rag-lab/
├── config.yaml               # models, chunking, retrieval knobs
├── data/docs/                # sample corpus (fictional company handbook)
├── data/index/               # built index (created by `ingest`)
├── evals/eval_set.jsonl       # questions + expected answers + source docs
├── experiments/
│   └── compare_chunking.py   # strategy shootout
├── tests/                    # offline pytest suite
└── rag_lab/
    ├── documents.py          # loaders
    ├── chunking.py           # fixed / recursive strategies
    ├── embeddings.py         # provider interface + Ollama
    ├── vectorstore.py        # numpy cosine store (persisted)
    ├── retrieval.py          # dense, BM25, RRF hybrid
    ├── generation.py         # citation-grounded prompting
    ├── pipeline.py           # ask() end-to-end
    ├── index_meta.py         # index provenance + staleness checks
    └── evaluate.py           # metrics + LLM judge
```

## Key design decisions

- **Local-first.** Ollama provides embeddings and generation, so the whole
  system runs offline. The `EmbeddingProvider` / `LLM` interfaces mean you
  can swap in OpenAI or others without touching the pipeline.
- **Hybrid retrieval, not dense-only.** Dense embeddings miss exact terms
  ("PTO-2024 policy"); BM25 catches them. RRF fusion gets the best of both —
  the eval set proves it (run `experiment` to see the numbers).
- **Chunking is a hyperparameter.** `fixed` vs `recursive` isn't a style
  choice — it's measurable. The experiment runner rebuilds the index per
  strategy and reports Hit@5/MRR so you pick with data.
- **Evals are first-class.** `evaluate` reports retrieval metrics always,
  and `--judge` adds LLM-graded faithfulness. Every future change (new
  chunker, reranker, prompt) gets measured against the same set.

## Extending it (good next steps)

1. Add a **reranker** (cross-encoder) stage in `retrieval.py` and measure
   the MRR delta on the eval set.
2. Add **query rewriting** (HyDE) in `pipeline.py`.
3. Swap the numpy store for **pgvector/Chroma** and note the tradeoffs.
4. Grow `evals/eval_set.jsonl` to 50+ items — evals are the moat.
