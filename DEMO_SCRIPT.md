# Grounded - 90-Second Demo Script

Record with QuickTime Player > File > New Screen Recording.
Have `streamlit run app.py` already running at http://localhost:8501.

---

## 0:00-0:10 - Setup shot
Terminal visible: `streamlit run app.py`. Browser opens to the app.

> "Grounded is a RAG system that answers questions strictly from your
documents - every claim cited, every refusal honest."

## 0:10-0:35 - Grounded answer with sources
In the Chat tab, ask:
"How many vacation days do employees with 4 years of tenure get?"

Wait for the answer, then expand Sources.

> "The answer comes with citations, and I can expand Sources to see
exactly which chunks were retrieved and their scores. Verifiable AI."

## 0:35-0:55 - Honest refusal (the anti-hallucination moment)
Ask: "What's the CEO's favorite color?"

The app replies it doesn't know based on the provided documents.

> "Ask it something the documents don't cover, and it refuses instead
of making something up. That behavior is by design - and measured."

## 0:55-1:15 - Retrieval is a visible engineering choice
In the sidebar, uncheck "Hybrid retrieval (RRF)", re-ask the first
question. Compare the sources/scores.

> "Hybrid retrieval is a toggle, not magic. Dense embeddings match
meaning, BM25 matches exact terms, RRF fuses both - and you can
see the difference live."

## 1:15-1:30 - Everything is measured
Open the Evals tab, click "Run evals". Metric cards appear:
Hit@k, MRR, phrase recall.

> "And none of this is vibes - every change is scored against the
same eval set. That's what makes it engineering."

---

## Tips
- 1080p, browser zoom at 100-110% so text is readable.
- Pause 2 seconds between sections for clean cuts.
- Total: ~90 seconds. No voiceover needed - captions work fine.
