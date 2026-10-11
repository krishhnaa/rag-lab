"""Evaluation: retrieval metrics always, LLM-judged answer quality on demand.

Metrics:
- hit@k: fraction of questions where a chunk from the source doc is retrieved
- mrr:   mean reciprocal rank of the first correct chunk
- phrase_recall: fraction of expected phrases present in the answer
- faithfulness (needs --judge): LLM grades 1-5 whether the answer is
  supported by the retrieved context only. Unparseable judge replies are
  counted in judge_failures and excluded from the average.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .generation import LLM
from .pipeline import RAGPipeline
from .retrieval import Hit


@dataclass
class EvalItem:
    question: str
    expected_phrases: list[str]
    source_doc: str


def load_eval_set(path: Path) -> list[EvalItem]:
    items = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            items.append(
                EvalItem(d["question"], d["expected_phrases"], d["source_doc"])
            )
    return items


def reciprocal_rank(results: list[Hit], source_doc: str) -> float:
    """1/rank of the first chunk from source_doc, or 0.0 if not retrieved."""
    for i, h in enumerate(results, 1):
        if h.chunk.doc_id == source_doc:
            return 1.0 / i
    return 0.0


def _summarize_rr(rrs: list[float], k: int) -> dict:
    n = max(1, len(rrs))
    return {
        "hit@k": sum(1 for r in rrs if r > 0) / n,
        "mrr": sum(rrs) / n,
        "n": len(rrs),
        "k": k,
    }


def retrieval_metrics(
    pipeline: RAGPipeline, items: list[EvalItem], k: int
) -> dict[str, float]:
    """Retrieval-only metrics (no LLM calls) - used by experiments."""
    rrs = []
    for item in items:
        results = pipeline.retriever.retrieve(
            item.question, k=k, hybrid=pipeline.hybrid,
            bm25_k=pipeline.bm25_k, rrf_k=pipeline.rrf_k,
        )
        rrs.append(reciprocal_rank(results, item.source_doc))
    return _summarize_rr(rrs, k)


def phrase_recall(answer: str, item: EvalItem) -> float:
    if not item.expected_phrases:
        return 1.0
    low = answer.lower()
    found = sum(1 for p in item.expected_phrases if p.lower() in low)
    return found / len(item.expected_phrases)


JUDGE_SYSTEM = """You grade whether an ANSWER is faithful to the CONTEXT.
Reply with exactly: SCORE: <1-5> REASON: <one sentence>.
5 = every claim supported by context. 1 = contradicts or invents facts."""


def judge_faithfulness(
    llm: LLM, question: str, context: str, answer: str
) -> float | None:
    """Score in (0, 1], or None if the judge's reply couldn't be parsed."""
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER: {answer}"
    out = llm.chat(JUDGE_SYSTEM, user, temperature=0.0)
    m = re.search(r"SCORE:\s*([1-5])", out)
    return float(m.group(1)) / 5.0 if m else None


def run_evaluation(
    pipeline: RAGPipeline,
    items: list[EvalItem],
    judge: LLM | None = None,
    verbose: bool = False,
) -> dict:
    # One ask() per question: retrieval metrics come from ans.retrieved, so
    # the query isn't embedded and searched twice.
    rrs, recalls, faiths, rows = [], [], [], []
    judge_failures = 0
    for item in items:
        ans = pipeline.ask(item.question)
        rr = reciprocal_rank(ans.retrieved, item.source_doc)
        rrs.append(rr)
        rec = phrase_recall(ans.text, item)
        recalls.append(rec)
        faith = None
        if judge is not None:
            ctx = "\n\n".join(c.text for c in ans.chunks)
            faith = judge_faithfulness(judge, item.question, ctx, ans.text)
            if faith is None:
                judge_failures += 1
            else:
                faiths.append(faith)
        rows.append(
            {"q": item.question[:60], "rr": round(rr, 2), "recall": round(rec, 2),
             "faith": round(faith, 2) if faith is not None else None}
        )
        if verbose:
            print(f"  Q: {item.question[:70]}")
            print(
                f"    rr={rr:.2f} recall={rec:.2f}"
                + (f" faith={faith:.2f}" if faith is not None else "")
            )
    result = {
        **_summarize_rr(rrs, pipeline.top_k),
        "phrase_recall": sum(recalls) / max(1, len(recalls)),
    }
    if faiths:
        result["faithfulness"] = sum(faiths) / len(faiths)
    if judge is not None:
        result["judge_failures"] = judge_failures
    result["rows"] = rows
    return result


def print_report(result: dict, label: str = "") -> None:
    head = f" — {label}" if label else ""
    print(f"\n\U0001f4ca Eval report{head}")
    print(f"  hit@{result['k']}:      {result['hit@k']:.2f}")
    print(f"  mrr:           {result['mrr']:.2f}")
    print(f"  phrase_recall: {result['phrase_recall']:.2f}")
    if "faithfulness" in result:
        print(f"  faithfulness:  {result['faithfulness']:.2f}")
    if result.get("judge_failures"):
        print(f"  judge_failures: {result['judge_failures']} (excluded from faithfulness)")
    print(f"  (n={result['n']})")
