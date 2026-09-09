"""
run_engine.py — End-to-end run of the COMPLETE RagEngine pipeline.

Exercises the whole chain for a set of representative queries:

    Question
      -> metadata filter -> Dense + BM25 -> RRF fusion
      -> cross-encoder rerank -> TABLE-AWARE BOOST -> answerability gate
      -> Llama 3.1 grounded generation -> citation mapping + verification

Prints, per query: resolved filter, answerability status, the grounded answer,
its citations, the verification verdict, and the final (post-boost) evidence
order the LLM actually received. Requires OPENROUTER_API_KEY to be set for the
configured LLM model; if it is unavailable the retrieval/rerank stages still
run and the answer field reports the unavailability.

Run:  python run_engine.py
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from backend.generation.llm import OllamaClient
from backend.rag_engine import get_engine

BAR = "=" * 100
SUB = "-" * 100

QUERIES = [
    "What were Sasol's direct Scope 1 GHG emissions in 2023?",
    "How much direct carbon output did Sasol produce in FY2023?",
    "Compare Scope 1 emissions between Sasol and Impala Platinum",
]


def _short(cid) -> str:
    return str(cid)[:14] if cid else "?"


def _has_answer_fig(text: str) -> bool:
    return "58644" in (text or "").replace(" ", "").replace(",", "")


def run_one(engine, q: str) -> None:
    print("\n" + BAR)
    print("QUESTION:", q)
    print(BAR)
    res = engine.answer(q)

    rd = res.get("retrieval_details", {})
    print("Resolved filter :", rd.get("applied_filter"))
    print("Status          :", res.get("status"), "| llm_available:", res.get("llm_available"))
    print("\nANSWER:")
    print(" ", res.get("answer"))

    cits = res.get("citations", [])
    print("\nCITATIONS:")
    if not cits:
        print("  (none)")
    for c in cits:
        print(f"  [{c.get('citation_id')}] {c.get('document_name')} p{c.get('page_number')} "
              f"chunk={_short(c.get('chunk_id'))} type={c.get('content_type')}")

    v = res.get("verification", {})
    print("\nVERIFICATION:", v.get("overall_status"), "-", v.get("summary"))

    print("\nFINAL EVIDENCE ORDER (post table-boost) handed to the LLM:")
    for e in rd.get("evidence", []):
        tag = "  <== 58 644" if _has_answer_fig(e.get("text_preview", "")) else ""
        boost = e.get("table_boost")
        print(f"  #{e.get('final_rank') or e.get('rank'):<2} {_short(e.get('chunk_id')):<14} "
              f"type={e.get('content_type'):<6} rerank={e.get('rerank_score')} "
              f"raw={e.get('rerank_score_raw')} boost={boost}{tag}")


def main() -> None:
    print(BAR)
    print("COMPLETE ENGINE RUN")
    print(BAR)
    health = OllamaClient().health()
    print(f"LLM health: reachable={health['reachable']} "
          f"model_available={health['model_available']} model={health['model']}")

    engine = get_engine()
    print("Engine stats:", engine.stats())

    for q in QUERIES:
        run_one(engine, q)

    print("\n" + BAR)
    print("COMPLETE ENGINE RUN — DONE")
    print(BAR)


if __name__ == "__main__":
    main()
