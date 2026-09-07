"""
Retrieval evaluation.

Uses the provided QA dataset (Data_ret.csv) purely as a benchmark - never as
part of the retrieval knowledge base. Each row has:
    - Question: the query
    - Context : the ground-truth supporting passage from the report
    - Value   : the ground-truth answer value

Because our chunks are re-derived from the PDFs (different boundaries than the
dataset's Context), we treat a retrieved chunk as a "hit" when it has strong
token overlap with the ground-truth Context. We then compute standard retrieval
metrics for each method:

    Dense | BM25 | Hybrid | Hybrid + Cross-Encoder Reranker

Metrics: Recall@K, Precision@K, MRR, Hit Rate. The purpose is to show whether
the hybrid architecture actually improves retrieval - no numbers are hardcoded.
"""
from __future__ import annotations

import json
import random
import re

import pandas as pd

from backend.config import EVAL_RESULTS_PATH, QA_CSV_PATH, settings
from backend.reranking.reranker import Reranker
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.store import ChunkStore

_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
         "is", "are", "was", "were", "be", "as", "at", "by", "it", "its"}


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(str(text).lower()) if t not in _STOP}


def _overlap_ratio(chunk_text: str, context_text: str) -> float:
    """Fraction of ground-truth context tokens present in the chunk."""
    ctx = _tokens(context_text)
    if not ctx:
        return 0.0
    ch = _tokens(chunk_text)
    return len(ctx & ch) / len(ctx)


def _is_hit(chunk_text: str, context_text: str, threshold: float = 0.5) -> bool:
    return _overlap_ratio(chunk_text, context_text) >= threshold


def _metrics_for_ranking(hit_flags: list[bool], k: int) -> dict:
    """Compute Recall@K, Precision@K, MRR, HitRate for a single query.

    With one ground-truth passage per query, Recall@K == HitRate@K, but we
    report both for clarity. Precision@K = hits_in_topk / k.
    """
    topk = hit_flags[:k]
    hit = any(topk)
    first_rank = next((i + 1 for i, f in enumerate(hit_flags) if f), None)
    return {
        "hit": 1.0 if hit else 0.0,
        "recall": 1.0 if hit else 0.0,
        "precision": (sum(topk) / k) if k else 0.0,
        "rr": (1.0 / first_rank) if first_rank else 0.0,
    }


def load_qa(sample_size: int | None = None, seed: int = 42) -> list[dict]:
    if not QA_CSV_PATH.exists():
        raise FileNotFoundError(f"QA dataset not found at {QA_CSV_PATH}.")
    df = pd.read_csv(QA_CSV_PATH)
    df = df.dropna(subset=["Question", "Context"])
    rows = [{"question": str(r["Question"]), "context": str(r["Context"])} for _, r in df.iterrows()]
    if sample_size and sample_size < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_size)
    return rows


def evaluate(sample_size: int | None = 150, k: int = None, verbose: bool = True) -> dict:
    """Run the retrieval benchmark and return a results dict."""
    k = k or settings.RERANK_TOP_K
    store = ChunkStore.load()
    dense = DenseRetriever(store).load()
    bm25 = BM25Retriever(store).load()
    hybrid = HybridRetriever(dense, bm25)
    reranker = Reranker()

    qa = load_qa(sample_size=sample_size)
    if verbose:
        print(f"Evaluating on {len(qa)} QA pairs (k={k}) ...", flush=True)

    methods = ["dense", "bm25", "hybrid", "hybrid_reranked"]
    agg = {m: {"hit": 0.0, "recall": 0.0, "precision": 0.0, "rr": 0.0} for m in methods}
    # Content-type breakdown of the reranked hits, so we can distinguish
    # table / figure / OCR / narrative evidence retrieval (not just an overall
    # number). Counts the content_type of the first hit chunk per query.
    hit_by_type: dict[str, int] = {}

    # Retrieve a wider pool so reranker has candidates.
    pool = max(settings.TOP_K_HYBRID, 10)

    for i, item in enumerate(qa):
        q, ctx = item["question"], item["context"]
        fused = hybrid.search(q, top_k=pool, top_k_dense=pool, top_k_bm25=pool)

        dense_flags = [_is_hit(r["text"], ctx) for r in fused["dense_results"]]
        bm25_flags = [_is_hit(r["text"], ctx) for r in fused["bm25_results"]]
        hybrid_flags = [_is_hit(r["text"], ctx) for r in fused["hybrid_results"]]

        # Rerank the SAME candidate pool production uses (the full fused union),
        # so the "Hybrid + Cross-Encoder" row reflects the real pipeline rather
        # than a narrower hybrid-top-k pool.
        reranked = reranker.rerank(q, fused["candidate_pool"], top_k=pool)
        rerank_flags = [_is_hit(r["text"], ctx) for r in reranked]

        for m, flags in zip(methods, [dense_flags, bm25_flags, hybrid_flags, rerank_flags]):
            mm = _metrics_for_ranking(flags, k)
            for key in agg[m]:
                agg[m][key] += mm[key]

        # Record the content type of the first reranked hit within top-k.
        for r in reranked[:k]:
            if _is_hit(r["text"], ctx):
                ct = r.get("content_type", "text")
                hit_by_type[ct] = hit_by_type.get(ct, 0) + 1
                break

        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(qa)} done", flush=True)

    n = len(qa)
    results = {
        "num_eval_questions": n,
        "k": k,
        "methods": {},
        "metric_names": ["Recall@K", "Precision@K", "MRR", "Hit Rate"],
    }
    label = {
        "dense": "Dense",
        "bm25": "BM25",
        "hybrid": "Hybrid",
        "hybrid_reranked": "Hybrid + Cross-Encoder",
    }
    for m in methods:
        results["methods"][m] = {
            "label": label[m],
            "recall_at_k": round(agg[m]["recall"] / n, 4),
            "precision_at_k": round(agg[m]["precision"] / n, 4),
            "mrr": round(agg[m]["rr"] / n, 4),
            "hit_rate": round(agg[m]["hit"] / n, 4),
        }
    # Reranked-hit breakdown by evidence content type (table/figure/ocr/text).
    results["reranked_hit_by_content_type"] = dict(
        sorted(hit_by_type.items(), key=lambda kv: kv[1], reverse=True)
    )
    return results


def save_results(results: dict) -> None:
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_results() -> dict | None:
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def dataset_stats() -> dict:
    """Return real dataset statistics for the Evaluation page."""
    store = ChunkStore.load()
    num_docs = len({c["document_file"] for c in store.chunks})
    num_questions = 0
    if QA_CSV_PATH.exists():
        try:
            df = pd.read_csv(QA_CSV_PATH).dropna(subset=["Question"])
            num_questions = int(df["Question"].nunique())
        except Exception:
            num_questions = 0
    return {
        "num_documents": num_docs,
        "num_chunks": len(store),
        "num_eval_questions": num_questions,
    }
