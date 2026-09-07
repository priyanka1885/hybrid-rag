"""
Evaluation command.

    python scripts/evaluate.py                 # default sample (150 QA pairs)
    python scripts/evaluate.py --sample 300     # larger sample
    python scripts/evaluate.py --full           # all QA pairs (slow)

Runs the retrieval benchmark comparing Dense / BM25 / Hybrid / Hybrid+Reranker
against the ground-truth Context passages in the QA dataset, then saves the
metrics to data/processed/evaluation.json for the frontend Evaluation page.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.evaluation.evaluator import evaluate, save_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on the QA dataset.")
    parser.add_argument("--sample", type=int, default=150, help="Number of QA pairs to sample.")
    parser.add_argument("--full", action="store_true", help="Use the full QA set (overrides --sample).")
    parser.add_argument("--k", type=int, default=None, help="Cutoff K for metrics (default RERANK_TOP_K).")
    args = parser.parse_args()

    sample = None if args.full else args.sample
    try:
        results = evaluate(sample_size=sample, k=args.k, verbose=True)
    except Exception as exc:
        print(f"\nEvaluation failed: {exc}", file=sys.stderr)
        return 1

    save_results(results)

    print("\n=== Retrieval comparison (higher is better) ===")
    header = f"{'Method':<26}{'Recall@K':>10}{'Prec@K':>10}{'MRR':>8}{'HitRate':>10}"
    print(header)
    print("-" * len(header))
    for m in results["methods"].values():
        print(
            f"{m['label']:<26}{m['recall_at_k']:>10.3f}{m['precision_at_k']:>10.3f}"
            f"{m['mrr']:>8.3f}{m['hit_rate']:>10.3f}"
        )
    print(f"\nEvaluated on {results['num_eval_questions']} questions at K={results['k']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
