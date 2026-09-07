"""
Ingestion command.

    python scripts/ingest.py            # build indexes (reuse if present)
    python scripts/ingest.py --force    # force a full rebuild

Runs: PDFs -> text -> chunks -> metadata -> embeddings -> FAISS -> BM25 -> persist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as `python scripts/ingest.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingestion.pipeline import run_ingestion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Hybrid RAG indexes from financial-report PDFs.")
    parser.add_argument("--force", action="store_true", help="Rebuild indexes even if they already exist.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    args = parser.parse_args()

    try:
        summary = run_ingestion(force=args.force, verbose=not args.quiet)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"\nIngestion failed: {exc}", file=sys.stderr)
        return 1

    print("\n=== Ingestion summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
