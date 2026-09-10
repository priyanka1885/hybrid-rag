# Hybrid RAG for Financial Reports

A complete, local-first **Hybrid Retrieval-Augmented Generation** system for
question answering over financial / sustainability reports. It combines dense
semantic retrieval, BM25 lexical retrieval, transparent hybrid fusion,
cross-encoder reranking, and a **local Llama 3.1 8B Instruct** model to produce
**grounded answers with verified citations** — no paid APIs, no cloud services.

---

## Problem statement

Financial and sustainability reports are long, dense, and full of exact figures
and domain-specific terminology. A user asking *"What is Absa's B-BBEE level?"*
needs an answer that is:

1. **Grounded** — derived only from the actual reports, not model memory.
2. **Cited** — traceable to a specific document and page.
3. **Verified** — the cited evidence should genuinely support the claim.
4. **Honest** — the system should refuse to answer when the reports don't
   contain the information, instead of hallucinating.

This project delivers exactly that, running entirely on `localhost`.

---

## Why Hybrid RAG?

| Component | Role | Why it matters |
|-----------|------|----------------|
| **Dense retrieval** | `Text → Embedding → Vector similarity` | Captures semantic similarity; finds relevant passages even when wording differs. |
| **BM25** | `Text → Tokenization → Lexical scoring` | Handles exact terminology, company names, numbers, and rare phrases. **Not** embedding based. |
| **Hybrid fusion** | `alpha·dense + (1-alpha)·bm25` | Dense and lexical retrieval have complementary strengths; fusion gets the best of both. |
| **Cross-encoder** | `(Question, Passage) → Relevance score` | Precise question–passage relevance scoring on the small candidate set. |
| **Llama 3.1 8B** | `Question + Evidence → Grounded answer` | Capable open-source local LLM for grounded generation without a paid API. |
| **Citation verification** | `Answer + Citation → SUPPORTED / PARTIAL / NOT` | A citation should not merely exist — the evidence should actually support the claim. |
| **Evaluation** | `QA pairs → retrieval metrics` | RAG quality should be measured, not judged from a single demo. |

---

## Architecture

```
Financial Report PDFs
        ↓ PDF Text Extraction (page numbers preserved)
        ↓ Chunking + Metadata
        ↓
  ┌─────────────────────┐
  ↓                     ↓
Dense Retrieval     BM25 Retrieval
(FAISS)             (rank_bm25)
  └──────────┬──────────┘
             ↓
       Hybrid Fusion (normalized score blend)
             ↓
   Cross-Encoder Reranking
             ↓
       Top Evidence
             ↓
   Llama 3.1 8B Instruct (Ollama, local)
             ↓
     Grounded Answer
             ↓
   Citation Verification
             ↓
  Answer + Citations + Evidence + Verification
```

---

## Tech stack

- **Retrieval:** FAISS (dense vector index), `rank_bm25` (lexical)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (configurable)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (configurable)
- **LLM:** Llama 3.1 8B Instruct via **Ollama** (configurable)
- **PDF:** `pdfplumber`
- **Backend:** FastAPI + Pydantic + Uvicorn
- **Frontend:** React + Vite + Tailwind CSS

Everything runs locally. No OpenAI / Gemini / Claude / cloud vector DB required.

---

## Dataset

The knowledge base is a fixed set of financial / sustainability report PDFs
located under `DOCUMENTS_DIR` (default `./Structured data-20250319T105519Z-001`).
PDFs are discovered **recursively**, so nested folders are fine, and **no file
names are hardcoded** — every `*.pdf` found becomes part of the index.

`Data_ret.csv` contains question / ground-truth-context / value triples and is
used **only for evaluation / benchmarking**, never as part of the retrieval
knowledge base.

> This is not a document-upload app. There is no upload, document management, or
> company-selection feature by design — the system works directly with the
> provided dataset.

---

## Installation

### 1. Python backend

```bash
# from the project root
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# local development: serving + ingestion + evaluation + tests
pip install -r requirements-offline.txt
```

Dependencies are split in two, because the deployed service is memory-bound:

| File | Contents | Used by |
| --- | --- | --- |
| `requirements.txt` | only what the API needs to answer a question | the deployed service |
| `requirements-offline.txt` | the above plus `pdfplumber`, `PyMuPDF`, `pandas`, `pytest` | ingestion, evaluation, tests |

Ingestion and evaluation run locally and write their output into `data/`, so the
server never needs their dependencies. Keeping them out saves roughly 117 MB of
resident memory on the instance — see [Deployment](#deployment-memory).

### 2. Frontend

```bash
cd frontend
npm install
```

### 3. Environment

```bash
# copy the example and adjust if needed
cp .env.example .env        # Windows: copy .env.example .env
```

---

## Local LLM setup (Ollama)

1. Install Ollama: https://ollama.com/download
2. Pull the model:

   ```bash
   ollama pull llama3.1:8b
   ```

3. Ollama serves on `http://localhost:11434` by default (matches `.env`).

If the model is unavailable, the app does **not** crash — it returns a helpful
message: *"Local LLM is unavailable. Please make sure Ollama is running and
llama3.1:8b is available."* Retrieval, reranking, and the retrieval-details view
still work without the LLM.

---

## Running locally

### Step 1 — Build the indexes (one time)

```bash
python scripts/ingest.py
```

This runs: `PDFs → text → chunks → metadata → embeddings → FAISS → BM25 → persist`.
Artifacts are written to `data/processed/` and `data/indexes/`. If they already
exist they are reused; use `--force` to rebuild.

### Step 2 — Run the evaluation (optional, for the Evaluation page)

```bash
python scripts/evaluate.py --sample 200     # or --full for all QA pairs
```

### Step 3 — Start the backend

```bash
uvicorn backend.main:app --reload
# Backend: http://localhost:8000   (docs at /docs)
```

### Step 4 — Start the frontend

```bash
cd frontend
npm run dev
# Frontend: http://localhost:5173
```

Open **http://localhost:5173** and ask a question.

---

## Example questions

- *What was Clicks Group's revenue in 2022?*
- *What are Sasol's total greenhouse gas emissions?*
- *What is Absa's B-BBEE level?*
- *How many retail locations does Pick n Pay report?*

Out-of-scope (*"What is the capital of France?"*) and unsupported financial
questions are refused rather than answered — see the answerability behavior
below.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | `{ "status": "ok", "llm": {...}, "indexes_loaded": true, "memory_mb": 331.4 }` |
| `POST` | `/ask` | Body `{ "question": "..." }` → answer, citations, verification, retrieval_details |
| `GET`  | `/evaluation` | Real dataset stats + Dense/BM25/Hybrid/Hybrid+Reranker metrics |

Interactive docs: `http://localhost:8000/docs`.

`memory_mb` is the serving process's resident memory. Watch it across requests to
tell normal steady-state usage from a genuine leak: it should settle and stay
flat, not climb.

---

<a id="deployment-memory"></a>
## Deployment and memory

The service is memory-bound, not CPU-bound. Two fp32 ONNX sessions dominate the
footprint, and on a 512 MB instance there is not much left over:

| Component | Resident |
| --- | --- |
| Python + FastAPI + Uvicorn + Pydantic | ~46 MB |
| NumPy + faiss + ONNX Runtime | ~36 MB |
| Chunk store + FAISS index + BM25 index | ~24 MB |
| Embedder ONNX session (fp32, 86 MB graph) | ~108 MB |
| Cross-encoder ONNX session (fp32, 87 MB graph) | ~95 MB |
| **Steady-state baseline** | **~310 MB** |

`render.yaml` captures the settings that keep it inside the limit. If the service
was created from the dashboard rather than from this blueprint, apply the same
values there — the dashboard's own start command wins.

**The one that matters most: `--workers 1`.** Each uvicorn worker is a complete
copy of the model sessions, so `--workers 2` doubles the baseline to ~620 MB and
the instance is killed on startup, every time.

The rest bound how much a *concurrent* burst can add on top of the baseline:

- `--limit-concurrency` sheds excess load with a 503 instead of running out of
  memory, so a traffic spike degrades rather than restarting the service.
- `MAX_REQUEST_THREADS` caps the thread pool that runs the sync endpoints
  (Starlette defaults to 40). Each in-flight request holds its own candidate pool
  and response tree, so this is a direct multiplier on peak memory.
- `RERANK_TOKEN_BUDGET` / `EMBED_TOKEN_BUDGET` bound transformer inference on
  `batch × seq²` rather than on batch size. Attention memory grows with the
  *square* of sequence length, so a fixed batch of 16 padded to the full
  512-token window allocates ~192 MB for one tensor, while the same budget keeps
  it near 12 MB by batching short table rows widely and long passages narrowly.
  Scores are unaffected — padding is excluded by the attention mask.
- `MALLOC_ARENA_MAX=2` stops glibc from parking freed memory in a separate arena
  per thread, which is what makes resident memory climb request after request and
  look like a leak.

Measured effect of the above on a 512 MB instance: baseline ~310 MB, and peak
under 12-way concurrent load ~370 MB instead of ~520 MB.

If you need more headroom than that, the honest options are to move to a larger
instance or to switch the two ONNX graphs to fp16/int8 exports. The latter halves
the ~200 MB of weights but shifts the vectors and the cross-encoder logits, so it
requires rebuilding the FAISS index and re-checking `MIN_RERANK_SCORE`.

---

## Answerability & hallucination control

The system refuses to answer when evidence is weak, using configurable
thresholds (`MIN_RERANK_SCORE`, `MIN_HYBRID_SCORE`):

- **Completely unrelated** → *"This question is outside the scope of the
  available financial reports."*
- **Financial but unsupported** → *"I couldn't find sufficient evidence in the
  provided financial reports to answer this reliably."*

The system prompt also forbids the LLM from using outside knowledge or inventing
figures, and numeric citations are checked against the source text during
verification.

---

## Project structure

```
hybrid-rag-financial-reports/
├── backend/
│   ├── main.py                 # FastAPI app (/health, /ask, /evaluation)
│   ├── config.py               # env-driven configuration (single source of truth)
│   ├── rag_engine.py           # orchestrates the full pipeline
│   ├── runtime.py              # process memory hygiene (allocator, thread caps)
│   ├── batching.py             # memory-bounded batch planning for inference
│   ├── api/schemas.py          # Pydantic request/response models
│   ├── ingestion/              # pdf_loader.py, chunker.py, pipeline.py
│   ├── embeddings/embedder.py  # local embedding model wrapper
│   ├── retrieval/              # store.py, dense.py, bm25.py, hybrid.py
│   ├── reranking/reranker.py   # cross-encoder reranking
│   ├── generation/llm.py       # Ollama (Llama 3.1) client + grounding prompt
│   ├── citations/              # mapper.py (citations), verifier.py (verification)
│   └── evaluation/evaluator.py # retrieval benchmark
├── frontend/                   # React + Vite + Tailwind (Ask + Evaluation pages)
├── scripts/
│   ├── ingest.py               # build indexes
│   └── evaluate.py             # run the retrieval benchmark
├── tests/                      # pytest suite
├── data/                       # generated: processed/ + indexes/ (gitignored)
├── .env.example
├── render.yaml                 # deploy config (memory limits; see Deployment)
├── requirements.txt            # serving dependencies only
├── requirements-offline.txt    # + ingestion, evaluation, tests
└── README.md
```

---

## Testing

```bash
python scripts/ingest.py        # tests that need indexes are skipped if missing
python -m pytest tests
```

The suite covers PDF ingestion, chunking + metadata, dense / BM25 / hybrid
retrieval, reranking, citation mapping + verification, answerability, and the
`/health`, `/ask`, `/evaluation` endpoints.

---

## Evaluation methodology

`scripts/evaluate.py` runs each QA question through Dense, BM25, Hybrid, and
Hybrid + Cross-Encoder retrieval, and compares the retrieved chunks against the
ground-truth `Context` passage from the dataset. A chunk counts as a **hit** when
it strongly overlaps the ground-truth passage. Metrics reported: **Recall@K,
Precision@K, MRR, Hit Rate**. Results are saved to
`data/processed/evaluation.json` and surfaced on the Evaluation page.

All evaluation numbers displayed in the UI come from this real pipeline — none
are hardcoded. Run the evaluation yourself to reproduce the figures.

---

## Future improvements

- Section-aware / table-aware chunking for cleaner financial tables.
- OCR fallback for image-only pages and RTL-text cleanup.
- Answer-level (faithfulness) evaluation in addition to retrieval metrics.
- Streaming token responses from the LLM.
- Approximate FAISS index (IVF/HNSW) for larger corpora.
```
