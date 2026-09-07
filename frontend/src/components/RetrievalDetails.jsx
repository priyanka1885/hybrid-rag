import { useState } from "react";

function fmt(n, digits = 3) {
  if (n === null || n === undefined) return "—";
  return Number(n).toFixed(digits);
}

function Table({ title, rows, columns }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="mb-4">
        <h4 className="mb-1 text-sm font-semibold text-slate-700">{title}</h4>
        <p className="text-xs text-slate-400">No results.</p>
      </div>
    );
  }
  return (
    <div className="mb-5">
      <h4 className="mb-2 text-sm font-semibold text-slate-700">{title}</h4>
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="min-w-full text-left text-xs">
          <thead className="bg-slate-100 text-slate-600">
            <tr>
              {columns.map((c) => (
                <th key={c.key} className="px-3 py-2 font-medium">
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.chunk_id + i} className="border-t border-slate-100">
                {columns.map((c) => (
                  <td key={c.key} className="px-3 py-2 text-slate-700">
                    {c.render ? c.render(r, i) : r[c.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function RetrievalDetails({ details }) {
  const [open, setOpen] = useState(false);
  if (!details) return null;

  return (
    <div className="mt-6">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 text-sm font-medium text-brand-600 hover:text-brand-700"
      >
        <span className={`transition-transform ${open ? "rotate-90" : ""}`}>▸</span>
        View Retrieval Details
      </button>

      {open && (
        <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
          <p className="mb-4 text-xs text-slate-500">
            Hybrid fusion alpha = {fmt(details.alpha, 2)} (weight on dense vs BM25).
          </p>

          <Table
            title="Dense Retrieval (semantic · FAISS)"
            rows={details.dense_results}
            columns={[
              { key: "rank", label: "Rank" },
              { key: "dense_score", label: "Score", render: (r) => fmt(r.dense_score) },
              { key: "document_name", label: "Document" },
              { key: "page_number", label: "Page" },
            ]}
          />

          <Table
            title="BM25 Retrieval (lexical · keyword)"
            rows={details.bm25_results}
            columns={[
              { key: "rank", label: "Rank" },
              { key: "bm25_score", label: "Score", render: (r) => fmt(r.bm25_score, 2) },
              { key: "document_name", label: "Document" },
              { key: "page_number", label: "Page" },
            ]}
          />

          <Table
            title="Hybrid Retrieval (fused score)"
            rows={details.hybrid_results}
            columns={[
              { key: "rank", label: "Rank" },
              { key: "hybrid_score", label: "Hybrid", render: (r) => fmt(r.hybrid_score) },
              { key: "dense_norm", label: "Dense(n)", render: (r) => fmt(r.dense_norm, 2) },
              { key: "bm25_norm", label: "BM25(n)", render: (r) => fmt(r.bm25_norm, 2) },
              { key: "document_name", label: "Document" },
              { key: "page_number", label: "Page" },
            ]}
          />

          <Table
            title="Cross-Encoder Reranking (final evidence)"
            rows={details.reranked_results}
            columns={[
              { key: "final_rank", label: "Final Rank" },
              { key: "rerank_score", label: "Reranker Score", render: (r) => fmt(r.rerank_score) },
              { key: "document_name", label: "Document" },
              { key: "page_number", label: "Page" },
            ]}
          />
        </div>
      )}
    </div>
  );
}
