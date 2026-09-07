// Small About modal (not a separate page, per the information architecture).
export default function AboutModal({ open, onClose }) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-2xl bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-800">About</h2>
          <button
            onClick={onClose}
            className="rounded-full px-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
          >
            ✕
          </button>
        </div>
        <div className="space-y-3 text-sm text-slate-600">
          <p>
            <strong>Hybrid RAG for Financial Reports</strong> answers questions over a fixed
            set of financial / sustainability reports using a hybrid retrieval pipeline.
          </p>
          <ul className="list-disc space-y-1 pl-5">
            <li><strong>Dense retrieval</strong> (FAISS) captures semantic similarity.</li>
            <li><strong>BM25</strong> captures exact terminology, figures and company names.</li>
            <li><strong>Hybrid fusion</strong> combines both with a transparent score.</li>
            <li><strong>Cross-encoder reranking</strong> sharpens question–passage relevance.</li>
            <li><strong>Llama 3.1 8B</strong> (local, via Ollama) generates grounded answers.</li>
            <li><strong>Citation verification</strong> checks that evidence supports the claim.</li>
          </ul>
          <p className="text-xs text-slate-400">
            Answers are grounded strictly in the retrieved evidence. The system refuses
            out-of-scope or unsupported questions instead of hallucinating.
          </p>
        </div>
      </div>
    </div>
  );
}
