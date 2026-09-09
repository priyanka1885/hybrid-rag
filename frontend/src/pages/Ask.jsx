import { useState } from "react";
import { ask } from "../api";
import VerificationBadge from "../components/VerificationBadge";
import RetrievalDetails from "../components/RetrievalDetails";

const EXAMPLES = [
  "What was Clicks Group's revenue in 2022?",
  "What are Sasol's total greenhouse gas emissions?",
  "What is Absa's B-BBEE level?",
  "How many employees does Pick n Pay report?",
];

// Renders answer text and turns [n] markers into small superscript chips.
function AnswerText({ text }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="leading-relaxed text-slate-800">
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/);
        if (m) {
          return (
            <sup
              key={i}
              className="mx-0.5 rounded bg-brand-100 px-1 text-[0.65rem] font-semibold text-brand-700"
            >
              {m[1]}
            </sup>
          );
        }
        return <span key={i}>{p}</span>;
      })}
    </p>
  );
}

export default function Ask() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function submit(q) {
    const query = (q ?? question).trim();
    if (!query) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await ask(query);
      setResult(data);
    } catch (e) {
      setError(e.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="text-xl font-semibold text-slate-800">Ask</h2>
      <p className="mt-1 text-sm text-slate-500">
        Ask a question about the financial reports.
      </p>

      <div className="mt-4">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
          }}
          rows={3}
          placeholder="e.g. What was Clicks Group's revenue in 2022?"
          className="w-full resize-none rounded-xl border border-slate-300 p-3 text-sm shadow-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
        />
        <div className="mt-3 flex items-center justify-between">
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => {
                  setQuestion(ex);
                  submit(ex);
                }}
                className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600 hover:border-brand-300 hover:text-brand-600"
              >
                {ex}
              </button>
            ))}
          </div>
          <button
            onClick={() => submit()}
            disabled={loading || !question.trim()}
            className="rounded-xl bg-brand-500 px-6 py-2 text-sm font-semibold text-white shadow-sm hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Thinking…" : "Ask"}
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-6 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          {error}
        </div>
      )}

      {result && result.llm_available === false && (
        <div className="mt-8 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          <p className="font-semibold">Answer generation unavailable</p>
          <p className="mt-1">{result.answer}</p>
          <p className="mt-3 text-xs text-amber-700">
            Retrieval still works — expand “View Retrieval Details” below to inspect the
            evidence the pipeline found.
          </p>
          <RetrievalDetails details={result.retrieval_details} />
        </div>
      )}

      {result && result.llm_available !== false && (
        <div className="mt-8">
          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Answer
            </h3>
            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <AnswerText text={result.answer} />
            </div>
          </section>

          {result.verification && (
            <div className="mt-4">
              <VerificationBadge
                status={result.verification.overall_status}
                summary={result.verification.summary}
              />
            </div>
          )}

          {result.citations && result.citations.length > 0 && (
            <section className="mt-6">
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Supporting Evidence
              </h3>
              <div className="space-y-3">
                {result.citations.map((c) => {
                  const pc =
                    result.verification?.per_citation?.find(
                      (p) => p.citation_id === c.citation_id
                    ) || null;
                  return (
                    <div
                      key={c.citation_id}
                      className="rounded-xl border border-slate-200 bg-white p-4"
                    >
                      <div className="mb-1 flex items-center justify-between">
                        <span className="text-sm font-semibold text-slate-700">
                          [{c.citation_id}] {c.document_name} — Page {c.page_number}
                        </span>
                        {pc && (
                          <span className="text-xs text-slate-400">
                            {pc.status.replace(/_/g, " ").toLowerCase()} ·{" "}
                            {(pc.coverage * 100).toFixed(0)}% overlap
                          </span>
                        )}
                      </div>
                      <p className="text-sm italic text-slate-600">"{c.supporting_text}"</p>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          <RetrievalDetails details={result.retrieval_details} />
        </div>
      )}
    </div>
  );
}
