import { useEffect, useState } from "react";
import { getEvaluation } from "../api";

function StatCard({ label, value }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-center">
      <div className="text-2xl font-semibold text-slate-800">{value}</div>
      <div className="mt-1 text-xs uppercase tracking-wide text-slate-400">{label}</div>
    </div>
  );
}

function pct(n) {
  return `${(Number(n) * 100).toFixed(1)}%`;
}

export default function Evaluation() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getEvaluation()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="text-sm text-slate-500">Loading evaluation…</p>;
  if (error)
    return (
      <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
        {error}
      </div>
    );

  const methods = data?.methods ? Object.values(data.methods) : [];

  return (
    <div className="mx-auto max-w-4xl">
      <h2 className="text-xl font-semibold text-slate-800">Evaluation</h2>
      <p className="mt-1 text-sm text-slate-500">
        Real dataset statistics and retrieval metrics measured on the provided QA pairs.
      </p>

      <section className="mt-6">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
          Dataset Statistics
        </h3>
        <div className="grid grid-cols-3 gap-4">
          <StatCard label="Financial Reports" value={data.num_documents} />
          <StatCard label="Chunks" value={data.num_chunks} />
          <StatCard label="Evaluation Questions" value={data.num_eval_questions} />
        </div>
      </section>

      <section className="mt-8">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
          Retrieval Comparison {data.k ? `(@K=${data.k})` : ""}
        </h3>

        {!data.available ? (
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-700">
            {data.message ||
              "No evaluation results yet. Run `python scripts/evaluate.py` to generate them."}
          </div>
        ) : (
          <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-slate-100 text-slate-600">
                <tr>
                  <th className="px-4 py-3 font-medium">Method</th>
                  <th className="px-4 py-3 font-medium">Recall@K</th>
                  <th className="px-4 py-3 font-medium">Precision@K</th>
                  <th className="px-4 py-3 font-medium">MRR</th>
                  <th className="px-4 py-3 font-medium">Hit Rate</th>
                </tr>
              </thead>
              <tbody>
                {methods.map((m) => (
                  <tr key={m.label} className="border-t border-slate-100">
                    <td className="px-4 py-3 font-medium text-slate-700">{m.label}</td>
                    <td className="px-4 py-3 text-slate-600">{pct(m.recall_at_k)}</td>
                    <td className="px-4 py-3 text-slate-600">{pct(m.precision_at_k)}</td>
                    <td className="px-4 py-3 text-slate-600">{m.mrr.toFixed(3)}</td>
                    <td className="px-4 py-3 text-slate-600">{pct(m.hit_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <p className="mt-3 text-xs text-slate-400">
          A retrieved chunk counts as a hit when it strongly overlaps the ground-truth
          evidence passage for the question. Metrics come from the real evaluation
          pipeline — no numbers are hardcoded.
        </p>
      </section>
    </div>
  );
}
