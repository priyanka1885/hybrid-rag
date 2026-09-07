import { useEffect, useState } from "react";
import Ask from "./pages/Ask";
import Evaluation from "./pages/Evaluation";
import AboutModal from "./components/AboutModal";
import { getHealth } from "./api";

function NavButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
        active ? "bg-brand-500 text-white" : "text-slate-600 hover:bg-slate-100"
      }`}
    >
      {children}
    </button>
  );
}

export default function App() {
  const [page, setPage] = useState("ask");
  const [aboutOpen, setAboutOpen] = useState(false);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: "down" }));
  }, []);

  const llmOk = health?.llm?.reachable && health?.llm?.model_available;

  return (
    <div className="min-h-full">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <h1 className="text-base font-semibold text-slate-800">
              Hybrid RAG for Financial Reports
            </h1>
            {health && (
              <span
                className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[0.7rem] font-medium ${
                  llmOk
                    ? "bg-emerald-50 text-emerald-600"
                    : "bg-amber-50 text-amber-600"
                }`}
                title={
                  llmOk
                    ? `LLM ready: ${health.llm.model}`
                    : "Local LLM not detected — start Ollama and pull the model"
                }
              >
                <span className="h-1.5 w-1.5 rounded-full bg-current" />
                {llmOk ? "LLM ready" : "LLM offline"}
              </span>
            )}
          </div>
          <nav className="flex items-center gap-1">
            <NavButton active={page === "ask"} onClick={() => setPage("ask")}>
              Ask
            </NavButton>
            <NavButton active={page === "evaluation"} onClick={() => setPage("evaluation")}>
              Evaluation
            </NavButton>
            <NavButton active={false} onClick={() => setAboutOpen(true)}>
              About
            </NavButton>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8">
        {page === "ask" ? <Ask /> : <Evaluation />}
      </main>

      <AboutModal open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  );
}
