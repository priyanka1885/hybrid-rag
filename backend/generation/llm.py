"""
Local LLM client (Ollama) for grounded answer generation.

    Question + Retrieved Evidence -> Llama 3.1 8B Instruct -> Grounded Answer

The model is instructed to answer ONLY from the supplied evidence and to cite
sources with [n] markers. The model name and base URL are configurable via
.env (LLM_MODEL, LLM_BASE_URL) - never hardcoded elsewhere.

If Ollama is not running or the model is missing, we raise LLMUnavailableError
so callers can show a friendly message instead of crashing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from backend.config import settings


class LLMUnavailableError(RuntimeError):
    """Raised when the local LLM cannot be reached or the model is missing."""


SYSTEM_PROMPT = """You are a careful financial-report analyst. You answer questions ONLY using the \
provided evidence extracted from financial/sustainability reports.

Strict rules:
- Use ONLY the supplied evidence. Do NOT use outside or prior knowledge.
- Do NOT invent facts, figures, dates, or company details.
- Preserve financial figures exactly as they appear in the evidence.
- Do NOT guess missing information.
- Cite every factual statement with a bracketed marker like [1] or [2] that \
refers to the numbered evidence used. Every factual claim must cite the \
evidence that directly supports THAT claim.

Reading tables, figures, and raw tabular rows:
- Pay equal attention to structured tabular data (TABLE chunks). Values in \
table rows represent factual evidence even if they are terse or lack full \
narrative sentences.
- Evidence may be a labelled TABLE or FIGURE, OR a raw tabular row embedded in \
plain-text evidence, e.g. a header like "(kilotons) 2023 2022" followed by a \
row "Direct scope 1 58 644 57 284". Treat such raw rows as valid ground truth, \
not as prose to be ignored.
- When a row of numbers follows year column headers that appear in the SAME \
evidence item, align the values to those columns left-to-right: the first \
value maps to the first year, the second value to the second year, and so on \
(so "Direct scope 1 58 644 57 284" under headers "2023 2022" means Scope 1 = \
58 644 in 2023 and 57 284 in 2022).
- When numbers are listed next to "Scope 1", "Scope 2", or another emission \
metric, extract and report them directly WITH their unit exactly as shown in \
the evidence (e.g. kilotons or tCO2e).
- Preserve the exact row/column relationship. A value belongs to the year in \
its own column and the metric in its own row. Never combine a value from one \
year with a different year just because both appear in the evidence.
- If the question asks about a specific year, report the value under that exact \
year column. For multiple years, keep each year-value pair exactly as shown.
- Do NOT claim information is missing simply because it is presented as a table \
row or list rather than a full prose sentence.
- Only decline the year-value mapping when the columns and values genuinely \
cannot be aligned from the evidence (headers absent, or the count of values \
does not match the count of year columns) - never merely because the format is \
tabular.

Using the evidence:
- When answering a numerical question, locate the metric, the requested year, \
and the value together in the same evidence item.
- If interpreting a table needs more than one evidence item (e.g. a header row \
and a value row), combine them.
- Never invent a missing value. Never calculate a value unless the question \
explicitly asks for a calculation AND every operand is present in the evidence.
- Do not cite evidence that is irrelevant to a claim just because it was \
retrieved. Cite the exact evidence supporting each claim.
- Prefer, in order: (1) exact table/figure evidence, (2) exact numeric \
evidence, (3) exact metric/entity evidence, (4) narrative evidence - over \
generic, loosely-similar text.

- If the evidence does not contain enough information to answer, respond with \
exactly: "I couldn't find sufficient evidence in the provided financial reports \
to answer this reliably."
- Keep the answer concise and professional (1-4 sentences)."""


# Appended to the user prompt only on the single controlled retry, when the
# model refused despite strong retrieved evidence. It re-focuses the model on
# extracting the specific value without loosening grounding.
FOCUS_SUFFIX = (
    "\n\nThe evidence above was retrieved as highly relevant to this question. "
    "Read every evidence item carefully, including tables and figures, and "
    "locate the specific metric, year, and value the question asks for. If the "
    "value is genuinely present, answer it with its [n] citation. Only reply "
    "with the insufficient-evidence message if the value is truly absent from "
    "ALL evidence items above."
)


@dataclass
class LLMResponse:
    text: str
    model: str


_EVIDENCE_LABEL = {
    "table": "TABLE EVIDENCE (preserve the exact row/column relationships)",
    "figure": "FIGURE EVIDENCE (preserve the exact axis/row/column relationships)",
    "ocr_page": "OCR PAGE EVIDENCE",
    "text": "Evidence",
}

# A text chunk this dense in numeric tokens is almost always a raw tabular row
# (a metric label followed by its per-year values), not narrative prose. We
# label such chunks explicitly so the model treats them as authoritative
# ground-truth data rows rather than skimming past them for fluent prose.
_NUMERIC_TABULAR_DENSITY = 0.12
_HAS_DIGIT_RE = __import__("re").compile(r"\d")

_NUMERIC_TABULAR_LABEL = (
    "NUMERIC / TABULAR DATA EVIDENCE (this is a raw table row: a metric label "
    "followed by its per-year values aligned to the year column headers shown "
    "in this same item; treat it as authoritative ground truth and read the "
    "value under the requested year)"
)


def _looks_tabular(text: str) -> bool:
    """True when a text chunk is number-dense enough to be a raw table row."""
    toks = text.split()
    if len(toks) < 4:
        return False
    numeric = sum(1 for t in toks if _HAS_DIGIT_RE.search(t))
    return (numeric / len(toks)) >= _NUMERIC_TABULAR_DENSITY


def _evidence_label(content_type: str, text: str) -> str:
    """Pick the evidence label, promoting number-dense text to tabular data."""
    if content_type == "text" and _looks_tabular(text):
        return _NUMERIC_TABULAR_LABEL
    return _EVIDENCE_LABEL.get(content_type, _EVIDENCE_LABEL["text"])


def build_context_block(evidence: list[dict]) -> str:
    """Render the reranked evidence into a numbered context block.

    Only the evidence CONTENT (plus the source document name for attribution and
    an explicit type label) is shown to the model. Internal retrieval metadata -
    chunk ids, page numbers, ranks, scores - is deliberately NOT included: it is
    bookkeeping, not answer content, and when the model echoes it the digits
    inside a chunk-id/page leak into verification as bogus "figures". The
    citation layer maps each [n] back to its document/page separately, so the
    model never needs to see or repeat that metadata.

    Structured evidence (tables/figures) is explicitly labelled and its Markdown
    layout is kept intact - never flattened - so the model can preserve
    year->value and metric->value relationships.
    """
    blocks = []
    for i, ev in enumerate(evidence, start=1):
        ctype = ev.get("content_type", "text")
        label = _evidence_label(ctype, ev.get("text", ""))
        blocks.append(
            f"[{i}] Source: {ev['document_name']}\n"
            f"{label}:\n{ev['text']}"
        )
    return "\n\n".join(blocks)


# Internal-metadata field markers that must never legitimately appear in a
# grounded answer. If the model echoes a chunk-id / content-type header, or a
# "Page: <n>" locator, the response is bookkeeping rather than an answer and
# should be regenerated (see rag_engine) rather than verified.
_METADATA_ECHO_RE = __import__("re").compile(
    r"(?i)(?:\b(?:chunk\s*id|content\s*type)\b\s*:|\bpage\s*:\s*\d)"
)


def looks_like_metadata_echo(text: str) -> bool:
    """True when an LLM response is echoing internal retrieval metadata.

    Conservative: it fires only on the concrete leaked-header patterns
    ("Chunk ID:", "Content type:", "Page: <n>"), so a genuine terse answer is
    never mistaken for metadata.
    """
    return bool(text) and bool(_METADATA_ECHO_RE.search(text))


def build_user_prompt(question: str, evidence: list[dict], focus: bool = False) -> str:
    context = build_context_block(evidence)
    prompt = (
        f"Evidence from the financial reports:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the evidence above, citing sources with [n] markers."
    )
    if focus:
        prompt += FOCUS_SUFFIX
    return prompt


class OllamaClient:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.LLM_MODEL
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.timeout = settings.LLM_TIMEOUT

    def health(self) -> dict:
        """Return a dict describing local LLM availability."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            names = {m.get("name", "") for m in tags}
            model_present = any(
                n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names
            )
            return {
                "reachable": True,
                "model_available": model_present,
                "model": self.model,
                "available_models": sorted(names),
            }
        except Exception:
            return {
                "reachable": False,
                "model_available": False,
                "model": self.model,
                "available_models": [],
            }

    def warmup(self) -> bool:
        """Preload the model into memory so the first real request is fast.

        Sends an empty-prompt generate request, which makes Ollama load the
        model without producing tokens. Best-effort: returns True on success,
        False otherwise (never raises).
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": "", "stream": False, "keep_alive": "30m"},
                timeout=self.timeout,
            )
            return resp.ok
        except requests.exceptions.RequestException:
            return False

    def generate(self, question: str, evidence: list[dict], focus: bool = False) -> LLMResponse:
        prompt = build_user_prompt(question, evidence, focus=focus)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "keep_alive": "30m",  # keep the model warm between requests
            "options": {"temperature": settings.LLM_TEMPERATURE},
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
        except requests.exceptions.RequestException as exc:
            raise LLMUnavailableError(
                "Local LLM is unavailable. Please make sure Ollama is running and "
                f"{self.model} is available."
            ) from exc

        if resp.status_code == 404:
            raise LLMUnavailableError(
                f"Model '{self.model}' not found in Ollama. Run: ollama pull {self.model}"
            )
        if not resp.ok:
            raise LLMUnavailableError(
                f"Local LLM returned an error ({resp.status_code}). "
                "Check that Ollama is running correctly."
            )
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMUnavailableError("Local LLM returned an invalid response.") from exc

        return LLMResponse(text=(data.get("response") or "").strip(), model=self.model)
