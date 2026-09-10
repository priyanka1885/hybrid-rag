"""
LLM client (OpenRouter) for grounded answer generation.

    Question + Retrieved Evidence -> Llama 3.1 8B Instruct -> Grounded Answer

Generation is served by OpenRouter's hosted, OpenAI-compatible chat-completions
API using the free Llama 3.1 8B Instruct model. The model is instructed to
answer ONLY from the supplied evidence and to cite sources with [n] markers.

The model name, base URL, and API key are configurable via .env
(LLM_MODEL, LLM_BASE_URL, OPENROUTER_API_KEY) - never hardcoded elsewhere and
the key is never committed.

The public class name (OllamaClient) and its interface (health/warmup/generate)
are kept unchanged so the rest of the pipeline does not need to be touched.

If the API key is missing, OpenRouter is unreachable, or the request times out,
we raise LLMUnavailableError so callers can show a friendly message instead of
crashing.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import requests

from backend.config import settings

# One pooled HTTP session for the whole process. requests.get/post at module
# level build a fresh connection pool, TLS context and adapter per call; reusing
# one session keeps that allocation out of the per-request path (and /health is
# polled continuously by the platform health check).
_HTTP_LOCK = threading.Lock()
_http_session: requests.Session | None = None


def _session() -> requests.Session:
    global _http_session
    with _HTTP_LOCK:
        if _http_session is None:
            sess = requests.Session()
            # Small pool: this process only ever talks to one host, and each
            # pooled connection holds its own buffers.
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=1, pool_maxsize=4, max_retries=0
            )
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            _http_session = sess
        return _http_session


# Health results are cached for this many seconds. The platform health check and
# the frontend both poll /health, and the previous implementation fetched and
# JSON-parsed OpenRouter's entire model catalogue on every single call. That
# repeatedly allocated and freed a large object graph, which fragments the heap
# and drives resident memory up over time on a small instance.
_HEALTH_TTL_SECONDS = 300.0
# Keyed by model name: two clients configured for different models must not read
# each other's status out of the cache.
_health_cache: dict[str, tuple[float, dict]] = {}
_HEALTH_CACHE_LOCK = threading.Lock()


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM API cannot be reached, is misconfigured, or errors."""


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
    """Client for the OpenRouter chat-completions API.

    The historical name is retained so nothing downstream needs to change; it
    now talks to OpenRouter's OpenAI-compatible endpoint rather than a local
    Ollama server. The model, base URL and API key all come from settings
    (.env) and can be overridden per-instance for local development/tests.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or settings.LLM_MODEL
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        # Never hardcoded: falls back to the env-sourced setting only.
        self.api_key = api_key if api_key is not None else settings.OPENROUTER_API_KEY
        self.timeout = settings.LLM_TIMEOUT

    # -- HTTP helpers --------------------------------------------------------
    @property
    def _chat_url(self) -> str:
        """OpenAI-compatible chat-completions endpoint on OpenRouter."""
        return f"{self.base_url}/v1/chat/completions"

    @property
    def _models_url(self) -> str:
        return f"{self.base_url}/v1/models"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health(self, use_cache: bool = True) -> dict:
        """Return a dict describing LLM (OpenRouter) availability.

        Does NOT depend on any local Ollama endpoint. Availability is gated on
        the API key being configured; when a key is present we additionally do a
        light-weight, short-timeout call to the models endpoint to confirm the
        API is reachable and that the configured model is offered.

        The result is cached for :data:`_HEALTH_TTL_SECONDS`. ``/health`` is
        polled continuously by the hosting platform, and the model catalogue this
        checks against is a large JSON document that changes rarely - parsing it
        on every poll was pure allocation churn. ``available_models`` is no
        longer returned in full for the same reason (no caller used it);
        ``num_available_models`` reports the count instead, and the key is kept
        as an empty list so the response shape stays backwards compatible.
        """
        if not self.api_key:
            return {
                "reachable": False,
                "model_available": False,
                "model": self.model,
                "available_models": [],
                "num_available_models": 0,
                "note": "OPENROUTER_API_KEY is not set.",
            }

        if use_cache:
            cached = _health_cache.get(self.model)
            if cached is not None and (time.monotonic() - cached[0]) < _HEALTH_TTL_SECONDS:
                return dict(cached[1])

        result = self._probe_models()
        with _HEALTH_CACHE_LOCK:
            _health_cache[self.model] = (time.monotonic(), result)
        return dict(result)

    def _probe_models(self) -> dict:
        """One live call to the models endpoint. Never raises."""
        try:
            resp = _session().get(self._models_url, headers=self._headers(), timeout=5)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            # Only the ids are needed. Extract them, then drop the decoded
            # catalogue immediately instead of holding the whole object graph.
            wanted = self.model.split(":")[0]
            count = 0
            model_present = False
            for m in data:
                if not isinstance(m, dict):
                    continue
                name = m.get("id", "")
                if not name:
                    continue
                count += 1
                if name == self.model or name.split(":")[0] == wanted:
                    model_present = True
            del data, resp
            return {
                "reachable": True,
                # An empty catalogue means the endpoint told us nothing useful,
                # so do not treat that as "model missing" (unchanged behaviour).
                "model_available": model_present or count == 0,
                "model": self.model,
                "available_models": [],
                "num_available_models": count,
            }
        except Exception:
            # Key is present but the API could not be reached right now. Report
            # unreachable rather than raising, so /health never crashes.
            return {
                "reachable": False,
                "model_available": False,
                "model": self.model,
                "available_models": [],
                "num_available_models": 0,
            }

    def warmup(self) -> bool:
        """No-op warmup for the hosted API.

        OpenRouter is a hosted service, so there is no local model to preload
        into memory and no cold start to hide. Kept for interface compatibility;
        best-effort and never raises. Returns True when a key is configured.
        """
        return bool(self.api_key)

    def generate(self, question: str, evidence: list[dict], focus: bool = False) -> LLMResponse:
        if not self.api_key:
            raise LLMUnavailableError(
                "LLM is not configured. Set OPENROUTER_API_KEY in your environment "
                "to enable answer generation."
            )

        prompt = build_user_prompt(question, evidence, focus=focus)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.LLM_TEMPERATURE,
            "stream": False,
        }
        try:
            resp = _session().post(
                self._chat_url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise LLMUnavailableError(
                f"The LLM request timed out after {self.timeout}s. Please try again."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMUnavailableError(
                "The LLM API (OpenRouter) is unreachable. Please check your network "
                "connection and OPENROUTER_API_KEY."
            ) from exc

        if resp.status_code in (401, 403):
            raise LLMUnavailableError(
                "OpenRouter rejected the request (authentication failed). Check that "
                "OPENROUTER_API_KEY is valid."
            )
        if resp.status_code == 404:
            raise LLMUnavailableError(
                f"Model '{self.model}' is not available on OpenRouter. "
                "Check the LLM_MODEL setting."
            )
        if resp.status_code == 429:
            raise LLMUnavailableError(
                "Free OpenRouter model is temporarily rate-limited. Please try again later."
            )
        if not resp.ok:
            raise LLMUnavailableError(
                f"The LLM API returned an error ({resp.status_code}). Please try again."
            )
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMUnavailableError("The LLM API returned an invalid response.") from exc

        # OpenRouter can return a top-level error object even with a 200 status.
        if isinstance(data, dict) and data.get("error"):
            msg = data["error"].get("message") if isinstance(data["error"], dict) else str(data["error"])
            raise LLMUnavailableError(f"The LLM API returned an error: {msg}")

        text = ""
        choices = data.get("choices") if isinstance(data, dict) else None
        if choices:
            message = choices[0].get("message") or {}
            text = message.get("content") or ""

        return LLMResponse(text=text.strip(), model=self.model)
