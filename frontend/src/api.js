// Thin API client for the FastAPI backend.
// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.js).
const BASE = import.meta.env.VITE_API_URL || "/api";
async function request(path, options = {}) {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    let detail = `Request failed (${resp.status})`;
    try {
      const body = await resp.json();
      detail = body.detail || detail;
    } catch (_) {
      /* ignore parse errors */
    }
    throw new Error(detail);
  }
  return resp.json();
}

export function getHealth() {
  return request("/health");
}

export function ask(question) {
  return request("/ask", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export function getEvaluation() {
  return request("/evaluation");
}
