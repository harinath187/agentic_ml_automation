const BASE = "/api";

async function handle(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function uploadDataset(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/datasets`, { method: "POST", body: form });
  return handle(res);
}

export async function startRun(payload) {
  const res = await fetch(`${BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handle(res);
}

export async function getRun(runId) {
  const res = await fetch(`${BASE}/runs/${runId}`);
  return handle(res);
}

export function reportUrl(runId) {
  return `${BASE}/runs/${runId}/report`;
}
