const BASE = "/api";

async function handle(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

function postJson(path, payload) {
  return fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(handle);
}

export async function uploadDataset(file, projectId) {
  const form = new FormData();
  form.append("file", file);
  if (projectId) form.append("project_id", projectId);
  const res = await fetch(`${BASE}/datasets`, { method: "POST", body: form });
  return handle(res);
}

export function createWorkspace(name) {
  return postJson("/workspaces", { name });
}

export function listWorkspaces() {
  return fetch(`${BASE}/workspaces`).then(handle);
}

export function deleteWorkspace(workspaceId) {
  return fetch(`${BASE}/workspaces/${workspaceId}`, { method: "DELETE" }).then(handle);
}

export function createProject(workspaceId, name, description = "") {
  return postJson(`/workspaces/${workspaceId}/projects`, { name, description });
}

export function deleteProject(projectId) {
  return fetch(`${BASE}/projects/${projectId}`, { method: "DELETE" }).then(handle);
}

export function listProjects(workspaceId) {
  return fetch(`${BASE}/workspaces/${workspaceId}/projects`).then(handle);
}

export function listWorkspaceDatasets(workspaceId) {
  return fetch(`${BASE}/workspaces/${workspaceId}/datasets`).then(handle);
}

export function listWorkspaceRuns(workspaceId) {
  return fetch(`${BASE}/workspaces/${workspaceId}/runs`).then(handle);
}

export function getDatasetEda(datasetId) {
  return fetch(`${BASE}/datasets/${datasetId}/eda`).then(handle);
}

export function updateProject(projectId, fields) {
  return fetch(`${BASE}/projects/${projectId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fields),
  }).then(handle);
}

export function listProjectDatasets(projectId) {
  return fetch(`${BASE}/projects/${projectId}/datasets`).then(handle);
}

export function listProjectRuns(projectId) {
  return fetch(`${BASE}/projects/${projectId}/runs`).then(handle);
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

export async function cancelRun(runId) {
  const res = await fetch(`${BASE}/runs/${runId}/cancel`, { method: "POST" });
  return handle(res);
}

export function reportUrl(runId) {
  return `${BASE}/runs/${runId}/report`;
}
