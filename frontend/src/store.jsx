import { createContext, useContext, useEffect, useState } from "react";
import {
  createProject as apiCreateProject,
  createWorkspace as apiCreateWorkspace,
  deleteProject as apiDeleteProject,
  deleteWorkspace as apiDeleteWorkspace,
  listProjectDatasets,
  listProjectRuns,
  listProjects,
  listWorkspaceDatasets,
  listWorkspaceRuns,
  listWorkspaces,
  updateProject as apiUpdateProject,
} from "./api";

export function fmtDate(ts) {
  return new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export function fmtDateTime(ts) {
  return new Date(ts).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

// Workspaces/projects/datasets/runs are all real, persisted server-side
// (api/db.py's workspaces/projects tables, plus workspace_id/project_id on
// datasets/runs) - this store just caches whatever's been fetched from the
// API so far, keyed by id, and normalizes snake_case API fields to the
// camelCase shape the UI components use.
function normalizeWorkspace(w) {
  return { id: w.workspace_id, name: w.name, createdAt: new Date(w.created_at).getTime() };
}
function normalizeProject(p) {
  return {
    id: p.project_id,
    workspaceId: p.workspace_id,
    name: p.name,
    description: p.description || "",
    createdAt: new Date(p.created_at).getTime(),
  };
}
function normalizeDataset(d) {
  return {
    id: d.dataset_id,
    workspaceId: d.workspace_id,
    projectId: d.project_id,
    name: d.original_filename,
    rows: d.row_count,
    cols: (d.columns || []).length,
    columns: d.columns || [],
    sizeBytes: d.size_bytes,
    createdAt: new Date(d.uploaded_at).getTime(),
  };
}
function normalizeRun(r) {
  return {
    id: r.run_id,
    workspaceId: r.workspace_id,
    projectId: r.project_id,
    datasetId: r.dataset_id,
    status: r.status,
    createdAt: new Date(r.created_at).getTime(),
  };
}

function mergeById(existing, incoming) {
  const byId = new Map(existing.map((item) => [item.id, item]));
  for (const item of incoming) byId.set(item.id, item);
  return Array.from(byId.values());
}

function initialState() {
  return {
    view: "workspace", // "workspaces" | "workspace" | "project"
    tab: "projects", // projects | datasets
    activeReportTab: "overview",
    currentWorkspaceId: null,
    currentProjectId: null,
    workspaces: [],
    projects: [],
    datasets: [],
    runs: [],
    loading: false,
    error: "",
  };
}

const StoreContext = createContext(null);

export function StoreProvider({ children }) {
  const [state, setState] = useState(initialState);

  function patch(fn) {
    setState((prev) => {
      const next = { ...prev };
      fn(next);
      return next;
    });
  }

  async function withLoading(fn) {
    patch((s) => {
      s.loading = true;
      s.error = "";
    });
    try {
      await fn();
    } catch (err) {
      patch((s) => {
        s.error = err.message;
      });
    } finally {
      patch((s) => {
        s.loading = false;
      });
    }
  }

  useEffect(() => {
    withLoading(async () => {
      const { workspaces } = await listWorkspaces();
      const normalizedWorkspaces = workspaces.map(normalizeWorkspace);
      patch((s) => {
        s.workspaces = normalizedWorkspaces;
        if (normalizedWorkspaces.length) {
          s.view = "workspace";
          s.currentWorkspaceId = normalizedWorkspaces[0].id;
          s.tab = "projects";
        } else {
          s.view = "workspaces";
        }
      });
      if (!normalizedWorkspaces.length) return;

      const workspaceId = normalizedWorkspaces[0].id;
      const [{ projects }, { datasets }, { runs }] = await Promise.all([
        listProjects(workspaceId),
        listWorkspaceDatasets(workspaceId),
        listWorkspaceRuns(workspaceId),
      ]);
      patch((s) => {
        s.projects = mergeById(s.projects, projects.map(normalizeProject));
        s.datasets = mergeById(s.datasets, datasets.map(normalizeDataset));
        s.runs = mergeById(s.runs, runs.map(normalizeRun));
      });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const actions = {
    goWorkspaces() {
      patch((s) => {
        s.view = "workspaces";
        s.currentWorkspaceId = null;
        s.currentProjectId = null;
      });
    },
    openWorkspace(wsId) {
      patch((s) => {
        s.view = "workspace";
        s.currentWorkspaceId = wsId;
        s.tab = "projects";
        s.currentProjectId = null;
      });
      withLoading(async () => {
        const [{ projects }, { datasets }, { runs }] = await Promise.all([
          listProjects(wsId),
          listWorkspaceDatasets(wsId),
          listWorkspaceRuns(wsId),
        ]);
        patch((s) => {
          s.projects = mergeById(s.projects, projects.map(normalizeProject));
          s.datasets = mergeById(s.datasets, datasets.map(normalizeDataset));
          s.runs = mergeById(s.runs, runs.map(normalizeRun));
        });
      });
    },
    setTab(tab) {
      patch((s) => {
        s.tab = tab;
        s.view = "workspace";
        s.currentProjectId = null;
      });
    },
    openProject(projId) {
      patch((s) => {
        s.view = "project";
        s.currentProjectId = projId;
        s.activeReportTab = "overview";
      });
      withLoading(async () => {
        const [{ datasets }, { runs }] = await Promise.all([listProjectDatasets(projId), listProjectRuns(projId)]);
        patch((s) => {
          s.datasets = mergeById(s.datasets, datasets.map(normalizeDataset));
          s.runs = mergeById(s.runs, runs.map(normalizeRun));
        });
      });
    },
    backToProjects() {
      patch((s) => {
        s.view = "workspace";
        s.tab = "projects";
        s.currentProjectId = null;
      });
    },
    setReportTab(tab) {
      patch((s) => {
        s.activeReportTab = tab;
      });
    },
    async createWorkspace(name) {
      const ws = normalizeWorkspace(await apiCreateWorkspace(name));
      patch((s) => {
        s.workspaces = [ws, ...s.workspaces];
      });
      return ws;
    },
    async createProject(workspaceId, name, description = "") {
      const proj = normalizeProject(await apiCreateProject(workspaceId, name, description));
      patch((s) => {
        s.projects = [proj, ...s.projects];
      });
      return proj;
    },
    async deleteWorkspace(workspaceId) {
      await apiDeleteWorkspace(workspaceId);
      patch((s) => {
        s.workspaces = s.workspaces.filter((w) => w.id !== workspaceId);
        s.projects = s.projects.filter((p) => p.workspaceId !== workspaceId);
        s.datasets = s.datasets.filter((d) => d.workspaceId !== workspaceId);
        s.runs = s.runs.filter((r) => r.workspaceId !== workspaceId);
        if (s.currentWorkspaceId === workspaceId) {
          s.view = "workspaces";
          s.currentWorkspaceId = null;
          s.currentProjectId = null;
        }
      });
    },
    async deleteProject(projectId) {
      const proj = state.projects.find((p) => p.id === projectId);
      await apiDeleteProject(projectId);
      patch((s) => {
        s.projects = s.projects.filter((p) => p.id !== projectId);
        s.datasets = s.datasets.filter((d) => d.projectId !== projectId);
        s.runs = s.runs.filter((r) => r.projectId !== projectId);
        if (s.currentProjectId === projectId) {
          s.view = "workspace";
          s.tab = "projects";
          s.currentProjectId = null;
          s.currentWorkspaceId = s.currentWorkspaceId || proj?.workspaceId || null;
        }
      });
    },
    setProjectDescription(projectId, description) {
      // Optimistic local update; persisted server-side in the background so
      // every keystroke doesn't block on a round trip.
      patch((s) => {
        const proj = s.projects.find((p) => p.id === projectId);
        if (proj) proj.description = description;
      });
      apiUpdateProject(projectId, { description }).catch((err) => {
        patch((s) => {
          s.error = err.message;
        });
      });
    },
    addDataset(dataset) {
      patch((s) => {
        s.datasets = [dataset, ...s.datasets];
      });
    },
    addRun(run) {
      patch((s) => {
        s.runs = [run, ...s.runs];
      });
    },
    updateRun(runId, fields) {
      patch((s) => {
        const run = s.runs.find((r) => r.id === runId);
        if (run) Object.assign(run, fields);
      });
    },
  };

  return <StoreContext.Provider value={{ state, actions }}>{children}</StoreContext.Provider>;
}

export function useStore() {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStore must be used within a StoreProvider");
  return ctx;
}

export function projectsIn(state, wsId) {
  return state.projects.filter((p) => p.workspaceId === wsId);
}
export function datasetsIn(state, wsId) {
  return state.datasets.filter((d) => d.workspaceId === wsId);
}
export function datasetsForProject(state, projId) {
  return state.datasets.filter((d) => d.projectId === projId);
}
export function runsIn(state, wsId) {
  return state.runs.filter((r) => r.workspaceId === wsId).sort((a, b) => b.createdAt - a.createdAt);
}
export function runsForProject(state, projId) {
  return state.runs.filter((r) => r.projectId === projId).sort((a, b) => b.createdAt - a.createdAt);
}
export function currentWorkspace(state) {
  return state.workspaces.find((w) => w.id === state.currentWorkspaceId);
}
export function currentProject(state) {
  return state.projects.find((p) => p.id === state.currentProjectId);
}
