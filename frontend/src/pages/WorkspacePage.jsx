import { useState } from "react";
import Breadcrumbs from "../components/Breadcrumbs";
import DatasetEdaModal from "../components/DatasetEdaModal";
import Modal from "../components/Modal";
import {
  currentWorkspace,
  datasetsForProject,
  datasetsIn,
  fmtBytes,
  fmtDate,
  fmtDateTime,
  projectsIn,
  runsIn,
  useStore,
} from "../store";

function NewProjectModal({ ws, onClose }) {
  const { actions } = useStore();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  async function submit() {
    const trimmed = name.trim();
    if (!trimmed) return;
    const proj = await actions.createProject(ws.id, trimmed, description.trim());
    onClose();
    actions.openProject(proj.id);
  }

  return (
    <Modal
      title="New project"
      description={
        <>
          Projects group a dataset, business problem, and pipeline runs in <strong>{ws.name}</strong>.
        </>
      }
      onClose={onClose}
    >
      <div className="field">
        <label htmlFor="npName">Project name</label>
        <input
          id="npName"
          type="text"
          autoFocus
          placeholder="e.g. churn-prediction"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
      </div>
      <div className="field">
        <label htmlFor="npDescription">Business problem (optional)</label>
        <textarea
          id="npDescription"
          rows={3}
          placeholder="e.g. Predict which customers will churn next month so retention offers can be targeted."
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </div>
      <div className="modal-actions">
        <button className="btn btn-outline" onClick={onClose}>
          Cancel
        </button>
        <button className="btn btn-primary" onClick={submit}>
          Create project
        </button>
      </div>
    </Modal>
  );
}

function RunStatusPill({ status }) {
  const map = {
    completed: { cls: "pill-success", label: "Completed" },
    succeeded: { cls: "pill-success", label: "Completed" },
    failed: { cls: "pill-running", label: "Failed" },
    cancelled: { cls: "pill-running", label: "Cancelled" },
    needs_clarification: { cls: "pill-running", label: "Needs input" },
    running: { cls: "pill-running", label: "Running…" },
    queued: { cls: "pill-running", label: "Queued…" },
  };
  const info = map[status] || { cls: "pill-running", label: status };
  return <span className={`pill ${info.cls}`}>{info.label}</span>;
}

function DashboardBody({ ws }) {
  const { state } = useStore();
  const projCount = projectsIn(state, ws.id).length;
  const runs = runsIn(state, ws.id);
  const expCount = projectsIn(state, ws.id).filter((p) => datasetsForProject(state, p.id).length > 0).length;
  const weekAgo = Date.now() - 1000 * 60 * 60 * 24 * 7;
  const runs7d = runs.filter((r) => r.createdAt >= weekAgo).length;

  return (
    <>
      <div className="tiles">
        <div className="tile">
          <div className="tile-label">Projects</div>
          <div className="tile-value">{projCount}</div>
        </div>
        <div className="tile">
          <div className="tile-label">Experiments</div>
          <div className="tile-value">{expCount}</div>
        </div>
        <div className="tile">
          <div className="tile-label">Runs (7d)</div>
          <div className="tile-value">{runs7d}</div>
        </div>
      </div>
      <div className="section">
        <div className="section-head">
          <h2>Recent runs</h2>
          <span className="hint">Most recent pipeline executions across this workspace</span>
        </div>
        {runs.length ? (
          <div className="list">
            {runs.slice(0, 8).map((r) => {
              const proj = state.projects.find((p) => p.id === r.projectId);
              return (
                <div className="list-row" key={r.id} style={{ cursor: "default" }}>
                  <div className="list-main">
                    <div className="list-title">{proj ? proj.name : "Unknown project"}</div>
                    <div className="list-sub">
                      Run · {fmtDateTime(r.createdAt)}
                                          Run · {fmtDateTime(r.createdAt)}
                    </div>
                  </div>
                  <div className="list-side">
                    <RunStatusPill status={r.status} />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="empty">No runs yet. Open a project and run the pipeline on an uploaded dataset.</div>
        )}
      </div>
    </>
  );
}

function ProjectsBody({ ws }) {
  const { state, actions } = useStore();
  const [showNew, setShowNew] = useState(false);
  const projs = projectsIn(state, ws.id);
  const dsCount = datasetsIn(state, ws.id).length;

  async function handleDeleteProject(e, proj) {
    e.stopPropagation();
    if (!window.confirm(`Delete project "${proj.name}"? This also deletes its datasets and runs.`)) {
      return;
    }
    try {
      await actions.deleteProject(proj.id);
    } catch (err) {
      window.alert(err.message);
    }
  }

  return (
    <>
      <div className="tiles">
        <div className="tile">
          <div className="tile-label">Projects</div>
          <div className="tile-value">{projs.length}</div>
        </div>
        <div className="tile">
          <div className="tile-label">Data sources</div>
          <div className="tile-value">{dsCount}</div>
        </div>
      </div>
      <div className="section">
        <div className="section-head">
          <h2>Projects</h2>
          <button className="btn btn-primary btn-sm" onClick={() => setShowNew(true)}>
            + New project
          </button>
        </div>
        {projs.length ? (
          <div className="grid-cards">
            {projs.map((p) => {
              const pDs = datasetsForProject(state, p.id);
              return (
                <div key={p.id} className="card card-clickable" onClick={() => actions.openProject(p.id)}>
                  <button
                    className="card-delete"
                    title="Delete project"
                    aria-label="Delete project"
                    onClick={(e) => handleDeleteProject(e, p)}
                  >
                    🗑
                  </button>
                  <div className="card-top">
                    <div>
                      <div className="card-title">{p.name}</div>
                      <div className="card-sub">
                        {pDs.length} dataset{pDs.length === 1 ? "" : "s"}
                      </div>
                    </div>
                  </div>
                  <div className="card-meta">Created {fmtDate(p.createdAt)}</div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="empty">
            No projects yet in this workspace.
            <br />
            <button className="btn btn-primary" onClick={() => setShowNew(true)}>
              + New project
            </button>
          </div>
        )}
      </div>
      {showNew && <NewProjectModal ws={ws} onClose={() => setShowNew(false)} />}
    </>
  );
}

function DatasetsBody({ ws }) {
  const { state } = useStore();
  const [previewDs, setPreviewDs] = useState(null);
  const ds = datasetsIn(state, ws.id).slice().sort((a, b) => b.createdAt - a.createdAt);

  return (
    <div className="section">
      <div className="section-head">
        <h2>Datasets</h2>
        <span className="hint">All uploads across projects in this workspace</span>
      </div>
      {ds.length ? (
        <div className="list">
          {ds.map((d) => {
            const proj = state.projects.find((p) => p.id === d.projectId);
            return (
              <button key={d.id} className="list-row" onClick={() => setPreviewDs(d)}>
                <div className="list-main">
                  <div className="list-title">{d.name}</div>
                  <div className="list-sub">
                    {proj ? proj.name : "—"} · {fmtDate(d.createdAt)}
                  </div>
                </div>
                <div className="list-side">
                  <div className="list-stat">
                    <div className="k">rows</div>
                    <div className="v">{d.rows.toLocaleString()}</div>
                  </div>
                  <div className="list-stat">
                    <div className="k">cols</div>
                    <div className="v">{d.cols}</div>
                  </div>
                  <div className="list-stat">
                    <div className="k">size</div>
                    <div className="v">{fmtBytes(d.sizeBytes)}</div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="empty">No datasets uploaded yet. Upload a CSV or Excel file from inside a project.</div>
      )}
      {previewDs && <DatasetEdaModal dataset={previewDs} onClose={() => setPreviewDs(null)} />}
    </div>
  );
}

export default function WorkspacePage() {
  const { state, actions } = useStore();
  const ws = currentWorkspace(state);

  if (!ws) {
    actions.goWorkspaces();
    return null;
  }

  const crumbs = [
    { label: "Workspaces", action: actions.goWorkspaces },
    { label: ws.name },
  ];

  return (
    <>
      <Breadcrumbs parts={crumbs} />
      {state.tab === "dashboard" && <DashboardBody ws={ws} />}
      {state.tab === "projects" && <ProjectsBody ws={ws} />}
      {state.tab === "datasets" && <DatasetsBody ws={ws} />}
    </>
  );
}
