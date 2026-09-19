import { useState } from "react";
import DatasetEdaModal from "../components/DatasetEdaModal";
import Modal from "../components/Modal";
import {
  currentWorkspace,
  datasetsForProject,
  datasetsIn,
  fmtBytes,
  fmtDate,
  projectsIn,
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

  return (
    <>
      {state.tab === "projects" && <ProjectsBody ws={ws} />}
      {state.tab === "datasets" && <DatasetsBody ws={ws} />}
    </>
  );
}
