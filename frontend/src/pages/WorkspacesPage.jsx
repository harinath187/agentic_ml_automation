import { useState } from "react";
import Modal from "../components/Modal";
import { datasetsIn, fmtDate, projectsIn, useStore } from "../store";

export default function WorkspacesPage() {
  const { state, actions } = useStore();
  const [showNew, setShowNew] = useState(false);
  const [name, setName] = useState("");

  async function submit() {
    const trimmed = name.trim();
    if (!trimmed) return;
    const ws = await actions.createWorkspace(trimmed);
    setShowNew(false);
    setName("");
    actions.openWorkspace(ws.id);
  }

  async function handleDelete(e, ws) {
    e.stopPropagation();
    if (!window.confirm(`Delete workspace "${ws.name}"? This also deletes its projects, datasets, and runs.`)) {
      return;
    }
    try {
      await actions.deleteWorkspace(ws.id);
    } catch (err) {
      window.alert(err.message);
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Workspaces</h1>
          <p>A workspace groups projects, datasets, pipelines, and runs for a team or environment.</p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowNew(true)}>
          + New workspace
        </button>
      </div>

      {state.error && <p className="error">{state.error}</p>}

      {!state.workspaces.length && state.loading ? (
        <p className="muted">Loading workspaces…</p>
      ) : state.workspaces.length ? (
        <div className="grid-cards">
          {state.workspaces.map((w) => {
            const pCount = projectsIn(state, w.id).length;
            const dCount = datasetsIn(state, w.id).length;
            return (
              <div key={w.id} className="card card-clickable" onClick={() => actions.openWorkspace(w.id)}>
                <button
                  className="card-delete"
                  title="Delete workspace"
                  aria-label="Delete workspace"
                  onClick={(e) => handleDelete(e, w)}
                >
                  🗑
                </button>
                <div className="card-top">
                  <div>
                    <div className="card-title">{w.name}</div>
                    <div className="card-sub">
                      {pCount} project{pCount === 1 ? "" : "s"} · {dCount} dataset{dCount === 1 ? "" : "s"}
                    </div>
                  </div>
                </div>
                <div className="card-meta">Created {fmtDate(w.createdAt)}</div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="empty">
          No workspaces yet.
          <br />
          <button className="btn btn-primary" onClick={() => setShowNew(true)}>
            + New workspace
          </button>
        </div>
      )}

      {showNew && (
        <Modal
          title="New workspace"
          description="A workspace groups projects, datasets, and runs for a team or environment."
          onClose={() => setShowNew(false)}
        >
          <div className="field">
            <label htmlFor="nwName">Workspace name</label>
            <input
              id="nwName"
              type="text"
              autoFocus
              placeholder="e.g. Growth analytics"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
            />
          </div>
          <div className="modal-actions">
            <button className="btn btn-outline" onClick={() => setShowNew(false)}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={submit}>
              Create workspace
            </button>
          </div>
        </Modal>
      )}
    </>
  );
}
