import { useState } from "react";
import Modal from "./components/Modal";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import ProjectPage from "./pages/ProjectPage";
import WorkspacePage from "./pages/WorkspacePage";
import WorkspacesPage from "./pages/WorkspacesPage";
import { useStore } from "./store";

function NewWorkspaceModal({ onClose }) {
  const { actions } = useStore();
  const [name, setName] = useState("");

  async function submit() {
    const trimmed = name.trim();
    if (!trimmed) return;
    const ws = await actions.createWorkspace(trimmed);
    onClose();
    actions.openWorkspace(ws.id);
  }

  return (
    <Modal
      title="New workspace"
      description="A workspace groups projects, datasets, and runs for a team or environment."
      onClose={onClose}
    >
      <div className="field">
        <label htmlFor="topbarNwName">Workspace name</label>
        <input
          id="topbarNwName"
          type="text"
          autoFocus
          placeholder="e.g. Growth analytics"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
      </div>
      <div className="modal-actions">
        <button className="btn btn-outline" onClick={onClose}>
          Cancel
        </button>
        <button className="btn btn-primary" onClick={submit}>
          Create workspace
        </button>
      </div>
    </Modal>
  );
}

export default function App() {
  const { state } = useStore();
  const [showNewWorkspace, setShowNewWorkspace] = useState(false);

  return (
    <div className="app">
      <TopBar onNewWorkspace={() => setShowNewWorkspace(true)} />
      <div className="shell">
        <Sidebar />
        <div className="main">
          {state.view === "workspaces" && <WorkspacesPage />}
          {state.view === "workspace" && <WorkspacePage />}
          {state.view === "project" && <ProjectPage />}
        </div>
      </div>
      {showNewWorkspace && <NewWorkspaceModal onClose={() => setShowNewWorkspace(false)} />}
    </div>
  );
}
