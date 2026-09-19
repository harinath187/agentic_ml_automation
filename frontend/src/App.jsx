import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import ProjectPage from "./pages/ProjectPage";
import WorkspacePage from "./pages/WorkspacePage";
import WorkspacesPage from "./pages/WorkspacesPage";
import { useStore } from "./store";

export default function App() {
  const { state } = useStore();

  return (
    <div className="app">
      <TopBar />
      <div className="shell">
        <Sidebar />
        <div className="main">
          {state.view === "workspaces" && <WorkspacesPage />}
          {state.view === "workspace" && <WorkspacePage />}
          {state.view === "project" && <ProjectPage />}
        </div>
      </div>
    </div>
  );
}
