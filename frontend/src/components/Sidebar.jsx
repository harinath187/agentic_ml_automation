import { useStore } from "../store";

const ITEMS = [
  { tab: "dashboard", label: "Dashboard" },
  { tab: "projects", label: "Projects" },
  { tab: "datasets", label: "Datasets" },
];

export default function Sidebar() {
  const { state, actions } = useStore();
  if (state.view === "workspaces") return null;
  const activeTab = state.view === "project" ? "projects" : state.tab;

  return (
    <div className="sidebar">
      <div>
        <div className="side-section-label">Workspace</div>
        <div className="side-nav">
          {ITEMS.map((item) => (
            <button
              key={item.tab}
              className={"nav-item" + (item.tab === activeTab ? " active" : "")}
              onClick={() => actions.setTab(item.tab)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
