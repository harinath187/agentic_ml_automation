import { useStore } from "../store";
import { REPORT_TABS } from "./ReportTabs";

const ITEMS = [
  { tab: "projects", label: "Projects" },
  { tab: "datasets", label: "Datasets" },
];

export default function Sidebar() {
  const { state, actions } = useStore();
  if (state.view === "workspaces") return null;
  const activeTab = state.view === "project" ? state.activeReportTab : state.tab;
  const hasResult = state.view === "project" && state.runs.some(
    (run) => run.projectId === state.currentProjectId && run.status === "completed",
  );

  return (
    <div className="sidebar">
      <div>
        <div className="side-nav">
          {state.view !== "project" && ITEMS.map((item) => (
            <button
              key={item.tab}
              className={"nav-item" + (item.tab === activeTab ? " active" : "")}
              onClick={() => actions.setTab(item.tab)}
            >
              {item.label}
            </button>
          ))}
          {state.view === "project" && (
            <>
              <div className="side-section-label">Result</div>
              {REPORT_TABS.map((tab) => (
                <button
                  key={tab.id}
                  className={"nav-item" + (hasResult && tab.id === activeTab ? " active" : "")}
                  disabled={!hasResult}
                  title={hasResult ? tab.label : "Run the pipeline to view results"}
                  aria-label={hasResult ? tab.label : `${tab.label}. Run the pipeline to view results`}
                  onClick={() => actions.setReportTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
