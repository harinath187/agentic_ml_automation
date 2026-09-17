import { useEffect, useRef, useState } from "react";
import { currentWorkspace, useStore } from "../store";
import ThemeToggle from "./ThemeToggle";

export default function TopBar({ onNewWorkspace }) {
  const { state, actions } = useStore();
  const ws = currentWorkspace(state);
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);

  useEffect(() => {
    function onDocClick(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("click", onDocClick);
    return () => document.removeEventListener("click", onDocClick);
  }, []);

  return (
    <div className="topbar">
      <div className="brand">
        <span className="mark">A</span> Agentic ML Console
      </div>
      <div className="topbar-right">
        <div className="ws-switcher-wrap" ref={wrapRef}>
          <button
            className="ws-switcher"
            aria-haspopup="true"
            aria-expanded={open}
            onClick={(e) => {
              e.stopPropagation();
              setOpen((v) => !v);
            }}
          >
            <span>{ws ? ws.name : "Pick a workspace"}</span>
            <span className="chev">▾</span>
          </button>
          {open && (
            <div className="ws-dropdown">
              {state.workspaces.map((w) => (
                <button
                  key={w.id}
                  className={"ws-dropdown-item" + (w.id === state.currentWorkspaceId ? " current" : "")}
                  onClick={() => {
                    setOpen(false);
                    actions.openWorkspace(w.id);
                  }}
                >
                  <span className="label">{w.name}</span>
                </button>
              ))}
              <div className="ws-dropdown-sep" />
              <button
                className="ws-dropdown-item"
                onClick={() => {
                  setOpen(false);
                  actions.goWorkspaces();
                }}
              >
                <span className="label">All workspaces</span>
              </button>
              <button
                className="ws-dropdown-item ws-dropdown-create"
                onClick={() => {
                  setOpen(false);
                  onNewWorkspace();
                }}
              >
                <span className="label">New workspace</span>
              </button>
            </div>
          )}
        </div>
        <ThemeToggle />
      </div>
    </div>
  );
}
