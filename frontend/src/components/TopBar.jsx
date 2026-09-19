import ThemeToggle from "./ThemeToggle";

export default function TopBar() {
  return (
    <div className="topbar">
      <div className="brand">
        <span className="mark">A</span> Agentic ML Console
      </div>
      <div className="topbar-right">
        <ThemeToggle />
      </div>
    </div>
  );
}
