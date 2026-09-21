import ThemeToggle from "./ThemeToggle";

export default function TopBar() {
  return (
    <div className="topbar">
      <div className="brand">
        <span className="mark">
          <svg
            viewBox="0 0 24 24"
            width="16"
            height="16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="4.5" cy="6" r="2.3" fill="currentColor" stroke="none" />
            <circle cx="4.5" cy="18" r="2.3" fill="currentColor" stroke="none" />
            <circle cx="12" cy="12" r="2.4" fill="currentColor" stroke="none" />
            <path d="M6.6 6.9 10 11" />
            <path d="M6.6 17.1 10 13" />
            <path d="M14.3 12h3.8" />
            <path d="M18.7 9.3l1.8 2.7-1.8 2.7" />
          </svg>
        </span>{" "}
        ML Pipeline
      </div>
      <div className="topbar-right">
        <ThemeToggle />
      </div>
    </div>
  );
}
