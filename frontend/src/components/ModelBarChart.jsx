export default function ModelBarChart({ models, bestModel }) {
  const entries = Object.entries(models || {});
  if (entries.length === 0) return null;

  const maxScore = Math.max(...entries.map(([, m]) => m.score_test ?? 0), 1e-9);

  return (
    <div className="bar-chart">
      {entries.map(([name, m]) => (
        <div className="bar-row" key={name}>
          <span className="bar-label">
            {name}
            {name === bestModel && <span className="best-badge">best</span>}
          </span>
          <div className="bar-track">
            <div
              className={`bar-fill ${name === bestModel ? "bar-fill-best" : ""}`}
              style={{ width: `${Math.max(2, (m.score_test / maxScore) * 100)}%` }}
            />
          </div>
          <span className="bar-value">{m.score_test}</span>
        </div>
      ))}
    </div>
  );
}
