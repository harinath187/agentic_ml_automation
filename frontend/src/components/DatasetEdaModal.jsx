import { useEffect, useMemo, useRef, useState } from "react";
import { getDatasetEda } from "../api";

const KIND_BADGE = {
  numeric: { label: "NUM", cls: "badge-num" },
  categorical: { label: "CAT", cls: "badge-cat" },
  datetime: { label: "DAT", cls: "badge-dat" },
  boolean: { label: "BOO", cls: "badge-boo" },
  text: { label: "TEX", cls: "badge-tex" },
};

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function fmtNum(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return typeof n === "number" ? n.toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(n);
}

function Sparkline({ values }) {
  if (!values || !values.length) return <span className="sparkline-empty">—</span>;
  const max = Math.max(...values, 1);
  return (
    <svg className="sparkline" viewBox={`0 0 ${values.length * 4} 20`} preserveAspectRatio="none">
      {values.map((v, i) => {
        const h = Math.max(1, (v / max) * 18);
        return <rect key={i} x={i * 4} y={20 - h} width={3} height={h} rx={0.5} />;
      })}
    </svg>
  );
}

function TypeCounts({ typeCounts }) {
  return (
    <div className="type-pills">
      {Object.entries(typeCounts).map(([kind, count]) => {
        const badge = KIND_BADGE[kind];
        if (!badge) return null;
        return (
          <span key={kind} className={`type-pill ${badge.cls}`}>
            {badge.label}·{count}
          </span>
        );
      })}
    </div>
  );
}

function ColumnDetail({ column }) {
  if (!column) return <p className="muted">Select a column to inspect it.</p>;
  const s = column.stats || {};

  return (
    <div className="col-detail">
      <div className="col-detail-head">
        <span className={`type-badge ${KIND_BADGE[column.kind]?.cls || ""}`}>
          {KIND_BADGE[column.kind]?.label || column.kind}
        </span>
        <h4>{column.name}</h4>
      </div>

      <div className="kv-grid">
        <div>
          <div className="k">Unique</div>
          <div className="v">{column.unique_count.toLocaleString()}</div>
        </div>
        <div>
          <div className="k">Cardinality</div>
          <div className="v">{column.cardinality_pct}%</div>
        </div>
        <div>
          <div className="k">Missing</div>
          <div className="v">
            {column.missing_count.toLocaleString()} ({column.missing_pct}%)
          </div>
        </div>
        <div>
          <div className="k">Dtype</div>
          <div className="v mono">{column.dtype}</div>
        </div>

        {column.kind === "numeric" && s.min !== undefined && (
          <>
            <div>
              <div className="k">Min</div>
              <div className="v">{fmtNum(s.min)}</div>
            </div>
            <div>
              <div className="k">Max</div>
              <div className="v">{fmtNum(s.max)}</div>
            </div>
            <div>
              <div className="k">Mean</div>
              <div className="v">{fmtNum(s.mean)}</div>
            </div>
            <div>
              <div className="k">Median</div>
              <div className="v">{fmtNum(s.median)}</div>
            </div>
            <div>
              <div className="k">Std</div>
              <div className="v">{fmtNum(s.std)}</div>
            </div>
            <div>
              <div className="k">Q1</div>
              <div className="v">{fmtNum(s.q1)}</div>
            </div>
            <div>
              <div className="k">Q3</div>
              <div className="v">{fmtNum(s.q3)}</div>
            </div>
            <div>
              <div className="k">Skew</div>
              <div className="v">{fmtNum(s.skew)}</div>
            </div>
            <div>
              <div className="k">Kurt.</div>
              <div className="v">{fmtNum(s.kurtosis)}</div>
            </div>
            <div>
              <div className="k">Zeros %</div>
              <div className="v">{fmtNum(s.zeros_pct)}</div>
            </div>
          </>
        )}

        {column.kind === "datetime" && s.min !== undefined && (
          <>
            <div>
              <div className="k">Min</div>
              <div className="v mono">{s.min}</div>
            </div>
            <div>
              <div className="k">Max</div>
              <div className="v mono">{s.max}</div>
            </div>
            <div>
              <div className="k">Distinct days</div>
              <div className="v">{s.distinct_days}</div>
            </div>
          </>
        )}

        {column.kind === "boolean" && s.true_pct !== undefined && (
          <>
            <div>
              <div className="k">True</div>
              <div className="v">
                {s.true_count.toLocaleString()} ({s.true_pct}%)
              </div>
            </div>
            <div>
              <div className="k">False</div>
              <div className="v">
                {s.false_count.toLocaleString()} ({s.false_pct}%)
              </div>
            </div>
          </>
        )}
      </div>

      {(column.kind === "categorical" || column.kind === "text") && s.top_values && (
        <div className="top-values">
          <div className="k" style={{ marginBottom: 6 }}>
            Top values
          </div>
          {s.top_values.map((tv) => (
            <div className="top-value-row" key={tv.value}>
              <span className="top-value-label" title={tv.value}>
                {tv.value}
              </span>
              <div className="top-value-track">
                <div className="top-value-fill" style={{ width: `${Math.min(100, tv.pct)}%` }} />
              </div>
              <span className="top-value-pct">{tv.pct}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CorrelationsPanel({ correlations }) {
  if (!correlations) {
    return <p className="muted">Not enough numeric columns for a correlation matrix.</p>;
  }
  const { columns, matrix } = correlations;
  return (
    <div className="corr-wrap">
      <table className="corr-table">
        <thead>
          <tr>
            <th></th>
            {columns.map((c) => (
              <th key={c} title={c}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={columns[i]}>
              <th title={columns[i]}>{columns[i]}</th>
              {row.map((value, j) => {
                const v = value === null ? 0 : value;
                const alpha = Math.min(1, Math.abs(v));
                const bg = v >= 0 ? `rgba(15, 122, 110, ${alpha * 0.55})` : `rgba(179, 69, 47, ${alpha * 0.55})`;
                return (
                  <td key={columns[j]} style={{ background: bg }} title={`${columns[i]} × ${columns[j]}: ${value ?? "n/a"}`}>
                    {value === null ? "—" : value.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OverviewPanel({ eda }) {
  const missingCols = eda.columns.filter((c) => c.missing_count > 0).sort((a, b) => b.missing_pct - a.missing_pct);
  return (
    <div className="overview-panel">
      <div className="overview-row">
        <span className="k">Memory</span>
        <span className="v">{fmtBytes(eda.memory_bytes)}</span>
      </div>
      <div className="overview-row">
        <span className="k">Duplicate rows</span>
        <span className="v">
          {eda.duplicate_row_count.toLocaleString()} ({eda.duplicate_row_pct}%)
        </span>
      </div>
      <div className="overview-row">
        <span className="k">Missing cells</span>
        <span className="v">{eda.missing_cells_pct}%</span>
      </div>
      <div className="section-head" style={{ marginTop: 16 }}>
        <h2 style={{ fontSize: 13 }}>Columns with missing values</h2>
      </div>
      {missingCols.length ? (
        <div className="top-values">
          {missingCols.map((c) => (
            <div className="top-value-row" key={c.name}>
              <span className="top-value-label" title={c.name}>
                {c.name}
              </span>
              <div className="top-value-track">
                <div className="top-value-fill" style={{ width: `${Math.min(100, c.missing_pct)}%` }} />
              </div>
              <span className="top-value-pct">{c.missing_pct}%</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">No missing values in this dataset.</p>
      )}
    </div>
  );
}

export default function DatasetEdaModal({ dataset, onClose }) {
  const [eda, setEda] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState("columns");
  const [selectedColumn, setSelectedColumn] = useState(null);
  const [filter, setFilter] = useState("");
  const overlayRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    getDatasetEda(dataset.id)
      .then((result) => {
        if (cancelled) return;
        setEda(result);
        setSelectedColumn(result.columns[0]?.name || null);
      })
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [dataset.id]);

  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const filteredColumns = useMemo(() => {
    if (!eda) return [];
    const q = filter.trim().toLowerCase();
    return q ? eda.columns.filter((c) => c.name.toLowerCase().includes(q)) : eda.columns;
  }, [eda, filter]);

  const selected = eda?.columns.find((c) => c.name === selectedColumn) || null;

  return (
    <div
      className="modal-overlay"
      ref={overlayRef}
      onClick={(e) => {
        if (e.target === overlayRef.current) onClose();
      }}
    >
      <div className="modal eda-modal">
        <div className="eda-header">
          <div>
            <h3>{dataset.name}</h3>
            <p className="desc">Interactive exploration — single-call EDA over the latest snapshot.</p>
          </div>
          <button className="eda-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {loading && <p className="muted">Computing EDA snapshot…</p>}
        {error && <p className="error">{error}</p>}

        {eda && (
          <>
            <div className="eda-tiles">
              <div className="eda-tile">
                <div className="tile-label">Rows</div>
                <div className="tile-value">{eda.row_count.toLocaleString()}</div>
              </div>
              <div className="eda-tile">
                <div className="tile-label">Columns</div>
                <div className="tile-value">{eda.column_count}</div>
              </div>
              <div className="eda-tile">
                <div className="tile-label">Memory</div>
                <div className="tile-value">{fmtBytes(eda.memory_bytes)}</div>
              </div>
              <div className="eda-tile">
                <div className="tile-label">Duplicates</div>
                <div className="tile-value">{eda.duplicate_row_count.toLocaleString()}</div>
              </div>
              <div className="eda-tile">
                <div className="tile-label">Missing</div>
                <div className="tile-value">{eda.missing_cells_pct}%</div>
              </div>
              <div className="eda-tile eda-tile-types">
                <div className="tile-label">Types</div>
                <TypeCounts typeCounts={eda.type_counts} />
              </div>
            </div>

            <div className="eda-body">
              <div className="eda-samples">
                <div className="eda-samples-head">
                  <span>SAMPLE ROWS ({eda.sample_rows.length})</span>
                  <span className="hint">Click a column header to inspect</span>
                </div>
                <div className="eda-table-wrap">
                  <table className="preview-table">
                    <thead>
                      <tr>
                        {eda.columns.map((c) => (
                          <th
                            key={c.name}
                            className={c.name === selectedColumn ? "selected" : ""}
                            onClick={() => {
                              setSelectedColumn(c.name);
                              setTab("columns");
                            }}
                          >
                            {c.name}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {eda.sample_rows.map((row, i) => (
                        <tr key={i}>
                          {eda.columns.map((c) => (
                            <td key={c.name} className={c.name === selectedColumn ? "selected" : ""}>
                              {row[c.name] === null || row[c.name] === undefined ? "" : String(row[c.name])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="eda-side">
                <div className="eda-tabs">
                  <button className={tab === "columns" ? "active" : ""} onClick={() => setTab("columns")}>
                    Columns ({eda.columns.length})
                  </button>
                  <button className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")}>
                    Overview
                  </button>
                  <button className={tab === "correlations" ? "active" : ""} onClick={() => setTab("correlations")}>
                    Correlations
                  </button>
                </div>

                {tab === "columns" && (
                  <>
                    <input
                      className="col-filter"
                      type="text"
                      placeholder="Filter columns…"
                      value={filter}
                      onChange={(e) => setFilter(e.target.value)}
                    />
                    <div className="col-list">
                      {filteredColumns.map((c) => (
                        <button
                          key={c.name}
                          className={"col-row" + (c.name === selectedColumn ? " selected" : "")}
                          onClick={() => setSelectedColumn(c.name)}
                        >
                          <span className={`type-badge ${KIND_BADGE[c.kind]?.cls || ""}`}>
                            {KIND_BADGE[c.kind]?.label || c.kind}
                          </span>
                          <span className="col-row-name">{c.name}</span>
                          <Sparkline values={c.histogram} />
                          <span className="col-row-missing">{c.missing_pct}%</span>
                        </button>
                      ))}
                    </div>
                    <ColumnDetail column={selected} />
                  </>
                )}

                {tab === "overview" && <OverviewPanel eda={eda} />}
                {tab === "correlations" && <CorrelationsPanel correlations={eda.correlations} />}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
