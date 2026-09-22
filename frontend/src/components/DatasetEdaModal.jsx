import { useEffect, useRef, useState } from "react";
import { getDatasetEda } from "../api";

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function TypeCounts({ typeCounts }) {
  const KIND_BADGE = {
    numeric: { label: "NUM", cls: "badge-num" },
    categorical: { label: "CAT", cls: "badge-cat" },
    datetime: { label: "DAT", cls: "badge-dat" },
    boolean: { label: "BOO", cls: "badge-boo" },
    text: { label: "TEX", cls: "badge-tex" },
  };
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

export default function DatasetEdaModal({ dataset, onClose }) {
  const [eda, setEda] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const overlayRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    getDatasetEda(dataset.id)
      .then((result) => {
        if (cancelled) return;
        setEda(result);
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

            <div className="eda-samples">
              <div className="eda-samples-head">
                <span>SAMPLE ROWS ({eda.sample_rows.length})</span>
              </div>
              <div className="eda-table-wrap">
                <table className="preview-table">
                  <thead>
                    <tr>
                      {eda.columns.map((c) => (
                        <th key={c.name}>{c.name}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {eda.sample_rows.map((row, i) => (
                      <tr key={i}>
                        {eda.columns.map((c) => (
                          <td key={c.name}>
                            {row[c.name] === null || row[c.name] === undefined ? "" : String(row[c.name])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
