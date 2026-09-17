import { useEffect, useState } from "react";
import { getDatasetEda } from "../api";
import ModelBarChart from "./ModelBarChart";
import PlanSummary from "./PlanSummary";

// Native, in-app replacement for embedding reports/template.html: every
// section below reads the same deterministic pipeline objects the HTML
// report is built from (api/run_store.py now persists them onto the run
// record), just rendered as React instead of an iframe.

function DefinitionList({ rows }) {
  const present = rows.filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (!present.length) return null;
  return (
    <dl className="plan-grid">
      {present.map(([label, value]) => (
        <div className="plan-row" key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Unavailable({ label }) {
  return <p className="muted">{label} is not available for this run.</p>;
}

function MissingValuesChart({ values }) {
  const entries = Object.entries(values || {}).sort(([, a], [, b]) => b - a);
  if (!entries.length) return null;
  const max = Math.max(...entries.map(([, value]) => value), 1);
  return (
    <div className="eda-chart-list">
      {entries.map(([column, value]) => (
        <div className="eda-chart-row" key={column}>
          <span className="eda-chart-label" title={column}>{column}</span>
          <div className="eda-chart-track"><div className="eda-chart-fill missing" style={{ width: `${(value / max) * 100}%` }} /></div>
          <span className="eda-chart-value">{value}%</span>
        </div>
      ))}
    </div>
  );
}

function OutlierBoxPlot({ values, outliers }) {
  const entries = Object.entries(values || {}).filter(([, info]) => info?.min !== undefined);
  if (!entries.length) return null;
  return (
    <div className="eda-boxplot-list">
      {entries.map(([column, info]) => {
        const range = Math.max(info.max - info.min, 1);
        const position = (value) => `${((value - info.min) / range) * 100}%`;
        return (
          <div className="eda-boxplot-row" key={column}>
            <span className="eda-chart-label" title={column}>{column}</span>
            <div className="eda-boxplot" aria-label={`${column}: ${info.count} outliers`}>
              <span className="eda-boxplot-whisker" style={{ left: position(info.min), width: `${((info.max - info.min) / range) * 100}%` }} />
              <span className="eda-boxplot-cap" style={{ left: position(info.min) }} />
              <span className="eda-boxplot-cap" style={{ left: position(info.max) }} />
              <span className="eda-boxplot-box" style={{ left: position(info.q1), width: `${((info.q3 - info.q1) / range) * 100}%` }} />
              <span className="eda-boxplot-median" style={{ left: position(info.median) }} />
            </div>
            <span className="eda-chart-value">{outliers?.[column]?.count || 0} outliers</span>
          </div>
        );
      })}
    </div>
  );
}

function DistributionHistogram({ distributions }) {
  const entries = Object.entries(distributions || {}).filter(([, info]) => info?.histogram?.counts?.length);
  if (!entries.length) return null;
  return (
    <div className="eda-histogram-grid">
      {entries.map(([column, info]) => {
        const counts = info.histogram.counts;
        const max = Math.max(...counts, 1);
        return (
          <figure className="eda-histogram" key={column}>
            <figcaption><strong>{column}</strong><span>skew {info.skew}</span></figcaption>
            <div className="eda-histogram-bars" aria-label={`${column} distribution histogram`}>
              {counts.map((count, index) => <span key={index} style={{ height: `${Math.max(3, (count / max) * 100)}%` }} title={`${count} values`} />)}
            </div>
            <div className="eda-histogram-axis"><span>{info.histogram.min.toFixed(2)}</span><span>{info.histogram.max.toFixed(2)}</span></div>
          </figure>
        );
      })}
    </div>
  );
}

function OverviewTab({ runRecord }) {
  const { plan, decision } = runRecord;
  return (
    <div>
      <PlanSummary plan={plan} />
      {decision ? (
        <DefinitionList
          rows={[
            ["Outcome", decision.outcome],
            ["Best model", decision.best_model],
            ["Reasoning", decision.reasoning],
          ]}
        />
      ) : (
        <Unavailable label="Evaluation decision" />
      )}
    </div>
  );
}

function DataQualityTab({ runRecord }) {
  const profile = runRecord.dataset_profile;
  const quality = runRecord.data_quality_report;
  if (!profile && !quality) return <Unavailable label="Dataset profile / quality report" />;

  return (
    <div>
      {profile && (
        <>
          <h3>Dataset profile</h3>
          <DefinitionList
            rows={[
              ["Rows", profile.row_count],
              ["Columns", profile.column_count],
              ["Duplicate rows", `${profile.duplicate_row_count} (${profile.duplicate_row_pct}%)`],
              ["Numerical columns", profile.numerical_columns?.join(", ")],
              ["Categorical columns", profile.categorical_columns?.join(", ")],
              ["Datetime columns", profile.datetime_columns?.join(", ")],
              ["Constant columns", profile.constant_columns?.join(", ")],
              ["Near-constant columns", profile.near_constant_columns?.join(", ")],
            ]}
          />
        </>
      )}
      {quality && (
        <>
          <h3>Data quality findings</h3>
          <DefinitionList
            rows={[
              [
                "Missing-value columns",
                Object.entries(quality.missing_value_columns || {})
                  .map(([col, pct]) => `${col} (${pct}%)`)
                  .join(", "),
              ],
              ["Duplicate rows", `${quality.duplicate_row_count} (${quality.duplicate_row_pct}%)`],
              ["Constant columns", quality.constant_columns?.join(", ")],
              ["Near-constant columns", quality.near_constant_columns?.join(", ")],
              [
                "Possible outlier columns",
                Object.entries(quality.possible_outlier_columns || {})
                  .map(([col, info]) => `${col} (${info.count}, ${info.pct}%)`)
                  .join(", "),
              ],
              ["Invalid dtype columns", quality.invalid_dtype_columns?.join(", ")],
            ]}
          />
        </>
      )}
    </div>
  );
}

function EdaTab({ runRecord }) {
  const eda = runRecord.eda_summary;
  const [snapshot, setSnapshot] = useState(null);

  const {
    missing_value_pct: missingValuePct,
    outliers_iqr: outliersIqr,
    correlation_matrix: correlationMatrix,
    distributions,
    datetime_columns: datetimeColumns,
    seasonality_notes: seasonalityNotes,
  } = eda || {};

  const hasHistogramData = Object.values(distributions || {}).some(
    (info) => info?.min !== undefined && info?.histogram?.counts?.length,
  );

  useEffect(() => {
    if (!runRecord.dataset_id || hasHistogramData) return undefined;
    let cancelled = false;
    getDatasetEda(runRecord.dataset_id).then((result) => {
      if (!cancelled) setSnapshot(result);
    }).catch(() => {
      // The report remains usable with its embedded summary if the snapshot is unavailable.
    });
    return () => {
      cancelled = true;
    };
  }, [runRecord.dataset_id, hasHistogramData]);

  if (!eda) return <Unavailable label="EDA summary" />;

  const snapshotDistributions = Object.fromEntries(
    (snapshot?.columns || [])
      .filter((column) => column.kind === "numeric" && column.stats)
      .map((column) => [
        column.name,
        {
          ...(distributions?.[column.name] || {}),
          ...column.stats,
          histogram: {
            counts: column.histogram || [],
            min: column.stats.min,
            max: column.stats.max,
          },
        },
      ]),
  );
  const chartDistributions = hasHistogramData ? distributions : snapshotDistributions;

  const numericCols = Object.keys(correlationMatrix || {});
  const boxPlotValues = Object.fromEntries(
    Object.entries(chartDistributions || {}).map(([column, info]) => [column, info]),
  );

  return (
    <div>
      {missingValuePct && Object.keys(missingValuePct).length > 0 && (
        <>
          <h3>Missing values</h3>
          <MissingValuesChart values={missingValuePct} />
        </>
      )}

      {chartDistributions && Object.keys(chartDistributions).length > 0 && (
        <>
          <h3>Outliers (IQR method)</h3>
          <OutlierBoxPlot values={boxPlotValues} outliers={outliersIqr} />
        </>
      )}

      {chartDistributions && Object.keys(chartDistributions).length > 0 && (
        <>
          <h3>Distribution histograms</h3>
          <DistributionHistogram distributions={chartDistributions} />
        </>
      )}

      {numericCols.length >= 2 && (
        <>
          <h3>Correlation matrix</h3>
          <div className="dataset-preview">
            <table>
              <thead>
                <tr>
                  <th></th>
                  {numericCols.map((col) => (
                    <th key={col}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {numericCols.map((rowCol) => (
                  <tr key={rowCol}>
                    <td>{rowCol}</td>
                    {numericCols.map((colCol) => (
                      <td key={colCol}>{correlationMatrix[rowCol]?.[colCol] ?? ""}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {datetimeColumns?.length > 0 && (
        <>
          <h3>Datetime / seasonality</h3>
          <DefinitionList
            rows={datetimeColumns.map((col) => {
              const note = seasonalityNotes?.[col];
              return [
                col,
                note
                  ? `${note.min_date} – ${note.max_date} (${note.distinct_months} distinct months)`
                  : "",
              ];
            })}
          />
        </>
      )}
    </div>
  );
}

function formatLogValue(value) {
  const isEmptyObject =
    value && typeof value === "object" && !Array.isArray(value) && !Object.keys(value).length;
  if (value === null || value === undefined || (Array.isArray(value) && !value.length) || isEmptyObject) {
    return <span className="plan-empty">None</span>;
  }
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function LogSection({ title, log }) {
  if (!log) return null;
  const entries = Object.entries(log);
  if (!entries.length) return null;
  return (
    <div className="plan-summary">
      <h3>{title}</h3>
      <dl className="plan-grid">
        {entries.map(([key, value]) => (
          <div className="plan-row" key={key}>
            <dt>{key}</dt>
            <dd>{formatLogValue(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function ClassDistributionChart({ distribution }) {
  const labels = Array.from(new Set([
    ...Object.keys(distribution?.train || {}),
    ...Object.keys(distribution?.test || {}),
  ]));
  if (!labels.length) return null;
  const max = Math.max(...labels.flatMap((label) => [
    distribution.train?.[label] || 0,
    distribution.test?.[label] || 0,
  ]), 1);

  return (
    <div className="class-distribution-chart">
      <div className="class-distribution-legend">
        <span><i className="class-distribution-swatch train" />Train</span>
        <span><i className="class-distribution-swatch test" />Test</span>
      </div>
      <div className="class-distribution-bars">
        {labels.map((label) => (
          <div className="class-distribution-group" key={label}>
            <div className="class-distribution-columns">
              {[["train", distribution.train?.[label] || 0], ["test", distribution.test?.[label] || 0]].map(([split, count]) => (
                <div className="class-distribution-column" key={split}>
                  <span className="class-distribution-count">{count}</span>
                  <div className={`class-distribution-bar ${split}`} style={{ height: `${Math.max(3, (count / max) * 150)}px` }} />
                </div>
              ))}
            </div>
            <span className="class-distribution-label" title={label}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function PreprocessingTab({ runRecord }) {
  const { cleaning_log, feature_log, feature_selection_log, split_log } = runRecord;
  if (!cleaning_log && !feature_log && !feature_selection_log && !split_log) {
    return <Unavailable label="Preprocessing logs" />;
  }
  return (
    <div>
      <LogSection title="Feature selection" log={feature_selection_log} />
      <LogSection title="Cleaning" log={cleaning_log} />
      <LogSection title="Feature engineering" log={feature_log} />
      <LogSection title="Train/test split" log={split_log} />
      {split_log?.class_distribution && (
        <section className="class-distribution-section">
          <h3>Class distribution</h3>
          <ClassDistributionChart distribution={split_log.class_distribution} />
        </section>
      )}
    </div>
  );
}

const METRIC_KEY_ORDER = [
  "score_test",
  "score_val",
  "accuracy",
  "precision",
  "recall",
  "f1",
  "roc_auc",
  "pr_auc",
  "r2",
  "rmse",
  "mae",
  "mape",
  "smape",
  "mase",
];

function formatMetricValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(4);
  return String(value);
}

function ModelMetricsTable({ results, bestModel }) {
  const metricKeys = [];
  for (const key of METRIC_KEY_ORDER) {
    if (results.some((r) => r.metrics && r.metrics[key] !== undefined)) metricKeys.push(key);
  }
  // Catch any metric key not in the known ordering rather than silently dropping it.
  for (const r of results) {
    for (const key of Object.keys(r.metrics || {})) {
      if (!metricKeys.includes(key) && key !== "confusion_matrix" && key !== "per_class") {
        metricKeys.push(key);
      }
    }
  }

  return (
    <div className="dataset-preview">
      <h3>Per-model metrics</h3>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Status</th>
            {metricKeys.map((key) => (
              <th key={key}>{key}</th>
            ))}
            <th>Training time (s)</th>
            <th>Errors</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.model_name} className={r.model_name === bestModel ? "best-row" : undefined}>
              <td>
                {r.model_name}
                {r.model_name === bestModel ? " ★" : ""}
              </td>
              <td>{r.status}</td>
              {metricKeys.map((key) => (
                <td key={key}>{formatMetricValue(r.metrics?.[key])}</td>
              ))}
              <td>{r.training_time !== null && r.training_time !== undefined ? r.training_time.toFixed(2) : ""}</td>
              <td>{r.errors || ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FlatModelMetricsTable({ models, bestModel }) {
  const entries = Object.entries(models || {});
  if (!entries.length) return null;
  const metricKeys = [];
  for (const key of METRIC_KEY_ORDER) {
    if (entries.some(([, m]) => m[key] !== undefined)) metricKeys.push(key);
  }
  return (
    <div className="dataset-preview">
      <h3>Per-model metrics</h3>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            {metricKeys.map((key) => (
              <th key={key}>{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, m]) => (
            <tr key={name} className={name === bestModel ? "best-row" : undefined}>
              <td>
                {name}
                {name === bestModel ? " ★" : ""}
              </td>
              {metricKeys.map((key) => (
                <td key={key}>{formatMetricValue(m[key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelsTab({ runRecord }) {
  const { metrics, decision } = runRecord;
  if (!metrics) return <Unavailable label="Model metrics" />;
  const comparisonResults = metrics.model_comparison?.results;
  return (
    <div>
      <p className="muted">
        Evaluation metric: {metrics.eval_metric}
        {metrics.per_entity && ` — averaged across ${metrics.entities_trained} entities`}
      </p>
      <ModelBarChart models={metrics.models} bestModel={decision?.best_model} />

      {comparisonResults?.length > 0 ? (
        <ModelMetricsTable results={comparisonResults} bestModel={decision?.best_model} />
      ) : (
        <FlatModelMetricsTable models={metrics.models} bestModel={decision?.best_model} />
      )}

      {metrics.per_entity && (
        <div className="dataset-preview">
          <h3>Per-entity breakdown</h3>
          <table>
            <thead>
              <tr>
                <th>Entity</th>
                <th>Best model</th>
                <th>Test score</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(metrics.per_entity).map(([entityId, m]) => (
                <tr key={entityId}>
                  <td>{entityId}</td>
                  <td>{m.autogluon_best_model}</td>
                  <td>{m.models?.[m.autogluon_best_model]?.score_test ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {metrics.entities_skipped?.length > 0 && (
            <p className="muted">
              Skipped {metrics.entities_skipped.length} entities:{" "}
              {metrics.entities_skipped.map((s) => `${s.entity} (${s.reason})`).join(", ")}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function RecommendationProseSection({ title, text }) {
  if (!text) return null;
  return (
    <div className="recommendation-section">
      <h4>{title}</h4>
      <p>{text}</p>
    </div>
  );
}

function RecommendationTab({ runRecord }) {
  const rec = runRecord.recommendation;
  if (!rec) return <Unavailable label="Recommendation" />;
  return (
    <div>
      {rec.flagged_for_review && (
        <div className="card warning">
          <strong>Flagged for review:</strong> {rec.flag_reason}
        </div>
      )}

      <div className="recommendation-hero">
        <div>
          <span className="muted">Recommended model</span>
          <h3>{rec.recommended_model}</h3>
        </div>
        {rec.confidence_statement && (
          <span className="confidence-badge" title={rec.confidence_statement}>
            {rec.confidence_statement}
          </span>
        )}
      </div>

      {rec.cited_top_features?.length > 0 && (
        <div className="recommendation-section">
          <h4>Top features</h4>
          <div className="chip-list">
            {rec.cited_top_features.map((f) => (
              <span className="chip" key={f}>{f}</span>
            ))}
          </div>
        </div>
      )}

      <RecommendationProseSection title="Reason" text={rec.reason} />
      <RecommendationProseSection title="Performance summary" text={rec.performance_summary} />
      <RecommendationProseSection title="Comparison to alternatives" text={rec.comparison_to_alternatives} />
      <RecommendationProseSection title="How it works" text={rec.explanation_narrative} />
      <RecommendationProseSection title="Limitations" text={rec.limitations} />
    </div>
  );
}

function BusinessInterpretationTab({ runRecord }) {
  const rec = runRecord.recommendation;
  if (!rec?.business_interpretation) return <Unavailable label="Business interpretation" />;
  return (
    <div>
      <h3>Recommended model: {rec.recommended_model}</h3>
      <p>{rec.business_interpretation}</p>
    </div>
  );
}

function ChartsTab({ runRecord }) {
  const charts = runRecord.report_charts;
  const entries = Object.entries(charts || {}).filter(([, uri]) => uri);
  if (!entries.length) return <Unavailable label="Charts" />;
  return (
    <div className="charts-grid">
      {entries.map(([name, uri]) => (
        <figure key={name}>
          <img src={uri} alt={name} />
          <figcaption className="muted">{name.replace(/_/g, " ")}</figcaption>
        </figure>
      ))}
    </div>
  );
}

const TABS = [
  { id: "overview", label: "Overview", Component: OverviewTab },
  { id: "data-quality", label: "Data Quality", Component: DataQualityTab },
  { id: "eda", label: "EDA", Component: EdaTab },
  { id: "preprocessing", label: "Preprocessing", Component: PreprocessingTab },
  { id: "models", label: "Models & Metrics", Component: ModelsTab },
  { id: "recommendation", label: "Recommendation", Component: RecommendationTab },
  { id: "business-interpretation", label: "Business Interpretation", Component: BusinessInterpretationTab },
  { id: "charts", label: "Charts", Component: ChartsTab },
];

export default function ReportTabs({ runRecord }) {
  const [activeTab, setActiveTab] = useState(TABS[0].id);
  if (!runRecord) return null;

  const Active = TABS.find((t) => t.id === activeTab)?.Component ?? TABS[0].Component;

  return (
    <div>
      <div className="report-tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`report-tab${activeTab === tab.id ? " active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="report-tab-panel">
        <Active runRecord={runRecord} />
      </div>
    </div>
  );
}
