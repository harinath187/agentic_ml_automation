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

function ReportChart({ uri, name, caption }) {
  if (!uri) return null;
  return (
    <figure className="report-chart">
      <img src={uri} alt={name} />
      <figcaption className="muted">{caption || name.replace(/_/g, " ")}</figcaption>
    </figure>
  );
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
  const { plan } = runRecord;
  return (
    <div className="overview-report">
      <div className="overview-panel">
        <PlanSummary plan={plan} />
      </div>
    </div>
  );
}

function DataQualityTab({ runRecord }) {
  const profile = runRecord.dataset_profile;
  const quality = runRecord.data_quality_report;
  if (!profile && !quality) return <Unavailable label="Dataset profile / quality report" />;

  const rowCount = Number(profile?.row_count ?? quality?.row_count ?? 0);
  const columnCount = Number(profile?.column_count ?? quality?.column_count ?? 0);
  const duplicateCount = Number(quality?.duplicate_row_count ?? profile?.duplicate_row_count ?? 0);
  const duplicatePct = Number(quality?.duplicate_row_pct ?? profile?.duplicate_row_pct ?? 0);
  const outlierEntries = Object.entries(quality?.possible_outlier_columns || {})
    .filter(([, info]) => info && typeof info === "object")
    .sort(([, a], [, b]) => Number(b?.pct ?? 0) - Number(a?.pct ?? 0));
  const score = Number(quality?.overall_quality_score ?? 100);

  const statusHeading =
    score >= 80 ? "Looks good" :
    score >= 60 ? "A few values worth a quick look" :
    "Needs attention";

  const narrative = [
    `Your dataset has ${rowCount.toLocaleString()} ${rowCount === 1 ? "record" : "records"} across ${columnCount} ${columnCount === 1 ? "column" : "columns"},`,
    duplicateCount > 0
      ? `with ${duplicateCount.toLocaleString()} duplicate row${duplicateCount === 1 ? "" : "s"} (${duplicatePct.toFixed(1)}%).`
      : "with no duplicate rows.",
    outlierEntries.length
      ? `A small number of entries in ${outlierEntries.length} ${outlierEntries.length === 1 ? "column" : "columns"} look unusually high or low compared to the rest — these are flagged below in case they're data entry mistakes.`
      : "No unusual outlier patterns were detected in the numeric fields.",
  ].join(" ");

  return (
    <div className="data-quality-report">
      <div className="quality-status-header">
        <h3>{statusHeading}</h3>
        <p>{narrative}</p>
      </div>

      <div className="quality-score-grid">
        <div className="quality-score-tile">
          <span className="quality-score-label">Records</span>
          <strong>{rowCount.toLocaleString()}</strong>
        </div>
        <div className="quality-score-tile">
          <span className="quality-score-label">Fields tracked</span>
          <strong>{columnCount.toLocaleString()}</strong>
        </div>
        <div className="quality-score-tile">
          <span className="quality-score-label">Duplicate rows</span>
          <strong>{duplicateCount.toLocaleString()}</strong>
        </div>
      </div>

      <div className="quality-outliers-panel">
        <h3>Unusual values by column</h3>
        <p>How many entries in each field fall far outside the typical range.</p>

        {outlierEntries.length ? (
          <div className="quality-outlier-list">
            {outlierEntries.map(([column, info]) => {
              const count = Number(info?.count ?? 0);
              const pct = Number(info?.pct ?? 0);
              return (
                <div className="quality-outlier-row" key={column}>
                  <div className="quality-outlier-name">{column}</div>
                  <div className="quality-outlier-metrics">
                    <span>{count.toLocaleString()} entries</span>
                    <span className="quality-outlier-percent">{pct.toFixed(1)}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="quality-empty-state">No unusual values detected in the numeric columns.</div>
        )}
      </div>
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

  if (Array.isArray(value)) {
    return (
      <span className="log-compact-list">
        {value.map((item, index) => (
          <span key={`${String(item)}-${index}`} className="log-compact-item">
            {typeof item === "object" && item !== null ? formatLogValue(item) : String(item)}
            {index < value.length - 1 ? "," : ""}
          </span>
        ))}
      </span>
    );
  }

  if (typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) return <span className="plan-empty">None</span>;
    return (
      <span className="log-object-list">
        {entries.map(([key, item]) => (
          <span key={key} className="log-object-item">
            <span className="log-object-key">{key}</span>
            <span className="log-object-colon">:</span>
            <span className="log-object-value">{formatLogValue(item)}</span>
          </span>
        ))}
      </span>
    );
  }

  return <span className="mono">{String(value)}</span>;
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

function PreprocessingSection({ title, log, summary, accentClass }) {
  const entries = log ? Object.entries(log) : [];
  const summaryItems = summary || [];

  return (
    <div className={`preprocessing-detail-panel preprocessing-detail-panel--${accentClass || "neutral"}`}>
      <div className="preprocessing-section-header">
        <h3>{title}</h3>
        {summaryItems.length > 0 && (
          <div className="preprocessing-section-summary">
            {summaryItems.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        )}
      </div>
      {entries.length ? <LogSection log={log} /> : <p className="muted">No data reported for this stage.</p>}
    </div>
  );
}

function PreprocessingTab({ runRecord }) {
  const { cleaning_log, feature_log, feature_selection_log, split_log } = runRecord;
  const charts = runRecord.report_charts || {};
  const hasExplainabilityCharts = charts.feature_importance || charts.permutation_importance || charts.shap_importance;
  if (!cleaning_log && !feature_log && !feature_selection_log && !split_log && !hasExplainabilityCharts) {
    return <Unavailable label="Preprocessing logs" />;
  }

  const kept = Array.isArray(feature_selection_log?.kept) ? feature_selection_log.kept : [];
  const dropped = Array.isArray(feature_selection_log?.dropped) ? feature_selection_log.dropped : [];
  const encodedCount = Array.isArray(cleaning_log?.encoded_columns) ? cleaning_log.encoded_columns.length : 0;
  const imputationCount = Object.keys(cleaning_log?.imputation || {}).length;
  const cappedTotal = Object.values(cleaning_log?.outliers_capped || {}).reduce((sum, value) => sum + Number(value || 0), 0);
  const duplicateRows = Number(cleaning_log?.duplicates_removed || 0);
  const trainRows = Number(split_log?.train_rows || 0);
  const testRows = Number(split_log?.test_rows || 0);
  const splitMethod = split_log?.method || "not reported";
  const lagCount = Array.isArray(feature_log?.lag_features) ? feature_log.lag_features.length : 0;
  const rollingCount = Array.isArray(feature_log?.rolling_features) ? feature_log.rolling_features.length : 0;

  return (
    <div className="preprocessing-report">
      <div className="preprocessing-detail-layout">
        <PreprocessingSection
          title="Feature selection"
          log={feature_selection_log}
          accentClass="primary"
          summary={[
            `${kept.length} columns kept`,
            `${dropped.length} excluded`,
          ]}
        />
        <PreprocessingSection
          title="Cleaning"
          log={cleaning_log}
          accentClass="secondary"
          summary={[
            `${imputationCount} imputations`,
            `${encodedCount} encoded columns`,
            `${cappedTotal} outliers capped`,
            duplicateRows ? `${duplicateRows} duplicates dropped` : "No duplicate rows",
          ].filter(Boolean)}
        />
        <PreprocessingSection
          title="Feature engineering"
          log={feature_log}
          accentClass="accent"
          summary={[
            `${lagCount} lag features`,
            `${rollingCount} rolling features`,
          ]}
        />
        <PreprocessingSection
          title="Train/test split"
          log={split_log}
          accentClass="neutral"
          summary={[
            `${splitMethod}`,
            `${trainRows} train / ${testRows} test`,
          ]}
        />
      </div>

      {hasExplainabilityCharts && (
        <section>
          <h3>Feature importance</h3>
          <div className="charts-grid">
            <ReportChart uri={charts.feature_importance} name="feature importance" caption="Native feature importance for the recommended model." />
            <ReportChart uri={charts.permutation_importance} name="permutation importance" caption="Permutation importance for the recommended model." />
            <ReportChart uri={charts.shap_importance} name="SHAP importance" caption="SHAP importance for the recommended model." />
          </div>
        </section>
      )}

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

function metricLabel(key) {
  const labels = {
    score_test: "Score",
    score_val: "Score val",
    accuracy: "Accuracy",
    precision: "Prec.",
    recall: "Recall",
    f1: "F1",
    roc_auc: "AUC",
    pr_auc: "PR AUC",
    r2: "R²",
    rmse: "RMSE",
    mae: "MAE",
    mape: "MAPE",
    smape: "SMAPE",
    mase: "MASE",
    kappa: "Kappa",
    mcc: "MCC",
  };
  return labels[key] || key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function modelShortCode(name) {
  const seed = String(name || "model").trim();
  if (!seed) return "mdl";
  const words = seed.split(/[^A-Za-z0-9]+/).filter(Boolean);
  if (!words.length) return "mdl";
  if (words.length === 1) return words[0].slice(0, 3).toLowerCase();
  const initials = words.slice(0, 3).map((word) => word[0]).join("");
  return initials.toLowerCase() || words[0].slice(0, 3).toLowerCase();
}

function getMetricKeys(rows) {
  const metricKeys = [];
  for (const key of METRIC_KEY_ORDER) {
    if (rows.some((row) => row.metrics && row.metrics[key] !== undefined && row.metrics[key] !== null)) {
      metricKeys.push(key);
    }
  }
  for (const row of rows) {
    for (const key of Object.keys(row.metrics || {})) {
      if (key !== "confusion_matrix" && key !== "per_class" && !metricKeys.includes(key)) {
        metricKeys.push(key);
      }
    }
  }
  return metricKeys;
}

function ModelMetricsTable({ results, bestModel }) {
  const rows = (results || []).map((result, index) => ({
    ...result,
    model_name: result.model_name || result.name || `Model ${index + 1}`,
    metrics: result.metrics || {},
  }));
  if (!rows.length) return null;

  const metricKeys = getMetricKeys(rows);
  const maxByKey = Object.fromEntries(metricKeys.map((key) => {
    const values = rows
      .map((row) => Number(row.metrics?.[key]))
      .filter((value) => Number.isFinite(value));
    const max = values.length ? Math.max(...values) : 0;
    return [key, max > 0 ? max : 1];
  }));

  return (
    <div className="dataset-preview model-metric-table-block">
      <h3>Per-model metrics</h3>
      <div className="model-metric-table-wrap">
        <table className="model-metric-table">
          <thead>
            <tr>
              <th className="metric-rank-col">#</th>
              <th className="metric-model-col">Model</th>
              {metricKeys.map((key) => (
                <th key={key}>{metricLabel(key)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={row.model_name} className={row.model_name === bestModel ? "best-row" : undefined}>
                <td className="metric-rank-col">{index + 1}</td>
                <td className="metric-model-col">
                  <div className="model-name-stack">
                    <span className="model-name">{row.model_name}</span>
                    <small className="model-tag">{modelShortCode(row.model_name)}</small>
                  </div>
                </td>
                {metricKeys.map((key) => {
                  const value = row.metrics?.[key];
                  return (
                    <td key={`${row.model_name}-${key}`} className="metric-value-cell">
                      <span className="metric-value">{formatMetricValue(value)}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FlatModelMetricsTable({ models, bestModel }) {
  const entries = Object.entries(models || {});
  if (!entries.length) return null;

  const metricKeys = getMetricKeys(entries.map(([name, metrics]) => ({ model_name: name, metrics })));
  const maxByKey = Object.fromEntries(metricKeys.map((key) => {
    const values = entries
      .map(([, metrics]) => Number(metrics?.[key]))
      .filter((value) => Number.isFinite(value));
    const max = values.length ? Math.max(...values) : 0;
    return [key, max > 0 ? max : 1];
  }));

  return (
    <div className="dataset-preview model-metric-table-block">
      <h3>Per-model metrics</h3>
      <div className="model-metric-table-wrap">
        <table className="model-metric-table">
          <thead>
            <tr>
              <th className="metric-rank-col">#</th>
              <th className="metric-model-col">Model</th>
              {metricKeys.map((key) => (
                <th key={key}>{metricLabel(key)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, metrics], index) => (
              <tr key={name} className={name === bestModel ? "best-row" : undefined}>
                <td className="metric-rank-col">{index + 1}</td>
                <td className="metric-model-col">
                  <div className="model-name-stack">
                    <span className="model-name">{name}</span>
                    <small className="model-tag">{modelShortCode(name)}</small>
                  </div>
                </td>
                {metricKeys.map((key) => {
                  const value = metrics?.[key];
                  return (
                    <td key={`${name}-${key}`} className="metric-value-cell">
                      <span className="metric-value">{formatMetricValue(value)}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FailureSummary({ summary, explanation }) {
  if (!summary) return null;
  const suitabilityLabel = {
    suitable: "Dataset checks passed",
    suitable_with_warnings: "Dataset is usable with warnings",
    not_suitable: "Dataset needs attention before modeling",
  }[summary.dataset_suitability] || "Issues detected";

  return (
    <section className="model-health-panel">
      <div className="model-health-header">
        <div>
          <h3>Model health</h3>
          <p className="muted">{suitabilityLabel}</p>
        </div>
        <span className={`status-pill status-pill--${summary.dataset_suitability === "not_suitable" ? "error" : "warning"}`}>
          {summary.successful_model_count} of {summary.candidate_count} models succeeded
        </span>
      </div>

      {explanation?.summary && (
        <div className="model-health-explanation">
          <strong>What this means</strong>
          <p>{explanation.summary}</p>
          {explanation.recommendations?.length > 0 && (
            <ul>
              {explanation.recommendations.map((recommendation) => <li key={recommendation}>{recommendation}</li>)}
            </ul>
          )}
        </div>
      )}

      {summary.dataset_issues?.length > 0 && (
        <div className="model-health-list">
          <h4>Dataset issues</h4>
          {summary.dataset_issues.map((issue) => (
            <div className="model-health-item" key={`${issue.category}-${issue.detail}`}>
              <strong>{issue.category.replaceAll("_", " ")}</strong>
              <span>{issue.detail}</span>
            </div>
          ))}
        </div>
      )}

      {summary.model_failures?.length > 0 && (
        <div className="model-health-list">
          <h4>Models that could not run</h4>
          {summary.model_failures.map((failure) => (
            <details className="model-health-item" key={failure.model_name}>
              <summary><strong>{failure.model_name}</strong><span>{failure.message}</span></summary>
              {failure.technical_error && <code>{failure.technical_error}</code>}
            </details>
          ))}
        </div>
      )}
    </section>
  );
}

function ModelsTab({ runRecord }) {
  const { metrics, decision } = runRecord;
  if (!metrics) return <Unavailable label="Model metrics" />;
  const modelComparisonChart = runRecord.report_charts?.model_comparison;
  const comparisonResults = metrics.model_comparison?.results;
  return (
    <div>
      <p className="muted">
        Evaluation metric: {metrics.eval_metric}
        {metrics.per_entity && ` — averaged across ${metrics.entities_trained} entities`}
      </p>
      <FailureSummary summary={metrics.failure_summary} explanation={metrics.failure_explanation} />
      {modelComparisonChart ? (
        <ReportChart
          uri={modelComparisonChart}
          name="model comparison"
          caption="Model comparison on the primary evaluation metric."
        />
      ) : (
        <ModelBarChart models={metrics.models} bestModel={decision?.best_model} />
      )}

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
    <div className="result-section-stack">
      {rec.flagged_for_review && (
        <div className="result-section-panel warning-panel">
          <strong>Flagged for review:</strong> {rec.flag_reason}
        </div>
      )}

      <div className="result-section-panel recommendation-hero-panel">
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
      </div>

      {rec.cited_top_features?.length > 0 && (
        <div className="result-section-panel recommendation-section-panel">
          <div className="recommendation-section">
            <h4>Top features</h4>
            <div className="chip-list">
              {rec.cited_top_features.map((f) => (
                <span className="chip" key={f}>{f}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="result-section-panel">
        <RecommendationProseSection title="Reason" text={rec.reason} />
        <RecommendationProseSection title="Performance summary" text={rec.performance_summary} />
        <RecommendationProseSection title="Comparison to alternatives" text={rec.comparison_to_alternatives} />
        <RecommendationProseSection title="How it works" text={rec.explanation_narrative} />
        <RecommendationProseSection title="Limitations" text={rec.limitations} />
      </div>
    </div>
  );
}

function BusinessInterpretationTab({ runRecord }) {
  const rec = runRecord.recommendation;
  if (!rec?.business_interpretation) return <Unavailable label="Business interpretation" />;
  return (
    <div className="result-section-panel">
      <div className="preprocessing-section-header">
        <h3>Recommended model</h3>
      </div>
      <h3 className="business-interpretation-title">{rec.recommended_model}</h3>
      <p className="business-interpretation-copy">{rec.business_interpretation}</p>
    </div>
  );
}

function ChartsTab({ runRecord }) {
  const charts = runRecord.report_charts;
  const entries = Object.entries(charts || {}).filter(
    ([name, uri]) => uri && !["model_comparison", "feature_importance", "permutation_importance", "shap_importance"].includes(name),
  );
  if (!entries.length) return <Unavailable label="Charts" />;
  return (
    <div className="result-section-panel">
      <div className="charts-grid">
        {entries.map(([name, uri]) => (
          <figure key={name}>
            <img src={uri} alt={name} />
            <figcaption className="muted">{name.replace(/_/g, " ")}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

export const REPORT_TABS = [
  { id: "overview", label: "Experiment Plan", Component: OverviewTab },
  { id: "data-quality", label: "Data Quality", Component: DataQualityTab },
  { id: "eda", label: "EDA", Component: EdaTab },
  { id: "preprocessing", label: "Preprocessing", Component: PreprocessingTab },
  { id: "models", label: "Models & Metrics", Component: ModelsTab },
  { id: "recommendation", label: "Recommendation", Component: RecommendationTab },
  { id: "business-interpretation", label: "Business Interpretation", Component: BusinessInterpretationTab },
  { id: "charts", label: "Charts", Component: ChartsTab },
];

export default function ReportTabs({ runRecord, activeTab, onTabChange, showTabs = true }) {
  const [internalActiveTab, setInternalActiveTab] = useState(REPORT_TABS[0].id);
  if (!runRecord) return null;

  const selectedTab = activeTab || internalActiveTab;
  const Active = REPORT_TABS.find((t) => t.id === selectedTab)?.Component ?? REPORT_TABS[0].Component;
  const selectTab = (tabId) => {
    setInternalActiveTab(tabId);
    onTabChange?.(tabId);
  };

  return (
    <div>
      {showTabs && (
        <div className="report-tabs">
          {REPORT_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`report-tab${selectedTab === tab.id ? " active" : ""}`}
              onClick={() => selectTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      )}
      <div className="report-tab-panel">
        <Active runRecord={runRecord} />
      </div>
    </div>
  );
}
