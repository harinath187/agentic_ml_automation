import { useEffect, useRef, useState } from "react";
import { cancelRun, getRun, reportUrl, startRun, uploadDataset } from "./api";
import ModelBarChart from "./components/ModelBarChart";
import PlanSummary from "./components/PlanSummary";
import RunProgress from "./components/RunProgress";

const POLL_INTERVAL_MS = 2000;

export default function App() {
  const [dataset, setDataset] = useState(null);
  const [uploadError, setUploadError] = useState("");
  const [uploading, setUploading] = useState(false);

  const [businessDescription, setBusinessDescription] = useState("");
  const [sensitiveColumns, setSensitiveColumns] = useState([]);
  const [maxRetries, setMaxRetries] = useState(2);
  const [timeLimit, setTimeLimit] = useState(60);

  const [runId, setRunId] = useState(null);
  const [runRecord, setRunRecord] = useState(null);
  const [runError, setRunError] = useState("");
  const [clarificationAnswer, setClarificationAnswer] = useState("");
  const pollRef = useRef(null);

  async function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError("");
    setDataset(null);
    setRunId(null);
    setRunRecord(null);
    try {
      const result = await uploadDataset(file);
      setDataset(result);
      setSensitiveColumns([]);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  function toggleSensitiveColumn(col) {
    setSensitiveColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  }

  async function runWithDescription(description) {
    setRunError("");
    setRunRecord(null);
    try {
      const { run_id } = await startRun({
        dataset_id: dataset.dataset_id,
        business_description: description,
        sensitive_columns: sensitiveColumns,
        max_retries: Number(maxRetries),
        time_limit_s: Number(timeLimit),
      });
      setRunId(run_id);
    } catch (err) {
      setRunError(err.message);
    }
  }

  async function handleRunPipeline() {
    await runWithDescription(businessDescription);
  }

  async function handleSubmitClarification() {
    if (!clarificationAnswer.trim()) return;
    const question = runRecord?.clarification_question || "";
    const updatedDescription = `${businessDescription}\n\nClarification - ${question} ${clarificationAnswer.trim()}`;
    setBusinessDescription(updatedDescription);
    setClarificationAnswer("");
    await runWithDescription(updatedDescription);
  }

  async function handleCancelRun() {
    if (!runId) return;
    try {
      const record = await cancelRun(runId);
      setRunRecord(record);
    } catch (err) {
      setRunError(err.message);
    }
  }

  useEffect(() => {
    if (!runId) return undefined;

    async function poll() {
      try {
        const record = await getRun(runId);
        setRunRecord(record);
        if (record.status === "queued" || record.status === "running") {
          pollRef.current = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (err) {
        setRunError(err.message);
      }
    }

    poll();
    return () => clearTimeout(pollRef.current);
  }, [runId]);

  const isActive = runRecord?.status === "queued" || runRecord?.status === "running";
  const canRun = dataset && businessDescription.trim().length > 0 && !isActive;

  return (
    <div className="page">
      <header>
        <h1>Agentic AI ML Pipeline</h1>
        <p className="subtitle">
          Upload a dataset, describe the business problem, and get a model
          recommendation with a report. Raw data never reaches the LLM - only
          schema, stats, and metrics do.
        </p>
      </header>

      <section className="card">
        <h2>1. Upload dataset</h2>
        <input type="file" accept=".csv,.xls,.xlsx" onChange={handleFileChange} disabled={uploading} />
        {uploading && <p className="muted">Uploading...</p>}
        {uploadError && <p className="error">{uploadError}</p>}

        {dataset && (
          <div className="dataset-preview">
            <p className="muted">
              {dataset.row_count} rows, {dataset.columns.length} columns
            </p>
            <table>
              <thead>
                <tr>
                  {dataset.columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataset.preview.map((row, i) => (
                  <tr key={i}>
                    {dataset.columns.map((c) => (
                      <td key={c}>{String(row[c] ?? "")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {dataset && (
        <section className="card">
          <h2>2. Describe the problem</h2>
          <textarea
            rows={3}
            placeholder="e.g. Predict which customers will churn next month so we can target retention offers."
            value={businessDescription}
            onChange={(e) => setBusinessDescription(e.target.value)}
          />

          <h3>Mark sensitive/PII columns to exclude</h3>
          <div className="checkbox-grid">
            {dataset.columns.map((col) => (
              <label key={col} className="checkbox-item">
                <input
                  type="checkbox"
                  checked={sensitiveColumns.includes(col)}
                  onChange={() => toggleSensitiveColumn(col)}
                />
                {col}
              </label>
            ))}
          </div>

          <div className="field-row">
            <label>
              Max retries
              <input
                type="number"
                min={0}
                max={5}
                value={maxRetries}
                onChange={(e) => setMaxRetries(e.target.value)}
              />
            </label>
            <label>
              AutoML time budget (seconds)
              <input
                type="number"
                min={10}
                max={600}
                value={timeLimit}
                onChange={(e) => setTimeLimit(e.target.value)}
              />
            </label>
          </div>

          <button onClick={handleRunPipeline} disabled={!canRun}>
            {isActive ? (runRecord?.status === "queued" ? "Queued..." : "Running...") : "Run Pipeline"}
          </button>
          {runError && <p className="error">{runError}</p>}
        </section>
      )}

      {runRecord?.status === "queued" && (
        <section className="card">
          <RunProgress status="queued" currentStep={null} />
          {typeof runRecord.queue_depth === "number" && (
            <p className="muted">{runRecord.queue_depth} run(s) queued/in progress.</p>
          )}
          <button onClick={handleCancelRun}>Cancel</button>
        </section>
      )}

      {runRecord?.status === "running" && (
        <section className="card">
          <RunProgress status="running" currentStep={runRecord.current_step} />
          {runRecord.started_at && <p className="muted">Started at {runRecord.started_at}.</p>}
          {runRecord.plan ? (
            <PlanSummary plan={runRecord.plan} />
          ) : (
            <p className="muted">Waiting for the Planner to decide an approach...</p>
          )}
          <button onClick={handleCancelRun}>Cancel</button>
        </section>
      )}

      {runRecord?.status === "cancelled" && (
        <section className="card warning">
          <h2>Run cancelled</h2>
          <p className="muted">This run was cancelled before it finished.</p>
        </section>
      )}

      {runRecord?.status === "needs_clarification" && (
        <section className="card warning">
          <h2>Clarification needed</h2>
          <p>{runRecord.clarification_question}</p>
          <textarea
            rows={2}
            placeholder="Type your answer here..."
            value={clarificationAnswer}
            onChange={(e) => setClarificationAnswer(e.target.value)}
          />
          <button onClick={handleSubmitClarification} disabled={!clarificationAnswer.trim()}>
            Submit answer &amp; rerun
          </button>
          {runError && <p className="error">{runError}</p>}
        </section>
      )}

      {runRecord?.status === "failed" && (
        <section className="card error-card">
          <h2>Pipeline failed</h2>
          <p>{runRecord.error}</p>
        </section>
      )}

      {runRecord?.status === "completed" && (
        <section className="card">
          <h2>3. Results</h2>

          <PlanSummary plan={runRecord.plan} />

          <p className="muted">
            Evaluation metric: {runRecord.metrics.eval_metric}
            {runRecord.metrics.per_entity && ` — averaged across ${runRecord.metrics.entities_trained} entities`}
          </p>
          <ModelBarChart models={runRecord.metrics.models} bestModel={runRecord.decision.best_model} />

          {runRecord.metrics.per_entity && (
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
                  {Object.entries(runRecord.metrics.per_entity).map(([entityId, m]) => (
                    <tr key={entityId}>
                      <td>{entityId}</td>
                      <td>{m.autogluon_best_model}</td>
                      <td>{m.models?.[m.autogluon_best_model]?.score_test ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {runRecord.metrics.entities_skipped?.length > 0 && (
                <p className="muted">
                  Skipped {runRecord.metrics.entities_skipped.length} entities:{" "}
                  {runRecord.metrics.entities_skipped.map((s) => `${s.entity} (${s.reason})`).join(", ")}
                </p>
              )}
            </div>
          )}

          <h3>Recommendation: {runRecord.decision.best_model}</h3>
          <p>{runRecord.decision.reasoning}</p>

          <h3>Report</h3>
          <div className="report-actions">
            <a href={reportUrl(runId)} target="_blank" rel="noreferrer">
              Open report in new tab
            </a>
            <a href={reportUrl(runId)} download={`report_${runId}.html`}>
              Download report
            </a>
          </div>
          <iframe className="report-frame" title="report" src={reportUrl(runId)} />
        </section>
      )}
    </div>
  );
}
