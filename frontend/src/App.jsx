import { useEffect, useRef, useState } from "react";
import { cancelRun, getRun, startRun, uploadDataset } from "./api";
import PlanSummary from "./components/PlanSummary";
import ReportTabs from "./components/ReportTabs";
import RunProgress from "./components/RunProgress";

const POLL_INTERVAL_MS = 2000;

export default function App() {
  const [dataset, setDataset] = useState(null);
  const [uploadError, setUploadError] = useState("");
  const [uploading, setUploading] = useState(false);

  const [businessDescription, setBusinessDescription] = useState("");
  const [maxRetries, setMaxRetries] = useState(2);

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
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function runWithDescription(description) {
    setRunError("");
    setRunRecord(null);
    try {
      const { run_id } = await startRun({
        dataset_id: dataset.dataset_id,
        business_description: description,
        max_retries: Number(maxRetries),
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

          <p className="muted">
            We are using an LLM in this ML pipeline, but the data will not be
            exposed to the LLM.
          </p>

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
          <ReportTabs runRecord={runRecord} />
        </section>
      )}
    </div>
  );
}
