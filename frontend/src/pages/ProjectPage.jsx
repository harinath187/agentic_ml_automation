import { useEffect, useRef, useState } from "react";
import { cancelRun, getRun, startRun, uploadDataset } from "../api";
import Breadcrumbs from "../components/Breadcrumbs";
import DatasetEdaModal from "../components/DatasetEdaModal";
import PlanSummary from "../components/PlanSummary";
import ReportTabs from "../components/ReportTabs";
import RunProgress from "../components/RunProgress";
import {
  currentProject,
  datasetsForProject,
  fmtBytes,
  fmtDate,
  fmtDateTime,
  runsForProject,
  useStore,
} from "../store";

const POLL_INTERVAL_MS = 2000;

function RunStatusPill({ status }) {
  const map = {
    completed: { cls: "pill-success", label: "Completed" },
    failed: { cls: "pill-running", label: "Failed" },
    cancelled: { cls: "pill-running", label: "Cancelled" },
    needs_clarification: { cls: "pill-running", label: "Needs input" },
    running: { cls: "pill-running", label: "Running…" },
    queued: { cls: "pill-running", label: "Queued…" },
  };
  const info = map[status] || { cls: "pill-running", label: status };
  return <span className={`pill ${info.cls}`}>{info.label}</span>;
}

function RunDetails({ runRecord, datasets, runError, onCancel, onSubmitClarification, clarificationAnswer, setClarificationAnswer }) {
  return (
    <div className="run-accordion-body">
      <div className="section-head">
        <h3>Run details</h3>
        <span className="hint">
          {runRecord.dataset_id ? datasets.find((dataset) => dataset.id === runRecord.dataset_id)?.name || "Dataset" : "Pipeline run"}
          {runRecord.created_at ? ` · ${fmtDateTime(runRecord.created_at)}` : ""}
        </span>
      </div>

      {runError && <p className="error">{runError}</p>}

      {runRecord.status === "queued" && (
        <>
          <RunProgress status="queued" currentStep={null} />
          {typeof runRecord.queue_depth === "number" && <p className="muted">{runRecord.queue_depth} run(s) queued/in progress.</p>}
          <button className="btn btn-outline btn-sm" onClick={onCancel}>Cancel</button>
        </>
      )}

      {runRecord.status === "running" && (
        <>
          <RunProgress status="running" currentStep={runRecord.current_step} />
          {runRecord.started_at && <p className="muted">Started at {runRecord.started_at}.</p>}
          {runRecord.plan ? <PlanSummary plan={runRecord.plan} /> : <p className="muted">Waiting for the Planner to decide an approach...</p>}
          <button className="btn btn-outline btn-sm" onClick={onCancel}>Cancel</button>
        </>
      )}

      {runRecord.status === "cancelled" && (
        <div className="card warning"><h3>Run cancelled</h3><p className="muted">This run was cancelled before it finished.</p></div>
      )}

      {runRecord.status === "needs_clarification" && (
        <div className="card warning">
          <h3>Clarification needed</h3>
          <p>{runRecord.clarification_question}</p>
          <textarea rows={2} placeholder="Type your answer here..." value={clarificationAnswer} onChange={(e) => setClarificationAnswer(e.target.value)} />
          <button className="btn btn-primary btn-sm" disabled={!clarificationAnswer.trim()} onClick={onSubmitClarification}>Submit answer &amp; rerun</button>
        </div>
      )}

      {runRecord.status === "failed" && (
        <div className="card error-card"><h3>Pipeline failed</h3><p>{runRecord.error}</p></div>
      )}

      {runRecord.status === "completed" && (
        <div className="card"><h3>Results</h3><ReportTabs runRecord={runRecord} /></div>
      )}
    </div>
  );
}

export default function ProjectPage() {
  const { state, actions } = useStore();
  const proj = currentProject(state);
  const ws = state.workspaces.find((w) => w.id === proj?.workspaceId);

  const [description, setDescription] = useState(proj ? proj.description || "" : "");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  const [previewDs, setPreviewDs] = useState(null);
  const [activeRunId, setActiveRunId] = useState(null);
  const [runRecord, setRunRecord] = useState(null);
  const [runError, setRunError] = useState("");
  const [clarificationAnswer, setClarificationAnswer] = useState("");
  const pollRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    setDescription(proj ? proj.description || "" : "");
  }, [proj?.id]);

  useEffect(() => {
    if (!activeRunId) return undefined;

    async function poll() {
      try {
        const record = await getRun(activeRunId);
        setRunRecord(record);
        actions.updateRun(activeRunId, { status: record.status });
        if (record.status === "queued" || record.status === "running") {
          pollRef.current = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (err) {
        setRunError(err.message);
      }
    }

    poll();
    return () => clearTimeout(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRunId]);

  if (!proj || !ws) {
    actions.backToProjects();
    return null;
  }

  const allDatasets = datasetsForProject(state, proj.id).slice().sort((a, b) => b.createdAt - a.createdAt);
  const runs = runsForProject(state, proj.id);
  const completedDatasetIds = new Set(
    runs.filter((run) => run.status === "completed").map((run) => run.datasetId),
  );
  const datasets = allDatasets.filter((dataset) => !completedDatasetIds.has(dataset.id));
  const hasCompletedRun = runs.some((run) => run.status === "completed");
  const hasDescription = !!description.trim().length;

  useEffect(() => {
    if (hasCompletedRun && description) {
      setDescription("");
      actions.setProjectDescription(proj.id, "");
    }
  }, [hasCompletedRun, proj.id]);

  const crumbs = [
    { label: "Workspaces", action: actions.goWorkspaces },
    { label: ws.name, action: () => actions.openWorkspace(ws.id) },
    { label: proj.name },
  ];

  function onDescriptionChange(value) {
    setDescription(value);
    actions.setProjectDescription(proj.id, value);
  }

  async function handleFile(file) {
    if (!file) return;
    setUploading(true);
    setUploadError("");
    try {
      const result = await uploadDataset(file, proj.id);
      actions.addDataset({
        id: result.dataset_id,
        workspaceId: proj.workspaceId,
        projectId: proj.id,
        name: file.name,
        rows: result.row_count,
        cols: result.columns.length,
        columns: result.columns,
        sizeBytes: file.size,
        createdAt: Date.now(),
        preview: result.preview,
      });
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function runWithDescription(datasetId, businessDescription) {
    setRunError("");
    setRunRecord(null);
    try {
      const { run_id } = await startRun({
        dataset_id: datasetId,
        business_description: businessDescription,
      });
      actions.addRun({
        id: run_id,
        workspaceId: proj.workspaceId,
        projectId: proj.id,
        datasetId,
        status: "queued",
        createdAt: Date.now(),
      });
      setActiveRunId(run_id);
    } catch (err) {
      setRunError(err.message);
    }
  }

  function selectRun(runId) {
    setRunError("");
    setRunRecord(null);
    setClarificationAnswer("");
    setActiveRunId((currentId) => (currentId === runId ? null : runId));
  }

  async function handleCancelRun() {
    if (!activeRunId) return;
    try {
      const record = await cancelRun(activeRunId);
      setRunRecord(record);
      actions.updateRun(activeRunId, { status: record.status });
    } catch (err) {
      setRunError(err.message);
    }
  }

  async function handleSubmitClarification() {
    if (!clarificationAnswer.trim() || !runRecord) return;
    const question = runRecord.clarification_question || "";
    const updatedDescription = `${description}\n\nClarification - ${question} ${clarificationAnswer.trim()}`;
    onDescriptionChange(updatedDescription);
    setClarificationAnswer("");
    await runWithDescription(runRecord.dataset_id, updatedDescription);
  }

  function isDatasetBusy(datasetId) {
    return runs.some((r) => r.datasetId === datasetId && (r.status === "queued" || r.status === "running"));
  }

  async function handleDeleteProject() {
    if (!window.confirm(`Delete project "${proj.name}"? This also deletes its datasets and runs.`)) {
      return;
    }
    try {
      await actions.deleteProject(proj.id);
    } catch (err) {
      window.alert(err.message);
    }
  }

  return (
    <>
      <Breadcrumbs parts={crumbs} />
      <div className="page-head">
        <div>
          <h1>{proj.name}</h1>
          <p>Upload a CSV data source and describe the business problem, then run the pipeline against it.</p>
        </div>
        <button className="btn btn-outline btn-sm" onClick={handleDeleteProject}>
          🗑 Delete project
        </button>
      </div>

      <div className="section">
          <div className="section-head">
            <h2>Business problem</h2>
            {/* <span className="hint">What decision or outcome should this model support?</span> */}
          </div>
          <div className="field" style={{ marginBottom: 0 }}>
            <textarea
              rows={3}
              placeholder="e.g. Predict which customers will churn next month so retention offers can be targeted."
              value={description}
              onChange={(e) => onDescriptionChange(e.target.value)}
            />
          </div>
          <p className="muted" style={{ marginTop: 8 }}>
            We are using an LLM in this ML pipeline, but the raw data will not be exposed to the LLM - only column
            names, dtypes, stats, and metrics are.
          </p>
      </div>

      <div className="section">
        <div className="section-head">
          <h2>Upload dataset</h2>
          <span className="hint">CSV file</span>
        </div>
        <div
          className={"dropzone" + (dragging ? " drag" : "")}
          onDragEnter={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setDragging(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file) handleFile(file);
          }}
        >
          <div className="ico">⬆️</div>
          <div className="title">Drag &amp; drop a CSV file here</div>
          <div className="sub">or</div>
          <button
            type="button"
            className="btn btn-outline btn-sm"
            style={{ marginTop: 10 }}
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
          >
            {uploading ? "Uploading…" : "Browse files"}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xls,.xlsx"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
              e.target.value = "";
            }}
          />
        </div>
        {uploadError && <p className="error">{uploadError}</p>}
      </div>

      {datasets.length > 0 && (
        <div className="section">
          <div className="section-head">
            {/* <h2>Datasets</h2> */}
          </div>
          <div className="list">
            {datasets.map((d) => {
              const busy = isDatasetBusy(d.id);
              return (
                <div className="list-row" key={d.id} style={{ cursor: "default" }}>
                  <div className="list-icon">🗄️</div>
                  <div className="list-main">
                    <div className="list-title" style={{ cursor: "pointer" }} onClick={() => setPreviewDs(d)}>
                      {d.name}
                    </div>
                    <div className="list-sub">
                      {d.rows.toLocaleString()} rows · {d.cols} cols · {fmtBytes(d.sizeBytes)}
                    </div>
                  </div>
                  <div className="list-side">
                    <button
                      className="btn btn-accent btn-sm"
                      disabled={busy || !hasDescription}
                      title={!hasDescription ? "Add a business problem description first" : ""}
                      onClick={() => runWithDescription(d.id, description)}
                    >
                      {busy ? "Running…" : "▶ Run pipeline"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="section">
        <div className="section-head">
          <h2>Runs</h2>
        </div>
        {runs.length ? (
          <div className="list">
            {runs.slice(0, 8).map((r) => {
              const expanded = activeRunId === r.id;
              return (
                <div className={`run-accordion${expanded ? " is-open" : ""}`} key={r.id}>
                  <button className="list-row" onClick={() => selectRun(r.id)} aria-expanded={expanded}>
                    <div className="list-icon">{expanded ? "▼" : "▶"}</div>
                    <div className="list-main">
                      <div className="list-title">{allDatasets.find((dataset) => dataset.id === r.datasetId)?.name || "Dataset"}</div>
                      <div className="list-sub">{fmtDateTime(r.createdAt)}</div>
                    </div>
                    <div className="list-side"><RunStatusPill status={r.status} /></div>
                  </button>
                  {expanded && runRecord && <RunDetails runRecord={runRecord} datasets={allDatasets} runError={runError} onCancel={handleCancelRun} onSubmitClarification={handleSubmitClarification} clarificationAnswer={clarificationAnswer} setClarificationAnswer={setClarificationAnswer} />}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="empty">No runs yet for this project.</div>
        )}
      </div>

      {previewDs && <DatasetEdaModal dataset={previewDs} onClose={() => setPreviewDs(null)} />}
    </>
  );
}
