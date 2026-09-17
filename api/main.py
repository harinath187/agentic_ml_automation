"""FastAPI backend for the React frontend.

Endpoints:
    POST   /api/workspaces                        - create a workspace
    GET    /api/workspaces                         - list workspaces
    DELETE /api/workspaces/{workspace_id}          - delete a workspace and everything under it
    POST   /api/workspaces/{workspace_id}/projects - create a project in a workspace (name + optional description)
    GET    /api/workspaces/{workspace_id}/projects - list projects in a workspace
    GET    /api/workspaces/{workspace_id}/datasets - list datasets uploaded anywhere in a workspace
    GET    /api/workspaces/{workspace_id}/runs     - list runs started anywhere in a workspace
    PATCH  /api/projects/{project_id}              - update a project (e.g. its business description)
    DELETE /api/projects/{project_id}              - delete a project and its datasets/runs
    GET    /api/projects/{project_id}/datasets     - list datasets uploaded to a project
    GET    /api/projects/{project_id}/runs         - list runs started for a project
    POST   /api/datasets              - upload a CSV/Excel file, get back columns + preview
    GET    /api/datasets/{dataset_id}/eda - single-call EDA snapshot (stats/sample rows/correlations)
                                             for the frontend's dataset-detail modal
    POST   /api/runs                  - queue a pipeline run (Phase 9: bounded background worker pool)
    GET    /api/runs                  - list recent runs (Phase 9: persistent run storage)
    GET    /api/runs/{run_id}         - poll run status/result
    POST   /api/runs/{run_id}/cancel  - request cancellation of a queued/running run (Phase 9)
    GET    /api/runs/{run_id}/report  - fetch the generated HTML report

Workspaces/projects are a purely organizational layer on top of the flat
datasets/runs the pipeline itself works with - see api/db.py's "workspaces"/
"projects" tables. A dataset or run's workspace_id/project_id is optional
everywhere: uploading a dataset or starting a run without a project_id still
works exactly as before.

Local single-user use only (see Section 9.6 of the implementation plan) -
no auth, no multi-tenancy. Phase 9 made the state behind these endpoints
persistent (SQLite, api/db.py) and execution bounded/cancellable
(api/job_queue.py) - see README's "Production readiness" section.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from api import db, job_queue, run_store
from data_ingestion.loader import load_dataset
from tools.eda_snapshot import compute_eda_snapshot
from tools.file_cleanup import cleanup_old_files
from tools.logging_config import configure_logging, get_logger

load_dotenv()
configure_logging()
logger = get_logger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Phase 9: secure/robust upload handling - a size cap enforced while
# streaming to disk (never buffer an arbitrarily large upload fully in
# memory), on top of the existing extension allowlist.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

# Phase 9: periodic file cleanup (uploads/reports/AutoGluon artifacts) - see
# tools/file_cleanup.py. A plain daemon thread loop is enough at this scale;
# no external scheduler.
CLEANUP_INTERVAL_S = int(os.environ.get("CLEANUP_INTERVAL_S", str(6 * 3600)))

db.init_db()
_recovered = db.recover_interrupted_runs()
if _recovered:
    logger.warning("recovered_interrupted_runs", extra={"run_ids": _recovered, "count": len(_recovered)})

app = FastAPI(title="Agentic AI ML Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cleanup_loop() -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_S)
        cleanup_old_files()


@app.on_event("startup")
def _start_background_tasks() -> None:
    threading.Thread(target=_cleanup_loop, daemon=True, name="file-cleanup").start()


class WorkspaceRequest(BaseModel):
    name: str


@app.post("/api/workspaces")
async def create_workspace(req: WorkspaceRequest) -> dict:
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Workspace name is required.")
    workspace_id = uuid.uuid4().hex
    db.create_workspace(workspace_id, name)
    return db.get_workspace(workspace_id)


@app.get("/api/workspaces")
async def list_workspaces() -> dict:
    return {"workspaces": db.list_workspaces()}


def _unlink_workspace_or_project_files(datasets: list[dict], runs: list[dict]) -> None:
    """Best-effort cleanup of files on disk before their DB rows disappear -
    an uploaded dataset's stored CSV/Excel file and a completed run's
    generated HTML report. Never raises: a missing/already-deleted file is
    not a reason to fail the delete request."""
    for dataset in datasets:
        stored_path = dataset.get("stored_path")
        if stored_path:
            Path(stored_path).unlink(missing_ok=True)
    for run in runs:
        report_path = run.get("report_path")
        if report_path:
            Path(report_path).unlink(missing_ok=True)


@app.delete("/api/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str) -> dict:
    if db.get_workspace(workspace_id) is None:
        raise HTTPException(404, "Unknown workspace_id.")
    datasets = db.list_datasets(workspace_id=workspace_id)
    runs = db.list_runs(limit=1_000_000, workspace_id=workspace_id)
    _unlink_workspace_or_project_files(datasets, runs)
    db.delete_workspace(workspace_id)
    logger.info("workspace_deleted", extra={"workspace_id": workspace_id})
    return {"deleted": True, "workspace_id": workspace_id}


class ProjectRequest(BaseModel):
    name: str
    description: str = ""


@app.post("/api/workspaces/{workspace_id}/projects")
async def create_project(workspace_id: str, req: ProjectRequest) -> dict:
    if db.get_workspace(workspace_id) is None:
        raise HTTPException(404, "Unknown workspace_id.")
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Project name is required.")
    project_id = uuid.uuid4().hex
    db.create_project(project_id, workspace_id, name, description=req.description.strip())
    return db.get_project(project_id)


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str) -> dict:
    if db.get_project(project_id) is None:
        raise HTTPException(404, "Unknown project_id.")
    datasets = db.list_datasets(project_id=project_id)
    runs = db.list_runs(limit=1_000_000, project_id=project_id)
    _unlink_workspace_or_project_files(datasets, runs)
    db.delete_project(project_id)
    logger.info("project_deleted", extra={"project_id": project_id})
    return {"deleted": True, "project_id": project_id}


@app.get("/api/workspaces/{workspace_id}/projects")
async def list_projects(workspace_id: str) -> dict:
    if db.get_workspace(workspace_id) is None:
        raise HTTPException(404, "Unknown workspace_id.")
    return {"projects": db.list_projects(workspace_id=workspace_id)}


@app.get("/api/workspaces/{workspace_id}/datasets")
async def list_workspace_datasets(workspace_id: str) -> dict:
    if db.get_workspace(workspace_id) is None:
        raise HTTPException(404, "Unknown workspace_id.")
    return {"datasets": db.list_datasets(workspace_id=workspace_id)}


@app.get("/api/workspaces/{workspace_id}/runs")
async def list_workspace_runs(workspace_id: str, limit: int = Query(default=50, le=200)) -> dict:
    if db.get_workspace(workspace_id) is None:
        raise HTTPException(404, "Unknown workspace_id.")
    return {"runs": run_store.list_runs(limit=limit, workspace_id=workspace_id)}


class ProjectUpdateRequest(BaseModel):
    description: Optional[str] = None
    name: Optional[str] = None


@app.patch("/api/projects/{project_id}")
async def update_project(project_id: str, req: ProjectUpdateRequest) -> dict:
    if db.get_project(project_id) is None:
        raise HTTPException(404, "Unknown project_id.")
    fields = {k: v for k, v in req.model_dump(exclude_unset=True).items() if v is not None}
    return db.update_project(project_id, **fields)


@app.get("/api/projects/{project_id}/datasets")
async def list_project_datasets(project_id: str) -> dict:
    if db.get_project(project_id) is None:
        raise HTTPException(404, "Unknown project_id.")
    return {"datasets": db.list_datasets(project_id=project_id)}


@app.get("/api/projects/{project_id}/runs")
async def list_project_runs(project_id: str, limit: int = Query(default=50, le=200)) -> dict:
    if db.get_project(project_id) is None:
        raise HTTPException(404, "Unknown project_id.")
    return {"runs": run_store.list_runs(limit=limit, project_id=project_id)}


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...), project_id: Optional[str] = Form(default=None)) -> dict:
    project = db.get_project(project_id) if project_id else None
    if project_id and project is None:
        raise HTTPException(404, "Unknown project_id.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".csv", ".xls", ".xlsx"):
        raise HTTPException(400, "Only .csv, .xls, .xlsx files are supported.")

    dataset_id = uuid.uuid4().hex
    # dataset_id (a fresh uuid4) is the only user-controlled input that ever
    # becomes part of the filesystem path - the original client filename is
    # never used for anything but its extension and the display name we
    # store, so there's no path-traversal surface here regardless of what a
    # client sends as `filename`.
    dest = UPLOAD_DIR / f"{dataset_id}{suffix}"

    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(500, "Failed to save the uploaded file.") from exc

    try:
        df = load_dataset(dest)
    except Exception as exc:  # noqa: BLE001 - a malformed/mislabeled file must not leave an orphan upload or a 500
        dest.unlink(missing_ok=True)
        logger.warning("upload_parse_failed", extra={"dataset_id": dataset_id, "error": str(exc)})
        raise HTTPException(400, f"Could not parse this file as {suffix}: {exc}") from exc

    columns = list(df.columns)
    row_count = int(len(df))
    db.save_dataset(
        dataset_id=dataset_id,
        original_filename=file.filename or "",
        stored_path=str(dest),
        size_bytes=size,
        row_count=row_count,
        columns=columns,
        workspace_id=project["workspace_id"] if project else None,
        project_id=project_id,
    )
    logger.info("dataset_uploaded", extra={"dataset_id": dataset_id, "row_count": row_count, "size_bytes": size})

    preview = df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")
    return {
        "dataset_id": dataset_id,
        "columns": columns,
        "row_count": row_count,
        "preview": preview,
    }


@app.get("/api/datasets/{dataset_id}/eda")
async def get_dataset_eda(dataset_id: str) -> dict:
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "Unknown dataset_id. Upload the dataset again.")

    stored_path = Path(dataset["stored_path"])
    if not stored_path.exists():
        raise HTTPException(404, "The uploaded file is no longer available on disk.")

    try:
        df = load_dataset(stored_path)
    except Exception as exc:  # noqa: BLE001 - the file existed at upload time; surface a clear error, never a 500
        raise HTTPException(400, f"Could not re-read this dataset: {exc}") from exc

    snapshot = compute_eda_snapshot(df)
    snapshot["dataset_id"] = dataset_id
    snapshot["name"] = dataset["original_filename"]
    return snapshot


class RunRequest(BaseModel):
    dataset_id: str
    business_description: str
    max_retries: int = 2


@app.post("/api/runs")
async def start_run(req: RunRequest) -> dict:
    dataset = db.get_dataset(req.dataset_id)
    if not dataset:
        raise HTTPException(404, "Unknown dataset_id. Upload the dataset again.")

    try:
        run_id = run_store.create_run(
            file_path=dataset["stored_path"],
            business_description=req.business_description,
            max_retries=req.max_retries,
            dataset_id=req.dataset_id,
            workspace_id=dataset.get("workspace_id"),
            project_id=dataset.get("project_id"),
        )
    except job_queue.QueueFullError as exc:
        raise HTTPException(429, str(exc)) from exc

    return {"run_id": run_id, "status": db.QUEUED}


@app.get("/api/runs")
async def list_runs(limit: int = Query(default=50, le=200), status: Optional[str] = None) -> dict:
    return {"runs": run_store.list_runs(limit=limit, status=status)}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    record = run_store.get_run(run_id)
    if record is None:
        raise HTTPException(404, "Unknown run_id.")
    return record


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    record = run_store.cancel_run(run_id)
    if record is None:
        raise HTTPException(404, "Unknown run_id.")
    return record


@app.get("/api/runs/{run_id}/report", response_class=HTMLResponse)
async def get_report(run_id: str) -> str:
    record = run_store.get_run(run_id)
    if record is None:
        raise HTTPException(404, "Unknown run_id.")
    if record.get("status") != db.COMPLETED:
        raise HTTPException(409, "Run is not complete yet.")
    report_path = Path(record["report_path"])
    if not report_path.exists():
        raise HTTPException(404, "Report file not found.")
    return report_path.read_text(encoding="utf-8")
