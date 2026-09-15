"""FastAPI backend for the React frontend.

Endpoints:
    POST   /api/datasets              - upload a CSV/Excel file, get back columns + preview
    POST   /api/runs                  - queue a pipeline run (Phase 9: bounded background worker pool)
    GET    /api/runs                  - list recent runs (Phase 9: persistent run storage)
    GET    /api/runs/{run_id}         - poll run status/result
    POST   /api/runs/{run_id}/cancel  - request cancellation of a queued/running run (Phase 9)
    GET    /api/runs/{run_id}/report  - fetch the generated HTML report

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
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from api import db, job_queue, run_store
from data_ingestion.loader import load_dataset
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


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
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
    )
    logger.info("dataset_uploaded", extra={"dataset_id": dataset_id, "row_count": row_count, "size_bytes": size})

    preview = df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")
    return {
        "dataset_id": dataset_id,
        "columns": columns,
        "row_count": row_count,
        "preview": preview,
    }


class RunRequest(BaseModel):
    dataset_id: str
    business_description: str
    sensitive_columns: list[str] = []
    max_retries: int = 2
    time_limit_s: int = 60


@app.post("/api/runs")
async def start_run(req: RunRequest) -> dict:
    dataset = db.get_dataset(req.dataset_id)
    if not dataset:
        raise HTTPException(404, "Unknown dataset_id. Upload the dataset again.")

    try:
        run_id = run_store.create_run(
            file_path=dataset["stored_path"],
            business_description=req.business_description,
            sensitive_columns=req.sensitive_columns,
            max_retries=req.max_retries,
            time_limit_s=req.time_limit_s,
            dataset_id=req.dataset_id,
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
