"""FastAPI backend for the React frontend.

Endpoints:
    POST /api/datasets        - upload a CSV/Excel file, get back columns + preview
    POST /api/runs            - start a pipeline run in the background
    GET  /api/runs/{run_id}   - poll run status/result
    GET  /api/runs/{run_id}/report - fetch the generated HTML report

Local single-user use only (see Section 9.6 of the implementation plan) -
no auth, no multi-tenancy.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from api import run_store
from data_ingestion.loader import load_dataset

load_dotenv()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Agentic AI ML Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_datasets: dict[str, str] = {}  # dataset_id -> file path


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".csv", ".xls", ".xlsx"):
        raise HTTPException(400, "Only .csv, .xls, .xlsx files are supported.")

    dataset_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{dataset_id}{suffix}"
    dest.write_bytes(await file.read())
    _datasets[dataset_id] = str(dest)

    df = load_dataset(dest)
    preview = df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")

    return {
        "dataset_id": dataset_id,
        "columns": list(df.columns),
        "row_count": int(len(df)),
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
    file_path = _datasets.get(req.dataset_id)
    if not file_path:
        raise HTTPException(404, "Unknown dataset_id. Upload the dataset again.")

    run_id = run_store.create_run(
        file_path=file_path,
        business_description=req.business_description,
        sensitive_columns=req.sensitive_columns,
        max_retries=req.max_retries,
        time_limit_s=req.time_limit_s,
    )
    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    record = run_store.get_run(run_id)
    if record is None:
        raise HTTPException(404, "Unknown run_id.")
    return record


@app.get("/api/runs/{run_id}/report", response_class=HTMLResponse)
async def get_report(run_id: str) -> str:
    record = run_store.get_run(run_id)
    if record is None:
        raise HTTPException(404, "Unknown run_id.")
    if record.get("status") != "complete":
        raise HTTPException(409, "Run is not complete yet.")
    report_path = Path(record["report_path"])
    if not report_path.exists():
        raise HTTPException(404, "Report file not found.")
    return report_path.read_text(encoding="utf-8")
