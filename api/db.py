"""Phase 9: persistent storage for the API layer (datasets + runs/jobs).

Replaces api/main.py's/api/run_store.py's previous in-memory dicts
(`_datasets`, `_runs`), which were lost on every process restart - a
single-process assumption identified while auditing this codebase for
Phase 9. Backed by SQLite (stdlib `sqlite3`, no new dependency): enough
persistence and concurrent-access safety for a single-machine app at this
scale, without standing up a database server.

Two tables:
  datasets - one row per uploaded file (api/main.py's upload_dataset).
  runs     - one row per pipeline run/job (api/run_store.py), including its
             job status (Phase 9's queued/running/completed/failed/cancelled/
             needs_clarification lifecycle), timing, error/traceback, and the
             serialized pipeline result.

Every write goes through a short-lived connection opened in WAL mode with a
busy_timeout, which lets SQLite itself serialize concurrent writers from
Phase 9's worker threads instead of requiring an application-level lock.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(os.environ.get("APP_DB_PATH", "reports/app.db"))


def configure(path: str | Path) -> None:
    """Test/host hook - repoints every function in this module at a
    different SQLite file. Must be called before init_db()/any CRUD call;
    _connect() always reads the current value of DB_PATH (a plain module
    global, not a function default parameter, which Python would otherwise
    freeze at import time and never see this update)."""
    global DB_PATH
    DB_PATH = Path(path)

# Job statuses (Phase 9 requirement): queued, running, completed, failed,
# cancelled, plus needs_clarification - a legitimate terminal state distinct
# from failure that predates Phase 9 (the Planner asking a clarifying
# question) and is kept for API-contract compatibility.
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
NEEDS_CLARIFICATION = "needs_clarification"

TERMINAL_STATUSES = frozenset({COMPLETED, FAILED, CANCELLED, NEEDS_CLARIFICATION})


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                original_filename TEXT,
                stored_path TEXT NOT NULL,
                size_bytes INTEGER,
                row_count INTEGER,
                columns_json TEXT,
                uploaded_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                file_path TEXT NOT NULL,
                business_description TEXT NOT NULL,
                sensitive_columns_json TEXT NOT NULL DEFAULT '[]',
                max_retries INTEGER NOT NULL,
                time_limit_s INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                clarification_question TEXT,
                error TEXT,
                traceback TEXT,
                report_path TEXT,
                result_json TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                current_step TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
        # Migration for DBs created before current_step existed - CREATE TABLE
        # IF NOT EXISTS above is a no-op against an already-existing table.
        existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
        if "current_step" not in existing_columns:
            conn.execute("ALTER TABLE runs ADD COLUMN current_step TEXT")


# --- datasets ------------------------------------------------------------


def save_dataset(
    dataset_id: str,
    original_filename: str,
    stored_path: str,
    size_bytes: int,
    row_count: int,
    columns: list[str],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, original_filename, stored_path, size_bytes, row_count, columns_json, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                stored_path=excluded.stored_path, size_bytes=excluded.size_bytes,
                row_count=excluded.row_count, columns_json=excluded.columns_json
            """,
            (dataset_id, original_filename, stored_path, size_bytes, row_count, json.dumps(columns), now_iso()),
        )


def get_dataset(dataset_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["columns"] = json.loads(record.pop("columns_json") or "[]")
    return record


# --- runs ------------------------------------------------------------------


def create_run(
    run_id: str,
    dataset_id: Optional[str],
    file_path: str,
    business_description: str,
    sensitive_columns: list[str],
    max_retries: int,
    time_limit_s: int,
) -> None:
    """Inserts a new run row with status=queued. The caller (api/job_queue.py
    via api/run_store.py) is responsible for actually submitting the work -
    this only records that a job was accepted, so it survives a crash/
    restart between acceptance and execution (see recover_interrupted_runs())."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, dataset_id, file_path, business_description, sensitive_columns_json,
                max_retries, time_limit_s, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                dataset_id,
                file_path,
                business_description,
                json.dumps(sensitive_columns or []),
                max_retries,
                time_limit_s,
                QUEUED,
                now_iso(),
            ),
        )


def update_run(run_id: str, **fields: Any) -> None:
    """Generic partial update - only columns present in `fields` are
    touched. `result_json`/`sensitive_columns_json` are accepted as already-
    JSON-encoded strings; pass plain dicts/lists via update_run_result()
    instead if encoding is still needed."""
    if not fields:
        return
    columns = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [run_id]
    with _connect() as conn:
        conn.execute(f"UPDATE runs SET {columns} WHERE run_id = ?", values)


def update_run_result(run_id: str, result: dict) -> None:
    update_run(run_id, result_json=json.dumps(result, default=str))


def get_run(run_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return _run_row_to_dict(row) if row is not None else None


def list_runs(limit: int = 50, status: Optional[str] = None) -> list[dict]:
    query = "SELECT * FROM runs"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_run_row_to_dict(row) for row in rows]


def request_cancel(run_id: str) -> Optional[dict]:
    """Marks cancel_requested=1. If the run is still queued (never started),
    also flips status straight to cancelled here, since a queued job has no
    running worker thread to cooperatively check the flag - see
    api/job_queue.py's cancel() for the "already running" half of this."""
    with _connect() as conn:
        row = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE runs SET cancel_requested = 1 WHERE run_id = ?", (run_id,))
        if row["status"] == QUEUED:
            conn.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
                (CANCELLED, now_iso(), run_id),
            )
    return get_run(run_id)


def is_cancel_requested(run_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT cancel_requested FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return bool(row and row["cancel_requested"])


def recover_interrupted_runs() -> list[str]:
    """Startup recovery (Phase 9): a run stuck in 'queued' or 'running' when
    the process last stopped has no live worker thread to finish it - Python
    threads don't survive a restart, and Phase 9 deliberately doesn't add a
    distributed/durable task queue that could actually resume them. Marking
    them failed makes that explicit instead of leaving the API reporting a
    run as perpetually "running". Returns the run_ids that were recovered.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_id FROM runs WHERE status IN (?, ?)", (QUEUED, RUNNING)
        ).fetchall()
        run_ids = [row["run_id"] for row in rows]
        if run_ids:
            conn.executemany(
                "UPDATE runs SET status = ?, error = ?, completed_at = ? WHERE run_id = ?",
                [
                    (FAILED, "Interrupted by a server restart before it could finish.", now_iso(), run_id)
                    for run_id in run_ids
                ],
            )
    return run_ids


def _run_row_to_dict(row: sqlite3.Row) -> dict:
    record = dict(row)
    record["sensitive_columns"] = json.loads(record.pop("sensitive_columns_json") or "[]")
    result_json = record.pop("result_json", None)
    record["result"] = json.loads(result_json) if result_json else None
    record["cancel_requested"] = bool(record["cancel_requested"])
    return record
