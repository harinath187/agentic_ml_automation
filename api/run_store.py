"""In-memory store for pipeline runs, executed in background threads.

Single-process, local-use only (per the v1 deployment scope decision) - no
persistence, no multi-worker support. A run's DataFrame/pydantic objects are
converted to plain dicts before being stored, since only serializable state
is exposed over the API.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from typing import Optional

from orchestration.graph import run_pipeline

_lock = threading.Lock()
_runs: dict[str, dict] = {}


def create_run(
    file_path: str,
    business_description: str,
    sensitive_columns: list[str],
    max_retries: int,
    time_limit_s: int,
) -> str:
    run_id = uuid.uuid4().hex
    with _lock:
        _runs[run_id] = {"status": "running"}

    thread = threading.Thread(
        target=_execute,
        args=(run_id, file_path, business_description, sensitive_columns, max_retries, time_limit_s),
        daemon=True,
    )
    thread.start()
    return run_id


def get_run(run_id: str) -> Optional[dict]:
    with _lock:
        record = _runs.get(run_id)
        return dict(record) if record is not None else None


def _execute(
    run_id: str,
    file_path: str,
    business_description: str,
    sensitive_columns: list[str],
    max_retries: int,
    time_limit_s: int,
) -> None:
    try:
        result = run_pipeline(
            file_path=file_path,
            business_description=business_description,
            sensitive_columns=sensitive_columns,
            max_retries=max_retries,
            time_limit_s=time_limit_s,
        )

        if result.get("needs_clarification"):
            record = {
                "status": "needs_clarification",
                "clarification_question": result["plan"].clarification_question,
            }
        else:
            record = {
                "status": "complete",
                "plan": result["plan"].model_dump(mode="json"),
                "eda_summary": result.get("eda_summary"),
                "cleaning_log": result.get("cleaning_log"),
                "feature_log": result.get("feature_log"),
                "split_log": result.get("split_log"),
                "metrics": result["metrics"],
                "decision": result["decision"].model_dump(mode="json"),
                "report_path": result["report_path"],
            }
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        record = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}

    with _lock:
        _runs[run_id] = record
