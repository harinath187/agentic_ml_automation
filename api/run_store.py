"""Phase 9: run/job orchestration for the API layer.

Previously this module WAS the storage (an in-memory dict) and the
concurrency model (one unbounded `threading.Thread` per run) - both
single-process assumptions that don't survive a restart and don't bound
resource usage. It now delegates:
  - persistence to api/db.py (SQLite - survives a restart)
  - execution to api/job_queue.py (a bounded worker pool - survives a burst
    of simultaneous run requests without spawning unbounded threads)

create_run()/get_run() keep their original names/shapes so api/main.py barely
changes; the new cancel_run() is Phase 9's addition, and the status vocabulary
returned by get_run() is now queued/running/completed/failed/cancelled/
needs_clarification (previously running/complete/error/needs_clarification -
see README's "Known limitations" for the compatibility note).
"""
from __future__ import annotations

import os
import threading
import traceback
import uuid
from typing import Optional

import json

from api import db, job_queue
from orchestration.graph import PipelineCancelled, run_pipeline
from tools.logging_config import get_logger

logger = get_logger(__name__)

# Phase 9: resource/time limit - an outer wall-clock backstop per run, on top
# of time_limit_s (which only bounds a single AutoML .fit() call). Guards
# against anything else hanging indefinitely (a stuck LLM retry loop, a slow
# per_entity/hierarchical run with many entities) by cooperatively cancelling
# the run the same way an explicit user cancel does - see PipelineCancelled.
RUN_TIMEOUT_S = int(os.environ.get("PIPELINE_MAX_RUNTIME_S", "3600"))

_watchdogs: dict[str, threading.Timer] = {}
_watchdogs_lock = threading.Lock()


def _start_watchdog(run_id: str) -> None:
    timer = threading.Timer(RUN_TIMEOUT_S, _on_watchdog_timeout, args=(run_id,))
    timer.daemon = True
    with _watchdogs_lock:
        _watchdogs[run_id] = timer
    timer.start()


def _stop_watchdog(run_id: str) -> None:
    with _watchdogs_lock:
        timer = _watchdogs.pop(run_id, None)
    if timer is not None:
        timer.cancel()


def _on_watchdog_timeout(run_id: str) -> None:
    logger.warning("run_timeout_exceeded", extra={"run_id": run_id, "timeout_s": RUN_TIMEOUT_S})
    job_queue.get_default_queue().cancel(run_id)
    db.request_cancel(run_id)


def create_run(
    file_path: str,
    business_description: str,
    sensitive_columns: list[str],
    max_retries: int,
    time_limit_s: int,
    dataset_id: Optional[str] = None,
) -> str:
    queue = job_queue.get_default_queue()
    if not queue.can_accept():
        raise job_queue.QueueFullError(
            f"Too many runs already queued or in progress (limit {queue.max_queued}). Try again shortly."
        )

    run_id = uuid.uuid4().hex
    db.create_run(
        run_id=run_id,
        dataset_id=dataset_id,
        file_path=file_path,
        business_description=business_description,
        sensitive_columns=sensitive_columns,
        max_retries=max_retries,
        time_limit_s=time_limit_s,
    )
    logger.info("run_queued", extra={"run_id": run_id, "dataset_id": dataset_id})

    try:
        queue.submit(
            run_id,
            _execute,
            run_id,
            file_path,
            business_description,
            sensitive_columns,
            max_retries,
            time_limit_s,
        )
    except job_queue.QueueFullError:
        db.update_run(run_id, status=db.FAILED, error="Too many runs already queued/in progress.", completed_at=db.now_iso())
        raise
    _start_watchdog(run_id)
    return run_id


def cancel_run(run_id: str) -> Optional[dict]:
    """Requests cancellation. A queued (not yet started) run is cancelled
    outright; a running one is asked to stop cooperatively at its next
    pipeline step (see orchestration/graph.py's PipelineCancelled) - the
    status will still read "running" for a bit after this call returns.
    Returns the run record (post-request), or None if run_id is unknown.
    """
    row = db.get_run(run_id)
    if row is None:
        return None
    if row["status"] in db.TERMINAL_STATUSES:
        return get_run(run_id)  # already finished - nothing to cancel

    outcome = job_queue.get_default_queue().cancel(run_id)
    db.request_cancel(run_id)
    logger.info("run_cancel_requested", extra={"run_id": run_id, "outcome": outcome})
    return get_run(run_id)


def _row_to_response(row: dict) -> dict:
    record: dict = {"run_id": row["run_id"], "status": row["status"], "created_at": row["created_at"]}
    if row["status"] == db.NEEDS_CLARIFICATION:
        record["clarification_question"] = row["clarification_question"]
    elif row["status"] == db.COMPLETED:
        record.update(row["result"] or {})
        record["report_path"] = row["report_path"]
    elif row["status"] == db.FAILED:
        record["error"] = row["error"]
        record["traceback"] = row["traceback"]
    else:
        # queued/running/cancelled: no pipeline result yet, but surface
        # whatever debugging signal is available so polling the API mid-run
        # shows more than a bare status - which node it's on and how long
        # it's been running, instead of an opaque "running" with nothing
        # else to inspect in, say, the browser network tab.
        if row.get("started_at"):
            record["started_at"] = row["started_at"]
        if row["status"] == db.RUNNING and row.get("current_step"):
            record["current_step"] = row["current_step"]
        if row["status"] == db.RUNNING and row.get("plan"):
            record["plan"] = row["plan"]
        if row["status"] == db.QUEUED:
            record["queue_depth"] = job_queue.get_default_queue().depth()
    return record


def get_run(run_id: str) -> Optional[dict]:
    row = db.get_run(run_id)
    return _row_to_response(row) if row is not None else None


def list_runs(limit: int = 50, status: Optional[str] = None) -> list[dict]:
    return [_row_to_response(row) for row in db.list_runs(limit=limit, status=status)]


def _on_progress(run_id: str, node_name: str, updates: dict) -> None:
    """orchestration/graph.py's progress_callback: always records which node
    just finished, and additionally persists the ExperimentPlan the moment
    it's decided (node_validate_plan's `updates["plan"]`) - well before
    training/evaluation finish - so a client polling GET /api/runs/{run_id}
    mid-run can already show the user what the Planner decided, not just
    which step it's on.
    """
    fields: dict = {"current_step": node_name}
    plan = updates.get("plan")
    if plan is not None:
        fields["plan_json"] = json.dumps(plan.model_dump(mode="json"))
    db.update_run(run_id, **fields)


def _execute(
    run_id: str,
    file_path: str,
    business_description: str,
    sensitive_columns: list[str],
    max_retries: int,
    time_limit_s: int,
    cancel_event,
) -> None:
    db.update_run(run_id, status=db.RUNNING, started_at=db.now_iso())
    logger.info("run_started", extra={"run_id": run_id})

    try:
        _execute_inner(run_id, file_path, business_description, sensitive_columns, max_retries, time_limit_s, cancel_event)
    finally:
        _stop_watchdog(run_id)


def _execute_inner(
    run_id: str,
    file_path: str,
    business_description: str,
    sensitive_columns: list[str],
    max_retries: int,
    time_limit_s: int,
    cancel_event,
) -> None:
    try:
        result = run_pipeline(
            file_path=file_path,
            business_description=business_description,
            sensitive_columns=sensitive_columns,
            max_retries=max_retries,
            time_limit_s=time_limit_s,
            run_id=run_id,
            cancel_event=cancel_event,
            progress_callback=lambda step, updates: _on_progress(run_id, step, updates),
        )

        if result.get("needs_clarification"):
            db.update_run(
                run_id,
                status=db.NEEDS_CLARIFICATION,
                clarification_question=result["plan"].clarification_question,
                completed_at=db.now_iso(),
            )
            logger.info("run_needs_clarification", extra={"run_id": run_id})
            return

        record = {
            "plan": result["plan"].model_dump(mode="json"),
            "eda_summary": result.get("eda_summary"),
            "cleaning_log": result.get("cleaning_log"),
            "feature_log": result.get("feature_log"),
            "split_log": result.get("split_log"),
            "metrics": result["metrics"],
            "decision": result["decision"].model_dump(mode="json"),
        }
        experiment_record = result.get("experiment_record")
        if experiment_record is not None:
            record["experiment_record"] = experiment_record.model_dump(mode="json")

        db.update_run(run_id, status=db.COMPLETED, report_path=result["report_path"], completed_at=db.now_iso())
        db.update_run_result(run_id, record)
        logger.info("run_completed", extra={"run_id": run_id, "best_model": result["decision"].best_model})

    except PipelineCancelled:
        db.update_run(run_id, status=db.CANCELLED, completed_at=db.now_iso())
        logger.info("run_cancelled", extra={"run_id": run_id})
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client, never crash the worker
        db.update_run(
            run_id,
            status=db.FAILED,
            error=str(exc),
            traceback=traceback.format_exc(),
            completed_at=db.now_iso(),
        )
        logger.error("run_failed", extra={"run_id": run_id, "error": str(exc)})
