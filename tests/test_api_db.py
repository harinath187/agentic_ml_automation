"""Tests for api/db.py (Phase 9): the SQLite-backed persistent store that
replaced api/main.py's/api/run_store.py's in-memory dicts. Each test points
db.DB_PATH at a fresh tmp_path file via db.configure() so tests never touch
the real reports/app.db.
"""
from __future__ import annotations

import pytest

from api import db


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    db.configure(tmp_path / "app.db")
    db.init_db()
    yield


# --- datasets ------------------------------------------------------------


def test_save_and_get_dataset_roundtrip():
    db.save_dataset("d1", "orig.csv", "uploads/d1.csv", size_bytes=123, row_count=10, columns=["a", "b"])
    record = db.get_dataset("d1")
    assert record["dataset_id"] == "d1"
    assert record["original_filename"] == "orig.csv"
    assert record["columns"] == ["a", "b"]
    assert record["row_count"] == 10


def test_get_dataset_missing_returns_none():
    assert db.get_dataset("missing") is None


def test_save_dataset_upsert_overwrites_existing_row():
    db.save_dataset("d1", "orig.csv", "uploads/d1.csv", size_bytes=1, row_count=1, columns=["a"])
    db.save_dataset("d1", "orig.csv", "uploads/d1_v2.csv", size_bytes=2, row_count=2, columns=["a", "b"])
    record = db.get_dataset("d1")
    assert record["stored_path"] == "uploads/d1_v2.csv"
    assert record["row_count"] == 2


# --- run lifecycle ---------------------------------------------------------


def test_create_run_starts_queued():
    db.create_run("r1", "d1", "uploads/d1.csv", "desc", ["ssn"], max_retries=2, time_limit_s=60)
    row = db.get_run("r1")
    assert row["status"] == db.QUEUED
    assert row["sensitive_columns"] == ["ssn"]
    assert row["result"] is None


def test_get_run_missing_returns_none():
    assert db.get_run("missing") is None


def test_create_run_has_no_plan_until_one_is_recorded():
    db.create_run("r1", None, "uploads/d1.csv", "desc", [], 2, 60)
    assert db.get_run("r1")["plan"] is None


def test_update_run_persists_plan_json_mid_run():
    # Mirrors api/run_store.py's _on_progress: the plan can be recorded while
    # status is still "running", well before the run completes - a client
    # polling GET /api/runs/{run_id} should be able to see it immediately.
    import json

    db.create_run("r1", None, "uploads/d1.csv", "desc", [], 2, 60)
    db.update_run("r1", status=db.RUNNING, started_at=db.now_iso())
    db.update_run("r1", plan_json=json.dumps({"problem_type": "classification", "target_column": "churned"}))

    row = db.get_run("r1")
    assert row["status"] == db.RUNNING
    assert row["plan"] == {"problem_type": "classification", "target_column": "churned"}


def test_update_run_transitions_through_lifecycle():
    db.create_run("r1", None, "uploads/d1.csv", "desc", [], 2, 60)

    db.update_run("r1", status=db.RUNNING, started_at=db.now_iso())
    assert db.get_run("r1")["status"] == db.RUNNING

    db.update_run_result("r1", {"metrics": {"a": 1}})
    db.update_run("r1", status=db.COMPLETED, report_path="reports/x.html", completed_at=db.now_iso())
    row = db.get_run("r1")
    assert row["status"] == db.COMPLETED
    assert row["result"] == {"metrics": {"a": 1}}
    assert row["report_path"] == "reports/x.html"


def test_update_run_records_error_and_traceback():
    db.create_run("r1", None, "uploads/d1.csv", "desc", [], 2, 60)
    db.update_run(
        "r1", status=db.FAILED, error="boom", traceback="Traceback...", completed_at=db.now_iso()
    )
    row = db.get_run("r1")
    assert row["status"] == db.FAILED
    assert row["error"] == "boom"
    assert row["traceback"] == "Traceback..."


def test_request_cancel_on_queued_run_flips_status_immediately():
    db.create_run("r1", None, "uploads/d1.csv", "desc", [], 2, 60)
    row = db.request_cancel("r1")
    assert row["status"] == db.CANCELLED
    assert row["cancel_requested"] is True


def test_request_cancel_on_running_run_only_sets_flag():
    db.create_run("r1", None, "uploads/d1.csv", "desc", [], 2, 60)
    db.update_run("r1", status=db.RUNNING, started_at=db.now_iso())
    row = db.request_cancel("r1")
    assert row["status"] == db.RUNNING  # unchanged - a running job must transition itself
    assert db.is_cancel_requested("r1") is True


def test_request_cancel_missing_run_returns_none():
    assert db.request_cancel("missing") is None


def test_list_runs_orders_newest_first_and_respects_limit():
    for i in range(5):
        db.create_run(f"r{i}", None, "uploads/d.csv", "desc", [], 2, 60)
    rows = db.list_runs(limit=3)
    assert len(rows) == 3
    assert [r["run_id"] for r in rows] == ["r4", "r3", "r2"]


def test_list_runs_filters_by_status():
    db.create_run("r1", None, "uploads/d.csv", "desc", [], 2, 60)
    db.create_run("r2", None, "uploads/d.csv", "desc", [], 2, 60)
    db.update_run("r2", status=db.RUNNING, started_at=db.now_iso())

    queued = db.list_runs(status=db.QUEUED)
    running = db.list_runs(status=db.RUNNING)
    assert [r["run_id"] for r in queued] == ["r1"]
    assert [r["run_id"] for r in running] == ["r2"]


def test_recover_interrupted_runs_marks_queued_and_running_as_failed():
    db.create_run("r1", None, "uploads/d.csv", "desc", [], 2, 60)  # stays queued
    db.create_run("r2", None, "uploads/d.csv", "desc", [], 2, 60)
    db.update_run("r2", status=db.RUNNING, started_at=db.now_iso())
    db.create_run("r3", None, "uploads/d.csv", "desc", [], 2, 60)
    db.update_run("r3", status=db.COMPLETED, completed_at=db.now_iso())  # already finished - untouched

    recovered = db.recover_interrupted_runs()

    assert set(recovered) == {"r1", "r2"}
    assert db.get_run("r1")["status"] == db.FAILED
    assert db.get_run("r2")["status"] == db.FAILED
    assert "restart" in db.get_run("r1")["error"]
    assert db.get_run("r3")["status"] == db.COMPLETED


def test_recover_interrupted_runs_is_noop_when_nothing_stuck():
    db.create_run("r1", None, "uploads/d.csv", "desc", [], 2, 60)
    db.update_run("r1", status=db.COMPLETED, completed_at=db.now_iso())
    assert db.recover_interrupted_runs() == []
