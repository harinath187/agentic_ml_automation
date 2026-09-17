"""Tests for api/run_store.py's progress/plan surfacing: _on_progress()
(orchestration/graph.py's progress_callback) and _row_to_response() (what
GET /api/runs/{run_id} actually returns). No LLM/pipeline run happens here -
see tests/test_pipeline_integration.py for a real run_pipeline() call.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from api import db, run_store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    db.configure(tmp_path / "app.db")
    db.init_db()
    yield


class _FakePlan(BaseModel):
    problem_type: str = "classification"
    target_column: str = "churned"


def test_on_progress_always_records_current_step():
    db.create_run("r1", None, "uploads/d.csv", "desc", [], 2, 60)
    run_store._on_progress("r1", "clean", {})
    assert db.get_run("r1")["current_step"] == "clean"


def test_on_progress_persists_plan_when_present_in_updates():
    db.create_run("r1", None, "uploads/d.csv", "desc", [], 2, 60)
    run_store._on_progress("r1", "validate_plan", {"plan": _FakePlan(), "needs_clarification": False})

    row = db.get_run("r1")
    assert row["current_step"] == "validate_plan"
    assert row["plan"] == {"problem_type": "classification", "target_column": "churned"}


def test_on_progress_does_not_touch_plan_when_absent_from_updates():
    db.create_run("r1", None, "uploads/d.csv", "desc", [], 2, 60)
    run_store._on_progress("r1", "validate_plan", {"plan": _FakePlan(), "needs_clarification": False})
    run_store._on_progress("r1", "eda", {"eda_summary": {}})  # later step, no plan key

    assert db.get_run("r1")["plan"] == {"problem_type": "classification", "target_column": "churned"}


def test_row_to_response_surfaces_plan_only_while_running():
    db.create_run("r1", None, "uploads/d.csv", "desc", [], 2, 60)
    db.update_run("r1", status=db.RUNNING, started_at=db.now_iso())
    run_store._on_progress("r1", "validate_plan", {"plan": _FakePlan()})

    running_response = run_store.get_run("r1")
    assert running_response["current_step"] == "validate_plan"
    assert running_response["plan"] == {"problem_type": "classification", "target_column": "churned"}

    db.update_run("r1", status=db.QUEUED)  # plan must not leak into a non-running status shape
    assert "plan" not in run_store.get_run("r1")


def test_row_to_response_running_without_plan_yet_has_no_plan_key():
    db.create_run("r1", None, "uploads/d.csv", "desc", [], 2, 60)
    db.update_run("r1", status=db.RUNNING, started_at=db.now_iso())
    run_store._on_progress("r1", "ingest", {"df": None, "schema_summary": {}})

    response = run_store.get_run("r1")
    assert "plan" not in response
