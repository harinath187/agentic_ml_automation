"""Tests for Phase 9's cooperative cancellation in orchestration/graph.py:
the _cancellable() node wrapper, PipelineCancelled, run_pipeline() recording
a "cancelled" ExperimentRecord instead of "error", and node_report naming
the output file after run_id (fixing a same-second filename collision
between two concurrent runs - see node_report's comment).
"""
from __future__ import annotations

import threading

import pytest

from orchestration import graph as graph_module
from orchestration.graph import PipelineCancelled, _cancellable
from tools import experiment_tracking


def test_cancellable_wrapper_passes_through_when_no_cancel_event():
    calls = []
    node = _cancellable(lambda state: calls.append(state) or {"ok": True})
    result = node({"cancel_event": None})
    assert result == {"ok": True}
    assert calls


def test_cancellable_wrapper_reports_node_start_before_running_inner_node():
    events = []

    def inner(state):
        events.append("inner")
        return {"ok": True}

    node = _cancellable(inner)
    result = node(
        {
            "cancel_event": None,
            "progress_callback": lambda name, update: events.append((name, update)),
        }
    )

    assert result == {"ok": True}
    assert events[0][0] == "inner"
    assert events[0][1] == {"progress_status": "started"}
    assert events[1] == "inner"


def test_cancellable_wrapper_passes_through_when_event_not_set():
    event = threading.Event()
    node = _cancellable(lambda state: {"ok": True})
    assert node({"cancel_event": event}) == {"ok": True}


def test_cancellable_wrapper_raises_when_event_is_set():
    event = threading.Event()
    event.set()
    node = _cancellable(lambda state: {"ok": True})
    with pytest.raises(PipelineCancelled):
        node({"cancel_event": event})


def test_cancellable_wrapper_never_calls_inner_node_once_cancelled():
    event = threading.Event()
    event.set()
    calls = []
    node = _cancellable(lambda state: calls.append(1))
    with pytest.raises(PipelineCancelled):
        node({"cancel_event": event})
    assert calls == []  # the wrapped node body never ran


def test_run_pipeline_records_cancelled_status_not_error(monkeypatch, tmp_path):
    """Simulates a run whose cancel_event is already set before the graph
    even starts (equivalent to a user cancelling immediately after
    queueing) - run_pipeline must raise PipelineCancelled and record status
    "cancelled", never "error"."""
    store = experiment_tracking.InMemoryExperimentStore()
    monkeypatch.setattr(experiment_tracking, "get_default_store", lambda: store)

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")

    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(PipelineCancelled):
        graph_module.run_pipeline(
            file_path=str(csv_path),
            business_description="test",
            run_id="cancel-test-run",
            cancel_event=cancel_event,
        )

    record = store.get("cancel-test-run")
    assert record is not None
    assert record.status == "cancelled"


def test_run_pipeline_uses_the_provided_run_id(monkeypatch, tmp_path):
    store = experiment_tracking.InMemoryExperimentStore()
    monkeypatch.setattr(experiment_tracking, "get_default_store", lambda: store)

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(PipelineCancelled):
        graph_module.run_pipeline(
            file_path=str(csv_path),
            business_description="test",
            run_id="my-custom-run-id",
            cancel_event=cancel_event,
        )

    assert store.get("my-custom-run-id") is not None
    assert store.get("some-other-id") is None


def test_node_report_names_output_file_after_run_id(monkeypatch):
    """Two concurrent runs finishing in the same second must not overwrite
    each other's report - the filename must be derived from run_id, not just
    a datetime string (see node_report / agents/reporter.py's OUTPUT_DIR)."""
    import agents.reporter as reporter_module
    from agents.schemas import EvaluatorDecision, ExperimentPlan, ProblemType

    captured = {}

    def fake_generate_report(**kwargs):
        captured.update(kwargs)
        return kwargs.get("output_path") or "fallback.html"

    monkeypatch.setattr(reporter_module, "generate_report", fake_generate_report)

    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="y")
    decision = EvaluatorDecision(decision="proceed", best_model="baseline", reasoning="test")
    state = {
        "run_id": "report-filename-test",
        "plan": plan,
        "metrics": {"models": {}},
        "decision": decision,
        "business_description": "test",
    }

    graph_module.node_report(state)

    assert captured["output_path"] is not None
    assert "report-filename-test" in captured["output_path"]


def test_node_report_falls_back_to_default_naming_without_run_id(monkeypatch):
    """No run_id in state (shouldn't normally happen via run_pipeline, but
    node_report must not crash if it is ever called without one) - passes
    output_path=None through, letting reporter.generate_report use its own
    timestamp-based default."""
    import agents.reporter as reporter_module
    from agents.schemas import EvaluatorDecision, ExperimentPlan, ProblemType

    captured = {}

    def fake_generate_report(**kwargs):
        captured.update(kwargs)
        return "fallback.html"

    monkeypatch.setattr(reporter_module, "generate_report", fake_generate_report)

    plan = ExperimentPlan(problem_type=ProblemType.REGRESSION, target_column="y")
    decision = EvaluatorDecision(decision="proceed", best_model="baseline", reasoning="test")
    state = {
        "plan": plan,
        "metrics": {"models": {}},
        "decision": decision,
        "business_description": "test",
    }

    result = graph_module.node_report(state)

    assert captured["output_path"] is None
    assert result["report_path"] == "fallback.html"
