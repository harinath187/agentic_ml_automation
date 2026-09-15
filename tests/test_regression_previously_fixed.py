"""Phase 10: regression tests for previously-fixed issues found earlier in
this project's development (Phase 9's production-readiness pass). Each test
documents the original defect in its docstring and reproduces the exact
conditions that triggered it, so a future change that reintroduces any of
these can't slip through silently.

tests/test_cleaning.py's `test_cleaning_preserves_{time,entity}_column_...`
also belong to this "previously fixed issues" set - a defect this Phase 10
effort found and fixed - but live there since tools/cleaning.py is their
natural home.
"""
from __future__ import annotations

import concurrent.futures as cf
import tempfile
import threading
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("autogluon.tabular")

from agents.schemas import EvaluatorDecision, ExperimentPlan, ProblemType, RecommendationOutput, ReportContent
from tools import experiment_tracking
from tools.report_charts import model_comparison_chart


# --- matplotlib pyplot global state is not thread-safe -----------------------
#
# tools/report_charts.py used to build charts with `plt.subplots()`/
# `plt.close(fig)`, which read/mutate pyplot's global current-figure
# registry. Once Phase 9 made concurrent pipeline runs possible (a bounded
# worker pool instead of one run at a time), two runs building report charts
# at the same time could corrupt each other's figure state. Fixed by
# building `Figure`/`FigureCanvasAgg` directly, which never touches that
# global registry.


def _comparison():
    return {
        "primary_metric": "rmse",
        "winner": "good",
        "results": [
            {"model_name": "good", "status": "success", "metrics": {"rmse": 1.0}},
            {"model_name": "bad", "status": "success", "metrics": {"rmse": 5.0}},
        ],
    }


def test_report_charts_are_thread_safe_under_concurrent_generation():
    def build_one(_i):
        return model_comparison_chart(_comparison())

    with cf.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(build_one, range(50)))

    assert len(results) == 50
    assert all(r is not None and r.startswith("data:image/png;base64,") for r in results)


# --- SQLite connections must not leak file handles ----------------------------
#
# tools/experiment_tracking.py's SQLiteExperimentStore originally did
# `with self._connect() as conn:` where _connect() returned a bare
# `sqlite3.connect(...)` - `with conn:` only commits/rolls back the
# transaction, it does NOT close the connection. On Windows this left the
# WAL/SHM files locked open, so a temp directory containing the DB file
# could not be cleaned up afterward (a real repro during this project's own
# development). Fixed by wrapping connect/close in a contextmanager that
# always closes the connection.


def test_sqlite_experiment_store_does_not_leak_connections_that_block_cleanup():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = experiment_tracking.SQLiteExperimentStore(Path(tmp_dir) / "experiments.db")
        for i in range(20):
            record = experiment_tracking.ExperimentRecord(
                run_id=f"r{i}", created_at=experiment_tracking.now_iso(),
                file_path="x.csv", business_description="t", max_retries=1, time_limit_s=10,
            )
            store.save(record)
        store.get("r5")
        store.list()
    # tempfile.TemporaryDirectory's __exit__ (just executed) tries to delete
    # every file in tmp_dir - on Windows this raises PermissionError if
    # anything (a leaked open sqlite3.Connection) still holds the DB/WAL/SHM
    # files open. Reaching this line at all is the assertion.


# --- concurrent pipeline runs must not overwrite each other's report --------
#
# agents/reporter.py's generate_report() used to default to
# f"report_{datetime.now():%Y%m%d_%H%M%S}.html" when no output_path was
# given. Two runs completing within the same second (only possible once
# Phase 9's bounded worker pool allowed more than one run at a time) would
# collide and one would silently overwrite the other's report file.
# orchestration/graph.py's node_report now names the file after run_id
# instead.


def test_concurrent_pipeline_runs_produce_distinct_report_files(regression_df, tmp_path):
    """Runs two real pipelines concurrently (real threads, real training -
    like two overlapping API requests would under Phase 9's worker pool) and
    asserts they land on two different report files, each containing that
    run's own run_id."""
    from orchestration.graph import run_pipeline

    csv_path = tmp_path / "regression.csv"
    regression_df.to_csv(csv_path, index=False)

    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION, target_column="price",
        candidate_model_families=["baseline", "linear_regression"],
    )

    # A single MonkeyPatch shared across both threads is fine here: every
    # patched target is a pure function of its arguments (no shared mutable
    # state), and pytest's monkeypatch fixture isn't itself thread-safe to
    # instantiate twice in the same test, so patch module attributes directly
    # via unittest.mock instead, undone manually at the end.
    import agents.evaluator as evaluator_module
    import agents.planner as planner_module
    import agents.recommender as recommender_module
    import agents.reporter as reporter_module

    def fake_evaluate(metrics, plan_, retry_count, max_retries):
        models = metrics.get("models", {})
        best = max(models.items(), key=lambda kv: kv[1]["score_test"])[0] if models else None
        return EvaluatorDecision(decision="proceed", best_model=best, reasoning="test")

    patches = [
        mock.patch.object(planner_module, "call_llm_json", lambda *a, **k: plan),
        mock.patch.object(evaluator_module, "evaluate_results", fake_evaluate),
        mock.patch.object(
            recommender_module, "call_llm_json",
            lambda *a, **k: RecommendationOutput(recommended_model="irrelevant", reason="test"),
        ),
        mock.patch.object(
            reporter_module, "call_llm_json",
            lambda *a, **k: ReportContent(executive_summary="test", approach_narrative="test"),
        ),
    ]
    for p in patches:
        p.start()
    try:
        run_ids = ["concurrent-run-a", "concurrent-run-b"]
        results = {}
        lock = threading.Lock()

        def run_one(run_id):
            result = run_pipeline(
                file_path=str(csv_path), business_description="Predict house price",
                max_retries=0, time_limit_s=15, run_id=run_id,
            )
            with lock:
                results[run_id] = result

        threads = [threading.Thread(target=run_one, args=(rid,)) for rid in run_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert set(results.keys()) == set(run_ids)
        report_paths = {rid: results[rid]["report_path"] for rid in run_ids}
        assert report_paths["concurrent-run-a"] != report_paths["concurrent-run-b"]
        for run_id, path in report_paths.items():
            assert run_id in path
            assert Path(path).exists()
            assert Path(path).read_text(encoding="utf-8").strip()  # not empty, not truncated by a collision
    finally:
        for p in patches:
            p.stop()
