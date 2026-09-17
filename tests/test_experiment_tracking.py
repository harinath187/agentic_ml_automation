"""Tests for tools/experiment_tracking.py (Phase 8): record assembly and the
ExperimentStore implementations. No pipeline run happens here - see
tests/test_pipeline_integration.py for run_pipeline() producing a real
ExperimentRecord end to end.
"""
from __future__ import annotations

import random

import numpy as np

from agents.schemas import (
    DataQualityReport,
    EvaluatorDecision,
    ExperimentPlan,
    ProblemType,
    ValidationStrategy,
    ValidationStrategyType,
)
from tools import experiment_tracking as et


# --- set_global_seeds / hash_file --------------------------------------------


def test_set_global_seeds_is_deterministic():
    et.set_global_seeds(123)
    a = (random.random(), np.random.rand())
    et.set_global_seeds(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_hash_file_is_stable_for_same_content(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n")
    assert et.hash_file(f) == et.hash_file(f)


def test_hash_file_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    f1.write_text("a,b\n1,2\n")
    f2.write_text("a,b\n1,3\n")
    assert et.hash_file(f1) != et.hash_file(f2)


def test_hash_file_returns_none_for_missing_file(tmp_path):
    assert et.hash_file(tmp_path / "missing.csv") is None


def test_capture_library_versions_includes_python_and_known_installed_libs():
    versions = et.capture_library_versions()
    assert "python" in versions
    assert "pandas" in versions  # a hard (non-optional) dependency, always installed


# --- build_record -------------------------------------------------------------


def _base_kwargs(tmp_path, state: dict) -> dict:
    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b\n1,2\n")
    return dict(
        run_id="run-1",
        started_at_iso=et.now_iso(),
        started_perf=0.0,
        file_path=str(dataset),
        business_description="predict something",
        sensitive_columns=["name"],
        max_retries=2,
        time_limit_s=60,
        state=state,
    )


def test_build_record_complete_run_captures_plan_metrics_and_winner(tmp_path):
    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="y",
        validation_strategy=ValidationStrategy(strategy_type=ValidationStrategyType.TRAIN_TEST_SPLIT),
    )
    decision = EvaluatorDecision(decision="proceed", best_model="lightgbm", reasoning="best score")
    metrics = {
        "eval_metric": "rmse",
        "models": {"lightgbm": {"score_test": -1.2}},
        "candidate_results": [
            {
                "model_name": "lightgbm",
                "status": "success",
                "parameters": {"n_estimators": 100},
                "metrics": {"rmse": 1.2},
                "score_test": -1.2,
                "training_time": 0.5,
                "errors": None,
            },
            {
                "model_name": "xgboost",
                "status": "failed",
                "parameters": {},
                "metrics": {},
                "score_test": None,
                "training_time": None,
                "errors": "boom",
            },
        ],
    }
    quality_report = DataQualityReport(constant_columns=[], possible_leakage_columns=[], overall_quality_score=100.0)
    state = {
        "plan": plan, "decision": decision, "metrics": metrics, "retry_count": 0, "report_path": "reports/x.html",
        "data_quality_report": quality_report,
    }

    record = et.build_record(**_base_kwargs(tmp_path, state))

    assert record.status == "complete"
    assert record.selected_model == "lightgbm"
    assert record.experiment_plan is not None
    assert record.validation_strategy == {"strategy_type": "train_test_split", "test_size": None, "folds": None, "notes": ""}
    assert len(record.candidate_models) == 2
    assert record.candidate_models[0].model_name == "lightgbm"
    assert record.errors == ["xgboost: boom"]
    assert record.report_path == "reports/x.html"
    assert record.dataset_hash is not None
    assert record.runtime_seconds is not None and record.runtime_seconds >= 0
    # quality_report must be present and non-null for every successful run,
    # independent of whether analyze_data_quality found anything to flag -
    # this is the "was it checked" audit trail alongside feature_selection's
    # "what was dropped as a result".
    assert record.quality_report is not None
    assert record.quality_report["overall_quality_score"] == 100.0


def test_build_record_quality_report_present_even_when_nothing_flagged(tmp_path):
    quality_report = DataQualityReport()  # every list empty - the "checked and clean" case
    state = {"data_quality_report": quality_report}

    record = et.build_record(**_base_kwargs(tmp_path, state))

    assert record.quality_report is not None
    assert record.quality_report["suspicious_columns"] == []
    assert record.quality_report["possible_sentinel_missing"] == {}


def test_build_record_needs_clarification_run(tmp_path):
    plan = ExperimentPlan(needs_clarification=True, clarification_question="Which column is the target?")
    state = {"plan": plan, "needs_clarification": True}

    record = et.build_record(**_base_kwargs(tmp_path, state))

    assert record.status == "needs_clarification"
    assert record.clarification_question == "Which column is the target?"
    assert record.selected_model is None


def test_build_record_error_run_captures_exception_message(tmp_path):
    state: dict = {}
    record = et.build_record(**_base_kwargs(tmp_path, state), exception=ValueError("dataset had no rows"))

    assert record.status == "error"
    assert "dataset had no rows" in record.errors


def test_build_record_never_raises_on_missing_optional_state(tmp_path):
    # No plan/decision/metrics at all - e.g. a failure before the planner ran.
    record = et.build_record(**_base_kwargs(tmp_path, {}))
    assert record.status == "complete"
    assert record.experiment_plan is None
    assert record.candidate_models == []


# --- stores --------------------------------------------------------------------


def _sample_record(run_id: str = "run-1") -> et.ExperimentRecord:
    return et.ExperimentRecord(
        run_id=run_id,
        created_at=et.now_iso(),
        file_path="data.csv",
        business_description="test",
        max_retries=2,
        time_limit_s=60,
    )


def test_in_memory_store_save_get_list_roundtrip():
    store = et.InMemoryExperimentStore()
    store.save(_sample_record("a"))
    store.save(_sample_record("b"))

    assert store.get("a").run_id == "a"
    assert store.get("missing") is None
    assert {r.run_id for r in store.list()} == {"a", "b"}


def test_json_lines_store_persists_across_instances(tmp_path):
    path = tmp_path / "experiments.jsonl"
    store = et.JSONLinesExperimentStore(path)
    store.save(_sample_record("a"))

    reopened = et.JSONLinesExperimentStore(path)
    fetched = reopened.get("a")
    assert fetched is not None
    assert fetched.run_id == "a"


def test_json_lines_store_keeps_latest_write_per_run_id(tmp_path):
    path = tmp_path / "experiments.jsonl"
    store = et.JSONLinesExperimentStore(path)

    first = _sample_record("a")
    store.save(first)
    second = _sample_record("a")
    second.status = "complete"
    second.selected_model = "lightgbm"
    store.save(second)

    fetched = store.get("a")
    assert fetched.selected_model == "lightgbm"
    assert len(store.list()) == 1


def test_json_lines_store_skips_corrupt_lines(tmp_path):
    path = tmp_path / "experiments.jsonl"
    path.write_text("not valid json\n")
    store = et.JSONLinesExperimentStore(path)
    assert store.list() == []


def test_sqlite_store_save_get_list_roundtrip(tmp_path):
    store = et.SQLiteExperimentStore(tmp_path / "experiments.db")
    store.save(_sample_record("a"))
    store.save(_sample_record("b"))

    assert store.get("a").run_id == "a"
    assert store.get("missing") is None
    assert {r.run_id for r in store.list()} == {"a", "b"}


def test_sqlite_store_upsert_overwrites_existing_run(tmp_path):
    store = et.SQLiteExperimentStore(tmp_path / "experiments.db")
    store.save(_sample_record("a"))

    updated = _sample_record("a")
    updated.status = "complete"
    updated.selected_model = "lightgbm"
    store.save(updated)

    assert len(store.list()) == 1
    assert store.get("a").selected_model == "lightgbm"


def test_sqlite_store_persists_across_instances(tmp_path):
    path = tmp_path / "experiments.db"
    et.SQLiteExperimentStore(path).save(_sample_record("a"))

    reopened = et.SQLiteExperimentStore(path)
    fetched = reopened.get("a")
    assert fetched is not None
    assert fetched.run_id == "a"


def test_sqlite_store_handles_concurrent_writes_from_multiple_threads(tmp_path):
    import concurrent.futures as cf

    store = et.SQLiteExperimentStore(tmp_path / "experiments.db")
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda i: store.save(_sample_record(f"r{i}")), range(30)))

    assert len(store.list()) == 30


def test_default_store_can_be_swapped():
    original = et.get_default_store()
    try:
        replacement = et.InMemoryExperimentStore()
        et.set_default_store(replacement)
        assert et.get_default_store() is replacement
    finally:
        et.set_default_store(original)
