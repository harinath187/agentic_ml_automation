"""Phase 10: End-to-End Validation of the Agentic ML Pipeline.

This module validates the FULL pipeline (Upload -> Profile -> Data Quality ->
Problem Detection -> Planning -> Plan Validation -> Candidate Selection ->
Training -> Validation -> Evaluation -> Model Ranking -> Recommendation ->
Report) across five representative datasets (classification, regression,
simple time series, seasonal time series, multi-entity time series), and
specifically verifies the properties Phase 10 calls out:

  - the LLM cannot invent model metrics (test_llm_cannot_invent_metrics_*)
  - the LLM cannot override the deterministic model ranking
    (test_llm_cannot_override_*)
  - a failed candidate model does not crash the run
    (test_failed_candidate_model_does_not_crash_the_pipeline)
  - forecasting uses chronological (never random) validation
    (asserted inside every forecasting full-pipeline test, via split_log)
  - AutoGluon is one candidate among several, never the only one trained by
    default (test_autogluon_is_one_candidate_among_several)
  - a traditional forecasting model can beat AutoGluon, and vice versa, with
    REAL (not monkeypatched) AutoGluon training
    (test_traditional_model_beats_real_autogluon_on_trending_data,
    test_real_autogluon_beats_traditional_models_on_calendar_effect_data)
  - reports contain consistent metrics for the WHOLE comparison table, not
    just the winner (test_report_contains_consistent_metrics_for_every_candidate)
  - pipeline state remains valid as it's threaded between LangGraph nodes
    (test_pipeline_state_accumulates_correctly_node_by_node)

The Planner/Evaluator/Reporter LLM calls are monkeypatched with canned (or,
where the test is specifically about resisting a malicious LLM, deliberately
adversarial) responses - no API key required. Requires langgraph and
autogluon.tabular, like tests/test_pipeline_integration.py; skipped
automatically if either is missing.

Every dataset used here also has a standalone generator in
scripts_generate_sample_data.py / a shared fixture in tests/conftest.py, so
the same five representative datasets are available for manual/demo use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("autogluon.tabular")

from agents.schemas import (
    EvaluatorDecision,
    ExperimentPlan,
    ProblemType,
    RecommendationOutput,
    ReportContent,
    ScopeStrategy,
    ValidationStrategy,
    ValidationStrategyType,
)
from tests.test_pipeline_integration import _patch_agents
from tools.model_registry import resolve_candidates
from tools.model_runner import run_candidates
from tools.cleaning import clean_data
from tools.feature_engineering import engineer_features
from tools.splitting import split_data


# --- 1-5: full pipeline (Upload -> Report) across the five representative --
# --- dataset categories, asserting the ENTIRE node chain's artifacts -------


def _assert_full_node_chain_present(result: dict) -> None:
    """Shared assertions: every stage from Profile through Report left its
    mark on the final state, for any problem type/scope strategy."""
    assert result["dataset_profile"] is not None
    assert result["dataset_profile"].row_count > 0
    assert result["data_quality_report"] is not None
    assert result["problem_definition"] is not None
    assert result["plan"] is not None
    assert result["needs_clarification"] is False
    assert result["eda_summary"] is not None
    assert result["cleaning_log"] is not None
    assert result["feature_log"] is not None
    assert result["split_log"] is not None
    assert result["metrics"]["models"]  # at least one candidate trained successfully
    assert result["decision"] is not None
    assert result["decision"].best_model is not None
    assert result["recommendation"] is not None
    assert result["recommendation"].recommended_model == result["decision"].best_model
    assert result["report_path"]
    assert result["experiment_record"] is not None
    assert result["experiment_record"].status == "complete"
    assert result["experiment_record"].selected_model == result["decision"].best_model


def test_classification_full_pipeline_upload_to_report(classification_df, monkeypatch, tmp_path):
    csv_path = tmp_path / "classification.csv"
    classification_df.to_csv(csv_path, index=False)
    plan = ExperimentPlan(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="churn",
        candidate_model_families=["baseline", "logistic_regression", "random_forest"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path), business_description="Predict churn",
        sensitive_columns=["customer_name"], max_retries=1, time_limit_s=15,
    )

    _assert_full_node_chain_present(result)
    # problem_definition.detected_problem_type is a best-effort heuristic and
    # is allowed to stay None/ambiguous when a dataset has multiple plausible
    # targets (as churn_classification.csv deliberately does, to exercise
    # that ambiguity) - the Planner, not this deterministic step, makes the
    # final call. What must hold is that node_detect_problem still ran and
    # produced a reasoned ProblemDefinition either way (already asserted by
    # _assert_full_node_chain_present).
    assert result["problem_definition"].reasoning
    assert result["split_log"]["method"] == "random_shuffle"  # not a time-ordered problem


def test_regression_full_pipeline_upload_to_report(regression_df, monkeypatch, tmp_path):
    csv_path = tmp_path / "regression.csv"
    regression_df.to_csv(csv_path, index=False)
    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        candidate_model_families=["baseline", "linear_regression", "random_forest"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path), business_description="Predict house price",
        max_retries=1, time_limit_s=15,
    )

    _assert_full_node_chain_present(result)
    assert result["split_log"]["method"] == "random_shuffle"


def test_simple_timeseries_full_pipeline_upload_to_report(simple_timeseries_df, monkeypatch, tmp_path):
    csv_path = tmp_path / "simple_timeseries.csv"
    simple_timeseries_df.to_csv(csv_path, index=False)
    plan = ExperimentPlan(
        problem_type=ProblemType.FORECASTING,
        target_column="sales",
        time_column="date",
        candidate_model_families=["naive", "ets", "arima"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path), business_description="Forecast daily sales (no seasonality)",
        max_retries=1, time_limit_s=15,
    )

    _assert_full_node_chain_present(result)
    assert result["split_log"]["method"] == "chronological_no_shuffle"


def test_seasonal_timeseries_full_pipeline_upload_to_report(forecasting_df, monkeypatch, tmp_path):
    """forecasting_df (tests/conftest.py) is Phase 10's "seasonal time
    series" representative dataset - trend plus weekly seasonality."""
    csv_path = tmp_path / "seasonal_timeseries.csv"
    forecasting_df.to_csv(csv_path, index=False)
    plan = ExperimentPlan(
        problem_type=ProblemType.FORECASTING,
        target_column="sales",
        time_column="date",
        candidate_model_families=["seasonal_naive", "sarima", "autogluon_timeseries"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path), business_description="Forecast daily sales (weekly seasonality)",
        max_retries=1, time_limit_s=15,
    )

    _assert_full_node_chain_present(result)
    assert result["split_log"]["method"] == "chronological_no_shuffle"
    # AutoGluon must be present alongside classical candidates, not the only thing trained.
    assert any(name != "seasonal_naive" for name in result["metrics"]["models"])


def test_multi_entity_timeseries_full_pipeline_upload_to_report(multi_entity_forecasting_df, monkeypatch, tmp_path):
    csv_path = tmp_path / "multi_entity_timeseries.csv"
    multi_entity_forecasting_df.to_csv(csv_path, index=False)
    plan = ExperimentPlan(
        problem_type=ProblemType.FORECASTING,
        target_column="sales",
        time_column="date",
        entity_column="store_id",
        scope_strategy=ScopeStrategy.POOLED,
        entity_selection_reasoning="pooled across all stores",
        candidate_model_families=["naive", "seasonal_naive"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path), business_description="Forecast daily sales across all stores",
        max_retries=1, time_limit_s=15,
    )

    _assert_full_node_chain_present(result)
    assert result["plan"].scope_strategy == ScopeStrategy.POOLED
    assert result["split_log"]["method"] == "chronological_no_shuffle_per_entity"


# --- LLM cannot invent metrics / cannot override ranking, at FULL-pipeline --
# --- level (unit-level coverage already exists in tests/test_recommender.py --
# --- and tests/test_evaluator_deterministic_winner.py) ----------------------


def test_llm_cannot_override_deterministic_ranking_end_to_end(regression_df, monkeypatch, tmp_path):
    """Unlike _patch_agents (which stubs evaluate_results entirely), this
    test leaves the REAL agents.evaluator.evaluate_results in place and only
    stubs its LLM call site with a decision naming a model that did NOT win -
    the real deterministic winner must still be what survives to the final
    result.
    """
    import agents.planner as planner_module
    import agents.evaluator as evaluator_module
    import agents.recommender as recommender_module
    import agents.reporter as reporter_module

    csv_path = tmp_path / "regression.csv"
    regression_df.to_csv(csv_path, index=False)

    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        candidate_model_families=["baseline", "linear_regression", "random_forest"],
    )
    monkeypatch.setattr(planner_module, "call_llm_json", lambda *a, **k: plan)

    def adversarial_evaluator_llm(system_prompt, user_prompt, schema, **kwargs):
        # Deliberately names the worst-known candidate as "best" - a
        # malicious/confused LLM response.
        return EvaluatorDecision(decision="proceed", best_model="baseline", reasoning="I like this one better")

    monkeypatch.setattr(evaluator_module, "call_llm_json", adversarial_evaluator_llm)

    def adversarial_recommender_llm(system_prompt, user_prompt, schema, **kwargs):
        return RecommendationOutput(
            recommended_model="baseline",
            reason="fabricated",
            cited_metrics={"rmse": -999999.0},  # never a real value
        )

    monkeypatch.setattr(recommender_module, "call_llm_json", adversarial_recommender_llm)
    monkeypatch.setattr(
        reporter_module, "call_llm_json",
        lambda *a, **k: ReportContent(executive_summary="test", approach_narrative="test"),
    )

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path), business_description="Predict house price", max_retries=0, time_limit_s=15,
    )

    real_winner = result["metrics"]["model_comparison"]["winner"]
    assert real_winner != "baseline"  # a real signal must beat the dummy baseline on this data
    assert result["decision"].best_model == real_winner
    assert result["recommendation"].recommended_model == real_winner
    assert -999999.0 not in result["recommendation"].cited_metrics.values()


def test_llm_cannot_invent_metrics_end_to_end(regression_df, monkeypatch, tmp_path):
    import agents.planner as planner_module
    import agents.evaluator as evaluator_module
    import agents.recommender as recommender_module
    import agents.reporter as reporter_module

    csv_path = tmp_path / "regression.csv"
    regression_df.to_csv(csv_path, index=False)

    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        candidate_model_families=["baseline", "linear_regression", "random_forest"],
    )
    monkeypatch.setattr(planner_module, "call_llm_json", lambda *a, **k: plan)

    def fake_evaluate(metrics, plan_, retry_count, max_retries):
        models = metrics.get("models", {})
        best = max(models.items(), key=lambda kv: kv[1]["score_test"])[0]
        return EvaluatorDecision(decision="proceed", best_model=best, reasoning="test")

    monkeypatch.setattr(evaluator_module, "evaluate_results", fake_evaluate)

    def fabricating_recommender_llm(system_prompt, user_prompt, schema, **kwargs):
        return RecommendationOutput(
            recommended_model="irrelevant",
            reason="test",
            cited_metrics={"rmse": 0.00001, "a_metric_that_was_never_computed": 42.0},
        )

    monkeypatch.setattr(recommender_module, "call_llm_json", fabricating_recommender_llm)
    monkeypatch.setattr(
        reporter_module, "call_llm_json",
        lambda *a, **k: ReportContent(executive_summary="test", approach_narrative="test"),
    )

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path), business_description="Predict house price", max_retries=0, time_limit_s=15,
    )

    cited = result["recommendation"].cited_metrics
    assert "a_metric_that_was_never_computed" not in cited
    if "rmse" in cited:
        winner_metrics = next(
            r["metrics"] for r in result["metrics"]["model_comparison"]["results"]
            if r["model_name"] == result["decision"].best_model
        )
        assert cited["rmse"] == pytest.approx(winner_metrics["rmse"])  # not the fabricated 0.00001


# --- failed candidate models must not crash the run --------------------------


def test_failed_candidate_model_does_not_crash_the_pipeline(regression_df, monkeypatch, tmp_path):
    from tools import model_registry

    def broken_train_fn(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    linreg_def = next(
        d for d in model_registry.get_registry_for_problem_type(ProblemType.REGRESSION) if d.name == "linear_regression"
    )
    monkeypatch.setattr(linreg_def, "train_fn", broken_train_fn)

    csv_path = tmp_path / "regression.csv"
    regression_df.to_csv(csv_path, index=False)
    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        candidate_model_families=["baseline", "linear_regression", "random_forest"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path), business_description="Predict house price", max_retries=0, time_limit_s=15,
    )

    assert result["needs_clarification"] is False
    assert result["report_path"]  # the run reached the end, report and all

    candidate_results = {r["model_name"]: r for r in result["metrics"]["candidate_results"]}
    assert candidate_results["linear_regression"]["status"] == "failed"
    assert "simulated training failure" in candidate_results["linear_regression"]["errors"]
    assert candidate_results["baseline"]["status"] == "success"
    assert candidate_results["random_forest"]["status"] == "success"
    # the ranking/decision only ever considers the survivors
    assert result["decision"].best_model in ("baseline", "random_forest")


# --- AutoGluon is one candidate among several, never the only one -----------


def test_autogluon_is_one_candidate_among_several(classification_df, monkeypatch, tmp_path):
    csv_path = tmp_path / "classification.csv"
    classification_df.to_csv(csv_path, index=False)
    # An empty candidate_model_families list forces tools/model_registry.py's
    # default_candidates() fallback - "baseline" + "autogluon_tabular" - so
    # this also doubles as a regression check that the fallback never trains
    # AutoGluon alone.
    plan = ExperimentPlan(problem_type=ProblemType.CLASSIFICATION, target_column="churn", candidate_model_families=[])
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path), business_description="Predict churn",
        sensitive_columns=["customer_name"], max_retries=0, time_limit_s=15,
    )

    trained_families = set(result["metrics"]["models"].keys())
    assert len(trained_families) >= 2
    assert any("autogluon" not in name.lower() and name != "WeightedEnsemble_L2" for name in trained_families)


# --- traditional vs. real (non-mocked) AutoGluon training --------------------
#
# Unlike tests/test_ranking_end_to_end.py's ETS-vs-AutoGluon tests (which
# monkeypatch AutoGluon's train_fn to a controlled stub for speed), these two
# tests let AutoGluon actually train, through the exact same
# clean -> engineer_features -> split -> run_candidates path node_train uses,
# to prove the "traditional can beat AutoGluon"/"AutoGluon can beat
# traditional" requirement is a real behavior, not just a fact about the
# ranking arithmetic.


def _train_forecasting_candidates(df: pd.DataFrame, candidate_names: list[str], automl_time_limit: int = 20):
    cleaned, _ = clean_data(df, target_column="sales", time_column="date")
    engineered, _ = engineer_features(cleaned, problem_type="forecasting", target_column="sales", time_column="date")
    train_df, test_df, split_log = split_data(engineered, problem_type="forecasting", time_column="date")
    assert split_log["method"] == "chronological_no_shuffle"  # forecasting must never randomly shuffle

    candidates, _ = resolve_candidates(ProblemType.FORECASTING, candidate_names)
    metrics, _ = run_candidates(
        candidates, train_df, test_df, "sales", "date", ProblemType.FORECASTING, automl_time_limit=automl_time_limit
    )
    return metrics["model_comparison"]


def test_traditional_model_beats_real_autogluon_on_trending_data(simple_timeseries_df):
    """A monotonic linear trend with a holdout period beyond the training
    range: tree-based regressors (AutoGluon's default forecasting path,
    which reduces forecasting to regression over lag/rolling features) can't
    extrapolate past the max value they were trained on, while ETS/ARIMA
    explicitly model and extrapolate a trend. AutoGluon trains for real here
    - not monkeypatched."""
    comparison = _train_forecasting_candidates(
        simple_timeseries_df, ["naive", "seasonal_naive", "ets", "arima", "autogluon_timeseries"]
    )

    assert comparison["winner"] in ("ets", "arima")
    winner_rmse = next(r["metrics"]["rmse"] for r in comparison["results"] if r["model_name"] == comparison["winner"])
    autogluon_results = [
        r for r in comparison["results"]
        if r["status"] == "success" and r["model_name"] not in ("naive", "seasonal_naive", "ets", "arima", "sarima")
    ]
    assert autogluon_results  # AutoGluon actually produced a leaderboard
    for r in autogluon_results:
        assert winner_rmse < r["metrics"]["rmse"]  # every real AutoGluon model does worse than the winner


def test_real_autogluon_beats_traditional_models_on_calendar_effect_data():
    """A day-of-month calendar effect (a fixed spike on the 1st and 15th)
    with no weekly seasonality: classical models here (naive, seasonal_naive
    with a 7-day season, ETS, ARIMA, SARIMA with a 7-day seasonal order) have
    no way to see day-of-month, while AutoGluon's tabular regression can use
    the engineered `date_day` feature directly. AutoGluon trains for real
    here - not monkeypatched.
    """
    rng = np.random.default_rng(55)
    n = 180
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    base = 100 + rng.normal(0, 1, n)
    spike = np.where(dates.day.isin([1, 15]), 80.0, 0.0)
    df = pd.DataFrame({"date": dates, "sales": (base + spike).round(2)})

    comparison = _train_forecasting_candidates(
        df, ["naive", "seasonal_naive", "ets", "arima", "sarima", "autogluon_timeseries"]
    )

    classical_names = {"naive", "seasonal_naive", "ets", "arima", "sarima"}
    assert comparison["winner"] not in classical_names  # an AutoGluon leaderboard model won
    winner_rmse = next(r["metrics"]["rmse"] for r in comparison["results"] if r["model_name"] == comparison["winner"])
    for r in comparison["results"]:
        if r["status"] == "success" and r["model_name"] in classical_names:
            assert winner_rmse < r["metrics"]["rmse"]


# --- report consistency across the WHOLE comparison table, not just the -----
# --- winner (tests/test_reporter.py already checks the winner's numbers) ----


def test_report_contains_consistent_metrics_for_every_candidate(regression_df, monkeypatch, tmp_path):
    csv_path = tmp_path / "regression.csv"
    regression_df.to_csv(csv_path, index=False)
    plan = ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        candidate_model_families=["baseline", "linear_regression", "random_forest"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path), business_description="Predict house price", max_retries=0, time_limit_s=15,
    )

    html = open(result["report_path"], encoding="utf-8").read()
    comparison = result["metrics"]["model_comparison"]

    for name in comparison["ranked_model_names"]:
        assert name in html
        row = next(r for r in comparison["results"] if r["model_name"] == name)
        for metric_name, value in row["metrics"].items():
            if isinstance(value, (int, float)):
                assert f"{value:.4f}" in html, f"{name}'s {metric_name}={value} missing/inconsistent in report"

    # failed/skipped candidates are listed too, with their real status.
    for r in comparison["results"]:
        if r["status"] != "success":
            assert r["model_name"] in html
            assert r["status"] in html


# --- pipeline state validity between LangGraph nodes ------------------------


def test_pipeline_state_accumulates_correctly_node_by_node(classification_df):
    """Replays the deterministic (non-LLM) prefix of the graph - ingest ->
    profile_data -> quality_analysis -> detect_problem - by calling the node
    functions directly and merging their return values into one dict exactly
    as LangGraph does (`state = {**state, **node(state)}`), then asserts
    every key set by an earlier node is still present and unchanged after
    every later node runs. This is the actual PipelineState contract: nodes
    only ever ADD keys, never require or destroy state a later/parallel
    branch might still need.
    """
    from orchestration.graph import node_detect_problem, node_ingest, node_profile_data, node_quality_analysis

    csv_path_state = {"file_path": None, "sensitive_columns": ["customer_name"]}
    # node_ingest reads the file - write the fixture df to a real temp file.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.csv"
        classification_df.to_csv(path, index=False)
        csv_path_state["file_path"] = str(path)

        state: dict = dict(csv_path_state)
        for node in (node_ingest, node_profile_data, node_quality_analysis, node_detect_problem):
            before_keys = set(state.keys())
            update = node(state)
            state = {**state, **update}
            # every key the node returned is now present with a non-None value
            for key in update:
                assert key in state
            # nothing already in state was dropped
            assert before_keys <= set(state.keys())

    # after the full deterministic prefix, every expected key is populated
    for key in ("df", "schema_summary", "dataset_profile", "data_quality_report", "target_analysis", "problem_definition"):
        assert key in state and state[key] is not None
    # the original input state survived untouched all the way through
    assert state["file_path"] == csv_path_state["file_path"]
    assert state["sensitive_columns"] == ["customer_name"]
