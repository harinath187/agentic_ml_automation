"""End-to-end pipeline tests across the 3 supported problem types.

The Planner/Evaluator/Reporter LLM calls are monkeypatched with canned
responses (no API key required to run these). Requires the full
requirements.txt (langgraph, autogluon.tabular) to be installed - these
tests are skipped automatically if those packages are missing.
"""
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("autogluon.tabular")

from agents.schemas import EvaluatorDecision, PipelinePlan, ProblemType, ReportContent


def _patch_agents(monkeypatch, plan: PipelinePlan):
    import agents.planner as planner_module
    import agents.evaluator as evaluator_module
    import agents.reporter as reporter_module

    monkeypatch.setattr(planner_module, "call_llm_json", lambda *a, **k: plan)

    def fake_evaluate(metrics, plan_, retry_count, max_retries):
        models = metrics.get("models", {})
        best = max(models.items(), key=lambda kv: kv[1]["score_test"])[0] if models else None
        return EvaluatorDecision(decision="proceed", best_model=best, reasoning="best test score")

    monkeypatch.setattr(evaluator_module, "evaluate_results", fake_evaluate)
    monkeypatch.setattr(
        reporter_module,
        "call_llm_json",
        lambda *a, **k: ReportContent(
            problem_summary="test",
            approach_taken="test",
            models_compared="test",
            final_recommendation="test",
            business_impact="test",
            caveats_and_limitations="test",
        ),
    )


def test_classification_pipeline_end_to_end(classification_df, monkeypatch, tmp_path):
    csv_path = tmp_path / "classification.csv"
    classification_df.to_csv(csv_path, index=False)

    plan = PipelinePlan(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="churn",
        pipeline_steps=["clean", "engineer_features", "split", "train"],
        candidate_models=["LightGBM"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path),
        business_description="Predict customer churn",
        sensitive_columns=["customer_name"],
        max_retries=1,
        time_limit_s=15,
    )

    assert result["decision"].best_model is not None
    assert result["metrics"]["models"]
    assert result["report_path"]


def test_regression_pipeline_end_to_end(regression_df, monkeypatch, tmp_path):
    csv_path = tmp_path / "regression.csv"
    regression_df.to_csv(csv_path, index=False)

    plan = PipelinePlan(
        problem_type=ProblemType.REGRESSION,
        target_column="price",
        pipeline_steps=["clean", "engineer_features", "split", "train"],
        candidate_models=["LightGBM"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path),
        business_description="Predict house price",
        max_retries=1,
        time_limit_s=15,
    )

    assert result["decision"].best_model is not None
    assert result["metrics"]["models"]


def test_forecasting_pipeline_end_to_end(forecasting_df, monkeypatch, tmp_path):
    csv_path = tmp_path / "forecasting.csv"
    forecasting_df.to_csv(csv_path, index=False)

    plan = PipelinePlan(
        problem_type=ProblemType.FORECASTING,
        target_column="sales",
        time_column="date",
        pipeline_steps=["clean", "engineer_features", "split", "train"],
        candidate_models=["LightGBM"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path),
        business_description="Forecast daily sales",
        max_retries=1,
        time_limit_s=15,
    )

    assert result["decision"].best_model is not None
    assert result["metrics"]["models"]
    assert result["split_log"]["method"] == "chronological_no_shuffle"
