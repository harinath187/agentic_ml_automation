"""End-to-end pipeline tests across the 3 supported problem types.

The Planner/Evaluator/Reporter LLM calls are monkeypatched with canned
responses (no API key required to run these). Requires the full
requirements.txt (langgraph, autogluon.tabular) to be installed - these
tests are skipped automatically if those packages are missing.
"""
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("autogluon.tabular")

from agents.schemas import EvaluatorDecision, ExperimentPlan, ProblemType, RecommendationOutput, ReportContent


def _patch_agents(monkeypatch, plan: ExperimentPlan):
    import agents.planner as planner_module
    import agents.evaluator as evaluator_module
    import agents.recommender as recommender_module
    import agents.reporter as reporter_module

    monkeypatch.setattr(planner_module, "call_llm_json", lambda *a, **k: plan)

    def fake_evaluate(metrics, plan_, retry_count, max_retries):
        # Delegate winner selection to the same deterministic ranking engine
        # production uses (tools/evaluation.py, via _get_or_build_comparison) -
        # picking by raw score_test here instead would let this stub disagree
        # with node_recommend's winner (which always uses the real ranking,
        # e.g. ROC-AUC for classification rather than plain accuracy) whenever
        # the two metrics don't agree on the same candidate.
        best = evaluator_module._get_or_build_comparison(metrics, plan_).winner
        return EvaluatorDecision(decision="proceed", best_model=best, reasoning="best test score")

    monkeypatch.setattr(evaluator_module, "evaluate_results", fake_evaluate)
    monkeypatch.setattr(
        recommender_module,
        "call_llm_json",
        lambda *a, **k: RecommendationOutput(
            recommended_model="irrelevant - overridden by validate_recommendation",
            reason="test",
            performance_summary="test",
            comparison_to_alternatives="test",
            business_interpretation="test",
            limitations="test",
            confidence_statement="test",
        ),
    )
    monkeypatch.setattr(
        reporter_module,
        "call_llm_json",
        lambda *a, **k: ReportContent(executive_summary="test", approach_narrative="test"),
    )


# classification/regression end-to-end coverage lives in
# tests/test_phase10_e2e_validation.py (test_classification_full_pipeline_upload_to_report /
# test_regression_full_pipeline_upload_to_report), which asserts a strict superset of what a
# basic pipeline smoke test here would - forecasting is kept here because its LightGBM/
# feature-engineering path isn't exercised by any phase10 forecasting variant.


def test_forecasting_pipeline_end_to_end(forecasting_df, monkeypatch, tmp_path):
    csv_path = tmp_path / "forecasting.csv"
    forecasting_df.to_csv(csv_path, index=False)

    plan = ExperimentPlan(
        problem_type=ProblemType.FORECASTING,
        target_column="sales",
        time_column="date",
        pipeline_steps=["clean", "engineer_features", "split", "train"],
        candidate_model_families=["LightGBM"],
    )
    _patch_agents(monkeypatch, plan)

    from orchestration.graph import run_pipeline

    result = run_pipeline(
        file_path=str(csv_path),
        business_description="Forecast daily sales",
        max_retries=1,
    )

    assert result["decision"].best_model is not None
    assert result["metrics"]["models"]
    assert result["split_log"]["method"] == "chronological_no_shuffle"
