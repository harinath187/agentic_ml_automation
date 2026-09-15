"""Tests for the Reporter Agent / report rendering (agents/reporter.py,
reports/template.html).

Core rule under test: every number in the rendered report must come from
the deterministic pipeline objects passed to generate_report(), never from
the LLM. The LLM (mocked here, exactly like tests/test_pipeline_integration.py
does) supplies only executive_summary/approach_narrative prose - so these
tests assert that (a) numbers that only exist in the deterministic objects
show up verbatim in the HTML, and (b) numbers/model names the mocked LLM
tries to inject where it has no business doing so do not appear.
"""
import pytest

from agents.schemas import (
    DataQualityIssue,
    DataQualityReport,
    DatasetProfile,
    EvaluatorDecision,
    ExperimentPlan,
    ProblemDefinition,
    ProblemType,
    RecommendationOutput,
    ReportContent,
    ValidationStrategy,
    ValidationStrategyType,
)
from tools.model_registry import resolve_candidates
from tools.model_runner import run_candidates
from tools.report_charts import build_report_charts

SECTION_HEADERS = [
    "1. Executive Summary",
    "2. Dataset Overview",
    "3. Data Quality Findings",
    "4. Problem Definition",
    "5. Experiment Plan",
    "6. Preprocessing Performed",
    "7. Models Evaluated",
    "8. Validation Strategy",
    "9. Model Comparison Table",
    "10. Recommended Model",
    "11. Why This Model Was Selected",
    "12. Model Explanation",
    "13. Business Interpretation",
    "14. Model Limitations",
    "15. Charts",
]


def _dataset_profile(row_count=137) -> DatasetProfile:
    return DatasetProfile(
        row_count=row_count,
        column_count=2,
        column_names=["f1", "target"],
        duplicate_row_count=3,
        duplicate_row_pct=2.19,
        numerical_columns=["f1"],
        categorical_columns=[],
        datetime_columns=[],
        boolean_columns=[],
        columns=[],
    )


def _quality_report() -> DataQualityReport:
    return DataQualityReport(
        overall_quality_score=63.5,
        issues=[DataQualityIssue(column="f1", issue_type="possible_outliers", severity="medium", detail="3 rows (2.1%) outside 1.5*IQR")],
    )


def _problem_definition() -> ProblemDefinition:
    return ProblemDefinition(detected_problem_type=ProblemType.REGRESSION, target_column="target", confidence="high", reasoning="only one candidate")


def _plan() -> ExperimentPlan:
    return ExperimentPlan(
        problem_type=ProblemType.REGRESSION,
        target_column="target",
        feature_columns=["f1"],
        candidate_model_families=["baseline", "linear_regression"],
        evaluation_metrics=["rmse"],
        validation_strategy=ValidationStrategy(strategy_type=ValidationStrategyType.TRAIN_TEST_SPLIT),
        reasoning="plan reasoning text",
    )


@pytest.fixture
def real_metrics_and_charts():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(0, 1, n)
    y = 3 * x + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"f1": x, "target": y})
    train_df, test_df = df.iloc[:80].reset_index(drop=True), df.iloc[80:].reset_index(drop=True)

    candidates, _ = resolve_candidates(ProblemType.REGRESSION, ["baseline", "linear_regression"])
    metrics, chart_data = run_candidates(candidates, train_df, test_df, "target", None, ProblemType.REGRESSION)
    charts = build_report_charts("regression", metrics.get("model_comparison"), chart_data)
    return metrics, charts


def _mock_llm(monkeypatch, executive_summary="exec summary", approach_narrative="approach narrative"):
    import agents.reporter as reporter_module

    monkeypatch.setattr(
        reporter_module,
        "call_llm_json",
        lambda *a, **k: ReportContent(executive_summary=executive_summary, approach_narrative=approach_narrative),
    )


def test_report_contains_all_fourteen_sections(monkeypatch, real_metrics_and_charts, tmp_path):
    import agents.reporter as reporter_module

    metrics, charts = real_metrics_and_charts
    _mock_llm(monkeypatch)

    winner = metrics["model_comparison"]["winner"]
    decision = EvaluatorDecision(decision="proceed", best_model=winner, reasoning="test")
    recommendation = RecommendationOutput(recommended_model=winner, reason="test", cited_metrics={})

    out_path = tmp_path / "report.html"
    reporter_module.generate_report(
        business_description="Predict something",
        plan=_plan(),
        metrics=metrics,
        decision=decision,
        dataset_profile=_dataset_profile(),
        data_quality_report=_quality_report(),
        problem_definition=_problem_definition(),
        recommendation=recommendation,
        cleaning_log={"duplicates_removed": 3, "imputation": {}, "outliers_capped": {}, "encoded_columns": []},
        feature_log={"lag_features": [], "rolling_features": [], "date_parts": [], "scaled_columns": ["f1"]},
        split_log={"method": "random_shuffle", "train_rows": 80, "test_rows": 20},
        charts=charts,
        output_path=str(out_path),
    )

    html = out_path.read_text(encoding="utf-8")
    for header in SECTION_HEADERS:
        assert header in html, f"missing section: {header}"


def test_deterministic_dataset_stats_appear_verbatim(monkeypatch, real_metrics_and_charts, tmp_path):
    """Numbers that ONLY exist in the deterministic DatasetProfile/
    DataQualityReport must appear in the rendered HTML exactly."""
    import agents.reporter as reporter_module

    metrics, charts = real_metrics_and_charts
    _mock_llm(monkeypatch)
    winner = metrics["model_comparison"]["winner"]
    decision = EvaluatorDecision(decision="proceed", best_model=winner, reasoning="test")
    recommendation = RecommendationOutput(recommended_model=winner)

    profile = _dataset_profile(row_count=137)
    quality = _quality_report()

    out_path = tmp_path / "report.html"
    reporter_module.generate_report(
        business_description="test",
        plan=_plan(),
        metrics=metrics,
        decision=decision,
        dataset_profile=profile,
        data_quality_report=quality,
        recommendation=recommendation,
        output_path=str(out_path),
    )

    html = out_path.read_text(encoding="utf-8")
    assert "137" in html  # row_count, only known to DatasetProfile
    assert "63.5" in html  # overall_quality_score, only known to DataQualityReport
    assert "2.19" in html  # duplicate_row_pct


def test_llm_cannot_inject_a_model_name_that_was_not_recommended(monkeypatch, real_metrics_and_charts, tmp_path):
    """The LLM (ReportContent) has no field for model names at all - it
    cannot smuggle a different recommendation into the report even if it
    tries to via executive_summary text; the Recommended Model section
    renders recommendation.recommended_model, which Phase 5 already
    validated deterministically.
    """
    import agents.reporter as reporter_module

    metrics, charts = real_metrics_and_charts
    winner = metrics["model_comparison"]["winner"]
    _mock_llm(monkeypatch, executive_summary="Actually you should use totally_fake_model_the_llm_invented instead")
    decision = EvaluatorDecision(decision="proceed", best_model=winner, reasoning="test")
    recommendation = RecommendationOutput(recommended_model=winner)  # already validated upstream

    out_path = tmp_path / "report.html"
    reporter_module.generate_report(
        business_description="test",
        plan=_plan(),
        metrics=metrics,
        decision=decision,
        recommendation=recommendation,
        output_path=str(out_path),
    )

    html = out_path.read_text(encoding="utf-8")
    section = html.split('id="recommended-model"')[1].split('id="why-selected"')[0]
    assert winner in section
    assert "totally_fake_model_the_llm_invented" not in section


def test_cited_metrics_in_recommendation_render_exactly(monkeypatch, real_metrics_and_charts, tmp_path):
    import agents.reporter as reporter_module

    metrics, charts = real_metrics_and_charts
    _mock_llm(monkeypatch)
    winner = metrics["model_comparison"]["winner"]
    winner_metrics = next(r["metrics"] for r in metrics["model_comparison"]["results"] if r["model_name"] == winner)
    decision = EvaluatorDecision(decision="proceed", best_model=winner, reasoning="test")
    recommendation = RecommendationOutput(recommended_model=winner, cited_metrics={"rmse": winner_metrics["rmse"]})

    out_path = tmp_path / "report.html"
    reporter_module.generate_report(
        business_description="test", plan=_plan(), metrics=metrics, decision=decision,
        recommendation=recommendation, output_path=str(out_path),
    )

    html = out_path.read_text(encoding="utf-8")
    assert f"{winner_metrics['rmse']:.4f}" in html


def test_report_handles_missing_optional_context_gracefully(monkeypatch, real_metrics_and_charts, tmp_path):
    """dataset_profile/data_quality_report/problem_definition/recommendation/
    logs/charts are all optional (e.g. genuinely absent for some scope
    strategies) - the report must still render without raising.
    """
    import agents.reporter as reporter_module

    metrics, _charts = real_metrics_and_charts
    _mock_llm(monkeypatch)
    decision = EvaluatorDecision(decision="proceed", best_model=metrics["model_comparison"]["winner"], reasoning="test")

    out_path = tmp_path / "report.html"
    path = reporter_module.generate_report(
        business_description="test",
        plan=_plan(),
        metrics=metrics,
        decision=decision,
        output_path=str(out_path),
    )

    assert path == str(out_path)
    html = out_path.read_text(encoding="utf-8")
    assert "not available for this run" in html


def test_report_handles_per_entity_style_metrics_without_model_comparison(monkeypatch, tmp_path):
    """per_entity/hierarchical scopes never populate metrics["model_comparison"]
    - the report must fall back to the pre-Phase-4 simple models table."""
    import agents.reporter as reporter_module

    _mock_llm(monkeypatch)
    metrics = {"eval_metric": "rmse", "models": {"AutoGluonBest": {"score_test": -1.2, "score_val": -1.1, "fit_time_s": 3.0}}}
    decision = EvaluatorDecision(decision="proceed", best_model="AutoGluonBest", reasoning="test")

    out_path = tmp_path / "report.html"
    reporter_module.generate_report(
        business_description="test", plan=_plan(), metrics=metrics, decision=decision, output_path=str(out_path),
    )

    html = out_path.read_text(encoding="utf-8")
    assert "AutoGluonBest" in html
    assert "not available for this run" in html or "not available for this run's scope strategy" in html


def test_flagged_for_review_shown_but_does_not_change_recommended_model(monkeypatch, real_metrics_and_charts, tmp_path):
    import agents.reporter as reporter_module

    metrics, _charts = real_metrics_and_charts
    _mock_llm(monkeypatch)
    winner = metrics["model_comparison"]["winner"]
    decision = EvaluatorDecision(decision="proceed", best_model=winner, reasoning="test")
    recommendation = RecommendationOutput(
        recommended_model=winner, flagged_for_review=True, flag_reason="dataset looked unusual"
    )

    out_path = tmp_path / "report.html"
    reporter_module.generate_report(
        business_description="test", plan=_plan(), metrics=metrics, decision=decision,
        recommendation=recommendation, output_path=str(out_path),
    )

    html = out_path.read_text(encoding="utf-8")
    assert "Flagged for review" in html
    assert "dataset looked unusual" in html
    section = html.split('id="recommended-model"')[1].split('id="why-selected"')[0]
    assert winner in section
