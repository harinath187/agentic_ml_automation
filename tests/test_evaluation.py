"""Tests for the deterministic Model Validation and Evaluation Engine
(tools/evaluation.py): problem-specific metrics, primary-metric selection,
and the ranking engine (winner identification, failed/missing-metric
handling, "preserve all results"). No LLM involved anywhere here.
"""
import json

import numpy as np
import pytest

from agents.schemas import ProblemType
from tools.evaluation import (
    EvaluationResult,
    ModelComparison,
    build_comparison_from_score_test,
    build_model_comparison,
    compute_classification_metrics,
    compute_forecasting_metrics,
    compute_regression_metrics,
    normalize_autogluon_metric,
    select_primary_metric,
    to_llm_summary,
)


# --- classification metrics --------------------------------------------------


def test_classification_metrics_perfect_predictions():
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 1, 1, 0]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_classification_metrics_known_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]  # 1 mistake -> accuracy 0.75
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_classification_metrics_roc_auc_only_when_binary_and_proba_given():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    without_proba = compute_classification_metrics(y_true, y_pred)
    assert "roc_auc" not in without_proba

    with_proba = compute_classification_metrics(y_true, y_pred, y_proba=[0.1, 0.9, 0.2, 0.4])
    assert "roc_auc" in with_proba
    assert 0.0 <= with_proba["roc_auc"] <= 1.0


def test_classification_metrics_no_roc_auc_for_multiclass():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred, y_proba=[0.1, 0.9, 0.5, 0.2, 0.8, 0.6])
    assert "roc_auc" not in metrics


# --- regression metrics -------------------------------------------------------


def test_regression_metrics_known_values():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(0.6123724, rel=1e-4)
    assert metrics["r2"] is not None
    assert metrics["smape"] is not None


def test_regression_metrics_mape_none_when_true_has_zero():
    y_true = [0.0, 1.0, 2.0]
    y_pred = [0.1, 1.1, 2.1]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mape"] is None  # undefined with a zero in y_true
    assert metrics["smape"] is not None  # sMAPE stays well-defined


def test_regression_metrics_mape_defined_without_zeros():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [11.0, 19.0, 33.0]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mape"] is not None
    assert metrics["mape"] > 0


# --- forecasting metrics -------------------------------------------------------


def test_forecasting_metrics_basic():
    y_true = [10.0, 11.0, 12.0]
    y_pred = [10.5, 10.5, 12.5]
    metrics = compute_forecasting_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(np.mean([0.5, 0.5, 0.5]))
    assert metrics["rmse"] is not None
    assert metrics["smape"] is not None


def test_forecasting_metrics_mase_none_without_training_series():
    metrics = compute_forecasting_metrics([10.0, 11.0], [10.5, 10.5], y_train=None)
    assert metrics["mase"] is None


def test_forecasting_metrics_mase_computed_with_training_series():
    y_train = list(range(1, 30))  # steadily increasing - naive errors are all 1
    y_true = [30.0, 31.0]
    y_pred = [30.0, 31.0]  # perfect forecast -> MASE should be 0
    metrics = compute_forecasting_metrics(y_true, y_pred, y_train=y_train, seasonal_period=1)
    assert metrics["mase"] == pytest.approx(0.0, abs=1e-9)


def test_forecasting_metrics_mase_none_for_degenerate_naive_baseline():
    # A constant training series makes the naive-baseline error 0, so scaling
    # by it is undefined - MASE must come back None, not divide-by-zero.
    y_train = [5.0] * 20
    metrics = compute_forecasting_metrics([5.0, 5.0], [5.1, 4.9], y_train=y_train, seasonal_period=1)
    assert metrics["mase"] is None


# --- AutoGluon metric sign/name normalization --------------------------------


def test_normalize_autogluon_metric_flips_sign_for_lower_is_better():
    key, value = normalize_autogluon_metric("root_mean_squared_error", -4.2)
    assert key == "rmse"
    assert value == pytest.approx(4.2)


def test_normalize_autogluon_metric_keeps_sign_for_higher_is_better():
    key, value = normalize_autogluon_metric("accuracy", 0.91)
    assert key == "accuracy"
    assert value == pytest.approx(0.91)


def test_normalize_autogluon_metric_passes_through_unknown_names():
    key, value = normalize_autogluon_metric("some_custom_metric", 0.5)
    assert key == "some_custom_metric"
    assert value == pytest.approx(0.5)


# --- primary metric selection --------------------------------------------------


def test_select_primary_metric_classification_prefers_roc_auc_when_universal():
    results = [
        EvaluationResult(model_name="a", problem_type="classification", status="success", metrics={"accuracy": 0.9, "roc_auc": 0.95}),
        EvaluationResult(model_name="b", problem_type="classification", status="success", metrics={"accuracy": 0.8, "roc_auc": 0.85}),
    ]
    metric, higher_is_better = select_primary_metric(ProblemType.CLASSIFICATION, results)
    assert metric == "roc_auc"
    assert higher_is_better is True


def test_select_primary_metric_classification_falls_back_to_accuracy_when_partial():
    results = [
        EvaluationResult(model_name="a", problem_type="classification", status="success", metrics={"accuracy": 0.9, "roc_auc": 0.95}),
        EvaluationResult(model_name="b", problem_type="classification", status="success", metrics={"accuracy": 0.8}),  # no roc_auc
    ]
    metric, _ = select_primary_metric(ProblemType.CLASSIFICATION, results)
    assert metric == "accuracy"


def test_select_primary_metric_regression_and_forecasting_use_rmse_lower_is_better():
    for problem_type in (ProblemType.REGRESSION, ProblemType.FORECASTING):
        metric, higher_is_better = select_primary_metric(problem_type, [])
        assert metric == "rmse"
        assert higher_is_better is False


def test_select_primary_metric_honors_preferred_metrics_over_default():
    # accuracy would normally lose to roc_auc under the default policy (both
    # are universally reported here), but plan.evaluation_metrics explicitly
    # prioritizes accuracy/f1 - that plan-stated priority must win.
    results = [
        EvaluationResult(model_name="a", problem_type="classification", status="success", metrics={"accuracy": 0.9, "roc_auc": 0.6}),
        EvaluationResult(model_name="b", problem_type="classification", status="success", metrics={"accuracy": 0.8, "roc_auc": 0.95}),
    ]
    metric, higher_is_better = select_primary_metric(ProblemType.CLASSIFICATION, results, preferred_metrics=["f1", "accuracy", "roc_auc"])
    assert metric == "accuracy"  # f1 is absent from every candidate, so it's skipped; accuracy is the next preferred metric present everywhere
    assert higher_is_better is True


def test_select_primary_metric_skips_preferred_metric_not_universally_available():
    results = [
        EvaluationResult(model_name="a", problem_type="classification", status="success", metrics={"accuracy": 0.9, "roc_auc": 0.6}),
        EvaluationResult(model_name="b", problem_type="classification", status="success", metrics={"accuracy": 0.8}),  # no roc_auc
    ]
    metric, _ = select_primary_metric(ProblemType.CLASSIFICATION, results, preferred_metrics=["roc_auc", "accuracy"])
    assert metric == "accuracy"  # roc_auc not usable by every candidate, falls through to the next preferred entry


def test_select_primary_metric_falls_back_to_default_when_no_preferred_metric_usable():
    results = [
        EvaluationResult(model_name="a", problem_type="classification", status="success", metrics={"accuracy": 0.9, "roc_auc": 0.95}),
        EvaluationResult(model_name="b", problem_type="classification", status="success", metrics={"accuracy": 0.8, "roc_auc": 0.85}),
    ]
    metric, _ = select_primary_metric(ProblemType.CLASSIFICATION, results, preferred_metrics=["precision_at_k"])
    assert metric == "roc_auc"  # nothing in preferred_metrics usable -> falls back to the default policy


def test_build_model_comparison_winner_changes_with_preferred_metrics():
    # Two candidates that disagree on which metric ranks them first: "a" wins
    # on accuracy, "b" wins on roc_auc. The winner must track whichever
    # metric plan.evaluation_metrics actually prioritizes, not a hardcoded
    # default - this is the regression test for the eval_metric vs
    # model_comparison.primary_metric divergence bug.
    results = [
        EvaluationResult(model_name="a", problem_type="classification", status="success", metrics={"accuracy": 0.9, "roc_auc": 0.60}),
        EvaluationResult(model_name="b", problem_type="classification", status="success", metrics={"accuracy": 0.7, "roc_auc": 0.95}),
    ]
    accuracy_first = build_model_comparison(ProblemType.CLASSIFICATION, results, preferred_metrics=["accuracy"])
    assert accuracy_first.primary_metric == "accuracy"
    assert accuracy_first.winner == "a"

    roc_auc_first = build_model_comparison(ProblemType.CLASSIFICATION, results, preferred_metrics=["roc_auc"])
    assert roc_auc_first.primary_metric == "roc_auc"
    assert roc_auc_first.winner == "b"


# --- ranking engine: build_model_comparison -----------------------------------


def test_build_model_comparison_ranks_and_identifies_winner_regression():
    results = [
        EvaluationResult(model_name="good", problem_type="regression", status="success", metrics={"rmse": 1.0}),
        EvaluationResult(model_name="bad", problem_type="regression", status="success", metrics={"rmse": 5.0}),
    ]
    comparison = build_model_comparison(ProblemType.REGRESSION, results)
    assert comparison.primary_metric == "rmse"
    assert comparison.higher_is_better is False
    assert comparison.winner == "good"
    assert comparison.ranked_model_names == ["good", "bad"]
    assert len(comparison.results) == 2  # preserve all model results


def test_build_model_comparison_ranks_and_identifies_winner_classification():
    results = [
        EvaluationResult(model_name="weaker", problem_type="classification", status="success", metrics={"accuracy": 0.7}),
        EvaluationResult(model_name="stronger", problem_type="classification", status="success", metrics={"accuracy": 0.95}),
    ]
    comparison = build_model_comparison(ProblemType.CLASSIFICATION, results)
    assert comparison.winner == "stronger"
    assert comparison.ranked_model_names == ["stronger", "weaker"]


def test_build_model_comparison_excludes_failed_models_from_ranking_but_keeps_them():
    results = [
        EvaluationResult(model_name="ok", problem_type="regression", status="success", metrics={"rmse": 2.0}),
        EvaluationResult(model_name="broken", problem_type="regression", status="failed", errors="boom"),
    ]
    comparison = build_model_comparison(ProblemType.REGRESSION, results)
    assert comparison.winner == "ok"
    assert "broken" not in comparison.ranked_model_names
    assert {r.model_name for r in comparison.results} == {"ok", "broken"}  # preserved


def test_build_model_comparison_excludes_success_with_missing_primary_metric():
    results = [
        EvaluationResult(model_name="scored", problem_type="regression", status="success", metrics={"rmse": 2.0}),
        # "success" but missing the primary metric key entirely - must not crash ranking.
        EvaluationResult(model_name="no_metric", problem_type="regression", status="success", metrics={}),
    ]
    comparison = build_model_comparison(ProblemType.REGRESSION, results)
    assert comparison.winner == "scored"
    assert "no_metric" not in comparison.ranked_model_names
    assert len(comparison.results) == 2


def test_build_model_comparison_no_winner_when_everything_failed():
    results = [
        EvaluationResult(model_name="a", problem_type="regression", status="failed", errors="x"),
        EvaluationResult(model_name="b", problem_type="regression", status="skipped_missing_dependency", errors="y"),
    ]
    comparison = build_model_comparison(ProblemType.REGRESSION, results)
    assert comparison.winner is None
    assert comparison.ranked_model_names == []
    assert len(comparison.results) == 2
    assert "no winner" in comparison.winner_reasoning.lower() or "no candidate" in comparison.winner_reasoning.lower()


def test_build_model_comparison_empty_input():
    comparison = build_model_comparison(ProblemType.CLASSIFICATION, [])
    assert comparison.winner is None
    assert comparison.results == []
    assert comparison.ranked_model_names == []


# --- fallback ranking (per_entity/hierarchical paths) -------------------------


def test_build_comparison_from_score_test_ranks_by_highest():
    models = {"a": {"score_test": 0.5, "fit_time_s": 1.0}, "b": {"score_test": 0.9, "fit_time_s": 1.0}}
    comparison = build_comparison_from_score_test(ProblemType.CLASSIFICATION, models)
    assert comparison.winner == "b"
    assert comparison.primary_metric == "score_test"


def test_build_comparison_from_score_test_handles_missing_score():
    models = {"a": {"score_test": None}, "b": {"score_test": 0.7}}
    comparison = build_comparison_from_score_test(ProblemType.REGRESSION, models)
    assert comparison.winner == "b"
    assert len(comparison.results) == 2
    a_result = next(r for r in comparison.results if r.model_name == "a")
    assert a_result.status == "failed"


def test_build_comparison_from_score_test_empty_models():
    comparison = build_comparison_from_score_test(ProblemType.REGRESSION, {})
    assert comparison.winner is None
    assert comparison.results == []


# --- to_llm_summary (Phase 9 413-fix: trimmed LLM-facing view) ---------------


def _explainability_with_n_features(n: int) -> dict:
    return {
        "model_name": "rf",
        "problem_type": "classification",
        "supported": True,
        "feature_importance_method": "impurity",
        "feature_importance": [{"feature": f"f{i}", "importance": float(n - i)} for i in range(n)],
        "permutation_importance": [],
        "shap_importance": [],
    }


def _comparison_with_many_features(n_features: int = 8) -> ModelComparison:
    results = [
        EvaluationResult(
            model_name="rf",
            problem_type="classification",
            status="success",
            validation_strategy="train_test_split",
            metrics={"accuracy": 0.9, "f1": 0.88},
            training_time=1.23,
            prediction_time=0.01,
            explainability=_explainability_with_n_features(n_features),
        ),
        EvaluationResult(
            model_name="baseline",
            problem_type="classification",
            status="failed",
            errors="boom",
        ),
    ]
    return ModelComparison(
        problem_type="classification",
        primary_metric="accuracy",
        higher_is_better=True,
        results=results,
        ranked_model_names=["rf"],
        winner="rf",
        winner_reasoning="'rf' has the highest accuracy.",
    )


def test_to_llm_summary_trims_json_size_versus_full_dump():
    comparison = _comparison_with_many_features(n_features=8)

    full_size = len(comparison.model_dump_json())
    summary_size = len(json.dumps(to_llm_summary(comparison, top_n_features=5)))

    assert summary_size < full_size


def test_to_llm_summary_preserves_winner_and_exact_metrics():
    comparison = _comparison_with_many_features(n_features=8)
    summary = to_llm_summary(comparison, top_n_features=5)

    assert summary["winner"] == "rf"
    assert summary["winner_reasoning"] == comparison.winner_reasoning

    rf_summary = next(r for r in summary["results"] if r["model_name"] == "rf")
    assert rf_summary["metrics"] == {"accuracy": 0.9, "f1": 0.88}  # unchanged/exact
    assert rf_summary["top_features"] == ["f0", "f1", "f2", "f3", "f4"]  # capped at top_n_features
    assert len(rf_summary["top_features"]) == 5 < 8  # actually trimmed, not just re-shaped

    baseline_summary = next(r for r in summary["results"] if r["model_name"] == "baseline")
    assert baseline_summary["metrics"] == {}
    assert baseline_summary["top_features"] == []


def test_to_llm_summary_does_not_mutate_original_comparison():
    comparison = _comparison_with_many_features(n_features=8)
    original_json = comparison.model_dump_json()

    to_llm_summary(comparison, top_n_features=2)

    assert comparison.model_dump_json() == original_json  # untouched, still fully available


def test_to_llm_summary_handles_missing_or_empty_explainability():
    results = [
        EvaluationResult(model_name="no_explain", problem_type="regression", status="success", metrics={"rmse": 1.0}, explainability=None),
        EvaluationResult(model_name="empty_explain", problem_type="regression", status="success", metrics={"rmse": 2.0}, explainability={}),
    ]
    comparison = ModelComparison(
        problem_type="regression", primary_metric="rmse", higher_is_better=False,
        results=results, ranked_model_names=["no_explain", "empty_explain"],
        winner="no_explain", winner_reasoning="lowest rmse",
    )

    summary = to_llm_summary(comparison)  # must not raise

    assert summary["results"][0]["top_features"] == []
    assert summary["results"][1]["top_features"] == []
