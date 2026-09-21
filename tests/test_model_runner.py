"""Tests for tools/model_runner.py: training/scoring of resolved candidates,
and - most importantly - failure handling. Every scenario here must produce
a ModelResult, never an exception, since one bad candidate must not sink an
entire training run.
"""
import numpy as np
import pandas as pd
import pytest

from agents.schemas import ProblemType, ValidationStrategyType
from tools.failure_analysis import annotate_model_failure, build_failure_summary
from tools.model_registry import ModelDefinition, is_dependency_available, resolve_candidates
from tools.model_runner import run_candidates, run_model


@pytest.fixture
def classification_train_test():
    rng = np.random.default_rng(0)
    n = 120
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    target = (f1 + f2 + rng.normal(0, 0.1, n) > 0).astype(int)
    df = pd.DataFrame({"f1": f1, "f2": f2, "target": target})
    return df.iloc[:90].reset_index(drop=True), df.iloc[90:].reset_index(drop=True)


@pytest.fixture
def forecasting_train_test():
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    values = np.linspace(100, 160, n) + 5 * np.sin(np.arange(n) * 2 * np.pi / 7)
    df = pd.DataFrame({"date": dates, "sales": values})
    return df.iloc[:45].reset_index(drop=True), df.iloc[45:].reset_index(drop=True)


# --- successful training ------------------------------------------------------


def test_run_model_baseline_classifier_succeeds(classification_train_test):
    train_df, test_df = classification_train_test
    (baseline,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])

    result = run_model(baseline, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert result.status == "success"
    assert result.eval_metric == "accuracy"


def test_run_model_classifier_preserves_string_target_preview():
    train_df = pd.DataFrame({"feature": [0, 1, 0, 1], "target": ["ham", "spam", "ham", "spam"]})
    test_df = pd.DataFrame({"feature": [0, 1], "target": ["ham", "spam"]})
    (baseline,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])

    result = run_model(baseline, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert result.status == "success"
    assert set(result.prediction).issubset({"ham", "spam"})


def test_run_model_uses_tuned_parameters(classification_train_test):
    train_df, test_df = classification_train_test
    (logistic,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["logistic_regression"])

    result = run_model(
        logistic,
        train_df,
        test_df,
        "target",
        None,
        ProblemType.CLASSIFICATION,
        tuned_params={"C": 0.1, "class_weight": "balanced"},
    )

    assert result.status == "success"
    assert result.parameters["C"] == 0.1
    assert result.parameters["class_weight"] == "balanced"
    assert result.score_test is not None
    assert 0.0 <= result.score_test <= 1.0
    assert result.training_time is not None


def test_run_model_better_classifier_beats_baseline(classification_train_test):
    train_df, test_df = classification_train_test
    (baseline,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])
    (logreg,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["logistic_regression"])

    baseline_result = run_model(baseline, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)
    logreg_result = run_model(logreg, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert logreg_result.score_test >= baseline_result.score_test


def test_run_model_forecasting_baseline_succeeds(forecasting_train_test):
    train_df, test_df = forecasting_train_test
    (naive,), _ = resolve_candidates(ProblemType.FORECASTING, ["naive"])

    result = run_model(naive, train_df, test_df, "sales", "date", ProblemType.FORECASTING)

    assert result.status == "success"
    assert result.eval_metric == "root_mean_squared_error"
    assert result.score_test is not None
    assert result.score_test <= 0  # negative RMSE convention (0 = perfect)


def test_run_model_seasonal_naive_beats_naive_on_seasonal_data():
    # No trend, strong weekly seasonality - seasonal_naive should clearly
    # beat naive here (unlike forecasting_train_test, which is trend-
    # dominated and would favor naive instead).
    n = 63
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    values = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 7)
    df = pd.DataFrame({"date": dates, "sales": values})
    train_df, test_df = df.iloc[:49].reset_index(drop=True), df.iloc[49:].reset_index(drop=True)

    (naive,), _ = resolve_candidates(ProblemType.FORECASTING, ["naive"])
    (seasonal,), _ = resolve_candidates(ProblemType.FORECASTING, ["seasonal_naive"])

    naive_result = run_model(naive, train_df, test_df, "sales", "date", ProblemType.FORECASTING)
    seasonal_result = run_model(seasonal, train_df, test_df, "sales", "date", ProblemType.FORECASTING)

    assert seasonal_result.score_test >= naive_result.score_test


# --- failure handling ---------------------------------------------------------


def test_run_model_missing_dependency_is_skipped_not_raised(classification_train_test):
    train_df, test_df = classification_train_test
    fake_definition = ModelDefinition(
        name="fake_model",
        problem_types=(ProblemType.CLASSIFICATION,),
        model_family="gradient_boosting",
        train_fn=lambda *a, **k: None,
        predict_fn=lambda *a, **k: None,
        required_dependencies=("definitely_not_a_real_package_xyz",),
    )

    result = run_model(fake_definition, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert result.status == "skipped_missing_dependency"
    assert "definitely_not_a_real_package_xyz" in result.errors


def test_failed_model_has_deterministic_failure_category(classification_train_test):
    train_df, test_df = classification_train_test
    fake_definition = ModelDefinition(
        name="fake_model",
        problem_types=(ProblemType.CLASSIFICATION,),
        model_family="tree",
        train_fn=lambda *a, **k: (_ for _ in ()).throw(ValueError("invalid feature matrix")),
        predict_fn=lambda *a, **k: None,
    )

    result = run_model(fake_definition, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)
    annotate_model_failure(result)

    assert result.status == "failed"
    assert result.failure_category == "invalid_input"
    assert result.failure_message


def test_failure_summary_keeps_model_and_dataset_issues_separate():
    metrics = {
        "candidate_results": [
            {
                "model_name": "lightgbm",
                "status": "skipped_missing_dependency",
                "errors": "Missing dependencies: lightgbm",
            }
        ]
    }
    summary = build_failure_summary(metrics, None, None, None)

    assert summary["model_failures"][0]["category"] == "missing_dependency"
    assert summary["dataset_issues"] == []


def test_run_model_unsupported_validation_strategy_is_skipped(classification_train_test):
    train_df, test_df = classification_train_test
    (baseline,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])

    result = run_model(
        baseline, train_df, test_df, "target", None, ProblemType.CLASSIFICATION,
        validation_strategy=ValidationStrategyType.TIME_SERIES_SPLIT,
    )

    assert result.status == "skipped_unsupported_validation_strategy"
    assert result.score_test is None


def test_run_model_k_fold_populates_score_val_and_is_labeled_k_fold(classification_train_test):
    """Regression coverage: plan.validation_strategy=k_fold must actually run
    cross-validation (score_val populated) and be labeled honestly - not just
    stamped onto a result that only ever saw a single train/test split."""
    train_df, test_df = classification_train_test  # 90 train rows: enough for 5 folds
    (baseline,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])

    result = run_model(
        baseline, train_df, test_df, "target", None, ProblemType.CLASSIFICATION,
        validation_strategy=ValidationStrategyType.STRATIFIED_K_FOLD,
    )

    assert result.status == "success"
    assert result.score_val is not None
    assert result.artifacts.get("validation_strategy_executed") == "stratified_k_fold"
    # score_test (the untouched holdout) must still be produced independently of CV.
    assert result.score_test is not None


def test_run_model_k_fold_skipped_when_too_few_rows_for_requested_folds(classification_train_test):
    """With more folds requested than the guard allows for this train_df size,
    CV must be skipped (score_val stays None) rather than crashing or lying
    about which strategy actually ran."""
    train_df, test_df = classification_train_test  # 90 rows
    (baseline,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])

    result = run_model(
        baseline, train_df, test_df, "target", None, ProblemType.CLASSIFICATION,
        validation_strategy=ValidationStrategyType.K_FOLD,
        validation_folds=20,  # would need >= 200 rows per the MIN_ROWS_PER_FOLD guard
    )

    assert result.status == "success"
    assert result.score_val is None
    assert "validation_strategy_executed" not in result.artifacts


def test_run_model_training_exception_becomes_failed_result(classification_train_test):
    train_df, test_df = classification_train_test

    def _raises(*args, **kwargs):
        raise RuntimeError("boom")

    broken_definition = ModelDefinition(
        name="broken_model",
        problem_types=(ProblemType.CLASSIFICATION,),
        model_family="baseline",
        train_fn=_raises,
        predict_fn=lambda *a, **k: None,
    )

    result = run_model(broken_definition, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert result.status == "failed"
    assert "boom" in result.errors
    assert result.score_test is None


def test_run_model_prediction_exception_becomes_failed_result(classification_train_test):
    train_df, test_df = classification_train_test

    def _predict_raises(*args, **kwargs):
        raise ValueError("predict exploded")

    broken_definition = ModelDefinition(
        name="broken_predict_model",
        problem_types=(ProblemType.CLASSIFICATION,),
        model_family="baseline",
        train_fn=lambda *a, **k: {"model": None},
        predict_fn=_predict_raises,
    )

    result = run_model(broken_definition, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert result.status == "failed"
    assert "predict exploded" in result.errors


def test_run_model_scoring_failure_becomes_failed_result(classification_train_test):
    train_df, test_df = classification_train_test

    # predict_fn returns a wrong-length array so scoring blows up.
    broken_definition = ModelDefinition(
        name="wrong_length_model",
        problem_types=(ProblemType.CLASSIFICATION,),
        model_family="baseline",
        train_fn=lambda *a, **k: {"model": None},
        predict_fn=lambda *a, **k: np.array([]),
    )

    result = run_model(broken_definition, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert result.status == "failed"
    assert result.errors


# --- run_candidates aggregation ----------------------------------------------


def test_run_candidates_aggregates_successful_models_only(classification_train_test):
    train_df, test_df = classification_train_test
    candidates, _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline", "logistic_regression"])

    metrics, chart_data = run_candidates(candidates, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert set(metrics["models"].keys()) == {"baseline", "logistic_regression"}
    assert metrics["eval_metric"] == "roc_auc"
    assert len(metrics["candidate_results"]) == 2
    # Single source of truth: the top-level eval_metric must always mirror
    # model_comparison.primary_metric - they must never independently disagree.
    assert metrics["eval_metric"] == metrics["model_comparison"]["primary_metric"]


def test_run_candidates_eval_metric_follows_plan_evaluation_metrics(classification_train_test):
    train_df, test_df = classification_train_test
    candidates, _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline", "logistic_regression"])

    metrics, _ = run_candidates(
        candidates, train_df, test_df, "target", None, ProblemType.CLASSIFICATION,
        evaluation_metrics=["precision", "accuracy"],
    )

    # "precision" is usable by every successful candidate here, so it must be
    # honored ahead of the problem-type default (roc_auc/accuracy) - and both
    # fields must still agree with each other.
    assert metrics["eval_metric"] == "precision"
    assert metrics["model_comparison"]["primary_metric"] == "precision"


def test_run_candidates_populates_dataset_explainability_for_classification(classification_train_test):
    train_df, test_df = classification_train_test
    # baseline is excluded from feature/permutation importance by design, so
    # use two candidates that both produce a usable importance ranking -
    # exactly the >= 2 models explain_dataset_consensus requires.
    candidates, _ = resolve_candidates(ProblemType.CLASSIFICATION, ["logistic_regression", "random_forest"])

    metrics, _ = run_candidates(candidates, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    dataset_explainability = metrics["model_comparison"]["dataset_explainability"]
    assert dataset_explainability is not None
    assert dataset_explainability["num_models_aggregated"] == 2
    assert dataset_explainability["consensus_ranking"]


def test_run_candidates_keeps_failed_candidates_out_of_models_but_in_results(classification_train_test):
    train_df, test_df = classification_train_test
    (baseline,), _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])
    broken = ModelDefinition(
        name="broken_model",
        problem_types=(ProblemType.CLASSIFICATION,),
        model_family="baseline",
        train_fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")),
        predict_fn=lambda *a, **k: None,
    )

    metrics, chart_data = run_candidates([baseline, broken], train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    assert "baseline" in metrics["models"]
    assert "broken_model" not in metrics["models"]
    result_names = {r["model_name"] for r in metrics["candidate_results"]}
    assert result_names == {"baseline", "broken_model"}
    broken_result = next(r for r in metrics["candidate_results"] if r["model_name"] == "broken_model")
    assert broken_result["status"] == "failed"


