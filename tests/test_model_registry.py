"""Tests for the model registry (tools/model_registry.py): registration and
candidate-selection/resolution. No training happens here - see
tests/test_model_runner.py for training/failure-handling behavior.
"""
from agents.schemas import ProblemType, ValidationStrategyType
from tools.model_registry import (
    REGISTRY,
    default_candidates,
    get_registry_for_problem_type,
    is_dependency_available,
    resolve_candidates,
)

EXPECTED_CLASSIFICATION = {"baseline", "logistic_regression", "random_forest", "xgboost", "lightgbm"}
EXPECTED_REGRESSION = {"baseline", "linear_regression", "random_forest", "xgboost", "lightgbm"}
EXPECTED_FORECASTING = {"naive", "seasonal_naive", "ets", "arima", "sarima"}


# --- registration ------------------------------------------------------------


def test_classification_models_are_registered():
    names = {d.name for d in get_registry_for_problem_type(ProblemType.CLASSIFICATION)}
    assert names == EXPECTED_CLASSIFICATION


def test_regression_models_are_registered():
    names = {d.name for d in get_registry_for_problem_type(ProblemType.REGRESSION)}
    assert names == EXPECTED_REGRESSION


def test_forecasting_models_are_registered():
    names = {d.name for d in get_registry_for_problem_type(ProblemType.FORECASTING)}
    assert names == EXPECTED_FORECASTING


def test_every_registry_entry_has_a_complete_definition():
    for definition in REGISTRY:
        assert definition.name
        assert definition.problem_types
        assert definition.model_family
        assert callable(definition.train_fn)
        assert callable(definition.predict_fn)
        assert isinstance(definition.supported_validation_strategies, tuple)
        assert isinstance(definition.required_dependencies, tuple)
        assert isinstance(definition.default_params, dict)


def test_registry_has_no_autogluon_entries():
    # AutoGluon is used only for the "hierarchical" forecasting scope
    # strategy, which trains via tools/automl_training.train_hierarchical_timeseries
    # directly from orchestration/graph.py, bypassing this registry entirely.
    autogluon_entries = [d for d in REGISTRY if d.name.startswith("autogluon")]
    assert autogluon_entries == []


def test_forecasting_models_only_declare_time_series_split():
    for definition in get_registry_for_problem_type(ProblemType.FORECASTING):
        assert definition.supported_validation_strategies == (ValidationStrategyType.TIME_SERIES_SPLIT,)


# --- candidate selection / resolution ----------------------------------------


def test_resolve_candidates_matches_exact_names():
    matched, unmatched = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline", "random_forest"])
    assert {d.name for d in matched} == {"baseline", "random_forest"}
    assert unmatched == []


def test_resolve_candidates_tolerates_llm_style_naming():
    matched, unmatched = resolve_candidates(
        ProblemType.CLASSIFICATION, ["LightGBM", "Random Forest", "XGBoost", "Logistic Regression"]
    )
    assert {d.name for d in matched} == {"lightgbm", "random_forest", "xgboost", "logistic_regression"}
    assert unmatched == []


def test_resolve_candidates_reports_unmatched_names_without_crashing():
    matched, unmatched = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline", "CatBoost", "NeuralNetworkFoo"])
    assert {d.name for d in matched} == {"baseline"}
    assert set(unmatched) == {"CatBoost", "NeuralNetworkFoo"}


def test_resolve_candidates_does_not_run_every_model_by_default():
    # Only what's requested comes back - never the full registry.
    matched, _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline"])
    assert len(matched) == 1


def test_resolve_candidates_deduplicates_repeated_requests():
    matched, _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline", "baseline", "Baseline"])
    assert len(matched) == 1


def test_resolve_candidates_scopes_to_problem_type():
    # "linear_regression" exists for regression, not forecasting.
    matched, unmatched = resolve_candidates(ProblemType.FORECASTING, ["linear_regression"])
    assert matched == []
    assert unmatched == ["linear_regression"]


def test_resolve_candidates_empty_request_matches_nothing():
    matched, unmatched = resolve_candidates(ProblemType.REGRESSION, [])
    assert matched == []
    assert unmatched == []


def test_default_candidates_is_a_safe_nonempty_fallback():
    for problem_type in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION, ProblemType.FORECASTING):
        fallback = default_candidates(problem_type)
        assert fallback
        assert all(problem_type in d.problem_types for d in fallback)


# --- dependency checks --------------------------------------------------------


def test_is_dependency_available_true_for_installed_package():
    assert is_dependency_available("statsmodels") is True


def test_is_dependency_available_false_for_missing_package():
    assert is_dependency_available("definitely_not_a_real_package_xyz") is False
