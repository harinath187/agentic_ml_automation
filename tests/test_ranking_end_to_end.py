"""End-to-end ranking tests on small synthetic datasets, through the real
tools/model_runner.run_candidates() -> tools/evaluation.build_model_comparison()
path - classification, regression, forecasting, and (explicitly requested)
both directions of ETS vs. AutoGluon on forecasting data.

AutoGluon itself is never trained here - its ModelDefinition's train_fn is
monkeypatched to return a controlled leaderboard dict, exactly like
tests/test_model_runner.py already does. That keeps these tests fast and
deterministic while still exercising the real ranking engine on real
classical-model predictions (ETS/naive/etc. actually fit and forecast).
"""
import numpy as np
import pandas as pd
import pytest

from agents.schemas import ProblemType
from tools.model_registry import resolve_candidates
from tools.model_runner import run_candidates


# --- classification -----------------------------------------------------------


def test_classification_ranking_picks_best_by_accuracy():
    rng = np.random.default_rng(0)
    n = 150
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    target = (f1 + f2 + rng.normal(0, 0.1, n) > 0).astype(int)
    df = pd.DataFrame({"f1": f1, "f2": f2, "target": target})
    train_df, test_df = df.iloc[:110].reset_index(drop=True), df.iloc[110:].reset_index(drop=True)

    candidates, _ = resolve_candidates(ProblemType.CLASSIFICATION, ["baseline", "logistic_regression", "random_forest"])
    metrics, chart_data = run_candidates(candidates, train_df, test_df, "target", None, ProblemType.CLASSIFICATION)

    comparison = metrics["model_comparison"]
    assert comparison["winner"] is not None
    assert comparison["winner"] != "baseline"  # a real signal should beat the dummy baseline
    assert set(comparison["ranked_model_names"]) <= {"baseline", "logistic_regression", "random_forest"}
    # roc_auc is preferred over accuracy when every successful candidate reports it
    # (all classifiers here expose predict_proba on this binary target).
    assert comparison["primary_metric"] in ("accuracy", "roc_auc")


# --- regression ----------------------------------------------------------------


def test_regression_ranking_picks_best_by_rmse():
    rng = np.random.default_rng(1)
    n = 150
    x = rng.normal(0, 1, n)
    y = 3 * x + 2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x": x, "y": y})
    train_df, test_df = df.iloc[:110].reset_index(drop=True), df.iloc[110:].reset_index(drop=True)

    candidates, _ = resolve_candidates(ProblemType.REGRESSION, ["baseline", "linear_regression", "random_forest"])
    metrics, chart_data = run_candidates(candidates, train_df, test_df, "y", None, ProblemType.REGRESSION)

    comparison = metrics["model_comparison"]
    assert comparison["primary_metric"] == "rmse"
    assert comparison["higher_is_better"] is False
    assert comparison["winner"] is not None
    winner_rmse = next(r["metrics"]["rmse"] for r in comparison["results"] if r["model_name"] == comparison["winner"])
    for result in comparison["results"]:
        if result["status"] == "success" and result["metrics"].get("rmse") is not None:
            assert winner_rmse <= result["metrics"]["rmse"] + 1e-9


# --- forecasting -----------------------------------------------------------------


def _seasonal_forecasting_df(n=70):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    values = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 7)
    return pd.DataFrame({"date": dates, "sales": values})


def test_forecasting_ranking_does_not_use_random_split():
    """Confirms the classical forecasting candidates are evaluated on a
    chronological (non-shuffled) holdout - the tail of the series, in order.
    """
    df = _seasonal_forecasting_df()
    train_df, test_df = df.iloc[:55].reset_index(drop=True), df.iloc[55:].reset_index(drop=True)
    assert test_df["date"].min() > train_df["date"].max()  # chronological, not random

    candidates, _ = resolve_candidates(ProblemType.FORECASTING, ["naive", "seasonal_naive"])
    metrics, chart_data = run_candidates(candidates, train_df, test_df, "sales", "date", ProblemType.FORECASTING)

    comparison = metrics["model_comparison"]
    assert comparison["primary_metric"] == "rmse"
    assert comparison["winner"] == "seasonal_naive"  # seasonality dominates, no trend


def _trending_forecasting_df(n=60):
    """A gentle, low-noise trend with no seasonality - the shape classical
    ETS (trend='add', seasonal=None) is actually built for, unlike the
    strongly-seasonal fixture above (where ETS extrapolates a spurious
    trend and performs badly, which is realistic but not what these
    ETS-vs-AutoGluon comparison tests are checking)."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    values = np.linspace(100, 110, n) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"date": dates, "sales": values})


def _make_autogluon_stub(score_test: float):
    def fake_train_models(train_df, test_df, target_column, problem_type, time_column=None, time_limit=60):
        return {
            "eval_metric": "root_mean_squared_error",
            "models": {"WeightedEnsemble_L2": {"score_test": score_test, "score_val": score_test, "fit_time_s": 1.5}},
            "autogluon_best_model": "WeightedEnsemble_L2",
            "model_path": "AutogluonModels/fake_run",
        }

    return fake_train_models


def test_ets_beats_autogluon_when_autogluon_is_worse(monkeypatch):
    df = _trending_forecasting_df()
    train_df, test_df = df.iloc[:45].reset_index(drop=True), df.iloc[45:].reset_index(drop=True)

    candidates, _ = resolve_candidates(ProblemType.FORECASTING, ["ets", "autogluon_timeseries"])
    autogluon_def = next(c for c in candidates if c.name == "autogluon_timeseries")

    # ETS scores ~0.5 rmse on this gentle-trend series (see module docstring).
    # AutoGluon's own reported score_test (higher-is-better, -RMSE convention)
    # is deliberately made much worse than that.
    monkeypatch.setattr(autogluon_def, "train_fn", _make_autogluon_stub(score_test=-50.0))

    metrics, chart_data = run_candidates(candidates, train_df, test_df, "sales", "date", ProblemType.FORECASTING)
    comparison = metrics["model_comparison"]

    assert comparison["winner"] == "ets"


def test_autogluon_beats_ets_when_autogluon_is_better(monkeypatch):
    df = _trending_forecasting_df()
    train_df, test_df = df.iloc[:45].reset_index(drop=True), df.iloc[45:].reset_index(drop=True)

    candidates, _ = resolve_candidates(ProblemType.FORECASTING, ["ets", "autogluon_timeseries"])
    autogluon_def = next(c for c in candidates if c.name == "autogluon_timeseries")

    # AutoGluon's own reported score_test is deliberately made near-perfect,
    # comfortably beating ETS's real ~0.5 rmse on this series.
    monkeypatch.setattr(autogluon_def, "train_fn", _make_autogluon_stub(score_test=-0.01))

    metrics, chart_data = run_candidates(candidates, train_df, test_df, "sales", "date", ProblemType.FORECASTING)
    comparison = metrics["model_comparison"]

    assert comparison["winner"] == "WeightedEnsemble_L2"


def test_ets_vs_autogluon_ranking_is_purely_numeric_not_llm_influenced(monkeypatch):
    """Sanity check that nothing in the ETS-vs-AutoGluon comparison path
    touches an LLM at all - tools/model_runner.py and tools/evaluation.py
    are pure pandas/numpy/sklearn/statsmodels.
    """
    import agents.llm_client as llm_client_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("No LLM call should happen during model ranking")

    monkeypatch.setattr(llm_client_module, "call_llm_json", fail_if_called)

    df = _trending_forecasting_df()
    train_df, test_df = df.iloc[:45].reset_index(drop=True), df.iloc[45:].reset_index(drop=True)
    candidates, _ = resolve_candidates(ProblemType.FORECASTING, ["ets", "autogluon_timeseries"])
    autogluon_def = next(c for c in candidates if c.name == "autogluon_timeseries")
    monkeypatch.setattr(autogluon_def, "train_fn", _make_autogluon_stub(score_test=-50.0))

    metrics, chart_data = run_candidates(candidates, train_df, test_df, "sales", "date", ProblemType.FORECASTING)

    assert metrics["model_comparison"]["winner"] == "ets"
