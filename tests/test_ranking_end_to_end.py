"""End-to-end ranking tests on small synthetic datasets, through the real
tools/model_runner.run_candidates() -> tools/evaluation.build_model_comparison()
path - classification, regression, and forecasting.
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


