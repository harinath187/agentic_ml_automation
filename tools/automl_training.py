"""AutoML training tool, wrapping AutoGluon's Tabular and TimeSeries predictors.

Classification and regression are trained directly via TabularPredictor.
Forecasting is handled as regression over lag/rolling/date features by
default (train_models), except for the "hierarchical" scope strategy, which
uses AutoGluon's TimeSeriesPredictor (train_hierarchical_timeseries) to
learn jointly across entities via item_id grouping.

Only a metrics dictionary is returned - never predictions on individual rows
or the underlying data.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

MODELS_DIR = Path("AutogluonModels")


def train_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    time_column: Optional[str] = None,
    time_limit: int = 60,
) -> dict:
    from autogluon.tabular import TabularPredictor

    train = train_df.copy()
    test = test_df.copy()
    if time_column and time_column in train.columns:
        train = train.drop(columns=[time_column])
        test = test.drop(columns=[time_column])

    ag_problem_type = {
        "classification": None,
        "regression": "regression",
        "forecasting": "regression",
    }.get(problem_type)

    run_dir = MODELS_DIR / f"run_{uuid.uuid4().hex[:8]}"
    predictor = TabularPredictor(
        label=target_column,
        problem_type=ag_problem_type,
        path=str(run_dir),
        verbosity=0,
    )
    predictor.fit(train_data=train, time_limit=time_limit)

    # Deferred imports: keep automl_training.py's module-load cost limited to
    # the AutoGluon dependency it always needed, matching the existing
    # `from autogluon.tabular import TabularPredictor` pattern above.
    from agents.schemas import ProblemType as _ProblemType
    from tools.evaluation import compute_metrics

    problem_type_enum = _ProblemType(problem_type)
    y_true = test[target_column]
    y_train = train[target_column] if problem_type == "forecasting" else None

    leaderboard = predictor.leaderboard(test, silent=True)
    metrics: dict = {}
    for _, row in leaderboard.iterrows():
        raw_name = row["model"]

        # Best-effort per-model MAE/R2/MAPE/etc breakdown (Fix 2) - AutoGluon's
        # leaderboard only exposes a single score_test/score_val per row, so
        # every candidate but the "best" one otherwise never gets more than
        # that one number. One bad row (e.g. a stacked model that can't be
        # re-predicted individually) must not sink the others.
        row_metrics: dict = {}
        try:
            preds = predictor.predict(test, model=raw_name)
            y_proba = None
            if problem_type == "classification":
                try:
                    proba = predictor.predict_proba(test, model=raw_name)
                    if proba.shape[1] == 2:
                        y_proba = proba.iloc[:, 1].to_numpy()
                except Exception:  # noqa: BLE001 - roc_auc is supplementary
                    y_proba = None
            row_metrics = compute_metrics(problem_type_enum, y_true, preds, y_proba=y_proba, y_train=y_train)
        except Exception:  # noqa: BLE001 - the leaderboard's own score_test/score_val is still authoritative
            row_metrics = {}

        # Namespaced (Fix 3) so this candidate can never collide with/be
        # confused for a same-family custom candidate (e.g. tools/model_registry.py's
        # own "lightgbm" vs AutoGluon's "LightGBM" leaderboard row) - every
        # downstream consumer (tools/model_runner.py, reports/template.html)
        # reads this same prefixed name, so nothing else needs to change.
        metrics[f"autogluon_{raw_name}"] = {
            "score_test": round(float(row["score_test"]), 4),
            "score_val": round(float(row["score_val"]), 4),
            "fit_time_s": round(float(row["fit_time"]), 3) if row["fit_time"] is not None else None,
            "metrics": row_metrics,
        }

    best_model = f"autogluon_{predictor.model_best}"
    eval_metric = str(predictor.eval_metric)

    # Best-effort, single predict() call on the already-fit best model - only
    # used for report charts (tools/report_charts.py); never breaks training
    # if it fails, and never changes any existing key's value.
    best_model_predictions = None
    try:
        preds = predictor.predict(test)
        best_model_predictions = (
            [str(v) for v in preds.tolist()] if problem_type == "classification" else [float(v) for v in preds.tolist()]
        )
    except Exception:  # noqa: BLE001 - predictions are supplementary, scores above are authoritative
        best_model_predictions = None

    return {
        "eval_metric": eval_metric,
        "models": metrics,
        "autogluon_best_model": best_model,
        "model_path": str(run_dir),
        "best_model_predictions": best_model_predictions,
    }


def train_hierarchical_timeseries(
    df: pd.DataFrame,
    entity_column: str,
    target_column: str,
    time_column: str,
    time_limit: int = 60,
    prediction_length: Optional[int] = None,
) -> dict:
    """Joint forecasting across all entities via AutoGluon TimeSeriesPredictor,
    using entity_column as item_id. Univariate (target + time only) - other
    columns are not passed as covariates, to keep the v1 scope manageable.
    """
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    working = df[[entity_column, time_column, target_column]].dropna().copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working = working.dropna(subset=[time_column])

    ts_df = TimeSeriesDataFrame.from_data_frame(
        working, id_column=entity_column, timestamp_column=time_column
    )

    if prediction_length is None:
        shortest_series = ts_df.num_timesteps_per_item().min()
        prediction_length = max(1, min(28, int(shortest_series) // 5 or 1))

    train_data, test_data = ts_df.train_test_split(prediction_length)

    run_dir = MODELS_DIR / f"run_ts_{uuid.uuid4().hex[:8]}"
    predictor = TimeSeriesPredictor(
        target=target_column,
        prediction_length=prediction_length,
        path=str(run_dir),
        verbosity=0,
    )
    predictor.fit(train_data, time_limit=time_limit)

    leaderboard = predictor.leaderboard(test_data, silent=True)
    metrics: dict = {}
    for _, row in leaderboard.iterrows():
        metrics[row["model"]] = {
            "score_test": round(float(row["score_test"]), 4),
            "score_val": round(float(row["score_val"]), 4) if pd.notna(row.get("score_val")) else None,
            "fit_time_s": round(float(row["fit_time_s"]), 3) if pd.notna(row.get("fit_time_s")) else None,
        }

    return {
        "eval_metric": str(predictor.eval_metric),
        "models": metrics,
        "autogluon_best_model": predictor.model_best,
        "model_path": str(run_dir),
        "prediction_length": prediction_length,
        "num_entities": int(ts_df.num_items),
    }


def cleanup_run(model_path: str) -> None:
    path = Path(model_path)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
