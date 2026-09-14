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

    leaderboard = predictor.leaderboard(test, silent=True)
    metrics: dict = {}
    for _, row in leaderboard.iterrows():
        metrics[row["model"]] = {
            "score_test": round(float(row["score_test"]), 4),
            "score_val": round(float(row["score_val"]), 4),
            "fit_time_s": round(float(row["fit_time"]), 3) if row["fit_time"] is not None else None,
        }

    best_model = predictor.model_best
    eval_metric = str(predictor.eval_metric)

    return {
        "eval_metric": eval_metric,
        "models": metrics,
        "autogluon_best_model": best_model,
        "model_path": str(run_dir),
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
