"""Feature engineering tool.

For forecasting: lag features, rolling averages, date/time parts.
For classification/regression: scaling of numeric features.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(
    df: pd.DataFrame,
    problem_type: str,
    target_column: Optional[str] = None,
    time_column: Optional[str] = None,
    entity_column: Optional[str] = None,
    text_columns: Optional[list[str]] = None,
    lags: tuple[int, ...] = (1, 7),
    rolling_windows: tuple[int, ...] = (7,),
) -> tuple[pd.DataFrame, dict]:
    """entity_column, when provided, keeps lag/rolling features scoped per
    entity (groupby) so values never leak across different stores/customers/
    etc. Used for the pooled scope strategy - single_entity/per_entity data
    is already reduced to one entity before this runs, so entity_column is
    None in those cases."""
    out = df.copy()
    log: dict = {"lag_features": [], "rolling_features": [], "date_parts": [], "scaled_columns": []}
    has_entity = entity_column and entity_column in out.columns

    if problem_type == "forecasting" and time_column and time_column in out.columns:
        out[time_column] = pd.to_datetime(out[time_column], errors="coerce")
        sort_cols = [entity_column, time_column] if has_entity else [time_column]
        out = out.sort_values(sort_cols).reset_index(drop=True)

        out[f"{time_column}_dayofweek"] = out[time_column].dt.dayofweek
        out[f"{time_column}_month"] = out[time_column].dt.month
        out[f"{time_column}_day"] = out[time_column].dt.day
        log["date_parts"] = [f"{time_column}_dayofweek", f"{time_column}_month", f"{time_column}_day"]

        if target_column and target_column in out.columns:
            grouped_target = out.groupby(entity_column)[target_column] if has_entity else None

            for lag in lags:
                col_name = f"{target_column}_lag_{lag}"
                out[col_name] = (
                    grouped_target.shift(lag) if has_entity else out[target_column].shift(lag)
                )
                log["lag_features"].append(col_name)

            for window in rolling_windows:
                col_name = f"{target_column}_rolling_mean_{window}"
                if has_entity:
                    out[col_name] = (
                        out.groupby(entity_column)[target_column]
                        .shift(1)
                        .groupby(out[entity_column])
                        .rolling(window=window)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                else:
                    out[col_name] = out[target_column].shift(1).rolling(window=window).mean()
                log["rolling_features"].append(col_name)

        out = out.dropna().reset_index(drop=True)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if problem_type in ("classification", "regression") and numeric_cols:
        scaler = StandardScaler()
        out[numeric_cols] = scaler.fit_transform(out[numeric_cols])
        log["scaled_columns"] = numeric_cols

    return out, log
