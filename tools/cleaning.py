"""Cleaning tool: imputation, outlier handling, categorical encoding, dedup.

Operates entirely on the local DataFrame. Returns (cleaned_df, log) where log
is a plain dict of what was done - safe to show an LLM/report, never the data.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def clean_data(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    time_column: Optional[str] = None,
    aggressive_outlier_handling: bool = False,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    log: dict = {"imputation": {}, "outliers_capped": {}, "encoded_columns": [], "duplicates_removed": 0}

    before_rows = len(out)
    out = out.drop_duplicates()
    log["duplicates_removed"] = before_rows - len(out)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    for col in numeric_cols:
        missing = int(out[col].isna().sum())
        if missing:
            if time_column and time_column in out.columns:
                out[col] = out[col].ffill().bfill()
                log["imputation"][col] = "forward_fill"
            else:
                median = out[col].median()
                out[col] = out[col].fillna(median)
                log["imputation"][col] = "median"

    categorical_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    for col in categorical_cols:
        missing = int(out[col].isna().sum())
        if missing:
            mode = out[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "missing"
            out[col] = out[col].fillna(fill_value)
            log["imputation"][col] = "mode"

    outlier_cols = numeric_cols if aggressive_outlier_handling else []
    for col in outlier_cols:
        series = out[col]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        capped = ((series < lower) | (series > upper)).sum()
        if capped:
            out[col] = series.clip(lower, upper)
            log["outliers_capped"][col] = int(capped)

    for col in categorical_cols:
        if out[col].nunique(dropna=True) <= 15:
            dummies = pd.get_dummies(out[col], prefix=col, drop_first=True)
            out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
            log["encoded_columns"].append(col)
        else:
            codes, _ = pd.factorize(out[col])
            out[col] = codes
            log["encoded_columns"].append(f"{col} (label_encoded: high cardinality)")

    return out, log
