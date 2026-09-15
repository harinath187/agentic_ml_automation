"""Exploratory Data Analysis tool. Returns aggregated summaries only, never raw rows."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from tools.cleaning import NUMERIC_TEXT_MIN_PARSE_RATIO, _parse_numeric_text


def run_eda(df: pd.DataFrame, sensitive_columns: Optional[list[str]] = None) -> dict:
    sensitive = set(sensitive_columns or [])
    cols = [c for c in df.columns if c not in sensitive]
    working = df[cols].copy()

    # Coerce messy-but-mostly-numeric text columns (e.g. "total_sqft" mixing
    # plain values with "2100 - 2850" ranges) the same way tools/cleaning.py
    # does, so outlier/correlation/distribution stats below don't silently
    # skip them just because EDA runs before cleaning's own conversion.
    text_cols = working.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in text_cols:
        parsed = _parse_numeric_text(working[col])
        non_null = working[col].notna().sum()
        if non_null and parsed.notna().sum() / non_null >= NUMERIC_TEXT_MIN_PARSE_RATIO:
            working[col] = parsed

    numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()

    missing_analysis = {
        col: round(float(working[col].isna().mean()) * 100, 2) for col in cols
    }

    outliers = {}
    for col in numeric_cols:
        series = working[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        if outlier_count:
            outliers[col] = {
                "count": outlier_count,
                "pct": round(outlier_count / len(series) * 100, 2),
            }

    correlation_matrix = {}
    if len(numeric_cols) >= 2:
        corr = working[numeric_cols].corr(numeric_only=True).round(3)
        correlation_matrix = corr.to_dict()

    distributions = {}
    for col in numeric_cols:
        series = working[col].dropna()
        if series.empty:
            continue
        distributions[col] = {
            "skew": round(float(series.skew()), 3),
            "kurtosis": round(float(series.kurtosis()), 3),
        }

    datetime_cols = []
    for col in cols:
        if pd.api.types.is_datetime64_any_dtype(working[col]):
            datetime_cols.append(col)

    seasonality_notes = {}
    for col in datetime_cols:
        dt = pd.to_datetime(working[col], errors="coerce").dropna()
        if dt.empty:
            continue
        seasonality_notes[col] = {
            "min_date": str(dt.min()),
            "max_date": str(dt.max()),
            "distinct_months": int(dt.dt.to_period("M").nunique()),
        }

    return {
        "missing_value_pct": missing_analysis,
        "outliers_iqr": outliers,
        "correlation_matrix": correlation_matrix,
        "distributions": distributions,
        "datetime_columns": datetime_cols,
        "seasonality_notes": seasonality_notes,
    }
