"""Cleaning tool: imputation, outlier handling, categorical encoding, dedup.

Operates entirely on the local DataFrame. Returns (cleaned_df, log) where log
is a plain dict of what was done - safe to show an LLM/report, never the data.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# A column passes as "numeric text" if at least this fraction of its
# non-null values parse as a plain number or a "low - high" range - below
# this it's left alone as genuinely categorical text (see _parse_numeric_text).
NUMERIC_TEXT_MIN_PARSE_RATIO = 0.8


def _parse_numeric_text(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion for an object-dtype column: handles
    plain numeric-as-text ("1200") and simple "low - high" range text
    ("2100 - 2850", parsed to the midpoint) - the common shape of messy
    numeric columns from a real-world CSV (e.g. a "total_sqft" column mixing
    single values and ranges). Returns a float series; NaN where parsing
    fails."""
    direct = pd.to_numeric(series, errors="coerce")
    range_match = series.astype(str).str.extract(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*$")
    range_mid = (range_match[0].astype(float) + range_match[1].astype(float)) / 2
    return direct.fillna(range_mid)


def clean_data(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    time_column: Optional[str] = None,
    entity_column: Optional[str] = None,
    text_columns: Optional[list[str]] = None,
    aggressive_outlier_handling: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """time_column/entity_column are excluded from the categorical
    imputation/encoding below (Phase 10 regression fix): a CSV/Excel upload
    always reads a date column as plain text (object dtype, never
    datetime64), so before this fix it fell into the same bucket as any
    other text column - high-cardinality (>15 unique values), so
    `pd.factorize` silently replaced real dates with meaningless integer
    codes before tools/feature_engineering.py ever got to parse them with
    `pd.to_datetime`. entity_column had the same problem for the "pooled"
    scope strategy: one-hot/label-encoding replaced it before
    tools/feature_engineering.py's/tools/splitting.py's groupby-by-entity
    logic ever saw it, silently disabling the "grouped per entity so lag/
    rolling features and the chronological split never leak across
    entities" guarantee orchestration/graph.py's module docstring promises.
    See tests/test_phase10_e2e_validation.py's full-pipeline tests (which
    upload via a real CSV, unlike most other tests here that build a
    DataFrame directly in memory) for the regression coverage.
    """
    out = df.copy()
    text_columns = text_columns or []
    log: dict = {
        "imputation": {},
        "outliers_capped": {},
        "encoded_columns": [],
        "duplicates_removed": 0,
        "numeric_text_converted": [],
    }

    before_rows = len(out)
    out = out.drop_duplicates()
    log["duplicates_removed"] = before_rows - len(out)

    # Convert messy-but-mostly-numeric text columns (e.g. "total_sqft" mixing
    # plain values with "2100 - 2850" ranges) to floats BEFORE the
    # numeric/categorical split below - otherwise such a column falls into
    # the object-dtype bucket and gets pd.factorize'd into meaningless
    # integer codes instead of treated as the numeric feature it actually is.
    for col in text_columns:
        if col in out.columns:
            out[col] = out[col].fillna("")

    text_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
    for excluded in (target_column, time_column, entity_column):
        if excluded in text_cols:
            text_cols.remove(excluded)
    text_cols = [col for col in text_cols if col not in text_columns]
    for col in text_cols:
        parsed = _parse_numeric_text(out[col])
        non_null = out[col].notna().sum()
        if non_null and parsed.notna().sum() / non_null >= NUMERIC_TEXT_MIN_PARSE_RATIO:
            out[col] = parsed
            log["numeric_text_converted"].append(col)

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
    for excluded in (target_column, time_column, entity_column):
        if excluded in categorical_cols:
            categorical_cols.remove(excluded)
    categorical_cols = [col for col in categorical_cols if col not in text_columns]

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
