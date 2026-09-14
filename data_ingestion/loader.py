"""Data ingestion + schema extraction.

This is the ONLY module that reads raw data off disk. It returns two things:

    df             - a pandas DataFrame. Stays local. Passed only to tools/*
                     (sandboxed, data-touching code). NEVER passed to an LLM.
    schema_summary - a plain dict safe to send to an LLM: column names,
                     dtypes, missing %, aggregated stats. Sensitive columns
                     (per the user-supplied denylist) are dropped entirely
                     from this summary - name and stats both - before it is
                     ever built.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame. Local only, never returned to an LLM."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .xls, or .xlsx.")


def extract_schema(
    df: pd.DataFrame, sensitive_columns: Optional[Iterable[str]] = None
) -> dict:
    """Build an LLM-safe schema summary, excluding sensitive_columns entirely.

    Sensitive columns are removed from the *output* only; the original df is
    left untouched so downstream tools can still use them if needed for
    processing (though the Planner Agent will never learn they exist).
    """
    sensitive = set(sensitive_columns or [])
    safe_columns = [c for c in df.columns if c not in sensitive]

    columns_info = []
    for col in safe_columns:
        series = df[col]
        missing_pct = round(float(series.isna().mean()) * 100, 2)
        info = {
            "name": col,
            "dtype": str(series.dtype),
            "missing_pct": missing_pct,
        }
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            info.update(
                {
                    "min": float(non_null.min()) if len(non_null) else None,
                    "max": float(non_null.max()) if len(non_null) else None,
                    "mean": float(non_null.mean()) if len(non_null) else None,
                    "std": float(non_null.std()) if len(non_null) else None,
                }
            )
        else:
            unique_count = int(series.nunique(dropna=True))
            info["unique_count"] = unique_count
            non_null_count = int(series.notna().sum())
            info["avg_rows_per_value"] = (
                round(non_null_count / unique_count, 2) if unique_count else None
            )
            if pd.api.types.is_datetime64_any_dtype(series) or _looks_like_datetime(series):
                info["looks_like_datetime"] = True
        columns_info.append(info)

    return {
        "row_count": int(len(df)),
        "column_count": len(safe_columns),
        "excluded_sensitive_column_count": len(sensitive & set(df.columns)),
        "columns": columns_info,
    }


def _looks_like_datetime(series: pd.Series, sample_size: int = 20) -> bool:
    if series.dtype != object:
        return False
    sample = series.dropna().head(sample_size)
    if sample.empty:
        return False
    try:
        pd.to_datetime(sample, errors="raise")
        return True
    except (ValueError, TypeError):
        return False
