"""Single-call, UI-facing EDA snapshot for the "click a dataset to explore
it" modal in the frontend (api/main.py's GET /api/datasets/{id}/eda).

Deliberately separate from tools/profiling.py: that module feeds the
Planner/Evaluator (pydantic ColumnProfile/DatasetProfile models, aggregated
stats only, never raw rows - see tests/test_llm_safety.py) and its output
shape is a contract those agents depend on. This module feeds the browser
directly - it's fine for it to return a small sample of raw rows (the LLM
never sees this endpoint's response), and its shape is free to evolve with
the UI without touching pipeline code.

Everything here is computed in one pass over an in-memory DataFrame and
returned as plain JSON-safe dicts (explicit float()/int() casts - numpy
scalars aren't JSON-serializable on their own).
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

SAMPLE_ROWS = 50
HISTOGRAM_BINS = 12
TOP_CATEGORIES = 10
# An object column made mostly of long strings reads as free text rather
# than a small set of repeated category labels - shown as its own "text"
# badge in the UI instead of "categorical".
TEXT_AVG_LENGTH_THRESHOLD = 25


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


def _looks_like_numeric_text(series: pd.Series, sample_size: int = 20) -> bool:
    if series.dtype != object:
        return False
    sample = series.dropna().head(sample_size)
    if sample.empty:
        return False
    try:
        pd.to_numeric(sample, errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series) or _looks_like_datetime(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series) or _looks_like_numeric_text(series):
        return "numeric"
    non_null = series.dropna().astype(str)
    if not non_null.empty and non_null.str.len().mean() >= TEXT_AVG_LENGTH_THRESHOLD:
        return "text"
    return "categorical"


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _histogram(series: pd.Series) -> list[int]:
    non_null = _as_numeric(series).dropna()
    if non_null.empty or non_null.nunique() <= 1:
        return []
    try:
        counts = pd.cut(non_null, bins=HISTOGRAM_BINS).value_counts(sort=False)
    except ValueError:
        return []
    return [int(c) for c in counts.tolist()]


def _numeric_column_stats(series: pd.Series) -> Optional[dict]:
    non_null = _as_numeric(series).dropna()
    if non_null.empty:
        return None
    zeros_pct = round(float((non_null == 0).mean()) * 100, 2)
    return {
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "mean": round(float(non_null.mean()), 4),
        "median": round(float(non_null.median()), 4),
        "std": round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0,
        "q1": round(float(non_null.quantile(0.25)), 4),
        "q3": round(float(non_null.quantile(0.75)), 4),
        "skew": round(float(non_null.skew()), 4) if len(non_null) > 2 else 0.0,
        "kurtosis": round(float(non_null.kurt()), 4) if len(non_null) > 3 else 0.0,
        "zeros_pct": zeros_pct,
    }


def _categorical_column_stats(series: pd.Series) -> Optional[dict]:
    non_null = series.dropna()
    if non_null.empty:
        return None
    counts = non_null.value_counts().head(TOP_CATEGORIES)
    total = len(non_null)
    return {
        "top_values": [
            {"value": str(value), "count": int(count), "pct": round(int(count) / total * 100, 2)}
            for value, count in counts.items()
        ]
    }


def _datetime_column_stats(series: pd.Series) -> Optional[dict]:
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if parsed.empty:
        return None
    return {
        "min": str(parsed.min()),
        "max": str(parsed.max()),
        "distinct_days": int(parsed.dt.normalize().nunique()),
    }


def _boolean_column_stats(series: pd.Series) -> Optional[dict]:
    non_null = series.dropna()
    if non_null.empty:
        return None
    total = len(non_null)
    true_count = int(non_null.sum())
    return {
        "true_count": true_count,
        "false_count": total - true_count,
        "true_pct": round(true_count / total * 100, 2),
        "false_pct": round((total - true_count) / total * 100, 2),
    }


def _column_stats(series: pd.Series, kind: str) -> Optional[dict]:
    if kind == "numeric":
        return _numeric_column_stats(series)
    if kind == "datetime":
        return _datetime_column_stats(series)
    if kind == "boolean":
        return _boolean_column_stats(series)
    return _categorical_column_stats(series)


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.where(pd.notnull(df), None)
    records = []
    for row in clean.to_dict(orient="records"):
        safe_row = {}
        for key, value in row.items():
            if value is None:
                safe_row[key] = None
            elif isinstance(value, (int, float, str, bool)):
                safe_row[key] = value
            else:
                safe_row[key] = str(value)
        records.append(safe_row)
    return records


def compute_eda_snapshot(df: pd.DataFrame) -> dict:
    row_count = int(len(df))
    column_count = int(len(df.columns))
    memory_bytes = int(df.memory_usage(deep=True).sum())
    duplicate_row_count = int(df.duplicated().sum()) if row_count else 0
    duplicate_row_pct = round(duplicate_row_count / row_count * 100, 2) if row_count else 0.0
    total_cells = row_count * column_count
    missing_cells = int(df.isna().sum().sum())
    missing_cells_pct = round(missing_cells / total_cells * 100, 2) if total_cells else 0.0

    type_counts = {"numeric": 0, "categorical": 0, "datetime": 0, "boolean": 0, "text": 0}
    columns = []
    for col in df.columns:
        series = df[col]
        kind = _infer_kind(series)
        type_counts[kind] += 1
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "kind": kind,
                "missing_count": missing_count,
                "missing_pct": round(missing_count / row_count * 100, 2) if row_count else 0.0,
                "unique_count": unique_count,
                "cardinality_pct": round(unique_count / row_count * 100, 2) if row_count else 0.0,
                "histogram": _histogram(series) if kind == "numeric" else [],
                "stats": _column_stats(series, kind),
            }
        )

    numeric_columns = [c["name"] for c in columns if c["kind"] == "numeric"]
    correlations = None
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True).round(3)
        correlations = {
            "columns": numeric_columns,
            "matrix": [[None if pd.isna(v) else float(v) for v in row] for row in corr.values.tolist()],
        }

    return {
        "row_count": row_count,
        "column_count": column_count,
        "memory_bytes": memory_bytes,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_row_pct": duplicate_row_pct,
        "missing_cells_pct": missing_cells_pct,
        "type_counts": type_counts,
        "columns": columns,
        "correlations": correlations,
        "sample_rows": _json_safe_records(df.head(SAMPLE_ROWS)),
    }
