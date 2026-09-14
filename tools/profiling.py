"""Data Intelligence tool: deterministic dataset profiling, target-selection
signals, and data-quality analysis - no LLM involved.

Runs before the Planner so it can reason over structured, pre-computed facts
(DatasetProfile / TargetAnalysis / DataQualityReport) instead of needing to
infer everything itself from a thin schema summary. Like tools/eda.py, this
only ever returns aggregated stats - never raw rows - and honors the same
sensitive_columns denylist used everywhere else in data_ingestion/tools.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from agents.schemas import (
    ColumnKind,
    ColumnProfile,
    DataQualityIssue,
    DataQualityReport,
    DatasetProfile,
    TargetAnalysis,
    TargetCandidate,
)

# A near-constant column is one dominated by a single value; a suspicious
# "ID-like" column is one where almost every row is a distinct value.
NEAR_CONSTANT_THRESHOLD = 0.99
ID_LIKE_UNIQUE_RATIO = 0.98
MAX_CATEGORICAL_TARGET_CARDINALITY = 20
MIN_ROWS_FOR_OUTLIER_CHECK = 4


def _visible_columns(df: pd.DataFrame, sensitive_columns: Optional[list[str]]) -> list[str]:
    sensitive = set(sensitive_columns or [])
    return [c for c in df.columns if c not in sensitive]


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


def _infer_kind(series: pd.Series) -> ColumnKind:
    if pd.api.types.is_bool_dtype(series):
        return ColumnKind.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series) or _looks_like_datetime(series):
        return ColumnKind.DATETIME
    if pd.api.types.is_numeric_dtype(series):
        return ColumnKind.NUMERICAL
    return ColumnKind.CATEGORICAL


def _numeric_stats(series: pd.Series) -> Optional[dict]:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return {
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "mean": round(float(non_null.mean()), 4),
        "median": round(float(non_null.median()), 4),
        "std": round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0,
    }


def _datetime_range(series: pd.Series) -> Optional[dict]:
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if parsed.empty:
        return None
    return {
        "min": str(parsed.min()),
        "max": str(parsed.max()),
        "distinct_days": int(parsed.dt.normalize().nunique()),
    }


def _build_column_profile(series: pd.Series, row_count: int) -> ColumnProfile:
    kind = _infer_kind(series)
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))
    non_missing = row_count - missing_count

    is_constant = unique_count <= 1
    is_near_constant = False
    if not is_constant and non_missing > 0:
        top_value_count = int(series.value_counts(dropna=True).iloc[0])
        is_near_constant = (top_value_count / non_missing) >= NEAR_CONSTANT_THRESHOLD

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        inferred_kind=kind,
        missing_count=missing_count,
        missing_pct=round((missing_count / row_count) * 100, 2) if row_count else 0.0,
        unique_count=unique_count,
        unique_pct=round((unique_count / row_count) * 100, 2) if row_count else 0.0,
        is_constant=is_constant,
        is_near_constant=is_near_constant,
        numeric_stats=_numeric_stats(series) if kind == ColumnKind.NUMERICAL else None,
        datetime_range=_datetime_range(series) if kind == ColumnKind.DATETIME else None,
    )


def profile_dataset(
    df: pd.DataFrame, sensitive_columns: Optional[list[str]] = None
) -> DatasetProfile:
    """Row/column counts, per-column dtype/cardinality/missingness, duplicate
    rows, and constant/near-constant column detection. Deterministic, pandas-only.
    """
    sensitive = set(sensitive_columns or [])
    visible_cols = _visible_columns(df, sensitive_columns)
    row_count = int(len(df))

    columns: list[ColumnProfile] = [
        _build_column_profile(df[col], row_count) for col in visible_cols
    ]

    duplicate_row_count = int(df.duplicated().sum()) if row_count else 0

    return DatasetProfile(
        row_count=row_count,
        column_count=len(visible_cols),
        column_names=visible_cols,
        duplicate_row_count=duplicate_row_count,
        duplicate_row_pct=round((duplicate_row_count / row_count) * 100, 2) if row_count else 0.0,
        numerical_columns=[c.name for c in columns if c.inferred_kind == ColumnKind.NUMERICAL],
        categorical_columns=[c.name for c in columns if c.inferred_kind == ColumnKind.CATEGORICAL],
        datetime_columns=[c.name for c in columns if c.inferred_kind == ColumnKind.DATETIME],
        boolean_columns=[c.name for c in columns if c.inferred_kind == ColumnKind.BOOLEAN],
        constant_columns=[c.name for c in columns if c.is_constant],
        near_constant_columns=[c.name for c in columns if c.is_near_constant],
        excluded_sensitive_column_count=len(sensitive & set(df.columns)),
        columns=columns,
    )


def analyze_target(
    df: pd.DataFrame,
    profile: DatasetProfile,
    sensitive_columns: Optional[list[str]] = None,
) -> TargetAnalysis:
    """Deterministic candidate-target detection. Never picks a target outright
    (that decision belongs to the Planner, informed by business context) -
    only narrows the field and surfaces cardinality/type signals.
    """
    if profile.row_count == 0:
        return TargetAnalysis(recommendation_reasoning="Dataset is empty; no target candidates.")

    by_name = {c.name: c for c in profile.columns}
    candidates: list[TargetCandidate] = []

    for col in profile.column_names:
        col_profile = by_name[col]
        if col_profile.is_constant:
            continue
        # ID-like columns (near-100% unique) are structurally unsuitable targets.
        is_id_like = col_profile.unique_pct >= ID_LIKE_UNIQUE_RATIO * 100
        if is_id_like and col_profile.inferred_kind != ColumnKind.NUMERICAL:
            continue
        if col_profile.inferred_kind == ColumnKind.DATETIME:
            continue

        likely_types: list[str] = []
        notes: list[str] = []

        if col_profile.inferred_kind == ColumnKind.CATEGORICAL:
            if col_profile.unique_count == 2:
                likely_types.append("classification")
                notes.append("binary categorical")
            elif col_profile.unique_count <= MAX_CATEGORICAL_TARGET_CARDINALITY:
                likely_types.append("classification")
                notes.append(f"low-cardinality categorical ({col_profile.unique_count} classes)")
            else:
                notes.append(
                    f"high-cardinality categorical ({col_profile.unique_count} classes) - "
                    "unlikely to be a useful classification target"
                )
        elif col_profile.inferred_kind == ColumnKind.BOOLEAN:
            likely_types.append("classification")
            notes.append("boolean")
        elif col_profile.inferred_kind == ColumnKind.NUMERICAL:
            is_integer_dtype = "int" in col_profile.dtype.lower()
            if is_id_like and is_integer_dtype:
                # A near-100%-unique float is a normal continuous target (price,
                # sales, ...); only near-100%-unique *integers* read as an ID
                # column (row index / primary key), so the exclusion is scoped
                # to integer dtypes only.
                notes.append("near-unique integer - likely an identifier, not a target")
            elif col_profile.unique_count == 2:
                likely_types.append("classification")
                notes.append("binary numeric (0/1-style) target")
            elif is_integer_dtype and col_profile.unique_count <= MAX_CATEGORICAL_TARGET_CARDINALITY:
                likely_types.extend(["classification", "regression"])
                notes.append(
                    f"low-cardinality integer ({col_profile.unique_count} distinct values) - "
                    "could be a classification label or a discrete regression target"
                )
            else:
                likely_types.append("regression")
                if profile.datetime_columns:
                    likely_types.append("forecasting")
                    notes.append("numeric, and a datetime column exists - forecasting is plausible")
                else:
                    notes.append("continuous numeric")

        if not likely_types:
            continue

        candidates.append(
            TargetCandidate(
                name=col,
                inferred_kind=col_profile.inferred_kind,
                cardinality=col_profile.unique_count,
                cardinality_ratio=round(col_profile.unique_count / profile.row_count, 4),
                missing_pct=col_profile.missing_pct,
                likely_problem_types=likely_types,
                signal_notes="; ".join(notes),
            )
        )

    recommended_target = None
    reasoning = "No unambiguous single candidate; the Planner should choose using business context."
    if len(candidates) == 1:
        recommended_target = candidates[0].name
        reasoning = f"Exactly one plausible target candidate found: {recommended_target!r}."
    elif not candidates:
        reasoning = "No plausible target candidates found via deterministic signals."

    return TargetAnalysis(
        candidate_targets=candidates,
        recommended_target=recommended_target,
        recommendation_reasoning=reasoning,
    )


def _detect_outliers(series: pd.Series) -> Optional[dict]:
    non_null = series.dropna()
    if len(non_null) < MIN_ROWS_FOR_OUTLIER_CHECK:
        return None
    q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((non_null < lower) | (non_null > upper)).sum())
    if not count:
        return None
    return {"count": count, "pct": round(count / len(non_null) * 100, 2)}


LEAKAGE_NAME_HINTS = ("target", "label", "outcome", "result", "leak", "actual", "y_true")


def analyze_data_quality(
    df: pd.DataFrame,
    profile: DatasetProfile,
    sensitive_columns: Optional[list[str]] = None,
) -> DataQualityReport:
    """Deterministic data-quality checks: missingness, duplicates, invalid
    dtypes, outliers, constant columns, suspicious/ID-like columns, and
    heuristic leakage indicators. No target column is known yet at this
    stage, so leakage detection is limited to naming hints and redundant
    (near-duplicate) feature pairs.
    """
    if profile.row_count == 0:
        return DataQualityReport(
            issues=[DataQualityIssue(issue_type="empty_dataset", severity="high", detail="Dataset has 0 rows.")],
            overall_quality_score=0.0,
        )

    visible_cols = _visible_columns(df, sensitive_columns)
    by_name = {c.name: c for c in profile.columns}
    issues: list[DataQualityIssue] = []

    missing_value_columns = {
        c.name: c.missing_pct for c in profile.columns if c.missing_count > 0
    }
    for col, pct in missing_value_columns.items():
        severity = "high" if pct >= 30 else "medium" if pct >= 5 else "low"
        issues.append(
            DataQualityIssue(column=col, issue_type="missing_values", severity=severity, detail=f"{pct}% missing")
        )

    if profile.duplicate_row_count:
        severity = "high" if profile.duplicate_row_pct >= 10 else "medium"
        issues.append(
            DataQualityIssue(
                issue_type="duplicate_rows",
                severity=severity,
                detail=f"{profile.duplicate_row_count} duplicate rows ({profile.duplicate_row_pct}%)",
            )
        )

    for col in profile.constant_columns:
        issues.append(DataQualityIssue(column=col, issue_type="constant_column", severity="low", detail="single unique value"))
    for col in profile.near_constant_columns:
        issues.append(
            DataQualityIssue(column=col, issue_type="near_constant_column", severity="low", detail="dominated by one value")
        )

    possible_outlier_columns: dict[str, dict] = {}
    for col in profile.numerical_columns:
        outliers = _detect_outliers(df[col])
        if outliers:
            possible_outlier_columns[col] = outliers
            severity = "medium" if outliers["pct"] >= 5 else "low"
            issues.append(
                DataQualityIssue(
                    column=col,
                    issue_type="possible_outliers",
                    severity=severity,
                    detail=f"{outliers['count']} rows ({outliers['pct']}%) outside 1.5*IQR",
                )
            )

    invalid_dtype_columns: list[str] = []
    for col in profile.categorical_columns:
        series = df[col]
        if _looks_like_numeric_text(series):
            invalid_dtype_columns.append(col)
            issues.append(
                DataQualityIssue(
                    column=col, issue_type="invalid_dtype", severity="medium",
                    detail="stored as text but values look numeric",
                )
            )

    suspicious_columns: list[str] = []
    for col in visible_cols:
        col_profile = by_name.get(col)
        if col_profile is None:
            continue
        if col_profile.unique_pct >= ID_LIKE_UNIQUE_RATIO * 100 and profile.row_count > 1:
            suspicious_columns.append(col)
            issues.append(
                DataQualityIssue(
                    column=col, issue_type="id_like_column", severity="low",
                    detail="near-unique per row; likely an identifier, not a useful feature",
                )
            )

    possible_leakage_columns: list[str] = []
    for col in visible_cols:
        if any(hint in col.lower() for hint in LEAKAGE_NAME_HINTS):
            possible_leakage_columns.append(col)
            issues.append(
                DataQualityIssue(
                    column=col, issue_type="possible_leakage", severity="medium",
                    detail="column name suggests it may encode the prediction outcome",
                )
            )

    numeric_cols = profile.numerical_columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        seen = set()
        for col_a in corr.columns:
            for col_b in corr.columns:
                if col_a == col_b or (col_b, col_a) in seen:
                    continue
                seen.add((col_a, col_b))
                value = corr.loc[col_a, col_b]
                if pd.notna(value) and value >= 0.995:
                    for c in (col_a, col_b):
                        if c not in possible_leakage_columns:
                            possible_leakage_columns.append(c)
                    issues.append(
                        DataQualityIssue(
                            issue_type="possible_leakage",
                            severity="medium",
                            detail=f"{col_a!r} and {col_b!r} are near-perfectly correlated (r={value:.3f})",
                        )
                    )

    high_severity_count = sum(1 for i in issues if i.severity == "high")
    medium_severity_count = sum(1 for i in issues if i.severity == "medium")
    low_severity_count = sum(1 for i in issues if i.severity == "low")
    penalty = high_severity_count * 15 + medium_severity_count * 7 + low_severity_count * 2
    overall_quality_score = max(0.0, round(100.0 - penalty, 2))

    return DataQualityReport(
        missing_value_columns=missing_value_columns,
        duplicate_row_count=profile.duplicate_row_count,
        duplicate_row_pct=profile.duplicate_row_pct,
        constant_columns=profile.constant_columns,
        near_constant_columns=profile.near_constant_columns,
        possible_outlier_columns=possible_outlier_columns,
        invalid_dtype_columns=invalid_dtype_columns,
        suspicious_columns=suspicious_columns,
        possible_leakage_columns=possible_leakage_columns,
        issues=issues,
        overall_quality_score=overall_quality_score,
    )
