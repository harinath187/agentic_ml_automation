"""Deterministic problem-type detection (Phase 2 of the Data Intelligence
layer) - no LLM involved.

Runs after tools/profiling.py (DatasetProfile/TargetAnalysis/DataQualityReport)
and before the Planner, so the Planner reasons over a structured
ProblemDefinition instead of re-deriving problem type / time-series signals
itself. Like the rest of tools/, this only ever returns aggregated
stats/flags - never raw rows.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from agents.schemas import (
    DataQualityReport,
    DatasetProfile,
    ProblemDefinition,
    ProblemType,
    TargetAnalysis,
    TimeSeriesSignals,
)

MIN_ITEM_AVG_ROWS_PER_VALUE = 2.0
SEASONALITY_CV_THRESHOLD = 0.1
TREND_CORRELATION_THRESHOLD = 0.3
MIN_ROWS_FOR_SEASONALITY = 14
MIN_ROWS_FOR_TREND = 5


def detect_problem(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_analysis: TargetAnalysis,
    quality_report: DataQualityReport,
    sensitive_columns: Optional[list[str]] = None,
) -> ProblemDefinition:
    """Deterministic classification/regression/forecasting detection, plus
    time-series signals (datetime column, item/entity column, frequency,
    time range, missing periods, seasonality, trend) when a datetime column
    is present. Never picks a target outright when ambiguous - only narrows
    the field, same contract as tools.profiling.analyze_target.
    """
    if profile.row_count == 0:
        return ProblemDefinition(
            confidence="low", reasoning="Dataset is empty; problem type cannot be determined."
        )

    detected_type, target_column, confidence, reasoning = _detect_problem_type(target_analysis, profile)

    time_series_signals = None
    if profile.datetime_columns:
        time_series_signals = _detect_time_series_signals(df, profile, target_column, quality_report)

    return ProblemDefinition(
        detected_problem_type=detected_type,
        target_column=target_column,
        confidence=confidence,
        reasoning=reasoning,
        time_series_signals=time_series_signals,
    )


def _detect_problem_type(
    target_analysis: TargetAnalysis, profile: DatasetProfile
) -> tuple[Optional[ProblemType], Optional[str], str, str]:
    forecasting_candidates = [
        c for c in target_analysis.candidate_targets if "forecasting" in c.likely_problem_types
    ]

    if profile.datetime_columns and forecasting_candidates:
        if target_analysis.recommended_target in {c.name for c in forecasting_candidates}:
            target_column = target_analysis.recommended_target
            confidence = "high"
            reasoning = (
                f"A datetime column is present and {target_column!r} is the sole plausible "
                "numeric target - forecasting is the deterministic best fit."
            )
        else:
            target_column = forecasting_candidates[0].name
            confidence = "medium" if len(forecasting_candidates) == 1 else "low"
            reasoning = (
                f"A datetime column is present; {len(forecasting_candidates)} numeric candidate(s) "
                f"could be forecasted. Defaulting to {target_column!r}, but the Planner should "
                "confirm the target using business context."
            )
        return ProblemType.FORECASTING, target_column, confidence, reasoning

    if target_analysis.recommended_target:
        target_column = target_analysis.recommended_target
        candidate = next(c for c in target_analysis.candidate_targets if c.name == target_column)
        if "classification" in candidate.likely_problem_types:
            detected_type = ProblemType.CLASSIFICATION
        elif "regression" in candidate.likely_problem_types:
            detected_type = ProblemType.REGRESSION
        else:
            detected_type = None
        reasoning = (
            f"Exactly one plausible target candidate found: {target_column!r} "
            f"({detected_type.value if detected_type else 'unclear type'})."
        )
        return detected_type, target_column, "high", reasoning

    non_forecasting_types = {
        t for c in target_analysis.candidate_targets for t in c.likely_problem_types if t != "forecasting"
    }
    if len(non_forecasting_types) == 1:
        detected_type = ProblemType(non_forecasting_types.pop())
        reasoning = (
            f"Multiple target candidates, but all point to {detected_type.value} - the specific "
            "target column is still ambiguous and left for the Planner to choose using business context."
        )
        return detected_type, None, "medium", reasoning

    if target_analysis.candidate_targets:
        reasoning = (
            "Multiple ambiguous target candidates spanning different problem types; the Planner "
            "must decide using business context."
        )
    else:
        reasoning = "No plausible target candidates found; the Planner must decide using business context."
    return None, None, "low", reasoning


def _detect_time_series_signals(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_column: Optional[str],
    quality_report: DataQualityReport,
) -> TimeSeriesSignals:
    datetime_column = profile.datetime_columns[0]
    by_name = {c.name: c for c in profile.columns}
    dt_profile = by_name[datetime_column]

    item_column = _detect_item_column(profile, quality_report, exclude={datetime_column, target_column})

    parsed = pd.to_datetime(df[datetime_column], errors="coerce").dropna()
    frequency = _infer_frequency(parsed)
    missing_count, missing_pct = _missing_periods(parsed, frequency)

    seasonality_detected, seasonality_notes = False, "Target column unavailable; seasonality not evaluated."
    trend_detected, trend_notes = False, "Target column unavailable; trend not evaluated."
    if target_column and target_column in df.columns:
        seasonality_detected, seasonality_notes = _detect_seasonality(df, datetime_column, target_column)
        trend_detected, trend_notes = _detect_trend(df, datetime_column, target_column)

    return TimeSeriesSignals(
        datetime_column=datetime_column,
        item_column=item_column,
        frequency=frequency,
        time_range=dt_profile.datetime_range,
        missing_periods_count=missing_count,
        missing_periods_pct=missing_pct,
        seasonality_detected=seasonality_detected,
        seasonality_notes=seasonality_notes,
        trend_detected=trend_detected,
        trend_notes=trend_notes,
    )


def _detect_item_column(
    profile: DatasetProfile, quality_report: DataQualityReport, exclude: set[Optional[str]]
) -> Optional[str]:
    """Best categorical column that repeats many rows per value (a store,
    customer, machine, ...), excluding ID-like/suspicious columns."""
    by_name = {c.name: c for c in profile.columns}
    best_col, best_avg_rows = None, MIN_ITEM_AVG_ROWS_PER_VALUE
    for col in profile.categorical_columns:
        if col in exclude or col in quality_report.suspicious_columns:
            continue
        col_profile = by_name[col]
        if col_profile.unique_count <= 1:
            continue
        avg_rows_per_value = profile.row_count / col_profile.unique_count
        if avg_rows_per_value >= best_avg_rows:
            best_avg_rows = avg_rows_per_value
            best_col = col
    return best_col


def _infer_frequency(parsed: pd.Series) -> Optional[str]:
    unique_sorted = pd.Series(parsed.unique()).sort_values()
    if len(unique_sorted) < 3:
        return None
    try:
        freq = pd.infer_freq(pd.DatetimeIndex(unique_sorted))
    except (ValueError, TypeError):
        freq = None
    if freq:
        return freq

    diffs = unique_sorted.diff().dropna().dt.days
    if diffs.empty:
        return None
    mode_days = diffs.mode()
    if mode_days.empty:
        return None
    days = int(mode_days.iloc[0])
    if days == 1:
        return "D"
    if days == 7:
        return "W"
    if 28 <= days <= 31:
        return "M"
    return None


def _missing_periods(parsed: pd.Series, frequency: Optional[str]) -> tuple[int, float]:
    """Global calendar-coverage check: how many expected periods between the
    min/max date have no row at all. When multiple entities are interleaved
    in one column, a date covered by any single entity counts as present -
    this is a dataset-wide signal, not a per-entity one.
    """
    if parsed.empty:
        return 0, 0.0
    freq = frequency or "D"
    expected = pd.date_range(parsed.min(), parsed.max(), freq=freq)
    if len(expected) == 0:
        return 0, 0.0
    actual_dates = set(parsed.dt.normalize().unique())
    expected_dates = set(pd.DatetimeIndex(expected).normalize())
    missing = len(expected_dates - actual_dates)
    pct = round(missing / len(expected_dates) * 100, 2)
    return missing, pct


def _detect_seasonality(df: pd.DataFrame, datetime_column: str, target_column: str) -> tuple[bool, str]:
    dt, values = _aligned_datetime_and_numeric_target(df, datetime_column, target_column)
    if dt is None or len(values) < MIN_ROWS_FOR_SEASONALITY or values.std() == 0:
        return False, "Not enough rows or numeric target variance to evaluate seasonality."

    dow_means = values.groupby(dt.dt.dayofweek).mean()
    if len(dow_means) < 2:
        return False, "Not enough distinct days-of-week to evaluate seasonality."
    cv = float(dow_means.std() / values.std())
    if cv >= SEASONALITY_CV_THRESHOLD:
        return True, f"Day-of-week means vary meaningfully relative to overall spread (cv={cv:.3f})."
    return False, f"No strong day-of-week pattern detected (cv={cv:.3f})."


def _detect_trend(df: pd.DataFrame, datetime_column: str, target_column: str) -> tuple[bool, str]:
    dt, values = _aligned_datetime_and_numeric_target(df, datetime_column, target_column)
    if dt is None or len(values) < MIN_ROWS_FOR_TREND or values.std() == 0:
        return False, "Not enough rows or numeric target variance to evaluate trend."

    order = dt.sort_values().index
    ordered_values = values.loc[order].reset_index(drop=True)
    time_index = pd.Series(range(len(ordered_values)))
    corr = time_index.corr(ordered_values)
    if pd.isna(corr):
        return False, "Could not compute a trend correlation."
    if abs(corr) >= TREND_CORRELATION_THRESHOLD:
        direction = "Increasing" if corr > 0 else "Decreasing"
        return True, f"{direction} trend detected (correlation={corr:.3f})."
    return False, f"No strong trend detected (correlation={corr:.3f})."


def _aligned_datetime_and_numeric_target(
    df: pd.DataFrame, datetime_column: str, target_column: str
) -> tuple[Optional[pd.Series], pd.Series]:
    dt = pd.to_datetime(df[datetime_column], errors="coerce")
    values = pd.to_numeric(df[target_column], errors="coerce")
    valid = dt.notna() & values.notna()
    if not valid.any():
        return None, values[valid]
    return dt[valid], values[valid]
