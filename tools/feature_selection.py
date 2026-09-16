"""Deterministic feature-selection tool - no LLM call.

Makes the ExperimentPlan.feature_columns / DataQualityReport exclusion lists
(computed in agents/planner.py, tools/plan_validation.py, tools/profiling.py)
actually load-bearing: this is the first thing that subsets the dataframe's
*columns* before cleaning/feature engineering ever runs.

Runs on the dataset's original column names, before tools/cleaning.py's
one-hot/label-encoding and tools/feature_engineering.py's lag/rolling/date
features exist - so it never has to reconcile its exclusion lists (computed
by tools/profiling.py on the original columns) against expanded dummy-column
names. This is a deliberate scope boundary, not an oversight: engineered-
feature quality (a newly created lag/rolling column that happens to be
near-constant, or a rare-category one-hot dummy that's mostly zero) is not
checked here, and would need a separate post-engineering pass to catch -
left as a natural follow-up rather than built into this function, since
running selection after engineering would reintroduce exactly the dummy-
column mismatch this ordering avoids.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from agents.schemas import DataQualityReport, ExperimentPlan

_REASON_SOURCES = (
    ("constant", "constant_columns"),
    ("near_constant", "near_constant_columns"),
    ("id_like", "suspicious_columns"),
    ("possible_leakage", "possible_leakage_columns"),
)


def apply_feature_selection(
    df: pd.DataFrame,
    plan: ExperimentPlan,
    quality_report: DataQualityReport,
) -> tuple[pd.DataFrame, dict]:
    """Keeps target/time/entity columns (structural, never candidates) plus
    everything in plan.feature_columns; drops the rest. Returns
    (selected_df, log) where log = {"kept": [...], "dropped": [{"column",
    "reason"}]}.
    """
    structural = {c for c in (plan.target_column, plan.time_column, plan.entity_column) if c}
    feature_set = set(plan.feature_columns)
    keep = [c for c in df.columns if c in structural or c in feature_set]

    reason_by_column: dict[str, str] = {}
    for reason, attr_name in _REASON_SOURCES:
        for column in getattr(quality_report, attr_name):
            reason_by_column.setdefault(column, reason)

    dropped = [
        {"column": c, "reason": reason_by_column.get(c, "not selected by plan")}
        for c in df.columns
        if c not in keep
    ]

    log = {"kept": keep, "dropped": dropped}
    return df[keep], log
