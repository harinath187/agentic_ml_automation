"""Cleaning tool: imputation, outlier handling, categorical encoding, dedup.

Operates entirely on the local DataFrame. Returns (cleaned_df, log) where log
is a plain dict of what was done - safe to show an LLM/report, never the data.

`clean_data()` below is the original single-pass implementation, kept as-is
for backward compatibility with existing callers/tests. `TabularCleaner` is
the leakage-safe replacement: `fit()` learns imputation/outlier statistics
from one dataframe only (the train split) and `transform()` reapplies those
*stored* statistics to any other dataframe (val/test/inference), instead of
recomputing them - the median/mode/IQR-bounds of `clean_data()` are otherwise
silently computed over train+test combined when it runs before the split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

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


@dataclass
class TabularCleaner:
    """Leakage-safe, fit/transform version of the cleaning logic above.

    `fit(train_df, ...)` learns and stores every statistic (numeric medians,
    categorical fill values, IQR/z-score outlier bounds, high-missingness
    columns to drop) from the dataframe passed to it - never from any other
    split. `transform(df)` re-applies those *stored* values to any dataframe
    (the same train data, a held-out val/test split, or new data at
    inference time), so val/test metrics are never inflated by statistics
    that peeked at them during fit.
    """

    target_column: Optional[str] = None
    time_column: Optional[str] = None
    entity_column: Optional[str] = None
    text_columns: list[str] = field(default_factory=list)
    missing_threshold: float = 0.7
    unknown_threshold: float = 0.05
    aggressive_outlier_handling: bool = False
    outlier_method: str = "iqr"  # "iqr" or "zscore"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0

    # Learned during fit() - populated, never hand-set.
    fitted_: bool = field(default=False, init=False)
    dropped_columns_: list[str] = field(default_factory=list, init=False)
    missingness_flag_columns_: list[str] = field(default_factory=list, init=False)
    numeric_columns_: list[str] = field(default_factory=list, init=False)
    categorical_columns_: list[str] = field(default_factory=list, init=False)
    numeric_text_converted_: list[str] = field(default_factory=list, init=False)
    numeric_fill_values_: dict = field(default_factory=dict, init=False)
    categorical_fill_values_: dict = field(default_factory=dict, init=False)
    outlier_bounds_: dict = field(default_factory=dict, init=False)
    report_: dict = field(default_factory=dict, init=False)

    def _excluded(self) -> set[str]:
        return {c for c in (self.target_column, self.time_column, self.entity_column) if c}

    def _new_report(self) -> dict:
        return {
            "rows_dropped_missing_target": 0,
            "dropped_columns_high_missingness": {},
            "missingness_flags_added": [],
            "imputation": {},
            "numeric_text_converted": [],
            "outliers_capped": {},
            "encoded_columns": [],
        }

    def _drop_missing_target_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.target_column or self.target_column not in df.columns:
            return df
        before = len(df)
        out = df[df[self.target_column].notna()].copy()
        self.report_["rows_dropped_missing_target"] = before - len(out)
        return out

    def _convert_numeric_text(self, df: pd.DataFrame, columns: list[str]) -> None:
        for col in columns:
            if col not in df.columns:
                continue
            parsed = _parse_numeric_text(df[col])
            non_null = df[col].notna().sum()
            if non_null and parsed.notna().sum() / non_null >= NUMERIC_TEXT_MIN_PARSE_RATIO:
                affected = int((parsed.notna() & df[col].notna()).sum())
                df[col] = parsed
                self.numeric_text_converted_.append(col)
                self.report_["numeric_text_converted"].append(
                    {"column": col, "rows_converted": affected}
                )

    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None,
            time_column: Optional[str] = None, entity_column: Optional[str] = None,
            text_columns: Optional[list[str]] = None) -> "TabularCleaner":
        """Learn all imputation/outlier statistics from `df` only."""
        if target_column is not None:
            self.target_column = target_column
        if time_column is not None:
            self.time_column = time_column
        if entity_column is not None:
            self.entity_column = entity_column
        if text_columns is not None:
            self.text_columns = text_columns

        self.report_ = self._new_report()
        work = df.copy()
        work = self._drop_missing_target_rows(work)

        excluded = self._excluded()
        for col in self.text_columns:
            if col in work.columns:
                work[col] = work[col].fillna("")

        candidate_text_cols = [
            c for c in work.select_dtypes(include=["object", "category"]).columns
            if c not in excluded and c not in self.text_columns
        ]
        self._convert_numeric_text(work, candidate_text_cols)

        # Missingness threshold: drop high-missingness columns (never the
        # target/time/entity columns, which are structurally required).
        self.dropped_columns_ = []
        for col in work.columns:
            if col in excluded:
                continue
            missing_frac = float(work[col].isna().mean()) if len(work) else 0.0
            if missing_frac > self.missing_threshold:
                self.dropped_columns_.append(col)
                self.report_["dropped_columns_high_missingness"][col] = round(missing_frac, 4)
        work = work.drop(columns=self.dropped_columns_)

        self.numeric_columns_ = [
            c for c in work.select_dtypes(include=[np.number]).columns if c not in excluded
        ]
        self.categorical_columns_ = [
            c for c in work.select_dtypes(include=["object", "category"]).columns
            if c not in excluded and c not in self.text_columns
        ]

        # Missingness indicator flags - any remaining column with >0% missing.
        self.missingness_flag_columns_ = [
            c for c in (self.numeric_columns_ + self.categorical_columns_)
            if work[c].isna().any()
        ]
        self.report_["missingness_flags_added"] = [
            f"{c}_was_missing" for c in self.missingness_flag_columns_
        ]

        # Numeric fill values (median), unless time-series (forward/back-fill
        # has no single stored value - transform re-derives it per-frame).
        self.numeric_fill_values_ = {}
        for col in self.numeric_columns_:
            if work[col].isna().any():
                if self.time_column and self.time_column in work.columns:
                    self.numeric_fill_values_[col] = "ffill_bfill"
                else:
                    median = float(work[col].median())
                    self.numeric_fill_values_[col] = median

        # Categorical fill values (mode, or "Unknown" above the threshold).
        self.categorical_fill_values_ = {}
        for col in self.categorical_columns_:
            missing_frac = float(work[col].isna().mean())
            if missing_frac == 0:
                continue
            if missing_frac > self.unknown_threshold:
                self.categorical_fill_values_[col] = "Unknown"
            else:
                mode = work[col].mode(dropna=True)
                self.categorical_fill_values_[col] = mode.iloc[0] if not mode.empty else "missing"

        # Outlier bounds, computed on fit data only.
        self.outlier_bounds_ = {}
        if self.aggressive_outlier_handling:
            for col in self.numeric_columns_:
                series = work[col]
                if self.outlier_method == "zscore":
                    mean, std = series.mean(), series.std()
                    if not std:
                        continue
                    lower, upper = mean - self.zscore_threshold * std, mean + self.zscore_threshold * std
                else:
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    if iqr == 0:
                        continue
                    lower = q1 - self.iqr_multiplier * iqr
                    upper = q3 + self.iqr_multiplier * iqr
                self.outlier_bounds_[col] = (float(lower), float(upper))

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Apply the statistics learned in `fit()` to `df` (train, val, test,
        or new inference data) - never recomputes them from `df` itself."""
        if not self.fitted_:
            raise RuntimeError("TabularCleaner.transform() called before fit()")

        report = self._new_report()
        out = self._drop_missing_target_rows_for_transform(df, report)
        excluded = self._excluded()

        for col in self.text_columns:
            if col in out.columns:
                out[col] = out[col].fillna("")

        for col in self.numeric_text_converted_:
            if col in out.columns:
                out[col] = _parse_numeric_text(out[col])
                report["numeric_text_converted"].append({"column": col, "rows_converted": None})

        out = out.drop(columns=[c for c in self.dropped_columns_ if c in out.columns])
        if self.dropped_columns_:
            report["dropped_columns_high_missingness"] = {
                c: self.report_["dropped_columns_high_missingness"].get(c) for c in self.dropped_columns_
            }

        # Missingness flags - added before imputing, using the columns/flags
        # decided at fit time so train/val/test share the same schema.
        for col in self.missingness_flag_columns_:
            if col in out.columns:
                out[f"{col}_was_missing"] = out[col].isna().astype(int)
                report["missingness_flags_added"].append(f"{col}_was_missing")

        if self.time_column and self.time_column in out.columns:
            if not out[self.time_column].is_monotonic_increasing:
                out = out.sort_values(self.time_column)
                report.setdefault("warnings", []).append(
                    f"'{self.time_column}' was not sorted before forward/back-fill; sorted defensively."
                )

        for col, fill in self.numeric_fill_values_.items():
            if col not in out.columns:
                continue
            missing = int(out[col].isna().sum())
            if not missing:
                continue
            if fill == "ffill_bfill":
                out[col] = out[col].ffill().bfill()
                report["imputation"][col] = "forward_fill"
            else:
                out[col] = out[col].fillna(fill)
                report["imputation"][col] = f"median ({fill:.4g})"

        for col, fill in self.categorical_fill_values_.items():
            if col not in out.columns:
                continue
            missing = int(out[col].isna().sum())
            if not missing:
                continue
            out[col] = out[col].fillna(fill)
            report["imputation"][col] = f"mode/unknown ({fill!r})"

        for col, (lower, upper) in self.outlier_bounds_.items():
            if col not in out.columns:
                continue
            series = out[col]
            capped = int(((series < lower) | (series > upper)).sum())
            if capped:
                out[col] = series.clip(lower, upper)
                report["outliers_capped"][col] = capped

        for col in self.categorical_columns_:
            if col not in out.columns:
                continue
            if out[col].nunique(dropna=True) <= 15:
                dummies = pd.get_dummies(out[col], prefix=col, drop_first=True)
                out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
                report["encoded_columns"].append(col)
            else:
                codes, _ = pd.factorize(out[col])
                out[col] = codes
                report["encoded_columns"].append(f"{col} (label_encoded: high cardinality)")

        self.report_ = report
        return out, report

    def _drop_missing_target_rows_for_transform(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        if not self.target_column or self.target_column not in df.columns:
            return df.copy()
        before = len(df)
        out = df[df[self.target_column].notna()].copy()
        report["rows_dropped_missing_target"] = before - len(out)
        return out

    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None,
                       time_column: Optional[str] = None, entity_column: Optional[str] = None,
                       text_columns: Optional[list[str]] = None) -> tuple[pd.DataFrame, dict]:
        self.fit(df, target_column=target_column, time_column=time_column,
                 entity_column=entity_column, text_columns=text_columns)
        return self.transform(df)

    def _state_dict(self) -> dict:
        return {
            "target_column": self.target_column,
            "time_column": self.time_column,
            "entity_column": self.entity_column,
            "text_columns": self.text_columns,
            "missing_threshold": self.missing_threshold,
            "unknown_threshold": self.unknown_threshold,
            "aggressive_outlier_handling": self.aggressive_outlier_handling,
            "outlier_method": self.outlier_method,
            "iqr_multiplier": self.iqr_multiplier,
            "zscore_threshold": self.zscore_threshold,
            "fitted_": self.fitted_,
            "dropped_columns_": self.dropped_columns_,
            "missingness_flag_columns_": self.missingness_flag_columns_,
            "numeric_columns_": self.numeric_columns_,
            "categorical_columns_": self.categorical_columns_,
            "numeric_text_converted_": self.numeric_text_converted_,
            "numeric_fill_values_": self.numeric_fill_values_,
            "categorical_fill_values_": self.categorical_fill_values_,
            "outlier_bounds_": self.outlier_bounds_,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Serialize all fitted state to JSON so inference-time preprocessing
        can reuse the exact same cleaner without retraining/refitting."""
        Path(path).write_text(json.dumps(self._state_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TabularCleaner":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        cleaner = cls(
            target_column=state["target_column"],
            time_column=state["time_column"],
            entity_column=state["entity_column"],
            text_columns=state["text_columns"],
            missing_threshold=state["missing_threshold"],
            unknown_threshold=state["unknown_threshold"],
            aggressive_outlier_handling=state["aggressive_outlier_handling"],
            outlier_method=state["outlier_method"],
            iqr_multiplier=state["iqr_multiplier"],
            zscore_threshold=state["zscore_threshold"],
        )
        cleaner.fitted_ = state["fitted_"]
        cleaner.dropped_columns_ = state["dropped_columns_"]
        cleaner.missingness_flag_columns_ = state["missingness_flag_columns_"]
        cleaner.numeric_columns_ = state["numeric_columns_"]
        cleaner.categorical_columns_ = state["categorical_columns_"]
        cleaner.numeric_text_converted_ = state["numeric_text_converted_"]
        cleaner.numeric_fill_values_ = state["numeric_fill_values_"]
        cleaner.categorical_fill_values_ = state["categorical_fill_values_"]
        cleaner.outlier_bounds_ = {
            col: tuple(bounds) for col, bounds in state["outlier_bounds_"].items()
        }
        return cleaner
