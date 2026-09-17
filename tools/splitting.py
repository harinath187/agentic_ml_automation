"""Splitting tool. Time-based split for forecasting (never shuffled), random otherwise."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    problem_type: str,
    test_size: float = 0.2,
    time_column: Optional[str] = None,
    entity_column: Optional[str] = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """entity_column, when provided (pooled forecasting across entities),
    splits chronologically per entity so every entity contributes to both
    train and test, instead of one global cutoff that could drop whole
    entities from train or test."""
    log: dict = {"method": None, "train_rows": None, "test_rows": None}
    has_entity = entity_column and entity_column in df.columns

    if problem_type == "forecasting":
        if time_column and time_column in df.columns:
            sort_cols = [entity_column, time_column] if has_entity else [time_column]
            df = df.sort_values(sort_cols).reset_index(drop=True)

        if has_entity:
            train_parts, test_parts = [], []
            for _, group in df.groupby(entity_column):
                split_idx = int(len(group) * (1 - test_size))
                train_parts.append(group.iloc[:split_idx])
                test_parts.append(group.iloc[split_idx:])
            train_df = pd.concat(train_parts).sort_values([entity_column, time_column]).reset_index(drop=True)
            test_df = pd.concat(test_parts).sort_values([entity_column, time_column]).reset_index(drop=True)
            log["method"] = "chronological_no_shuffle_per_entity"
        else:
            split_idx = int(len(df) * (1 - test_size))
            train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
            log["method"] = "chronological_no_shuffle"
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=True
        )
        log["method"] = "random_shuffle"

    log["train_rows"] = int(len(train_df))
    log["test_rows"] = int(len(test_df))
    return train_df, test_df, log


def _safe_stratify(y: pd.Series, test_fraction: float) -> Optional[pd.Series]:
    """sklearn's stratify=y raises ("least populated class ... too few
    members") when a class doesn't have at least 1 member on each side of
    the split; this returns None (falls back to a plain random split)
    instead of letting that raise, for a class small enough that
    stratification at this particular ratio isn't possible."""
    counts = y.value_counts()
    min_side_fraction = min(test_fraction, 1 - test_fraction)
    if counts.min() < 2 or counts.min() * min_side_fraction < 1:
        return None
    return y


def split_train_val_test(
    df: pd.DataFrame,
    target_column: str,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Classification-oriented 3-way split: TRAIN / VALIDATION / TEST, each
    carved off with stratification (falling back to a plain random split
    only when a class is too small to stratify at the requested ratio - see
    _safe_stratify). Used by tools/classification_cycle.py: TEST is carved
    off ONCE here and never touched again until that module's final
    test-set evaluation; every improvement cycle only ever sees TRAIN/VAL.

    val_size/test_size are both fractions of the ORIGINAL dataframe (e.g.
    0.2/0.2 leaves 0.6 for TRAIN) - not of an intermediate remainder.
    """
    y = df[target_column]
    log: dict = {"method": "stratified", "train_rows": None, "val_rows": None, "test_rows": None}

    stratify = _safe_stratify(y, test_size)
    if stratify is None:
        log["method"] = "random_shuffle_fallback"
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True, stratify=stratify
    )

    remainder_val_fraction = val_size / (1 - test_size)
    y_remainder = train_val_df[target_column]
    stratify_remainder = _safe_stratify(y_remainder, remainder_val_fraction)
    if stratify_remainder is None and log["method"] == "stratified":
        log["method"] = "stratified_test_only"
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=remainder_val_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_remainder,
    )

    log["train_rows"] = int(len(train_df))
    log["val_rows"] = int(len(val_df))
    log["test_rows"] = int(len(test_df))
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        log,
    )
