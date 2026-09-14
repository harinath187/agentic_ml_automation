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
