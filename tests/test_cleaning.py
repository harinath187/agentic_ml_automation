import pandas as pd

from tools.cleaning import clean_data


def test_cleaning_imputes_missing_numeric(classification_df):
    cleaned, log = clean_data(classification_df, target_column="churn")
    assert cleaned["monthly_charge"].isna().sum() == 0
    assert "monthly_charge" in log["imputation"]


def test_cleaning_encodes_categoricals(classification_df):
    cleaned, log = clean_data(classification_df, target_column="churn")
    assert "contract_type" not in cleaned.columns
    assert any(c.startswith("contract_type_") for c in cleaned.columns)
    assert "contract_type" in log["encoded_columns"]


def test_cleaning_removes_duplicates(classification_df):
    df_with_dupes = pd.concat([classification_df, classification_df.iloc[[0]]], ignore_index=True)
    cleaned, log = clean_data(df_with_dupes, target_column="churn")
    assert log["duplicates_removed"] >= 1


def test_cleaning_forward_fill_for_time_series(forecasting_df):
    df = forecasting_df.copy()
    df.loc[5, "sales"] = None
    cleaned, log = clean_data(df, target_column=None, time_column="date")
    assert cleaned["sales"].isna().sum() == 0
    assert log["imputation"]["sales"] == "forward_fill"
