import io

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


# --- messy numeric-text columns (e.g. "total_sqft" mixing plain values and
# "low - high" ranges) must be parsed to numeric, not factorized as
# categorical text - see tools/cleaning.py's _parse_numeric_text.


def test_cleaning_converts_plain_numeric_text_column_to_numeric():
    df = pd.DataFrame({
        "sqft": ["1000", "1200", "1500", "1800", "2000"],
        "price": [100, 120, 150, 180, 200],
    })
    cleaned, log = clean_data(df, target_column="price")

    assert "sqft" in log["numeric_text_converted"]
    assert "sqft" not in log["encoded_columns"]
    assert pd.api.types.is_numeric_dtype(cleaned["sqft"])
    assert cleaned["sqft"].tolist() == [1000.0, 1200.0, 1500.0, 1800.0, 2000.0]


def test_cleaning_converts_numeric_range_text_column_to_its_midpoint():
    df = pd.DataFrame({
        "total_sqft": ["1000", "2100 - 2850", "1500", "1800 - 2000", "2000"],
        "price": [100, 120, 150, 180, 200],
    })
    cleaned, log = clean_data(df, target_column="price")

    assert "total_sqft" in log["numeric_text_converted"]
    assert "total_sqft" not in log["encoded_columns"]
    assert not any(str(c).startswith("total_sqft") for c in log["encoded_columns"])
    assert cleaned["total_sqft"].tolist() == [1000.0, 2475.0, 1500.0, 1900.0, 2000.0]


def test_cleaning_leaves_genuinely_categorical_text_column_alone():
    df = pd.DataFrame({
        "city": ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Pune"],
        "price": [100, 120, 150, 180, 200],
    })
    cleaned, log = clean_data(df, target_column="price")

    assert "city" not in log["numeric_text_converted"]
    assert "city" in log["encoded_columns"]


def test_cleaning_forward_fill_for_time_series(forecasting_df):
    df = forecasting_df.copy()
    df.loc[5, "sales"] = None
    cleaned, log = clean_data(df, target_column=None, time_column="date")
    assert cleaned["sales"].isna().sum() == 0
    assert log["imputation"]["sales"] == "forward_fill"


# --- regression: time_column/entity_column must survive a real CSV upload ---
#
# forecasting_df builds "date" with pd.date_range() directly in memory
# (dtype datetime64), which the categorical-encoding loop below already
# skipped even before this fix (datetime64 isn't "object"/"category"). A
# real upload (data_ingestion/loader.py's pd.read_csv, with no parse_dates)
# always reads a date column back as plain text instead - that's the case
# these two tests reproduce with an actual CSV round-trip via io.BytesIO,
# rather than a DataFrame built directly in memory like every fixture above.
# Before the fix, clean_data's blind categorical encoding swept time_column/
# entity_column up as ordinary high-cardinality text and replaced real dates
# (and per-entity groupings) with meaningless `pd.factorize` integer codes -
# see tools/cleaning.py's clean_data docstring.


def _csv_roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(df.to_csv(index=False).encode()))


def test_cleaning_preserves_time_column_values_after_a_real_csv_upload(forecasting_df):
    uploaded = _csv_roundtrip(forecasting_df)
    assert uploaded["date"].dtype == object  # confirms this actually reproduces the bug's precondition

    cleaned, log = clean_data(uploaded, target_column="sales", time_column="date")

    assert cleaned["date"].tolist() == uploaded["date"].tolist()
    assert "date" not in log["encoded_columns"]
    assert not any(str(c).startswith("date") for c in log["encoded_columns"])


def test_cleaning_preserves_entity_column_values_after_a_real_csv_upload(multi_entity_forecasting_df):
    uploaded = _csv_roundtrip(multi_entity_forecasting_df)
    assert uploaded["store_id"].dtype == object

    cleaned, log = clean_data(uploaded, target_column="sales", time_column="date", entity_column="store_id")

    assert cleaned["store_id"].tolist() == uploaded["store_id"].tolist()
    assert "store_id" in cleaned.columns  # not replaced by one-hot/label-encoded columns
    assert not any(str(c).startswith("store_id_") for c in cleaned.columns)
    assert not any(str(c).startswith("store_id") for c in log["encoded_columns"])
