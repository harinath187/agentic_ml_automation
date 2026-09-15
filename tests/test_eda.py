import pandas as pd

from tools.eda import run_eda


# --- messy numeric-text columns (e.g. "total_sqft" mixing plain values and
# "low - high" ranges) must be visible to outlier/correlation stats, not
# silently skipped just because EDA runs before tools/cleaning.py's own
# numeric-text conversion - see tools/eda.py's run_eda.


def test_eda_detects_outliers_in_numeric_range_text_column():
    # Varied enough that IQR isn't degenerate (0), with one clear outlier -
    # mixes plain numeric text and "low - high" range text, same as a real
    # "total_sqft" column.
    values = [
        "1000 - 1010", "1005", "998 - 1010", "1002", "1004 - 1010",
        "1001", "999 - 1005", "1003", "1000", "9000 - 9200",
    ]
    df = pd.DataFrame({"total_sqft": values, "price": list(range(10))})

    summary = run_eda(df)

    assert "total_sqft" in summary["outliers_iqr"]
    assert summary["outliers_iqr"]["total_sqft"]["count"] == 1


def test_eda_leaves_genuinely_categorical_text_column_out_of_numeric_stats():
    df = pd.DataFrame({"city": ["Bengaluru", "Mumbai", "Delhi", "Chennai"], "price": [1, 2, 3, 4]})

    summary = run_eda(df)

    assert "city" not in summary["outliers_iqr"]
    assert "city" not in summary["correlation_matrix"]


def test_eda_excludes_sensitive_columns(classification_df):
    summary = run_eda(classification_df, sensitive_columns=["customer_name"])
    assert "customer_name" not in summary["missing_value_pct"]


def test_eda_detects_missing_values(classification_df):
    summary = run_eda(classification_df)
    assert summary["missing_value_pct"]["monthly_charge"] > 0


def test_eda_correlation_matrix_numeric_only(regression_df):
    summary = run_eda(regression_df)
    assert "sqft" in summary["correlation_matrix"]
    assert "neighborhood" not in summary["correlation_matrix"]


def test_eda_datetime_detection(forecasting_df):
    df = forecasting_df.copy()
    df["date"] = df["date"]  # already datetime64
    summary = run_eda(df)
    assert "date" in summary["datetime_columns"]
    assert "date" in summary["seasonality_notes"]
