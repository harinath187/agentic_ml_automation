from tools.eda import run_eda


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
