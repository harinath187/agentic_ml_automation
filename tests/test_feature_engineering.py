import pandas as pd

from tools.feature_engineering import engineer_features


def test_forecasting_lag_and_date_features(forecasting_df):
    engineered, log = engineer_features(
        forecasting_df, problem_type="forecasting", target_column="sales", time_column="date"
    )
    assert "sales_lag_1" in engineered.columns
    assert "sales_rolling_mean_7" in engineered.columns
    assert "date_dayofweek" in engineered.columns
    assert engineered["sales_lag_1"].isna().sum() == 0  # dropped after shifting


def test_forecasting_preserves_chronological_order(forecasting_df):
    engineered, _ = engineer_features(
        forecasting_df, problem_type="forecasting", target_column="sales", time_column="date"
    )
    assert engineered["date"].is_monotonic_increasing


def test_regression_scales_numeric_columns(regression_df):
    df = regression_df.dropna()
    engineered, log = engineer_features(df, problem_type="regression", target_column="price")
    assert abs(engineered["sqft"].mean()) < 1e-6
    assert "sqft" in log["scaled_columns"]
    assert "price" not in log["scaled_columns"]


def test_text_column_survives_feature_engineering_unchanged():
    df = pd.DataFrame({
        "notes": ["first long note", "second long note"],
        "value": [1.0, 2.0],
        "target": [0, 1],
    })
    engineered, _ = engineer_features(
        df, problem_type="regression", target_column="target", text_columns=["notes"]
    )

    assert engineered["notes"].tolist() == df["notes"].tolist()
