"""Tests for entity/grouping-aware feature engineering and splitting - the
mechanisms that make the pooled/hierarchical scope strategies safe (no
lag-feature leakage across entities, every entity represented in train+test).
"""
from tools.feature_engineering import engineer_features
from tools.splitting import split_data


def test_feature_engineering_groups_lag_rolling_per_entity(multi_entity_forecasting_df):
    engineered, log = engineer_features(
        multi_entity_forecasting_df,
        problem_type="forecasting",
        target_column="sales",
        time_column="date",
        entity_column="store_id",
    )

    # Each store's first max(lag, rolling_window)=7 rows get dropped independently -
    # if lag/rolling were computed globally (no grouping), the drop wouldn't be
    # evenly distributed per store like this.
    rows_per_store = engineered.groupby("store_id").size()
    assert (rows_per_store == 53).all()  # 60 - 7
    assert len(engineered) == 159
    assert "sales_lag_1" in engineered.columns
    assert "sales_rolling_mean_7" in engineered.columns
    assert engineered.isna().sum().sum() == 0


def test_feature_engineering_without_entity_column_ignores_grouping(multi_entity_forecasting_df):
    # Sanity check: omitting entity_column falls back to global (ungrouped) behavior.
    engineered, _ = engineer_features(
        multi_entity_forecasting_df,
        problem_type="forecasting",
        target_column="sales",
        time_column="date",
    )
    rows_per_store = engineered.groupby("store_id").size()
    assert not (rows_per_store == 53).all()


def test_split_data_per_entity_covers_every_entity_in_train_and_test(multi_entity_forecasting_df):
    train_df, test_df, log = split_data(
        multi_entity_forecasting_df,
        problem_type="forecasting",
        time_column="date",
        entity_column="store_id",
        test_size=0.2,
    )

    assert log["method"] == "chronological_no_shuffle_per_entity"
    stores = set(multi_entity_forecasting_df["store_id"].unique())
    assert set(train_df["store_id"].unique()) == stores
    assert set(test_df["store_id"].unique()) == stores

    for store in stores:
        train_dates = train_df.loc[train_df["store_id"] == store, "date"]
        test_dates = test_df.loc[test_df["store_id"] == store, "date"]
        assert train_dates.max() <= test_dates.min()
