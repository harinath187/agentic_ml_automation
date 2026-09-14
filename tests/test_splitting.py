from tools.splitting import split_data


def test_random_split_for_classification(classification_df):
    train, test, log = split_data(classification_df, problem_type="classification", test_size=0.25)
    assert log["method"] == "random_shuffle"
    assert len(test) == round(len(classification_df) * 0.25)
    assert len(train) + len(test) == len(classification_df)


def test_chronological_split_for_forecasting_never_shuffles(forecasting_df):
    train, test, log = split_data(
        forecasting_df, problem_type="forecasting", time_column="date", test_size=0.2
    )
    assert log["method"] == "chronological_no_shuffle"
    assert train["date"].max() <= test["date"].min()
    assert train["date"].is_monotonic_increasing
    assert test["date"].is_monotonic_increasing
