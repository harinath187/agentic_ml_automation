from tools.splitting import split_data, split_train_val_test


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


# --- split_train_val_test (tools/classification_cycle.py's 3-way split) ----


def test_split_train_val_test_sizes_and_no_overlap(classification_df):
    train, val, test, log = split_train_val_test(classification_df, "churn", val_size=0.2, test_size=0.2)
    total = len(classification_df)
    assert len(train) + len(val) + len(test) == total
    assert len(test) == round(total * 0.2)
    assert len(val) == round(total * 0.2)
    assert log["train_rows"] == len(train)
    assert log["val_rows"] == len(val)
    assert log["test_rows"] == len(test)


def test_split_train_val_test_is_stratified(classification_df):
    _, _, test, log = split_train_val_test(classification_df, "churn", val_size=0.2, test_size=0.2)
    assert log["method"] == "stratified"
    full_rate = classification_df["churn"].mean()
    test_rate = test["churn"].mean()
    assert abs(full_rate - test_rate) < 0.15


def test_split_train_val_test_falls_back_when_class_too_small():
    import pandas as pd

    df = pd.DataFrame({"x": range(30), "y": [0] * 29 + [1]})
    train, val, test, log = split_train_val_test(df, "y", val_size=0.2, test_size=0.2)
    assert log["method"] == "random_shuffle_fallback"
    assert len(train) + len(val) + len(test) == 30
