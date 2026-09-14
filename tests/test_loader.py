from data_ingestion.loader import extract_schema


def test_schema_excludes_sensitive_columns(classification_df):
    schema = extract_schema(classification_df, sensitive_columns=["customer_name"])
    names = [c["name"] for c in schema["columns"]]
    assert "customer_name" not in names
    assert schema["excluded_sensitive_column_count"] == 1
    assert schema["column_count"] == len(classification_df.columns) - 1


def test_schema_reports_missing_pct_and_stats(classification_df):
    schema = extract_schema(classification_df)
    by_name = {c["name"]: c for c in schema["columns"]}
    assert by_name["monthly_charge"]["missing_pct"] > 0
    assert "mean" in by_name["monthly_charge"]
    assert "unique_count" in by_name["contract_type"]


def test_schema_row_column_counts(regression_df):
    schema = extract_schema(regression_df)
    assert schema["row_count"] == len(regression_df)
    assert schema["column_count"] == len(regression_df.columns)
