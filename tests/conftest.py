from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    tenure_months = rng.integers(1, 72, n)
    monthly_charge = rng.normal(70, 20, n).round(2)
    contract_type = rng.choice(["month-to-month", "one-year", "two-year"], n)
    support_calls = rng.poisson(1.5, n)
    customer_name = [f"Customer {i}" for i in range(n)]

    churn_score = (
        -0.05 * tenure_months
        + 0.02 * monthly_charge
        + 0.6 * (contract_type == "month-to-month")
        + 0.3 * support_calls
        + rng.normal(0, 1, n)
    )
    churn = (churn_score > np.median(churn_score)).astype(int)

    df = pd.DataFrame(
        {
            "customer_name": customer_name,
            "tenure_months": tenure_months,
            "monthly_charge": monthly_charge,
            "contract_type": contract_type,
            "support_calls": support_calls,
            "churn": churn,
        }
    )
    df.loc[rng.choice(n, 15, replace=False), "monthly_charge"] = np.nan
    return df


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 250
    sqft = rng.integers(500, 4000, n)
    bedrooms = rng.integers(1, 6, n)
    age_years = rng.integers(0, 80, n)
    neighborhood = rng.choice(["A", "B", "C"], n)

    price = (
        150 * sqft
        + 8000 * bedrooms
        - 500 * age_years
        + rng.normal(0, 15000, n)
        + 20000
    )
    df = pd.DataFrame(
        {
            "sqft": sqft,
            "bedrooms": bedrooms,
            "age_years": age_years,
            "neighborhood": neighborhood,
            "price": price.round(2),
        }
    )
    df.loc[rng.choice(n, 10, replace=False), "age_years"] = np.nan
    return df


@pytest.fixture
def forecasting_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(100, 200, n)
    weekly_seasonality = 15 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = rng.normal(0, 5, n)
    sales = trend + weekly_seasonality + noise

    df = pd.DataFrame({"date": dates, "sales": sales.round(2)})
    return df


@pytest.fixture
def multi_entity_forecasting_df() -> pd.DataFrame:
    """Sales by store - a small entity/grouping-structure dataset for scope-strategy tests."""
    rng = np.random.default_rng(21)
    n_per_store = 60
    stores = ["store_1", "store_2", "store_3"]
    frames = []
    for i, store in enumerate(stores):
        dates = pd.date_range("2023-01-01", periods=n_per_store, freq="D")
        base = 50 + i * 100  # distinct baseline per store
        trend = np.linspace(0, 20, n_per_store)
        weekly = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
        noise = rng.normal(0, 3, n_per_store)
        sales = base + trend + weekly + noise
        frames.append(pd.DataFrame({"store_id": store, "date": dates, "sales": sales.round(2)}))
    return pd.concat(frames, ignore_index=True)
