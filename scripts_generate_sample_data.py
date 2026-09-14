"""One-off script to generate sample_data/*.csv for manual/demo testing.

Not part of the pipeline itself - run once: python scripts_generate_sample_data.py
"""
import numpy as np
import pandas as pd

OUT_DIR = "sample_data"


def make_churn_classification():
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

    return pd.DataFrame(
        {
            "customer_name": customer_name,
            "tenure_months": tenure_months,
            "monthly_charge": monthly_charge,
            "contract_type": contract_type,
            "support_calls": support_calls,
            "churn": churn,
        }
    )


def make_house_price_regression():
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
    return pd.DataFrame(
        {
            "sqft": sqft,
            "bedrooms": bedrooms,
            "age_years": age_years,
            "neighborhood": neighborhood,
            "price": price.round(2),
        }
    )


def make_sales_forecasting():
    rng = np.random.default_rng(11)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(100, 200, n)
    weekly_seasonality = 15 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = rng.normal(0, 5, n)
    sales = trend + weekly_seasonality + noise
    return pd.DataFrame({"date": dates, "sales": sales.round(2)})


if __name__ == "__main__":
    make_churn_classification().to_csv(f"{OUT_DIR}/churn_classification.csv", index=False)
    make_house_price_regression().to_csv(f"{OUT_DIR}/house_price_regression.csv", index=False)
    make_sales_forecasting().to_csv(f"{OUT_DIR}/sales_forecasting.csv", index=False)
    print("Sample datasets written to sample_data/")
