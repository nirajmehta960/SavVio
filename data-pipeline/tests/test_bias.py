import pandas as pd
import pytest
from preprocess_scripts.preprocess.financial import preprocess_financial_data


def test_bias_region_distribution(tmp_path):
    """
    Test that preprocessing does not disproportionately remove
    records from specific regions.
    """

    # Create a balanced dataset across regions
    regions = ["North", "South", "East", "West"]
    test_rows = []

    # Create 100 rows total (25 per region)
    for i, region in enumerate(regions * 25):
        test_rows.append({
            "user_id": i,
            "monthly_income_usd": 5000,
            "monthly_expenses_usd": 1000,
            "savings_usd": 500,
            "has_loan": "No",
            "loan_amount_usd": 0,
            "monthly_emi_usd": 0,
            "loan_interest_rate_pct": 0,
            "loan_term_months": 0,
            "credit_score": 700,
            "employment_status": "Full-time",
            "region": region
        })

    in_path = tmp_path / "bias_in.csv"
    out_path = tmp_path / "bias_out.csv"

    pd.DataFrame(test_rows).to_csv(in_path, index=False)

    processed = preprocess_financial_data(str(in_path), str(out_path))

    # Validation 1: Ensure all regions still exist after preprocessing
    unique_regions = processed["region"].unique()
    assert set(unique_regions) == set(regions)

    # Validation 2: Ensure each region still has at least one record
    region_counts = processed["region"].value_counts()
    for region in regions:
        assert region_counts[region] > 0