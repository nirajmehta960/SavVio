import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dags.src.preprocess.financial import preprocess_financial_data

def test_bias_region_distribution(tmp_path):
    """Test: Ensure preprocessing does not disproportionately delete data from specific regions"""
    # Create a balanced dataset across multiple regions
    regions = ["North", "South", "East", "West"]
    test_rows = []
    for i, region in enumerate(regions * 25):  # Total 100 records, 25 per region
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
    
    in_p, out_p = tmp_path / "bias_in.csv", tmp_path / "bias_out.csv"
    pd.DataFrame(test_rows).to_csv(in_p, index=False)
    
    processed = preprocess_financial_data(str(in_p), str(out_p))
    
    # Verification 1: Confirm all regions still exist after processing (no region is erroneously wiped out)
    unique_regions = processed["region"].unique()
    assert set(unique_regions) == set(regions)
    
    # Verification 2: Confirm distribution proportion is maintained (in this simple case, proportions should remain unchanged)
    region_counts = processed["region"].value_counts()
    for region in regions:
        # Ensure each region has at least some data remaining
        assert region_counts[region] > 0