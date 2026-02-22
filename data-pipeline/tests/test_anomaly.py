import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dags.src.preprocess.financial import preprocess_financial_data

def test_anomaly_extreme_values(tmp_path):
    """Test: Verify that data is dropped when values are outside business logic boundaries (e.g. credit score is negative or too high)"""
    test_data = {
        "user_id": [1, 2],
        "monthly_income_usd": [5000, 5000],
        "monthly_expenses_usd": [1000, 1000],
        "savings_usd": [500, 500],
        "has_loan": ["No", "No"],
        "loan_amount_usd": [0, 0],
        "monthly_emi_usd": [0, 0],
        "loan_interest_rate_pct": [0, 0],
        "loan_term_months": [0, 0],
        "credit_score": [999, -50],  # Both values are outside the 300-850 standard range
        "employment_status": ["FT", "FT"],
        "region": ["US", "US"]
    }
    
    in_p, out_p = tmp_path / "anomaly_in.csv", tmp_path / "anomaly_out.csv"
    pd.DataFrame(test_data).to_csv(in_p, index=False)
    
    processed = preprocess_financial_data(str(in_p), str(out_p))
    
    # Verification: Both anomalous records should be filtered out, resulting in empty output
    assert len(processed) == 0