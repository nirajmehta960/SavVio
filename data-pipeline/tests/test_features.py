import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dags.src.preprocess.financial import preprocess_financial_data

def test_feature_creation_logic(tmp_path):
    """Test: Verify if preprocessed data contains the correct feature columns and types"""
    # Create original column data matching financial.py expectations
    test_data = {
        "user_id": [1],
        "monthly_income_usd": [10000],
        "monthly_expenses_usd": [2000],
        "savings_usd": [5000],
        "has_loan": ["Yes"],
        "loan_amount_usd": [1000],
        "monthly_emi_usd": [100],
        "loan_interest_rate_pct": [5.0],
        "loan_term_months": [12],
        "credit_score": [750],
        "employment_status": ["Full-time"],
        "region": ["North"]
    }
    
    # Create temporary input and output paths
    in_p = tmp_path / "in.csv"
    out_p = tmp_path / "out.csv"
    pd.DataFrame(test_data).to_csv(in_p, index=False)
    
    # Execute preprocessing function
    processed = preprocess_financial_data(str(in_p), str(out_p))
    
    # Verification 1: Whether values are correctly translated (e.g. monthly_income corresponds to original 10000)
    assert processed.iloc[0]["monthly_income"] == 10000
    
    # Verification 2: Whether categories are correctly translated ('Yes' becomes 1)
    assert processed.iloc[0]["has_loan"] == 1
    
    # Verification 3: Type checking (fixes NumPy type compatibility issues)
    # Use np.integer and np.floating to simultaneously support Python built-ins and NumPy numeric types
    assert isinstance(processed.iloc[0]["credit_score"], (int, float, np.integer, np.floating))

    # Verification 4: Ensure the output file actually exists
    assert out_p.exists()