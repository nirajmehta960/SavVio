import pandas as pd
import pytest
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dags.src.preprocess.financial import preprocess_financial_data

def test_database_export_format(tmp_path):
    """Test: Verify if preprocessed data format meets database injection requirements"""
    test_data = {
        "user_id": [101],
        "monthly_income_usd": [8000],
        "monthly_expenses_usd": [2000],
        "savings_usd": [15000],
        "has_loan": ["No"],
        "loan_amount_usd": [0],
        "monthly_emi_usd": [0],
        "loan_interest_rate_pct": 0.0,
        "loan_term_months": 0,
        "credit_score": [800],
        "employment_status": ["Self-employed"],
        "region": ["West"]
    }
    
    in_p = tmp_path / "db_in.csv"
    out_p = tmp_path / "db_out.csv"
    pd.DataFrame(test_data).to_csv(in_p, index=False)
    
    processed = preprocess_financial_data(str(in_p), str(out_p))
    
    # Verification 1: Whether output file was successfully created (ready to be read by upload-to-db.py)
    assert os.path.exists(out_p)
    
    # Verification 2: Check for any Null values that would cause database insert failure
    assert processed.isnull().sum().sum() == 0
    
    # Verification 3: Check if user_id is a unique Primary Key
    assert processed["user_id"].is_unique