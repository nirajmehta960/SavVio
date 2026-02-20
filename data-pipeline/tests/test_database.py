import pandas as pd
import pytest
import os
from preprocess_scripts.preprocess.financial import preprocess_financial_data


def test_database_export_format(tmp_path):
    """
    Test that the processed dataset conforms to expected
    database-ready export requirements.
    """

    test_data = {
        "user_id": [101],
        "monthly_income_usd": [8000],
        "monthly_expenses_usd": [2000],
        "savings_usd": [15000],
        "has_loan": ["No"],
        "loan_amount_usd": [0],
        "monthly_emi_usd": [0],
        "loan_interest_rate_pct": [0.0],
        "loan_term_months": [0],
        "credit_score": [800],
        "employment_status": ["Self-employed"],
        "region": ["West"],
    }

    in_path = tmp_path / "db_in.csv"
    out_path = tmp_path / "db_out.csv"

    pd.DataFrame(test_data).to_csv(in_path, index=False)

    processed = preprocess_financial_data(str(in_path), str(out_path))

    # Validation 1: Ensure output file is created successfully
    assert os.path.exists(out_path)

    # Validation 2: Ensure no NULL values exist (DB-safe)
    assert processed.isnull().sum().sum() == 0

    # Validation 3: Ensure user_id remains unique (Primary Key constraint)
    assert processed["user_id"].is_unique