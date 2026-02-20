import pandas as pd
import pytest
from preprocess_scripts.preprocess.financial import preprocess_financial_data


def test_validation_invalid_types(tmp_path):
    """
    Test that preprocessing correctly handles invalid data types
    (e.g., string values in numeric fields).
    """

    bad_data = pd.DataFrame({
        "user_id": [1],
        "monthly_income_usd": ["invalid_string"],  # Invalid numeric type
        "monthly_expenses_usd": [1000],
        "savings_usd": [500],
        "has_loan": ["No"],
        "loan_amount_usd": [0],
        "monthly_emi_usd": [0],
        "loan_interest_rate_pct": [0],
        "loan_term_months": [0],
        "credit_score": [700],
        "employment_status": ["employed"],
        "region": ["US"]
    })

    in_p = tmp_path / "bad_type.csv"
    out_p = tmp_path / "output.csv"

    bad_data.to_csv(in_p, index=False)

    processed = preprocess_financial_data(str(in_p), str(out_p))

    # The row should be removed due to invalid numeric input
    assert len(processed) == 0