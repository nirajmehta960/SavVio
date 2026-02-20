import pandas as pd
import pytest
from preprocess_scripts.preprocess.financial import preprocess_financial_data


def test_financial_columns_exist(tmp_path):
    """
    Test that preprocessing produces a non-empty dataset
    and includes expected transformed columns.
    """

    # 1. Create sample input data
    data = pd.DataFrame({
        "user_id": [1],
        "monthly_income_usd": [5000],
        "monthly_expenses_usd": [1000],
        "savings_usd": [500],
        "has_loan": [0],
        "loan_amount_usd": [0],
        "monthly_emi_usd": [0],
        "loan_interest_rate_pct": [0],
        "loan_term_months": [0],
        "credit_score": [700],
        "employment_status": ["employed"],
        "region": ["US"]
    })

    # 2. Define temporary file paths
    input_csv = tmp_path / "test_input.csv"
    output_csv = tmp_path / "test_output.csv"

    # 3. Save input data as CSV
    data.to_csv(input_csv, index=False)

    # 4. Run preprocessing
    processed = preprocess_financial_data(str(input_csv), str(output_csv))

    # 5. Validate results
    assert not processed.empty
    assert "income_usd" in processed.columns