import pandas as pd
import numpy as np
import pytest
from preprocess_scripts.preprocess.financial import preprocess_financial_data


def test_feature_creation_logic(tmp_path):
    """
    Test that preprocessing correctly creates expected features
    and applies proper type conversions.
    """

    # Create input data matching expected financial schema
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

    # Create temporary input/output paths
    in_p = tmp_path / "in.csv"
    out_p = tmp_path / "out.csv"

    pd.DataFrame(test_data).to_csv(in_p, index=False)

    # Run preprocessing
    processed = preprocess_financial_data(str(in_p), str(out_p))

    # Validation 1: Numeric transformation correctness
    # income_usd should match original monthly_income_usd
    assert processed.iloc[0]["income_usd"] == 10000

    # Validation 2: Categorical encoding correctness
    # 'Yes' should be converted to 1
    assert processed.iloc[0]["has_loan"] == 1

    # Validation 3: Type compatibility check
    # Ensure compatibility with both Python native and NumPy numeric types
    assert isinstance(
        processed.iloc[0]["credit_score"],
        (int, float, np.integer, np.floating)
    )

    # Validation 4: Ensure output file is generated
    assert out_p.exists()