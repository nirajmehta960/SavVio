import pandas as pd
import pytest
from preprocess_scripts.preprocess.financial import preprocess_financial_data

def test_anomaly_extreme_values(tmp_path):
    """測試：驗證當數值超出業務邏輯範圍時（例如信用分數為負或過高），資料是否被剔除"""
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
        "credit_score": [999, -50],  # 這兩個數值都超出了 300-850 的規範範圍
        "employment_status": ["FT", "FT"],
        "region": ["US", "US"]
    }
    
    in_p, out_p = tmp_path / "anomaly_in.csv", tmp_path / "anomaly_out.csv"
    pd.DataFrame(test_data).to_csv(in_p, index=False)
    
    processed = preprocess_financial_data(str(in_p), str(out_p))
    
    # 驗證：這兩筆異常資料應該都要被過濾掉，結果應該為空
    assert len(processed) == 0