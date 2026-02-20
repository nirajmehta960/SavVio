import pandas as pd
import pytest
from preprocess_scripts.preprocess.financial import preprocess_financial_data

def test_validation_invalid_types(tmp_path):
    """測試：當數值欄位包含文字時，系統應該能正確處理（轉為 NaN 或報錯）"""
    bad_data = pd.DataFrame({
        "user_id": [1],
        "monthly_income_usd": ["invalid_string"],  # 錯誤的型別
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
    # 驗證該筆資料是否被剔除（因為收入不是有效數字）
    assert len(processed) == 0