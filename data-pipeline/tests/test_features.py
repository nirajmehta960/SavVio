import pandas as pd
import numpy as np
import pytest
from preprocess_scripts.preprocess.financial import preprocess_financial_data

def test_feature_creation_logic(tmp_path):
    """測試：驗證預處理後的資料是否包含正確的特徵欄位與型別"""
    # 建立符合 financial.py 預期的原始欄位資料
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
    
    # 建立臨時的輸入與輸出路徑
    in_p = tmp_path / "in.csv"
    out_p = tmp_path / "out.csv"
    pd.DataFrame(test_data).to_csv(in_p, index=False)
    
    # 執行預處理函式
    processed = preprocess_financial_data(str(in_p), str(out_p))
    
    # 驗證 1：數值是否正確轉換（例如 income_usd 是否對應原始的 10000）
    assert processed.iloc[0]["income_usd"] == 10000
    
    # 驗證 2：類別是否正確轉換（'Yes' 應變為 1）
    assert processed.iloc[0]["has_loan"] == 1
    
    # 驗證 3：型別檢查（修正 NumPy 型別相容性問題）
    # 使用 np.integer 與 np.floating 以同時支援 Python 內建與 NumPy 的數字型別
    assert isinstance(processed.iloc[0]["credit_score"], (int, float, np.integer, np.floating))

    # 驗證 4：確保輸出檔案確實存在
    assert out_p.exists()