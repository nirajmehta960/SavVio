import pandas as pd
import pytest
import os
from preprocess_scripts.preprocess.financial import preprocess_financial_data

def test_database_export_format(tmp_path):
    """測試：驗證預處理後的資料格式是否符合資料庫匯入要求"""
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
    
    # 驗證 1：檔案是否成功產出（準備好被 upload-to-db.py 讀取）
    assert os.path.exists(out_p)
    
    # 驗證 2：檢查是否有任何 Null 值會導致資料庫寫入失敗
    assert processed.isnull().sum().sum() == 0
    
    # 驗證 3：檢查 user_id 是否為唯一的 Primary Key
    assert processed["user_id"].is_unique