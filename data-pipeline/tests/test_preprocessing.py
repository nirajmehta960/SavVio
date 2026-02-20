import pandas as pd
import pytest
from preprocess_scripts.preprocess.financial import preprocess_financial_data

def test_financial_columns_exist(tmp_path):
    # 1. 建立假資料
    data = pd.DataFrame({
        "user_id": [1],
        "monthly_income_usd": [5000],  # 注意：原始碼裡是用這個名稱
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

    # 2. 定義臨時的檔案路徑
    input_csv = tmp_path / "test_input.csv"
    output_csv = tmp_path / "test_output.csv"

    # 3. 將假資料存成 CSV (因為你的函式是讀取 CSV)
    data.to_csv(input_csv, index=False)

    # 4. 呼叫函式並傳入「輸入路徑」與「輸出路徑」
    processed = preprocess_financial_data(str(input_csv), str(output_csv))

    # 5. 驗證結果
    assert not processed.empty
    assert "income_usd" in processed.columns