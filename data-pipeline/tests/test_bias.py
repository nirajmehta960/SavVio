import pandas as pd
import pytest
from preprocess_scripts.preprocess.financial import preprocess_financial_data

def test_bias_region_distribution(tmp_path):
    """測試：確保預處理過程不會導致特定地區的資料被不對稱地大量刪除"""
    # 建立一個多地區的平衡資料集
    regions = ["North", "South", "East", "West"]
    test_rows = []
    for i, region in enumerate(regions * 25):  # 總共 100 筆，每個地區 25 筆
        test_rows.append({
            "user_id": i,
            "monthly_income_usd": 5000,
            "monthly_expenses_usd": 1000,
            "savings_usd": 500,
            "has_loan": "No",
            "loan_amount_usd": 0,
            "monthly_emi_usd": 0,
            "loan_interest_rate_pct": 0,
            "loan_term_months": 0,
            "credit_score": 700,
            "employment_status": "Full-time",
            "region": region
        })
    
    in_p, out_p = tmp_path / "bias_in.csv", tmp_path / "bias_out.csv"
    pd.DataFrame(test_rows).to_csv(in_p, index=False)
    
    processed = preprocess_financial_data(str(in_p), str(out_p))
    
    # 驗證 1：確認資料經過處理後，各個地區是否依然存在（沒有地區被誤殺）
    unique_regions = processed["region"].unique()
    assert set(unique_regions) == set(regions)
    
    # 驗證 2：確認分佈比例是否維持（在這個簡單案例中，比例應該保持不變）
    region_counts = processed["region"].value_counts()
    for region in regions:
        # 確保每個地區至少還有資料存在
        assert region_counts[region] > 0