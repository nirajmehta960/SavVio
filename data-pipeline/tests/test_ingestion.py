import pytest
import os
from preprocess_scripts.preprocess.financial import preprocess_financial_data

def test_ingestion_file_not_found():
    """測試：當 CSV 檔案不存在時，應該拋出 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        # 給一個絕對不存在的路徑
        preprocess_financial_data("non_existent_file.csv", "output.csv")

def test_ingestion_empty_file(tmp_path):
    """測試：當檔案是空的，應該會因為無法解析 CSV 而報錯"""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("") # 建立一個空檔案
    
    with pytest.raises(Exception):
        preprocess_financial_data(str(empty_file), "output.csv")