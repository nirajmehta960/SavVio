import pytest
import pandas as pd
import os
import sys
from pathlib import Path
from unittest.mock import patch

# --- Magic trick: Ensure Python can locate dags/src ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dags.src.preprocess.financial import _load_input_to_df, preprocess_financial_data

# =============================================================================
# 1. Tests for _load_input_to_df
# =============================================================================

def test_load_input_missing_file():
    """Scenario: Should raise FileNotFoundError when file does not exist"""
    with pytest.raises(FileNotFoundError):
        _load_input_to_df("nonexistent_file.csv")

def test_load_input_csv(tmp_path):
    """Scenario: Successfully load a valid CSV file"""
    p = tmp_path / "test.csv"
    p.write_text("user_id,val\n1,10", encoding="utf-8")
    df = _load_input_to_df(str(p))
    assert len(df) == 1
    assert df.iloc[0]["user_id"] == 1

def test_load_input_jsonl(tmp_path):
    """Scenario: Successfully load a valid JSONL file"""
    p = tmp_path / "test.jsonl"
    p.write_text('{"user_id": 1, "val": 10}\n{"user_id": 2, "val": 20}', encoding="utf-8")
    df = _load_input_to_df(str(p))
    assert len(df) == 2

# =============================================================================
# 2. Tests for preprocess_financial_data
# =============================================================================

@patch("dags.src.preprocess.financial.pd.DataFrame.to_csv")
def test_preprocess_credit_score_out_of_range(mock_to_csv, tmp_path):
    """Scenario: Credit scores outside 300–850 should be removed"""
    p = tmp_path / "data.csv"
    p.write_text(
        "user_id,monthly_income,monthly_expenses,credit_score\n"
        "1,1000,500,700\n"
        "2,1000,500,200\n"
        "3,1000,500,900",
        encoding="utf-8"
    )
    
    out_path = str(tmp_path / "out.csv")
    df = preprocess_financial_data(str(p), out_path)
    
    assert len(df) == 1
    assert df.iloc[0]["user_id"] == 1
    mock_to_csv.assert_called_once()

@patch("dags.src.preprocess.financial.pd.DataFrame.to_csv")
def test_preprocess_invalid_critical_fields(mock_to_csv, tmp_path):
    """Scenario: Rows with invalid numeric values should be removed"""
    p = tmp_path / "data.csv"
    p.write_text(
        "user_id,monthly_income,monthly_expenses,credit_score\n"
        "1,1000,500,700\n"
        "2,invalid,500,700\n"
        "3,1000,,700",
        encoding="utf-8"
    )
    
    out_path = str(tmp_path / "out.csv")
    df = preprocess_financial_data(str(p), out_path)
    
    assert len(df) == 1
    assert df.iloc[0]["user_id"] == 1

@patch("dags.src.preprocess.financial.pd.DataFrame.to_csv")
def test_preprocess_full_pipeline_renaming_and_alias(mock_to_csv, tmp_path):
    """Scenario: Test column renaming, has_loan conversion, and alias generation"""
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"user_id": 1, "income_usd": 5000, "expenses_usd": 2000, "credit_score": 750, "has_loan": "Yes"}',
        encoding="utf-8"
    )
    
    out_path = str(tmp_path / "out.csv")
    df = preprocess_financial_data(str(p), out_path)
    
    assert len(df) == 1
    assert "monthly_income" in df.columns
    assert df.iloc[0]["monthly_income"] == 5000
    assert df.iloc[0]["has_loan"] == 1
    assert "income_usd" in df.columns