import pytest
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dags.src.preprocess.financial import preprocess_financial_data

def test_ingestion_file_not_found():
    """Test: When CSV file does not exist, should raise FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        # Provide a path that definitely does not exist
        preprocess_financial_data("non_existent_file.csv", "output.csv")

def test_ingestion_empty_file(tmp_path):
    """Test: When file is empty, should error out due to failure to parse CSV"""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("") # Create an empty file
    
    with pytest.raises(Exception):
        preprocess_financial_data(str(empty_file), "output.csv")