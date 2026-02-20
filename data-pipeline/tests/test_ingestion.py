import pytest
import os
from preprocess_scripts.preprocess.financial import preprocess_financial_data


def test_ingestion_file_not_found():
    """
    Test that a FileNotFoundError is raised
    when the input CSV file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        preprocess_financial_data("non_existent_file.csv", "output.csv")


def test_ingestion_empty_file(tmp_path):
    """
    Test that preprocessing raises an exception
    when the input CSV file is empty.
    """
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")  # Create an empty file

    with pytest.raises(Exception):
        preprocess_financial_data(str(empty_file), "output.csv")