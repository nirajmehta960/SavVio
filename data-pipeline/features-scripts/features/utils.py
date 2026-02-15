"""Shared utilities for feature engineering."""
import logging
import os
import sys

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for feature scripts."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

def get_processed_path(filename: str, base_dir: str = None) -> str:
    """Return path to a processed data file."""
    if base_dir is None:
        # Resolve project root relative to this file (data-pipeline root).
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    
    return os.path.join(base_dir, "data/processed", filename)

def get_validated_path(filename: str, base_dir: str = None) -> str:
    """Return path to a validated data file (output of feature engineering)."""
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    
    return os.path.join(base_dir, "data/validated", filename)

def ensure_output_dir(file_path: str) -> None:
    """Ensure the directory for file_path exists."""
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
