#!/usr/bin/env python3
"""
Orchestrator script for running all preprocessing tasks (Financial, Product, Review).
Run this script to execute the full preprocessing pipeline in order.
"""

import sys
import os
import logging
from pathlib import Path

# Fix path to ensure we can import 'preprocess' package
# This script is located at data_pipeline/dags/src/preprocess/run_preprocessing.py
current_script_path = Path(__file__).resolve()
src_dir = current_script_path.parent.parent          # .../dags/src/
dags_root = src_dir.parent                           # .../dags/

# Ensure running from dags root so relative data paths (data/raw, data/processed) work correctly
if os.getcwd() != str(dags_root):
    print(f"Changing working directory to: {dags_root}")
    os.chdir(dags_root)

# Add 'src' to python path to allow imports
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import preprocessing modules
# Note: These imports work because we added 'src' to sys.path
try:
    from preprocess import financial, product, review
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Try importing as if we are running as a module from dags root
    try:
        from src.preprocess import financial, product, review
    except ImportError:
        print("Critical: Could not import preprocessing modules. Check your python path.")
        sys.exit(1)

# Configure centralized logging
from src.utils import setup_logging
setup_logging()
logger = logging.getLogger("run_preprocessing")

def run_pipeline():
    logger.info("=" * 60)
    logger.info("STARTING DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # 1. Financial Data
    try:
        logger.info("\n>>> [1/3] Processing Financial Data...")
        financial.main()
        logger.info(">>> Financial Data Preprocessing: SUCCESS")
    except Exception as e:
        logger.error(f">>> Financial Data Preprocessing: FAILED\nError: {e}")
        sys.exit(1)

    # 2. Product Data
    try:
        logger.info("\n>>> [2/3] Processing Product Data...")
        product.main()
        logger.info(">>> Product Data Preprocessing: SUCCESS")
    except Exception as e:
        logger.error(f">>> Product Data Preprocessing: FAILED\nError: {e}")
        sys.exit(1)

    # 3. Review Data
    try:
        logger.info("\n>>> [3/3] Processing Review Data...")
        review.main()
        logger.info(">>> Review Data Preprocessing: SUCCESS")
    except Exception as e:
        logger.error(f">>> Review Data Preprocessing: FAILED\nError: {e}")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("ALL PREPROCESSING TASKS COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# Individual Airflow task functions (for parallel preprocessing)
# ──────────────────────────────────────────────────────────────

def preprocess_financial_task(**context):
    """Airflow task: preprocess financial data only."""
    logger.info(">>> Preprocessing Financial Data...")
    financial.main()
    logger.info(">>> Financial Data Preprocessing: SUCCESS")


def preprocess_product_task(**context):
    """Airflow task: preprocess product data only."""
    logger.info(">>> Preprocessing Product Data...")
    product.main()
    logger.info(">>> Product Data Preprocessing: SUCCESS")


def preprocess_review_task(**context):
    """Airflow task: preprocess review data only."""
    logger.info(">>> Preprocessing Review Data...")
    review.main()
    logger.info(">>> Review Data Preprocessing: SUCCESS")


if __name__ == "__main__":
    run_pipeline()
