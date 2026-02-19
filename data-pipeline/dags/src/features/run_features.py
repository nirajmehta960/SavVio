"""
Feature Engineering Orchestrator.

Runs the complete feature engineering pipeline:
1. Financial Feature Engineering
2. Review Feature Engineering
3. Affordability Feature Engineering

Usage:
    python3 src/features/run_features.py
"""

import sys
import os
import logging
import argparse

# Add current script directory to import path.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from financial_features import run_financial_features
from review_features import run_review_features
from utils import setup_logging

# Configure module logging.
setup_logging()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Feature Engineering Pipeline")
    parser.add_argument("--skip-financial", action="store_true", help="Skip financial features")
    parser.add_argument("--skip-reviews", action="store_true", help="Skip review features")
    parser.add_argument("--skip-affordability", action="store_true", help="Skip affordability features")
    args = parser.parse_args()

    # Resolve data paths.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Financial input/output files.
    FIN_INPUT = os.path.join(DATA_DIR, "processed/financial_preprocessed.csv")
    FIN_OUTPUT = os.path.join(DATA_DIR, "features/financial_featured.csv")
    
    # Review and product input/output files.
    REV_INPUT = os.path.join(DATA_DIR, "processed/review_preprocessed.jsonl")
    REV_OUTPUT = os.path.join(DATA_DIR, "features/product_rating_variance.csv")
    
    PROD_INPUT = os.path.join(DATA_DIR, "processed/product_preprocessed.jsonl")
    
    # Consolidated affordability output.
    AFF_OUTPUT = os.path.join(DATA_DIR, "features/features.csv")

    # Step 1: Financial features.
    if not args.skip_financial:
        logger.info("--- Starting Financial Feature Engineering ---")
        try:
            run_financial_features(FIN_INPUT, FIN_OUTPUT)
            logger.info("Financial features complete.")
        except Exception as e:
            logger.error(f"Financial feature engineering failed: {e}")

    # Step 2: Review features.
    if not args.skip_reviews:
        logger.info("--- Starting Review Feature Engineering ---")
        try:
            run_review_features(REV_INPUT, REV_OUTPUT)
            logger.info("Review features complete.")
        except Exception as e:
            logger.error(f"Review feature engineering failed: {e}")

    logger.info("Feature engineering pipeline finished.")


# ---------- Airflow task wrappers ----------

def feature_financial_task(**context):
    """Airflow task: run financial feature engineering."""
    logger.info(">>> Running Financial Feature Engineering...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    run_financial_features(
        input_path=os.path.join(DATA_DIR, "processed/financial_preprocessed.csv"),
        output_path=os.path.join(DATA_DIR, "features/financial_featured.csv"),
    )
    logger.info(">>> Financial Feature Engineering: SUCCESS")


def feature_review_task(**context):
    """Airflow task: run review feature engineering."""
    logger.info(">>> Running Review Feature Engineering...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    run_review_features(
        reviews_path=os.path.join(DATA_DIR, "processed/review_preprocessed.jsonl"),
        output_path=os.path.join(DATA_DIR, "features/product_rating_variance.csv"),
    )
    logger.info(">>> Review Feature Engineering: SUCCESS")


if __name__ == "__main__":
    main()
