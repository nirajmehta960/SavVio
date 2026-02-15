"""
Feature Engineering Orchestrator.

Runs the complete feature engineering pipeline:
1. Financial Feature Engineering
2. Review Feature Engineering
3. Affordability Feature Engineering

Usage:
    python3 features-scripts/run_features.py
"""

import sys
import os
import logging
import argparse

# Add current script directory to import path.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from features.financial_features import run_financial_features
from features.review_features import run_review_features
from features.affordability_features import run_affordability_features
from features.utils import setup_logging

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
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Financial input/output files.
    FIN_INPUT = os.path.join(DATA_DIR, "processed/financial_preprocessed.csv")
    FIN_OUTPUT = os.path.join(DATA_DIR, "features/financial_featured.csv")
    
    # Review and product input/output files.
    REV_INPUT = os.path.join(DATA_DIR, "processed/review_preprocessed.jsonl")
    REV_OUTPUT = os.path.join(DATA_DIR, "features/reviews_featured.jsonl")
    
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

    # Step 3: Affordability features.
    if not args.skip_affordability:
        logger.info("--- Starting Affordability Feature Engineering ---")
        # Affordability requires financial + review feature outputs and product input data.
        
        # Validate required upstream outputs exist.
        if not (os.path.exists(REV_OUTPUT) and os.path.exists(FIN_OUTPUT)):
            logger.warning("Skipping affordability features because previous outputs are missing. Run full pipeline.")
        else:
            try:
                run_affordability_features(REV_OUTPUT, FIN_OUTPUT, PROD_INPUT, AFF_OUTPUT)
                logger.info("Affordability features complete.")
            except Exception as e:
                logger.error(f"Affordability feature engineering failed: {e}")

    logger.info("Feature engineering pipeline finished.")

if __name__ == "__main__":
    main()
