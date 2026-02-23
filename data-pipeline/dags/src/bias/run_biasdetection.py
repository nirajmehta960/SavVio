"""
Bias Detection Orchestrator for SavVio Pipeline.

Runs all three bias detectors in sequence:
1) Financial bias (financial_featured.csv)
2) Product bias (product_featured.jsonl)
3) Review bias (review_featured.jsonl)

Usage:
    python3 src/bias/run_biasdetection.py
"""

import sys
import os
import logging
import argparse

# Add current script directory to import path.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from financial_bias import run_financial_bias
from product_bias import run_product_bias
from review_bias import run_review_bias

# Configure module logging.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run all bias detectors.")
    parser.add_argument("--skip-financial", action="store_true", help="Skip financial bias detection.")
    parser.add_argument("--skip-product", action="store_true", help="Skip product bias detection.")
    parser.add_argument("--skip-review", action="store_true", help="Skip review bias detection.")
    args = parser.parse_args()

    # Resolve data paths (data is at dags/data).
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    logger.info("=" * 60)
    logger.info("STARTING BIAS DETECTION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Financial bias.
    if not args.skip_financial:
        logger.info("\n>>> [1/3] Running Financial Bias Detection...")
        try:
            run_financial_bias(
                processed_path=os.path.join(DATA_DIR, "processed/financial_preprocessed.csv"),
                featured_path=os.path.join(DATA_DIR, "features/financial_featured.csv"),
            )
            logger.info(">>> Financial Bias Detection: SUCCESS")
        except Exception as e:
            logger.error(f">>> Financial Bias Detection: FAILED\nError: {e}")

    # Step 2: Product bias.
    if not args.skip_product:
        logger.info("\n>>> [2/3] Running Product Bias Detection...")
        try:
            run_product_bias(
                preprocessed_path=os.path.join(DATA_DIR, "processed/product_preprocessed.jsonl"),
                featured_path=os.path.join(DATA_DIR, "features/product_featured.jsonl"),
            )
            logger.info(">>> Product Bias Detection: SUCCESS")
        except Exception as e:
            logger.error(f">>> Product Bias Detection: FAILED\nError: {e}")

    # Step 3: Review bias.
    if not args.skip_review:
        logger.info("\n>>> [3/3] Running Review Bias Detection...")
        try:
            run_review_bias(
                preprocessed_path=os.path.join(DATA_DIR, "processed/review_preprocessed.jsonl"),
                featured_path=os.path.join(DATA_DIR, "features/review_featured.jsonl"),
            )
            logger.info(">>> Review Bias Detection: SUCCESS")
        except Exception as e:
            logger.error(f">>> Review Bias Detection: FAILED\nError: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("BIAS DETECTION PIPELINE FINISHED")
    logger.info("=" * 60)


# ---------- Airflow task wrappers ----------

def bias_financial_task(**context):
    """Airflow task: run financial bias detection."""
    logger.info(">>> Running Financial Bias Detection...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    run_financial_bias(
        processed_path=os.path.join(DATA_DIR, "processed/financial_preprocessed.csv"),
        featured_path=os.path.join(DATA_DIR, "features/financial_featured.csv"),
    )
    logger.info(">>> Financial Bias Detection: SUCCESS")


def bias_product_task(**context):
    """Airflow task: run product bias detection."""
    logger.info(">>> Running Product Bias Detection...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    run_product_bias(
        preprocessed_path=os.path.join(DATA_DIR, "processed/product_preprocessed.jsonl"),
        featured_path=os.path.join(DATA_DIR, "features/product_featured.jsonl"),
    )
    logger.info(">>> Product Bias Detection: SUCCESS")


def bias_review_task(**context):
    """Airflow task: run review bias detection."""
    logger.info(">>> Running Review Bias Detection...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    run_review_bias(
        preprocessed_path=os.path.join(DATA_DIR, "processed/review_preprocessed.jsonl"),
        featured_path=os.path.join(DATA_DIR, "features/review_featured.jsonl"),
    )
    logger.info(">>> Review Bias Detection: SUCCESS")


if __name__ == "__main__":
    main()
