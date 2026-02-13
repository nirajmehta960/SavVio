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
# This assumes the script is located at data-pipeline/scripts/run_preprocessing.py
current_script_path = Path(__file__).resolve()
scripts_dir = current_script_path.parent
project_root = scripts_dir.parent

# Ensure running from project root so relative data paths (data/raw, data/processed) work correctly
if os.getcwd() != str(project_root):
    print(f"Changing working directory to: {project_root}")
    os.chdir(project_root)

# Add 'scripts' to python path to allow imports
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import preprocessing modules
# Note: These imports work because we added 'scripts' to sys.path
try:
    from preprocess import financial, product, review
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Try importing as if we are running as a module from root
    try:
        from scripts.preprocess import financial, product, review
    except ImportError:
        print("Critical: Could not import preprocessing modules. Check your python path.")
        sys.exit(1)

# Configure centralized logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
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

if __name__ == "__main__":
    run_pipeline()
