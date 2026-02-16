"""
Feature Engineering for Financial Data.

Transforms preprocessed financial records into actionable risk metrics.
Calculates ratios and scores used for downstream affordability analysis.

Input: data/processed/financial_preprocessed.csv
Output: data/features/financial_featured.csv
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

# Add parent script directory to import path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from features.utils import setup_logging, ensure_output_dir

# Configure module logging.
setup_logging()
logger = logging.getLogger(__name__)

def calculate_discretionary_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Discretionary Income (Monthly Income - Monthly Expenses).
    Definition: Money left after all monthly obligations (Income - (Expenses + EMI)).
    """
    df["discretionary_income"] = df["monthly_income"] - (df["monthly_expenses"] + df["monthly_emi"])
    return df

def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes financial risk ratios.
    Handles division by zero by assigning np.nan where denominators are invalid.
    """
    # Metric 1: Debt-to-income ratio.
    # Definition: Percentage of gross monthly income that goes toward paying debts.
    df["debt_to_income_ratio"] = np.where(
        df["monthly_income"] > 0,
        df["monthly_emi"] / df["monthly_income"],
        np.nan
    )

    # Metric 2: Savings rate.
    # Definition: Ratio of total savings wealth relative to monthly income.
    df["saving_to_income_ratio"] = np.where(
        df["monthly_income"] > 0,
        df["savings_balance"] / df["monthly_income"],
        np.nan
    )

    # Metric 3: Monthly Expense Burden Ratio.
    # Definition: Percentage of income consumed by all monthly outflows (living costs + debt).
    df["monthly_expense_burden_ratio"] = np.where(
        df["monthly_income"] > 0,
        (df["monthly_expenses"] + df["monthly_emi"]) / df["monthly_income"],
        np.nan
    )

    # Metric 4: Financial Runway.
    # Definition: Number of months a user could survive on their current savings if they lost their income.
    df["financial_runway"] = np.where(
        (df["monthly_expenses"] + df["monthly_emi"]) > 0,
        df["savings_balance"] / (df["monthly_expenses"] + df["monthly_emi"]),
        np.nan
    )

    return df

def run_financial_features(input_path: str, output_path: str) -> None:
    """
    Executes the financial feature engineering pipeline.
    """
    logger.info("Starting financial feature engineering pipeline...")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")

        # Compute feature columns.
        df = calculate_discretionary_income(df)
        df = calculate_ratios(df)

        # Cleanup values before persisting output.
        # Keep undefined ratios as NaN; only normalize arithmetic output column.
        ratio_cols = ["debt_to_income_ratio", "saving_to_income_ratio", "monthly_expense_burden_ratio", "financial_runway"]
        df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], np.nan)
        df["discretionary_income"] = df["discretionary_income"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        numeric_cols = ["discretionary_income", *ratio_cols]

        # Persist output dataset.
        ensure_output_dir(output_path)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved featured data to {output_path}")
        logger.info(f"Sample features:\n{df[numeric_cols].head()}")

    except Exception as e:
        logger.error(f"Failed to process financial features: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Default local run paths.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_FILE = os.path.join(BASE_DIR, "data/processed/financial_preprocessed.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/features/financial_featured.csv")

    run_financial_features(INPUT_FILE, OUTPUT_FILE)
