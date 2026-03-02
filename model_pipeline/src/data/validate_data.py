"""
Data Validation for the Model Pipeline.

Validates that data loaded from either CSV or PostgreSQL meets the schema
and quality expectations required by the training pipeline. Run this
after loading data and before passing it to feature engineering / training.

Checks performed:
    1. Required columns exist
    2. No unexpected nulls in critical columns
    3. Data types are correct (numeric vs categorical)
    4. Value ranges are reasonable (no negative incomes, credit scores in range, etc.)
    5. Minimum row count to avoid training on empty/tiny datasets
    6. Target variable distribution is not completely degenerate

Usage:
    from data.validate_data import validate_financial_data, validate_products

    df = load_data()  # or load_financial_profiles()
    validate_financial_data(df)  # raises ValidationError on failure
"""

import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when loaded data fails validation checks."""
    pass


# ---------------------------------------------------------------------------
# Expected schemas
# ---------------------------------------------------------------------------

FINANCIAL_REQUIRED_COLUMNS = [
    "user_id",
    "monthly_income",
    "monthly_expenses",
    "savings_balance",
    "has_loan",
    "credit_score",
    "employment_status",
    "region",
    "discretionary_income",
    "debt_to_income_ratio",
    "saving_to_income_ratio",
    "monthly_expense_burden_ratio",
    "emergency_fund_months",
]

FINANCIAL_NUMERIC_COLUMNS = [
    "monthly_income",
    "monthly_expenses",
    "savings_balance",
    "credit_score",
    "discretionary_income",
    "debt_to_income_ratio",
    "saving_to_income_ratio",
    "monthly_expense_burden_ratio",
    "emergency_fund_months",
]

FINANCIAL_NO_NULL_COLUMNS = [
    "user_id",
    "monthly_income",
    "monthly_expenses",
    "credit_score",
]

PRODUCT_REQUIRED_COLUMNS = [
    "product_id",
    "product_name",
    "price",
]

PRODUCT_NUMERIC_COLUMNS = [
    "price",
    "average_rating",
    "rating_number",
]

# Reasonable value ranges
VALUE_RANGES = {
    "monthly_income": (0, 1_000_000),
    "monthly_expenses": (0, 1_000_000),
    "credit_score": (300, 850),
    "debt_to_income_ratio": (0, 100),
    "saving_to_income_ratio": (-50, 500),
    "emergency_fund_months": (-50, 500),
    "price": (0, 10_000_000),
    "average_rating": (0, 5),
}

MIN_ROWS_FINANCIAL = 100
MIN_ROWS_PRODUCTS = 10


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def _check_required_columns(df: pd.DataFrame, required: list, table_name: str) -> list:
    """Check that all required columns are present. Returns list of errors."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        return [f"[{table_name}] Missing required columns: {missing}"]
    return []


def _check_nulls(df: pd.DataFrame, no_null_cols: list, table_name: str) -> list:
    """Check for unexpected nulls in critical columns."""
    errors = []
    for col in no_null_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                pct = null_count / len(df) * 100
                errors.append(
                    f"[{table_name}] Column '{col}' has {null_count} nulls ({pct:.1f}%)"
                )
    return errors


def _check_numeric_types(df: pd.DataFrame, numeric_cols: list, table_name: str) -> list:
    """Check that numeric columns are actually numeric."""
    errors = []
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(
                f"[{table_name}] Column '{col}' expected numeric, got {df[col].dtype}"
            )
    return errors


def _check_value_ranges(df: pd.DataFrame, table_name: str) -> list:
    """Check that values fall within reasonable ranges."""
    errors = []
    for col, (lo, hi) in VALUE_RANGES.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            below = (df[col] < lo).sum()
            above = (df[col] > hi).sum()
            if below > 0:
                errors.append(
                    f"[{table_name}] Column '{col}' has {below} values below {lo}"
                )
            if above > 0:
                errors.append(
                    f"[{table_name}] Column '{col}' has {above} values above {hi}"
                )
    return errors


def _check_min_rows(df: pd.DataFrame, min_rows: int, table_name: str) -> list:
    """Check minimum row count."""
    if len(df) < min_rows:
        return [f"[{table_name}] Only {len(df)} rows, minimum expected: {min_rows}"]
    return []


def _check_duplicates(df: pd.DataFrame, id_col: str, table_name: str) -> list:
    """Check for duplicate IDs."""
    if id_col in df.columns:
        dups = df[id_col].duplicated().sum()
        if dups > 0:
            return [f"[{table_name}] {dups} duplicate values in '{id_col}'"]
    return []


def validate_financial_data(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> dict:
    """
    Validate a financial profiles DataFrame.

    Args:
        df: Financial data loaded from CSV or PostgreSQL.
        raise_on_error: If True, raises DataValidationError on any failure.

    Returns:
        dict with 'valid' (bool), 'errors' (list[str]), 'warnings' (list[str]),
        and 'summary' (dict of stats).
    """
    errors = []
    warnings = []

    # Schema checks
    errors += _check_required_columns(df, FINANCIAL_REQUIRED_COLUMNS, "financial")
    errors += _check_nulls(df, FINANCIAL_NO_NULL_COLUMNS, "financial")
    errors += _check_numeric_types(df, FINANCIAL_NUMERIC_COLUMNS, "financial")
    errors += _check_min_rows(df, MIN_ROWS_FINANCIAL, "financial")
    errors += _check_duplicates(df, "user_id", "financial")

    # Range checks (warnings, not hard errors)
    range_issues = _check_value_ranges(df, "financial")
    warnings += range_issues

    # Null fraction in optional columns
    for col in df.columns:
        if col not in FINANCIAL_NO_NULL_COLUMNS:
            null_pct = df[col].isnull().mean() * 100
            if null_pct > 50:
                warnings.append(
                    f"[financial] Column '{col}' is >50% null ({null_pct:.1f}%)"
                )

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "null_pct": df.isnull().mean().to_dict(),
    }

    is_valid = len(errors) == 0

    # Log results
    if is_valid:
        logger.info("Financial data validation PASSED — %d rows", len(df))
        print(f"[VALIDATION] Financial data PASSED — {len(df)} rows, {len(df.columns)} cols")
    else:
        logger.error("Financial data validation FAILED:\n%s", "\n".join(errors))
        print(f"[VALIDATION] Financial data FAILED — {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")

    if warnings:
        for w in warnings:
            logger.warning(w)
            print(f"  WARNING: {w}")

    if raise_on_error and not is_valid:
        raise DataValidationError(
            f"Financial data validation failed with {len(errors)} error(s):\n"
            + "\n".join(errors)
        )

    return {"valid": is_valid, "errors": errors, "warnings": warnings, "summary": summary}


def validate_products(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> dict:
    """
    Validate a products DataFrame.

    Args:
        df: Product data loaded from CSV or PostgreSQL.
        raise_on_error: If True, raises DataValidationError on any failure.

    Returns:
        dict with 'valid', 'errors', 'warnings', 'summary'.
    """
    errors = []
    warnings = []

    errors += _check_required_columns(df, PRODUCT_REQUIRED_COLUMNS, "products")
    errors += _check_nulls(df, ["product_id", "price"], "products")
    errors += _check_numeric_types(df, PRODUCT_NUMERIC_COLUMNS, "products")
    errors += _check_min_rows(df, MIN_ROWS_PRODUCTS, "products")
    errors += _check_duplicates(df, "product_id", "products")

    range_issues = _check_value_ranges(df, "products")
    warnings += range_issues

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
    }

    is_valid = len(errors) == 0

    if is_valid:
        logger.info("Products validation PASSED — %d rows", len(df))
        print(f"[VALIDATION] Products data PASSED — {len(df)} rows")
    else:
        logger.error("Products validation FAILED:\n%s", "\n".join(errors))
        print(f"[VALIDATION] Products data FAILED — {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")

    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")

    if raise_on_error and not is_valid:
        raise DataValidationError(
            f"Products validation failed with {len(errors)} error(s):\n"
            + "\n".join(errors)
        )

    return {"valid": is_valid, "errors": errors, "warnings": warnings, "summary": summary}


def validate_target_distribution(
    y: pd.Series,
    min_minority_pct: float = 5.0,
    raise_on_error: bool = True,
) -> dict:
    """
    Check that the target variable is not completely degenerate.

    Args:
        y: Binary or multi-class target series.
        min_minority_pct: Minimum percentage for the rarest class.
        raise_on_error: If True, raises on degenerate target.

    Returns:
        dict with 'valid', 'distribution', 'errors'.
    """
    dist = y.value_counts(normalize=True) * 100
    errors = []

    if len(dist) < 2:
        errors.append(f"Target has only {len(dist)} class(es) — model cannot learn")

    minority_pct = dist.min()
    if minority_pct < min_minority_pct:
        errors.append(
            f"Minority class is only {minority_pct:.1f}% — severe class imbalance "
            f"(threshold: {min_minority_pct}%)"
        )

    is_valid = len(errors) == 0

    if is_valid:
        print(f"[VALIDATION] Target distribution OK:\n{dist.to_string()}")
    else:
        print(f"[VALIDATION] Target distribution WARNING:")
        for e in errors:
            print(f"  {e}")

    if raise_on_error and not is_valid:
        raise DataValidationError("\n".join(errors))

    return {"valid": is_valid, "distribution": dist.to_dict(), "errors": errors}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO)

    # Add src/ to path so we can import config and data_loader
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from config import Config
    from data_loader import load_data, define_target

    print("=" * 60)
    print("Running Data Validation on temp_data/financial_featured.csv")
    print("=" * 60)

    try:
        df = load_data()
        result = validate_financial_data(df, raise_on_error=False)

        y = define_target(df)
        target_result = validate_target_distribution(y, raise_on_error=False)

        if result["valid"] and target_result["valid"]:
            print("\nAll validations PASSED.")
        else:
            print("\nSome validations FAILED — review errors above.")
            sys.exit(1)

    except FileNotFoundError:
        print("Data file not found. Run the Airflow pipeline first or place CSV in temp_data/.")
        sys.exit(1)
