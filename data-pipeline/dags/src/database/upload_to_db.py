"""
Upload finalized (feature-engineered) data to PostgreSQL.

Loads three datasets into their respective tables:
    - financial_featured.csv   (CSV)   → financial_profiles
    - products_featured.jsonl  (JSONL) → products
    - reviews_featured.jsonl   (JSONL) → reviews

Products must be loaded before reviews (FK dependency).
"""

import json
import os
import logging
import pandas as pd
from sqlalchemy import text

from src.database.db_connection import get_engine
from src.database.db_schema import create_tables

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mapping: source field → DB column
# Only mapped columns get pushed to the database.
# Adjust keys if your file headers differ.
# ---------------------------------------------------------------------------

'''
Column mapping is quite important because it's your safety net — it ensures only the columns you expect end up in the database,
and it handles any naming mismatches between your JSONL field names and your DB column names. 
Without it, you risk pushing extra/unexpected columns into to_sql() which would throw errors.
'''

FINANCIAL_COLS = {
    "user_id": "user_id",
    "monthly_income": "monthly_income",
    "monthly_expenses": "monthly_expenses",
    "savings_balance": "savings_balance",
    "has_loan": "has_loan",
    "loan_amount": "loan_amount",
    "monthly_emi": "monthly_emi",
    "loan_interest_rate": "loan_interest_rate",
    "loan_term_months": "loan_term_months",
    "credit_score": "credit_score",
    "employment_status": "employment_status",
    "region": "region",
    # Feature-engineered (included if present)
    "discretionary_income": "discretionary_income",
    "debt_to_income_ratio": "debt_to_income_ratio",
    "saving_to_income_ratio": "saving_to_income_ratio",
    "monthly_expense_burden_ratio": "monthly_expense_burden_ratio",
    "emergency_fund_months": "emergency_fund_months",
}

PRODUCT_COLS = {
    "product_id": "product_id",
    "product_name": "product_name",
    "price": "price",
    "average_rating": "average_rating",
    "rating_number": "rating_number",
    "rating_variance": "rating_variance",
    "description": "description",
    "features": "features",
    "details": "details",       # stored as JSONB
    "category": "category",
}

REVIEW_COLS = {
    "user_id": "user_id",
    "asin": "asin",
    "product_id": "product_id",
    "rating": "rating",
    "review_title": "review_title",
    "review_text": "review_text",
    "verified_purchase": "verified_purchase",
    "helpful_vote": "helpful_vote",
}


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file."""
    df = pd.read_csv(path)
    logger.info("Read %d rows from CSV: %s", len(df), path)
    return df


def _read_jsonl(path: str) -> pd.DataFrame:
    """Read a JSONL file (one JSON object per line)."""
    # df = pd.read_json(path, lines=True)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb > 100:
        chunks = pd.read_json(path, lines=True, chunksize=50_000)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_json(path, lines=True)
    logger.info("Read %d rows from JSONL: %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_and_rename(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Keep only columns that exist in both the source and the mapping."""
    available = {k: v for k, v in col_map.items() if k in df.columns}
    missing = [k for k in col_map if k not in df.columns]
    if missing:
        logger.warning("Columns not found in source (skipped): %s", missing)
    return df[list(available.keys())].rename(columns=available)


def _truncate_table(engine, table_name: str):
    """Truncate a table (cascade to handle FKs)."""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
    logger.info("Truncated table: %s", table_name)


def _ensure_jsonb(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Ensure a column contains valid JSON strings for JSONB storage.
    If values are already dicts, serialize them. If strings, validate.
    """
    if col not in df.columns:
        return df

    def to_json_str(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        if isinstance(val, str):
            return val  # assume already valid JSON string
        return json.dumps(val)

    df[col] = df[col].apply(to_json_str)
    return df


# ---------------------------------------------------------------------------
# Per-table loaders
# ---------------------------------------------------------------------------

def load_financial(engine, csv_path: str, truncate: bool = True) -> int:
    """Load financial profiles from CSV into financial_profiles table."""
    df = _read_csv(csv_path)
    df = _select_and_rename(df, FINANCIAL_COLS)

    if truncate:
        _truncate_table(engine, "financial_profiles")

    df.to_sql(
        "financial_profiles",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    logger.info("Loaded %d financial profiles", len(df))
    return len(df)


def load_products(engine, jsonl_path: str, truncate: bool = True) -> int:
    """Load products from JSONL into products table."""
    df = _read_jsonl(jsonl_path)
    df = _select_and_rename(df, PRODUCT_COLS)

    # Ensure text columns are strings
    for col in ["description", "features"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Serialize details as JSON string for JSONB column
    df = _ensure_jsonb(df, "details")

    if truncate:
        _truncate_table(engine, "products")

    df.to_sql(
        "products",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    logger.info("Loaded %d products", len(df))
    return len(df)


def load_reviews(engine, jsonl_path: str, truncate: bool = True) -> int:
    """Load reviews from JSONL into reviews table."""
    df = _read_jsonl(jsonl_path)
    df = _select_and_rename(df, REVIEW_COLS)

    # Type fixes
    if "verified_purchase" in df.columns:
        df["verified_purchase"] = df["verified_purchase"].astype(bool)
    if "helpful_vote" in df.columns:
        df["helpful_vote"] = df["helpful_vote"].fillna(0).astype(int)

    # Filter out reviews whose product_id doesn't exist in the products table
    # to avoid ForeignKeyViolation errors
    with engine.connect() as conn:
        existing_ids = pd.read_sql(
            text("SELECT product_id FROM products"), conn
        )["product_id"].tolist()
    existing_ids_set = set(existing_ids)
    before_count = len(df)
    df = df[df["product_id"].isin(existing_ids_set)]
    dropped = before_count - len(df)
    if dropped:
        logger.warning(
            "Dropped %d orphaned reviews (product_id not in products table)",
            dropped,
        )

    if truncate:
        _truncate_table(engine, "reviews")

    df.to_sql(
        "reviews",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    logger.info("Loaded %d reviews", len(df))
    return len(df)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def load_all(
    financial_path: str,
    products_path: str,
    reviews_path: str,
    env: str = "dev",
    truncate: bool = True,
):
    """
    Load all three datasets into PostgreSQL in FK-safe order.
    Products first, then reviews (reviews reference products).
    """
    engine = get_engine(env)
    create_tables(engine)

    n_fin  = load_financial(engine, financial_path, truncate)
    n_prod = load_products(engine, products_path, truncate)
    n_rev  = load_reviews(engine, reviews_path, truncate)

    summary = {
        "financial_profiles": n_fin,
        "products": n_prod,
        "reviews": n_rev,
    }
    logger.info("Upload complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI - optional, for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Upload featured data to PostgreSQL")
    parser.add_argument("--financial", required=True, help="Path to financial CSV (e.g., data/featured/financial_featured.csv)")
    parser.add_argument("--products", required=True, help="Path to products JSONL (e.g., data/featured/products_featured.jsonl)")
    parser.add_argument("--reviews", required=True, help="Path to reviews JSONL (e.g., data/featured/reviews_featured.jsonl)")
    parser.add_argument("--env", default="dev", choices=["dev", "prod"])
    parser.add_argument("--no-truncate", action="store_true", help="Append instead of replace")
    args = parser.parse_args()

    result = load_all(
        args.financial,
        args.products,
        args.reviews,
        env=args.env,
        truncate=not args.no_truncate,
    )
    print("Upload summary:", result)
