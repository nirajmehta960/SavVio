"""
Feature Engineering for Affordability.

Unifies User Financial Data, Product Data, and User Interactions (Reviews) to
calculate granular purchase affordability metrics.

Input:
    - data/features/reviews_featured.jsonl
    - data/features/financial_featured.csv
    - data/processed/product_preprocessed.jsonl

Output:
    - data/features/features.csv
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

def calculate_affordability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes affordability metrics for each User-Product interaction row.
    Metrics: Price-to-Income Ratio, Affordability Score, Residual Utility Score.
    """
    # Metric 1: Price-To-Income Ratio.
    # Definition: Percentage of one month's income required to buy this product.
    df["price_to_income_ratio"] = np.where(
        df["monthly_income"] > 0,
        df["price"] / df["monthly_income"],
        np.nan
    )

    # Metric 2: Affordability Score.
    # Definition: Remaining discretionary budget after purchasing the product.
    df["affordability_score"] = df["discretionary_income"] - df["price"]

    # Metric 3: Residual Utility Score.
    # Definition: The new 'financial_runway' (months) after the purchase is made.
    df["residual_utility_score"] = np.where(
        (df["monthly_expenses"] + df["monthly_emi"]) > 0,
        (df["savings_balance"] - df["price"]) / (df["monthly_expenses"] + df["monthly_emi"]),
        np.nan
    )
    
    return df

def run_affordability_features(reviews_path: str, financial_path: str, products_path: str, output_path: str) -> None:
    """
    Executes the Affordability Feature Pipeline.
    Merges interactions, financials, and products to compute final affordability scores.
    """
    logger.info("Starting affordability feature engineering pipeline...")
    
    if not (os.path.exists(reviews_path) and os.path.exists(financial_path) and os.path.exists(products_path)):
        logger.error(f"Missing input files. Checked: {reviews_path}, {financial_path}, {products_path}")
        return

    try:
        # Load input datasets.
        logger.info("Loading input datasets...")
        reviews_df = pd.read_json(reviews_path, lines=True)
        financial_df = pd.read_csv(financial_path)
        products_df = pd.read_json(products_path, lines=True)
        
        # Select columns required for downstream merge and feature computation.
        # Note: num_reviews is a product-level signal measuring popularity/volume.
        rev_cols = ["user_id", "product_id", "num_reviews", "rating_variance"] if "review_id" not in reviews_df.columns else ["review_id", "user_id", "product_id", "num_reviews", "rating_variance"]

        fin_cols = [
            "user_id", "monthly_income", "discretionary_income", "savings_balance", "monthly_expenses", "monthly_emi",
            "debt_to_income_ratio", "savings_rate", "monthly_expense_burden_ratio", "financial_runway"
        ]

        prod_cols = ["product_id", "price"]

        # Map review users to financial profiles for synthetic cross-dataset linkage.
        
        unique_fin_users = financial_df["user_id"].unique()
        n_fin_users = len(unique_fin_users)
        
        if n_fin_users > 0:
            logger.info(f"Mapping {len(reviews_df)} reviews to {n_fin_users} financial profiles...")
            # Deterministic hash mapping from review user_id -> financial user_id.
            def map_user(uid):
                idx = hash(uid) % n_fin_users
                return unique_fin_users[idx]
            
            reviews_df["mapped_fin_user_id"] = reviews_df["user_id"].apply(map_user)
        else:
            logger.error("No financial users found. Cannot map.")
            return

        # Merge datasets.
        logger.info("Merging datasets...")
        
        # Merge 1: reviews + financial on mapped id.
        merged_df = pd.merge(
            reviews_df[rev_cols + ["mapped_fin_user_id"]], 
            financial_df[fin_cols], 
            left_on="mapped_fin_user_id", 
            right_on="user_id", 
            how="inner"
        )
        
        # Merge 2: merged interactions + products on product_id.
        merged_df = pd.merge(merged_df, products_df[prod_cols], on="product_id", how="inner")
        
        logger.info(f"Merged features for {len(merged_df)} interactions.")

        # Compute affordability metrics.
        logger.info("Calculating affordability metrics...")
        merged_df = calculate_affordability(merged_df)
        
        # Cleanup values before persisting output.
        # Preserve NaN for undefined affordability ratios.
        affordability_cols = [
            "price_to_income_ratio",
            "affordability_score",
            "residual_utility_score",
        ]
        merged_df[affordability_cols] = merged_df[affordability_cols].replace([np.inf, -np.inf], np.nan)

        # Keep num_reviews numerically stable if present.
        if "num_reviews" in merged_df.columns:
            merged_df["num_reviews"] = pd.to_numeric(merged_df["num_reviews"], errors="coerce").fillna(0).astype(int)

        if "rating_variance" in merged_df.columns:
            merged_df["rating_variance"] = merged_df["rating_variance"].fillna(0.0)

        logger.info(
            "NaN summary - price_to_income_ratio: %d, residual_utility_score: %d",
            int(merged_df["price_to_income_ratio"].isna().sum()),
            int(merged_df["residual_utility_score"].isna().sum()),
        )

        # Persist output dataset.
        ensure_output_dir(output_path)
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Saved affordability features to {output_path}")
        logger.info(f"Sample:\n{merged_df[['price_to_income_ratio', 'affordability_score']].head()}")

    except Exception as e:
        logger.error(f"Failed to process affordability features: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Local execution stub (orchestrator is the primary entrypoint).
    from features.utils import get_processed_path, get_validated_path

    pass 
