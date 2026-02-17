"""
Feature Engineering for Reviews.

Computes product-level quality signals derived from individual reviews.
The only metric that requires review-level data is rating_variance,
since the product metadata already provides average_rating and rating_number.

Input:  data/processed/review_preprocessed.jsonl
Output: data/features/product_rating_variance.csv
        (One row per product: product_id, rating_variance)

This output is merged onto the products table during database loading (Phase 14).
"""

import os
import logging
import pandas as pd

from features.utils import setup_logging, ensure_output_dir

# Configure module logging.
setup_logging()
logger = logging.getLogger(__name__)


def compute_rating_variance(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-product rating variance (standard deviation).

    A high variance indicates polarizing opinions (mixed quality signal).
    A low variance indicates consensus among reviewers.

    Args:
        reviews_df: DataFrame with at least 'product_id' and 'rating' columns.

    Returns:
        DataFrame with columns: product_id, rating_variance
    """
    if "product_id" not in reviews_df.columns or "rating" not in reviews_df.columns:
        logger.error("Required columns 'product_id' and 'rating' not found.")
        raise ValueError("Missing required columns: product_id, rating")

    product_variance = (
        reviews_df
        .groupby("product_id")["rating"]
        .std()
        .reset_index()
        .rename(columns={"rating": "rating_variance"})
    )

    # Products with a single review have NaN std dev; default to 0.0.
    product_variance["rating_variance"] = product_variance["rating_variance"].fillna(0.0)

    return product_variance


def run_review_features(reviews_path: str, output_path: str) -> None:
    """
    Executes the review feature engineering pipeline.
    Reads individual reviews, computes product-level rating variance,
    and saves the result for downstream DB loading.
    """
    logger.info("Starting review feature engineering pipeline...")

    if not os.path.exists(reviews_path):
        logger.error(f"Input file not found: {reviews_path}")
        return

    try:
        logger.info(f"Loading reviews from {reviews_path}...")
        reviews_df = pd.read_json(reviews_path, lines=True)
        logger.info(f"Loaded {len(reviews_df)} reviews.")

        # Compute product-level rating variance.
        product_variance = compute_rating_variance(reviews_df)
        logger.info(f"Computed rating_variance for {len(product_variance)} products.")

        logger.info(
            "rating_variance summary — mean: %.4f, median: %.4f, max: %.4f",
            float(product_variance["rating_variance"].mean()),
            float(product_variance["rating_variance"].median()),
            float(product_variance["rating_variance"].max()),
        )

        # Persist output.
        ensure_output_dir(output_path)
        product_variance.to_csv(output_path, index=False)
        logger.info(f"Saved product rating variance to {output_path}")

    except Exception as e:
        logger.error(f"Failed to process review features: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    REVIEWS_FILE = os.path.join(BASE_DIR, "data/processed/review_preprocessed.jsonl")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/features/product_rating_variance.csv")

    run_review_features(REVIEWS_FILE, OUTPUT_FILE)
