"""
Feature Engineering for Reviews.

Extracts quantifiable signals from unstructured review data (e.g., review length).
Prepares interaction logs for downstream merging.

Input: data/processed/review_preprocessed.jsonl
Output: data/features/reviews_featured.jsonl
"""

import os
import logging
import pandas as pd

from features.utils import setup_logging, ensure_output_dir

# Configure module logging.
setup_logging()
logger = logging.getLogger(__name__)

def run_review_features(reviews_path: str, output_reviews_path: str) -> None:
    """
    Executes the review feature engineering pipeline.
    Computes text-based features and validates data integrity.
    """
    logger.info("Starting review feature engineering pipeline...")
    
    if not os.path.exists(reviews_path):
        logger.error(f"Input file not found: {reviews_path}")
        return

    try:
        # Load input reviews.
        logger.info(f"Loading reviews from {reviews_path}...")
        reviews_df = pd.read_json(reviews_path, lines=True)
        
        # Compute review-level feature columns.
        # Metric 1: Number of Reviews.
        # Definition: Total count of reviews for the product (proxy for popularity).
        if "product_id" in reviews_df.columns:
            review_counts = reviews_df.groupby("product_id")["product_id"].transform("count")
            reviews_df["num_reviews"] = review_counts.fillna(0).astype(int)
        else:
            reviews_df["num_reviews"] = 0

        # Metric 2: Rating Variance.
        # Definition: Standard deviation of ratings for a product (proxy for consensus).
        if "rating" in reviews_df.columns and "product_id" in reviews_df.columns:
            # Calculate variance (std dev) per product
            rating_stats = reviews_df.groupby("product_id")["rating"].std().rename("rating_variance")
            
            # Merit back to original dataframe (1:1 map from product_id)
            reviews_df["rating_variance"] = reviews_df["product_id"].map(rating_stats)
            reviews_df["rating_variance"] = reviews_df["rating_variance"].fillna(0.0)
        else:
            logger.warning("Rating or Product ID column missing. Skipping rating_variance.")
            reviews_df["rating_variance"] = 0.0

        # Cleanup values before persisting output.
        reviews_df["num_reviews"] = pd.to_numeric(reviews_df["num_reviews"], errors="coerce").fillna(0).astype(int)
        reviews_df["num_reviews"] = reviews_df["num_reviews"].clip(lower=0)

        logger.info(
            "num_reviews summary -> min: %d, max: %d, mean: %.2f",
            int(reviews_df["num_reviews"].min()),
            int(reviews_df["num_reviews"].max()),
            float(reviews_df["num_reviews"].mean()),
        )
        
        logger.info(
            "rating_variance summary -> mean: %.4f, max: %.4f",
            float(reviews_df["rating_variance"].mean()),
            float(reviews_df["rating_variance"].max()),
        )
        
        # Persist output dataset.
        ensure_output_dir(output_reviews_path)
        
        # Save reviews.
        reviews_df.to_json(output_reviews_path, orient="records", lines=True)
        logger.info(f"Saved featured reviews to {output_reviews_path}")

    except Exception as e:
        logger.error(f"Failed to process review features: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Default local run paths.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    REVIEWS_FILE = os.path.join(BASE_DIR, "data/processed/review_preprocessed.jsonl")
    OUTPUT_REVIEWS = os.path.join(BASE_DIR, "data/features/reviews_featured.jsonl")

    run_review_features(REVIEWS_FILE, OUTPUT_REVIEWS)
