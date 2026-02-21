"""
Entry point for SavVio bias detection analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

from financial_bias import analyze_financial_bias
from review_bias import analyze_review_bias


def setup_logging() -> logging.Logger:
    """
    Configure deterministic logging format for the analysis run.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger(__name__)


def resolve_repo_root() -> Path:
    """
    Resolve repository root reliably based on current file location.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Unable to resolve repository root from bias_analysis.py")


def main() -> None:
    logger = setup_logging()
    repo_root = resolve_repo_root()

    financial_path = repo_root / "data-pipeline" / "data" / "features" / "financial_featured.csv"
    review_path = repo_root / "data-pipeline" / "data" / "features" / "reviews_featured.jsonl"

    logger.info("Starting bias analysis...")
    logger.info("Financial feature input: %s", financial_path)
    logger.info("Review feature input: %s", review_path)

    financial_distributions, financial_flags = analyze_financial_bias(input_path=str(financial_path))
    review_distribution, review_flags, low_confidence_count = analyze_review_bias(input_path=str(review_path))

    logger.info("---- Financial Bias Distributions ----")
    for slice_name, dist_df in financial_distributions.items():
        logger.info("%s\n%s", slice_name, dist_df.to_string(index=False))
        if financial_flags[slice_name]:
            logger.warning("Underrepresented vulnerable groups in %s: %s", slice_name, ", ".join(financial_flags[slice_name]))

    logger.info("---- Review Bias Distribution ----")
    logger.info("\n%s", review_distribution.to_string(index=False))
    if review_flags:
        logger.warning("Underrepresented review groups: %s", ", ".join(review_flags))

    logger.info("Low-confidence review count (rating_variance == 0.0): %d", low_confidence_count)
    logger.info("Bias analysis completed successfully.")


if __name__ == "__main__":
    main()
