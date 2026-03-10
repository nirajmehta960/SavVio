"""
Training Data Generator — Scenario Creation & Labeling.

Generates synthetic user-product scenarios by pairing real financial
profiles with real products, computes the 6 financial features for each
pair, and labels each scenario GREEN/YELLOW/RED using the DecisionEngine.

Supports two sampling strategies:
    - stratified (default): Equal representation across income × price
      bracket combinations (3 income × 3 price = 9 cells) so the model
      sees balanced edge cases (e.g., low-income + premium product).
    - random: Pure uniform random pairing (legacy / quick experiments).

Output becomes the training dataset for downstream ML models.

Usage:
    from features.training_data_generator import generate_scenarios

    scenarios_df = generate_scenarios(financial_df, products_df, n_scenarios=50000)
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from deterministic_engine.financial_engine import DecisionEngine
from deterministic_engine.downgrade_engine import DowngradeEngine
from features.product_features import compute_product_features_batch
from features.review_features import compute_review_features_batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bracket definitions for stratified sampling
# ---------------------------------------------------------------------------

INCOME_BINS: List[float] = [0, 3_000, 7_000, float("inf")]
INCOME_LABELS: List[str] = ["low", "mid", "high"]

PRICE_BINS: List[float] = [0, 25, 200, float("inf")]
PRICE_LABELS: List[str] = ["budget", "mid", "premium"]


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def _sample_random(
    financial_profiles: pd.DataFrame,
    products: pd.DataFrame,
    n_scenarios: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pure uniform random pairing — legacy behaviour.
    
    Args:
        financial_profiles: DataFrame containing user financial profiles.
        products: DataFrame containing product data.
        n_scenarios: Total number of scenarios to generate.
        rng: NumPy random generator instance for reproducibility.
        
    Returns:
        A tuple of (sampled_users_df, sampled_products_df), both of length n_scenarios.
    """
    user_idx = rng.integers(0, len(financial_profiles), size=n_scenarios)
    prod_idx = rng.integers(0, len(products), size=n_scenarios)
    return (
        financial_profiles.iloc[user_idx].reset_index(drop=True),
        products.iloc[prod_idx].reset_index(drop=True),
    )


def _sample_stratified(
    financial_profiles: pd.DataFrame,
    products: pd.DataFrame,
    n_scenarios: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified pairing across income × price bracket combinations.

    Divides users into 3 income brackets and products into 3 price
    brackets, then samples equally from each of the 9 (income, price)
    cells so edge-case combinations are well represented.

    Empty cells are skipped and their quota is redistributed evenly
    across remaining cells.
    
    Args:
        financial_profiles: DataFrame containing user financial profiles.
        products: DataFrame containing product data.
        n_scenarios: Total number of scenarios to generate.
        rng: NumPy random generator instance for reproducibility.
        
    Returns:
        A tuple of (sampled_users_df, sampled_products_df), both of length n_scenarios.
    """
    # Assign brackets.
    income_bracket = pd.cut(
        financial_profiles["monthly_income"],
        bins=INCOME_BINS, labels=INCOME_LABELS, include_lowest=True,
    )
    price_bracket = pd.cut(
        products["price"],
        bins=PRICE_BINS, labels=PRICE_LABELS, include_lowest=True,
    )

    income_groups = {
        label: financial_profiles[income_bracket == label]
        for label in INCOME_LABELS
    }
    price_groups = {
        label: products[price_bracket == label]
        for label in PRICE_LABELS
    }

    # Identify non-empty cells.
    valid_cells = [
        (ig, pg)
        for ig in INCOME_LABELS
        for pg in PRICE_LABELS
        if len(income_groups[ig]) > 0 and len(price_groups[pg]) > 0
    ]

    if not valid_cells:
        raise ValueError(
            "No valid (income, price) bracket combinations — check your data."
        )

    # Distribute n_scenarios evenly; spread remainder across first cells.
    base, remainder = divmod(n_scenarios, len(valid_cells))
    cell_counts = [base + (1 if i < remainder else 0) for i in range(len(valid_cells))]

    user_chunks: List[pd.DataFrame] = []
    prod_chunks: List[pd.DataFrame] = []

    for (ig_label, pg_label), count in zip(valid_cells, cell_counts):
        # Draw reproducible integer seeds from the generator so
        # pd.DataFrame.sample() stays deterministic per cell.
        seed = int(rng.integers(0, 2**31))

        user_sample = income_groups[ig_label].sample(
            n=count, replace=True, random_state=seed,
        )
        prod_sample = price_groups[pg_label].sample(
            n=count, replace=True, random_state=seed + 1,
        )
        user_chunks.append(user_sample)
        prod_chunks.append(prod_sample)

        logger.debug(
            "Cell (%s income, %s price): %d scenarios",
            ig_label, pg_label, count,
        )

    users = pd.concat(user_chunks, ignore_index=True)
    prods = pd.concat(prod_chunks, ignore_index=True)

    logger.info(
        "Stratified sampling: %d valid cells out of 9, %d total scenarios",
        len(valid_cells), len(users),
    )
    return users, prods


# ---------------------------------------------------------------------------
# Feature computation & labeling
# ---------------------------------------------------------------------------

def _compute_features_and_label(
    users: pd.DataFrame,
    prods: pd.DataFrame,
    reviews: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Given paired user and product DataFrames of equal length, compute the
    6 financial features, apply the Layer 1 financial engine, and
    optionally apply Layer 2 downgrade logic using product/review
    features when reviews are provided.
    """
    engine = DecisionEngine()
    downgrade_engine = DowngradeEngine()

    scenarios = users.copy()
    scenarios["product_id"] = prods["product_id"].values
    scenarios["product_price"] = prods["price"].values

    # Keep product metadata for downstream use (but NOT for decision engine).
    for col in ("average_rating", "rating_number", "rating_variance"):
        if col in prods.columns:
            scenarios[col] = prods[col].values

    # Vectorized helper references for readability.
    price = scenarios["product_price"]
    income = scenarios["monthly_income"].replace(0, np.nan)
    savings = scenarios["savings_balance"]
    discretionary = scenarios["discretionary_income"]
    expenses = scenarios["monthly_expenses"]
    emi = scenarios["monthly_emi"]
    loan_amount = scenarios["loan_amount"].fillna(0)
    credit_score = scenarios["credit_score"].fillna(0)
    total_obligations = (expenses + emi).replace(0, np.nan)
    safe_price = price.replace(0, np.nan)

    # ── Financial features (6 computed) ──────────────────────────────────

    # Remaining discretionary budget after subtracting the product price.
    scenarios["affordability_score"] = discretionary - price

    # Product price as a fraction of monthly income.
    scenarios["price_to_income_ratio"] = price / income

    # Months of financial runway remaining after purchasing from savings.
    scenarios["residual_utility_score"] = (savings - price) / total_obligations

    # How many times over can savings cover the product price?
    scenarios["savings_to_price_ratio"] = savings / safe_price

    # Normalized net worth: positive = savings exceed debt.
    scenarios["net_worth_indicator"] = (savings - loan_amount) / income

    # Credit score projected onto 0–1 range.
    scenarios["credit_risk_indicator"] = (credit_score - 299) / 550.0

    # Label each scenario using the multi-condition deterministic engine (Layer 1).
    logger.info("Labeling %d scenarios with DecisionEngine (Layer 1)...", len(scenarios))
    scenarios["financial_label"] = scenarios.apply(engine.decide_row, axis=1)

    # ── Layer 2: Product & Review Features + Downgrade ──────────────────────
    if reviews is not None:
        logger.info("Computing Layer 2 product and review features for %d scenarios...", len(scenarios))

        # Product features — one row per product.
        unique_prods = prods.drop_duplicates(subset=["product_id"]).copy()
        product_feats_df = compute_product_features_batch(unique_prods)
        product_feats_df = product_feats_df.set_index("product_id")

        # Review features — aggregated one row per product.
        review_feats_df = compute_review_features_batch(reviews)

        # Merge Layer 2 features onto scenarios via product_id.
        scenarios = scenarios.merge(
            product_feats_df[
                [
                    "value_density",
                    "review_confidence",
                    "rating_polarization",
                    "quality_risk_score",
                    "cold_start_flag",
                    "price_category_rank",
                    "category_rating_deviation",
                ]
            ],
            left_on="product_id",
            right_index=True,
            how="left",
        )

        scenarios = scenarios.merge(
            review_feats_df[
                [
                    "verified_purchase_ratio",
                    "helpful_concentration",
                    "sentiment_spread",
                    "review_depth_score",
                    "reviewer_diversity",
                    "extreme_rating_ratio",
                ]
            ],
            left_on="product_id",
            right_index=True,
            how="left",
        )

        logger.info("Applying DowngradeEngine (Layer 2)...")

        def _apply_downgrade(row: pd.Series) -> str:
            class PF:
                pass

            class RF:
                pass

            pf = PF()
            pf.value_density = row["value_density"]
            pf.review_confidence = row["review_confidence"]
            pf.rating_polarization = row["rating_polarization"]
            pf.quality_risk_score = row["quality_risk_score"]
            pf.cold_start_flag = int(row["cold_start_flag"])
            pf.price_category_rank = row["price_category_rank"]
            pf.category_rating_deviation = row["category_rating_deviation"]

            rf = RF()
            rf.verified_purchase_ratio = row["verified_purchase_ratio"]
            rf.helpful_concentration = row["helpful_concentration"]
            rf.sentiment_spread = row["sentiment_spread"]
            rf.review_depth_score = row["review_depth_score"]
            rf.reviewer_diversity = row["reviewer_diversity"]
            rf.extreme_rating_ratio = row["extreme_rating_ratio"]

            result = downgrade_engine.evaluate(
                financial_label=row["financial_label"],
                product_features=pf,
                review_features=rf,
            )
            return result.final_label

        scenarios["label"] = scenarios.apply(_apply_downgrade, axis=1)
    else:
        # Legacy behaviour: no Layer 2, label equals financial label.
        scenarios["label"] = scenarios["financial_label"]

    return scenarios


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_scenarios(
    financial_profiles: pd.DataFrame,
    products: pd.DataFrame,
    reviews_df: Optional[pd.DataFrame] = None,
    n_scenarios: int = 10_000,
    random_state: int = 42,
    stratified: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic user-product scenarios.

    Pairs real user financial profiles with real products, computes the 6
    financial features for each pair, and labels each scenario
    GREEN/YELLOW/RED using the DecisionEngine.

    Args:
        financial_profiles: DataFrame from the financial_profiles table.
        products: DataFrame from the products table.
        n_scenarios: Number of (user, product) pairs to generate.
        random_state: Seed for reproducibility.
        stratified: If True (default), sample equally across 9
            (income × price) bracket cells for balanced representation.
            If False, use pure uniform random pairing.

    Returns:
        DataFrame with one row per scenario containing user features,
        product columns, 6 computed features, and a rule-based label.
    """
    rng = np.random.default_rng(random_state)

    if stratified:
        users, prods = _sample_stratified(
            financial_profiles, products, n_scenarios, rng,
        )
    else:
        users, prods = _sample_random(
            financial_profiles, products, n_scenarios, rng,
        )

    scenarios = _compute_features_and_label(users, prods, reviews=reviews_df)

    logger.info(
        "Generated %d scenarios (stratified=%s) — label distribution:\n%s",
        len(scenarios), stratified,
        scenarios["label"].value_counts().to_string(),
    )
    return scenarios
