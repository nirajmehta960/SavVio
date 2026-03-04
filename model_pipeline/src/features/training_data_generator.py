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
from typing import List, Tuple

import numpy as np
import pandas as pd

from deterministic_engine.decision_logic import DecisionEngine

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
    """Pure uniform random pairing — legacy behaviour."""
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
) -> pd.DataFrame:
    """
    Given paired user and product DataFrames of equal length, compute the
    6 financial features and label each row with the DecisionEngine.
    """
    engine = DecisionEngine()

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
    scenarios["credit_risk_indicator"] = (credit_score - 300) / 550.0

    # Label each scenario using the multi-condition deterministic engine.
    logger.info("Labeling %d scenarios with DecisionEngine...", len(scenarios))
    scenarios["label"] = scenarios.apply(engine.decide_row, axis=1)

    return scenarios


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_scenarios(
    financial_profiles: pd.DataFrame,
    products: pd.DataFrame,
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

    scenarios = _compute_features_and_label(users, prods)

    logger.info(
        "Generated %d scenarios (stratified=%s) — label distribution:\n%s",
        len(scenarios), stratified,
        scenarios["label"].value_counts().to_string(),
    )
    return scenarios
