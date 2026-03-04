"""
Training Data Generator — Scenario Creation & Labeling.

Generates synthetic user-product scenarios by randomly pairing real
financial profiles with real products, computes the 6 financial features
for each pair, and labels each scenario GREEN/YELLOW/RED using the
DecisionEngine.

Output becomes the training dataset for downstream ML models.

Usage (batch — ML training label generation):
    from features.scenario_generator import generate_scenarios

    scenarios_df = generate_scenarios(financial_profiles_df, products_df, n_scenarios=50000)
"""

import logging

import numpy as np
import pandas as pd

from deterministic_engine.decision_logic import DecisionEngine

logger = logging.getLogger(__name__)


def generate_scenarios(
    financial_profiles: pd.DataFrame,
    products: pd.DataFrame,
    n_scenarios: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic user-product scenarios by randomly sampling pairs.

    Pairs real user financial profiles with real products, computes the 6
    financial features for each pair, and labels each scenario GREEN/YELLOW/RED
    using the DecisionEngine.

    Args:
        financial_profiles: DataFrame from the financial_profiles table.
        products: DataFrame from the products table.
        n_scenarios: Number of random (user, product) pairs to generate.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with one row per scenario containing user features,
        product columns, 6 computed features, and a rule-based label.
    """

    rng = np.random.default_rng(random_state)
    engine = DecisionEngine()

    user_indices = rng.integers(0, len(financial_profiles), size=n_scenarios)
    product_indices = rng.integers(0, len(products), size=n_scenarios)

    users = financial_profiles.iloc[user_indices].reset_index(drop=True)
    prods = products.iloc[product_indices].reset_index(drop=True)

    # Combine user financial data with product columns into one table.
    scenarios = users.copy()
    scenarios["product_id"] = prods["product_id"].values
    scenarios["product_price"] = prods["price"].values

    # Keep product metadata for downstream use (but NOT for decision engine).
    if "average_rating" in prods.columns:
        scenarios["average_rating"] = prods["average_rating"].values
    if "rating_number" in prods.columns:
        scenarios["rating_number"] = prods["rating_number"].values
    if "rating_variance" in prods.columns:
        scenarios["rating_variance"] = prods["rating_variance"].values

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

    logger.info(
        "Generated %d scenarios — label distribution:\n%s",
        len(scenarios),
        scenarios["label"].value_counts().to_string(),
    )
    return scenarios
