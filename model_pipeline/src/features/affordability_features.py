"""
Affordability Feature Computation — Inference-Time Utility.

This module provides stateless functions for computing affordability metrics
at query time. It is called by the Deterministic Financial Logic Engine
when a user asks "Should I buy this?".

NOTE: This is NOT a batch pipeline script. It does not read/write files.
Financial features and product quality features are pre-computed and stored
in PostgreSQL by their respective pipeline modules:
    - financial_features.py  → financial health metrics per user
    - review_features.py     → quality metrics per product

This module combines them ON DEMAND for a specific user-product pair.

Usage (by Decision API / Deterministic Engine):
    from features.affordability_features import compute_affordability

    result = compute_affordability(
        user_financial_profile={
            "monthly_income": 5000.0,
            "discretionary_income": 1200.0,
            "savings_balance": 8000.0,
            "monthly_expenses": 2800.0,
            "monthly_emi": 1000.0,
        },
        product_price=799.99
    )
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AffordabilityResult:
    """Container for computed affordability metrics."""
    price_to_income_ratio: Optional[float]
    affordability_score: Optional[float]
    residual_utility_score: Optional[float]

    def to_dict(self) -> dict:
        return {
            "price_to_income_ratio": self.price_to_income_ratio,
            "affordability_score": self.affordability_score,
            "residual_utility_score": self.residual_utility_score,
        }


def compute_affordability(
    user_financial_profile: dict,
    product_price: float,
) -> AffordabilityResult:
    """
    Computes affordability metrics for a single user-product pair.

    Called at inference time by the Deterministic Financial Logic Engine
    when a user queries a specific product.

    Args:
        user_financial_profile: Dict containing pre-computed financial features.
            Required keys: monthly_income, discretionary_income, savings_balance,
                           monthly_expenses, monthly_emi
        product_price: Price of the product being evaluated.

    Returns:
        AffordabilityResult with three metrics:
            - price_to_income_ratio: % of monthly income required for purchase.
            - affordability_score: Remaining discretionary budget after purchase.
            - residual_utility_score: Months of financial runway remaining after purchase.
    """
    income = user_financial_profile.get("monthly_income", 0.0)
    discretionary = user_financial_profile.get("discretionary_income", 0.0)
    savings = user_financial_profile.get("savings_balance", 0.0)
    expenses = user_financial_profile.get("monthly_expenses", 0.0)
    emi = user_financial_profile.get("monthly_emi", 0.0)

    # Metric 1: Price-To-Income Ratio.
    # What percentage of one month's income does this product cost?
    price_to_income = None
    if income > 0:
        price_to_income = round(product_price / income, 4)
    else:
        logger.warning("Cannot compute price_to_income_ratio: monthly_income is 0.")

    # Metric 2: Affordability Score.
    # How much discretionary budget remains after buying this product?
    affordability_score = round(discretionary - product_price, 2)

    # Metric 3: Residual Utility Score (RUS).
    # How many months of financial runway remain if user spends savings on this?
    residual_utility = None
    total_obligations = expenses + emi
    if total_obligations > 0:
        residual_utility = round((savings - product_price) / total_obligations, 4)
    else:
        logger.warning("Cannot compute residual_utility_score: total obligations is 0.")

    result = AffordabilityResult(
        price_to_income_ratio=price_to_income,
        affordability_score=affordability_score,
        residual_utility_score=residual_utility,
    )

    logger.info(
        "Affordability computed — price: %.2f, score: %.2f, RUS: %s",
        product_price, affordability_score, residual_utility,
    )

    return result


# =============================================================================
# Synthetic Scenario Generation & Labeling (Deterministic Engine)
# =============================================================================
# Used to create labeled training data by pairing real user financial profiles
# with real products and applying rule-based labeling (GREEN / YELLOW / RED).
# This bridges the gap between the pre-computed features in PostgreSQL and the
# supervised ML model that needs labeled examples.
# =============================================================================

import pandas as pd
import numpy as np


def generate_scenarios(
    financial_profiles: pd.DataFrame,
    products: pd.DataFrame,
    n_scenarios: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic user-product scenarios by randomly sampling pairs.

    Args:
        financial_profiles: DataFrame from `financial_profiles` table.
            Required columns: monthly_income, discretionary_income,
                              savings_balance, monthly_expenses, monthly_emi,
                              emergency_fund_months
        products: DataFrame from `products` table.
            Required columns: price, product_id
        n_scenarios: Number of random (user, product) pairs to generate.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with one row per scenario containing user features,
        product price, computed affordability metrics, and a rule-based label.
    """
    rng = np.random.default_rng(random_state)

    user_indices = rng.integers(0, len(financial_profiles), size=n_scenarios)
    product_indices = rng.integers(0, len(products), size=n_scenarios)

    users = financial_profiles.iloc[user_indices].reset_index(drop=True)
    prods = products.iloc[product_indices].reset_index(drop=True)

    # Compute affordability metrics for each scenario
    scenarios = users.copy()
    scenarios["product_id"] = prods["product_id"].values
    scenarios["product_price"] = prods["price"].values

    scenarios["affordability_score"] = (
        scenarios["discretionary_income"] - scenarios["product_price"]
    )
    scenarios["price_to_income_ratio"] = (
        scenarios["product_price"] / scenarios["monthly_income"].replace(0, np.nan)
    )
    scenarios["residual_utility_score"] = (
        (scenarios["savings_balance"] - scenarios["product_price"])
        / (scenarios["monthly_expenses"] + scenarios["monthly_emi"]).replace(0, np.nan)
    )

    # Label each scenario using deterministic rules
    scenarios["label"] = scenarios.apply(label_scenario, axis=1)

    logger.info(
        "Generated %d scenarios — label distribution:\n%s",
        len(scenarios),
        scenarios["label"].value_counts().to_string(),
    )
    return scenarios


def label_scenario(row: pd.Series) -> str:
    """
    Deterministic labeling rules for a user-product scenario.

    Rules:
        RED    — Cannot afford AND no emergency cushion
                 (affordability_score < 0 AND emergency_fund_months < 1)
        YELLOW — Marginal: can't quite afford OR thin emergency cushion
                 (affordability_score < 0 OR emergency_fund_months < 3)
        GREEN  — Comfortably affordable with adequate safety net

    Args:
        row: A single scenario row with at least `affordability_score`
             and `emergency_fund_months`.

    Returns:
        One of "RED", "YELLOW", "GREEN".
    """
    if row["affordability_score"] < 0 and row.get("emergency_fund_months", 0) < 1:
        return "RED"
    elif row["affordability_score"] < 0 or row.get("emergency_fund_months", 0) < 3:
        return "YELLOW"
    else:
        return "GREEN"
