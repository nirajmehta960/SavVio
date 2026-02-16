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
