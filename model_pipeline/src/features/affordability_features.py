"""
Affordability & Financial Health Feature Computation.

Transforms raw financial profiles and product data into 6 model-ready
financial features that capture a user's ability to absorb a purchase.

Feature groups:
    Financial (6) — Measures of user affordability and resilience.
    No product review features — the decision engine is purely financial.

NOTE: This is NOT a batch pipeline script.  It does not read/write files.
Financial features are pre-computed and stored in PostgreSQL by the
data pipeline (financial_features.py).

This module combines them ON DEMAND for a specific user-product pair.

Additionally, this module provides batch scenario generation for ML training:
    - generate_scenarios() pairs real users with real products randomly.
    - Labels are assigned by the DecisionEngine (deterministic_engine/decision_logic.py).
    - Output becomes the training dataset for downstream ML models.

Usage (single pair — Decision API / Deterministic Engine):
    from features.affordability_features import compute_affordability

    result = compute_affordability(
        user_financial_profile={...},
        product_price=799.99,
    )

Usage (batch — ML training label generation):
    from features.affordability_features import generate_scenarios

    scenarios_df = generate_scenarios(financial_profiles_df, products_df, n_scenarios=50000)
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference-time feature computation
# ---------------------------------------------------------------------------

@dataclass
class AffordabilityResult:
    """Container for the 6 computed financial features returned by compute_affordability()."""

    # Financial features — how well can this user absorb the purchase?
    affordability_score: Optional[float]
    price_to_income_ratio: Optional[float]
    residual_utility_score: Optional[float]
    savings_to_price_ratio: Optional[float]
    net_worth_indicator: Optional[float]
    credit_risk_indicator: Optional[float]

    def to_dict(self) -> dict:
        return {
            "affordability_score": self.affordability_score,
            "price_to_income_ratio": self.price_to_income_ratio,
            "residual_utility_score": self.residual_utility_score,
            "savings_to_price_ratio": self.savings_to_price_ratio,
            "net_worth_indicator": self.net_worth_indicator,
            "credit_risk_indicator": self.credit_risk_indicator,
        }


def compute_affordability(
    user_financial_profile: dict,
    product_price: float,
    product_info: Optional[dict] = None,
) -> AffordabilityResult:
    """
    Computes 6 financial features for a single user-product pair.

    Called at inference time by the Deterministic Financial Logic Engine
    when a user queries a specific product.

    Args:
        user_financial_profile: Dict with pre-computed financial fields from DB.
        product_price: Price of the product being evaluated.
        product_info:  (Unused — kept for API compatibility.)

    Returns:
        AffordabilityResult with 6 financial features.
    """
    income = user_financial_profile.get("monthly_income", 0.0)
    discretionary = user_financial_profile.get("discretionary_income", 0.0)
    savings = user_financial_profile.get("savings_balance", 0.0)
    expenses = user_financial_profile.get("monthly_expenses", 0.0)
    emi = user_financial_profile.get("monthly_emi", 0.0)
    loan_amount = user_financial_profile.get("loan_amount", 0.0)
    credit_score = user_financial_profile.get("credit_score", 0)

    total_obligations = expenses + emi

    # ── Financial features ───────────────────────────────────────────────

    # Remaining discretionary budget after subtracting the product price.
    affordability_score = round(discretionary - product_price, 2)

    # What fraction of monthly income does this product cost?
    price_to_income = round(product_price / income, 4) if income > 0 else None

    # How many months of financial runway remain if the user buys this
    # from savings? (savings − price) / total monthly obligations.
    residual_utility = None
    if total_obligations > 0:
        residual_utility = round((savings - product_price) / total_obligations, 4)

    # How many times over can savings cover the product price?
    savings_to_price = round(savings / product_price, 4) if product_price > 0 else None

    # Normalized net worth: (savings − outstanding_loan) / monthly_income.
    # Negative means the user is underwater on their loan.
    net_worth = round((savings - loan_amount) / income, 4) if income > 0 else None

    # Credit score normalized to the 0–1 range: (score − 300) / 550.
    credit_risk = round((credit_score - 300) / 550, 4) if credit_score else None

    result = AffordabilityResult(
        affordability_score=affordability_score,
        price_to_income_ratio=price_to_income,
        residual_utility_score=residual_utility,
        savings_to_price_ratio=savings_to_price,
        net_worth_indicator=net_worth,
        credit_risk_indicator=credit_risk,
    )

    logger.info(
        "Affordability computed — price: %.2f, score: %.2f, RUS: %s",
        product_price, affordability_score, residual_utility,
    )

    return result


# ---------------------------------------------------------------------------
# Batch scenario generation & labeling (for ML training data)
# ---------------------------------------------------------------------------

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
    from deterministic_engine.decision_logic import DecisionEngine

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
