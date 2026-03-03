"""
Unit tests for affordability features and scenario generation.

Tests compute_affordability() (inference-time) and generate_scenarios()
(batch label generation with DecisionEngine).
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.affordability_features import (
    compute_affordability,
    AffordabilityResult,
    generate_scenarios,
)


# ── compute_affordability (inference-time) ───────────────────────────────────

class TestComputeAffordability:

    def test_basic_computation(self):
        profile = {
            "monthly_income": 5000.0,
            "discretionary_income": 1200.0,
            "savings_balance": 8000.0,
            "monthly_expenses": 2800.0,
            "monthly_emi": 1000.0,
        }
        result = compute_affordability(profile, product_price=200.0)

        assert isinstance(result, AffordabilityResult)
        assert result.price_to_income_ratio == pytest.approx(200 / 5000, abs=1e-3)
        assert result.affordability_score == pytest.approx(1200 - 200, abs=0.01)
        assert result.residual_utility_score == pytest.approx(
            (8000 - 200) / (2800 + 1000), abs=1e-3
        )

    def test_zero_income(self):
        profile = {
            "monthly_income": 0,
            "discretionary_income": 0,
            "savings_balance": 1000,
            "monthly_expenses": 500,
            "monthly_emi": 0,
        }
        result = compute_affordability(profile, product_price=100.0)
        assert result.price_to_income_ratio is None  # can't divide by zero
        assert result.affordability_score == -100.0

    def test_zero_obligations(self):
        profile = {
            "monthly_income": 5000,
            "discretionary_income": 5000,
            "savings_balance": 10000,
            "monthly_expenses": 0,
            "monthly_emi": 0,
        }
        result = compute_affordability(profile, product_price=50.0)
        assert result.residual_utility_score is None  # can't divide by zero

    def test_to_dict(self):
        profile = {
            "monthly_income": 5000,
            "discretionary_income": 1000,
            "savings_balance": 5000,
            "monthly_expenses": 2000,
            "monthly_emi": 500,
        }
        result = compute_affordability(profile, product_price=100)
        d = result.to_dict()
        assert set(d.keys()) == {
            "price_to_income_ratio",
            "affordability_score",
            "residual_utility_score",
        }


# ── generate_scenarios (batch) ───────────────────────────────────────────────

def _make_financial_df(n=50):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "user_id": [f"U{i}" for i in range(n)],
        "monthly_income": rng.uniform(2000, 10000, n),
        "monthly_expenses": rng.uniform(1000, 5000, n),
        "savings_balance": rng.uniform(500, 50000, n),
        "has_loan": rng.choice([True, False], n),
        "loan_amount": rng.uniform(0, 200000, n),
        "monthly_emi": rng.uniform(0, 2000, n),
        "loan_interest_rate": rng.uniform(0, 20, n),
        "loan_term_months": rng.choice([0, 60, 120, 240], n),
        "credit_score": rng.integers(300, 850, n),
        "employment_status": rng.choice(["employed", "self-employed", "unemployed"], n),
        "region": rng.choice(["east", "west", "south", "north"], n),
        "discretionary_income": rng.uniform(-2000, 5000, n),
        "debt_to_income_ratio": rng.uniform(0, 0.6, n),
        "saving_to_income_ratio": rng.uniform(0.1, 5.0, n),
        "monthly_expense_burden_ratio": rng.uniform(0.3, 1.2, n),
        "emergency_fund_months": rng.uniform(0.5, 12, n),
    })


def _make_products_df(n=30):
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "product_id": [f"P{i}" for i in range(n)],
        "product_name": [f"Product {i}" for i in range(n)],
        "price": rng.uniform(5, 2000, n),
        "average_rating": rng.uniform(1, 5, n),
        "rating_number": rng.integers(1, 5000, n),
        "rating_variance": rng.uniform(0, 2, n),
        "description": ["desc"] * n,
        "features": ["feat"] * n,
        "details": ["det"] * n,
        "category": ["cat"] * n,
    })


class TestGenerateScenarios:

    def test_correct_row_count(self):
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=100
        )
        assert len(scenarios) == 100

    def test_has_required_columns(self):
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=50
        )
        required = [
            "product_id", "product_price",
            "average_rating", "rating_number", "rating_variance",
            "affordability_score", "price_to_income_ratio",
            "residual_utility_score", "label",
        ]
        for col in required:
            assert col in scenarios.columns, f"Missing column: {col}"

    def test_labels_are_valid(self):
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=200
        )
        assert set(scenarios["label"].unique()).issubset({"GREEN", "YELLOW", "RED"})

    def test_not_all_one_label(self):
        """With varied synthetic data, we should get at least 2 label classes."""
        scenarios = generate_scenarios(
            _make_financial_df(100), _make_products_df(50), n_scenarios=500
        )
        assert scenarios["label"].nunique() >= 2

    def test_reproducibility(self):
        """Same seed → same output."""
        fin = _make_financial_df()
        prod = _make_products_df()
        s1 = generate_scenarios(fin, prod, n_scenarios=50, random_state=123)
        s2 = generate_scenarios(fin, prod, n_scenarios=50, random_state=123)
        pd.testing.assert_frame_equal(s1, s2)

    def test_different_seed_different_output(self):
        fin = _make_financial_df()
        prod = _make_products_df()
        s1 = generate_scenarios(fin, prod, n_scenarios=50, random_state=1)
        s2 = generate_scenarios(fin, prod, n_scenarios=50, random_state=2)
        assert not s1["product_price"].equals(s2["product_price"])
