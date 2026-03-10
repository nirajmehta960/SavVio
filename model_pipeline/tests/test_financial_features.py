"""
Unit tests for affordability features and scenario generation.

Tests compute_affordability() (inference-time, 6 financial features) and
generate_scenarios() (batch label generation with DecisionEngine).
"""

import os
import sys

import pytest
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.financial_features import (
    compute_affordability,
    AffordabilityResult,
)
from features.training_data_generator import generate_scenarios


# ── compute_affordability ────────────────────────────────────────────────────

class TestComputeAffordability:

    def _default_profile(self):
        return {
            "monthly_income": 5000.0,
            "discretionary_income": 1200.0,
            "savings_balance": 8000.0,
            "monthly_expenses": 2800.0,
            "monthly_emi": 1000.0,
            "loan_amount": 15000.0,
            "credit_score": 720,
        }

    def test_affordability_score(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.affordability_score == pytest.approx(1200 - 200, abs=0.01)

    def test_price_to_income_ratio(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.price_to_income_ratio == pytest.approx(200 / 5000, abs=1e-3)

    def test_residual_utility_score(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.residual_utility_score == pytest.approx(
            (8000 - 200) / (2800 + 1000), abs=1e-3
        )

    def test_savings_to_price_ratio(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.savings_to_price_ratio == pytest.approx(8000 / 200, abs=1e-3)

    def test_net_worth_indicator(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.net_worth_indicator == pytest.approx((8000 - 15000) / 5000, abs=1e-3)

    def test_credit_risk_indicator(self):
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.credit_risk_indicator == pytest.approx((720 - 299) / 550, abs=1e-3)

    def test_zero_income_guards(self):
        profile = self._default_profile()
        profile["monthly_income"] = 0
        profile["discretionary_income"] = 0
        result = compute_affordability(profile, product_price=100.0)
        assert result.price_to_income_ratio is None
        assert result.affordability_score == -100.0

    def test_zero_obligations_guards(self):
        profile = self._default_profile()
        profile["monthly_expenses"] = 0
        profile["monthly_emi"] = 0
        result = compute_affordability(profile, product_price=50.0)
        assert result.residual_utility_score is None

    def test_to_dict_has_all_6_keys(self):
        result = compute_affordability(self._default_profile(), 100)
        d = result.to_dict()
        assert len(d) == 6
        expected_keys = {
            "affordability_score", "price_to_income_ratio", "residual_utility_score",
            "savings_to_price_ratio", "net_worth_indicator", "credit_risk_indicator",
        }
        assert set(d.keys()) == expected_keys

    def test_no_product_info_defaults(self):
        """compute_affordability without product_info should not crash."""
        result = compute_affordability(self._default_profile(), 200.0)
        assert result.affordability_score is not None


# ── generate_scenarios ───────────────────────────────────────────────────────

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

    def test_has_all_6_feature_columns(self):
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=50
        )
        required = [
            "product_id", "product_price",
            # 6 financial features (no product review features).
            "affordability_score", "price_to_income_ratio", "residual_utility_score",
            "savings_to_price_ratio", "net_worth_indicator", "credit_risk_indicator",
            "label",
        ]
        for col in required:
            assert col in scenarios.columns, f"Missing column: {col}"

    def test_no_review_features_computed(self):
        """review_confidence_score and review_polarization_index are NOT computed."""
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=50
        )
        assert "review_confidence_score" not in scenarios.columns
        assert "review_polarization_index" not in scenarios.columns

    def test_labels_are_valid(self):
        scenarios = generate_scenarios(
            _make_financial_df(), _make_products_df(), n_scenarios=200
        )
        assert set(scenarios["label"].unique()).issubset({"GREEN", "YELLOW", "RED"})

    def test_not_all_one_label(self):
        """With varied synthetic data we should get at least 2 label classes."""
        scenarios = generate_scenarios(
            _make_financial_df(100), _make_products_df(50), n_scenarios=500
        )
        assert scenarios["label"].nunique() >= 2

    def test_reproducibility(self):
        """Same seed produces identical output."""
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
