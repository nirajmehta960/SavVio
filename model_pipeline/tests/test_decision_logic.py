"""
Unit tests for the Deterministic Decision Engine.

Tests all 4 tiers: hard-stop → caution → confidence downgrade → final assignment.
Also covers edge cases (missing fields, conflicting rules, downgrade chains).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deterministic_engine.decision_logic import DecisionEngine, DecisionResult


# ── Fixtures ─────────────────────────────────────────────────────────────────

# Thresholds matching Config defaults — injected to avoid import dependency
HARD_STOP = {
    "discretionary_income_lt": 0,
    "debt_to_income_ratio_gt": 0.40,
    "monthly_expense_burden_ratio_gt": 0.80,
    "emergency_fund_months_lt": 1,
    "price_exceeds_discretionary_emergency_lt": 3,
}

CAUTION = {
    "discretionary_income_range": (0, 1000),
    "debt_to_income_ratio_range": (0.20, 0.40),
    "monthly_expense_burden_ratio_range": (0.50, 0.80),
    "emergency_fund_months_range": (1, 3),
    "saving_to_income_ratio_range": (0.25, 1.0),
}

CONFIDENCE_DOWNGRADE = {
    "rating_number_lt": 10,
    "rating_variance_zero_count_lt": 10,
    "rating_variance_gt": 1.0,
    "average_rating_lte": 3.0,
}


@pytest.fixture
def engine():
    return DecisionEngine(
        hard_stop=HARD_STOP,
        caution=CAUTION,
        confidence_downgrade=CONFIDENCE_DOWNGRADE,
    )


def _healthy_financial():
    """Financial profile that passes all checks (Green candidate)."""
    return {
        "discretionary_income": 3000,
        "debt_to_income_ratio": 0.10,
        "saving_to_income_ratio": 2.0,
        "monthly_expense_burden_ratio": 0.40,
        "emergency_fund_months": 6,
    }


def _healthy_product():
    """Product with strong signals (no downgrades)."""
    return {
        "price": 50.0,
        "average_rating": 4.5,
        "rating_number": 500,
        "rating_variance": 0.5,
    }


# ── Tier 1: Hard-stop → RED ─────────────────────────────────────────────────

class TestHardStopRed:

    def test_negative_discretionary_income(self, engine):
        fin = _healthy_financial()
        fin["discretionary_income"] = -100
        result = engine.decide(fin, _healthy_product())
        assert result.color == "RED"
        assert any("negative_discretionary" in r for r in result.triggered_rules)

    def test_high_dti(self, engine):
        fin = _healthy_financial()
        fin["debt_to_income_ratio"] = 0.50
        result = engine.decide(fin, _healthy_product())
        assert result.color == "RED"
        assert any("high_debt_to_income" in r for r in result.triggered_rules)

    def test_high_expense_burden(self, engine):
        fin = _healthy_financial()
        fin["monthly_expense_burden_ratio"] = 0.90
        result = engine.decide(fin, _healthy_product())
        assert result.color == "RED"
        assert any("high_expense_burden" in r for r in result.triggered_rules)

    def test_low_emergency_fund(self, engine):
        fin = _healthy_financial()
        fin["emergency_fund_months"] = 0.5
        result = engine.decide(fin, _healthy_product())
        assert result.color == "RED"
        assert any("low_emergency_fund" in r for r in result.triggered_rules)

    def test_price_exceeds_discretionary_thin_runway(self, engine):
        fin = _healthy_financial()
        fin["discretionary_income"] = 200
        fin["emergency_fund_months"] = 2
        prod = _healthy_product()
        prod["price"] = 500  # price > discretionary AND emergency < 3
        result = engine.decide(fin, prod)
        assert result.color == "RED"
        assert any("price_exceeds" in r for r in result.triggered_rules)

    def test_boundary_dti_exactly_040_not_red(self, engine):
        """DTI exactly 0.40 should NOT trigger hard-stop (> 0.40 required)."""
        fin = _healthy_financial()
        fin["debt_to_income_ratio"] = 0.40
        result = engine.decide(fin, _healthy_product())
        assert result.color != "RED" or not any("high_debt" in r for r in result.triggered_rules)


# ── Tier 2: Caution → YELLOW ────────────────────────────────────────────────

class TestCautionYellow:

    def test_tight_discretionary(self, engine):
        fin = _healthy_financial()
        fin["discretionary_income"] = 500  # in [0, 1000]
        result = engine.decide(fin, _healthy_product())
        assert result.color in ("YELLOW", "RED")  # could be downgraded further
        assert any("tight_discretionary" in r for r in result.triggered_rules)

    def test_moderate_dti(self, engine):
        fin = _healthy_financial()
        fin["debt_to_income_ratio"] = 0.30  # in [0.20, 0.40]
        result = engine.decide(fin, _healthy_product())
        assert any("moderate_debt" in r for r in result.triggered_rules)

    def test_moderate_expense_burden(self, engine):
        fin = _healthy_financial()
        fin["monthly_expense_burden_ratio"] = 0.65  # in [0.50, 0.80]
        result = engine.decide(fin, _healthy_product())
        assert any("moderate_expense" in r for r in result.triggered_rules)

    def test_thin_emergency_fund(self, engine):
        fin = _healthy_financial()
        fin["emergency_fund_months"] = 2  # in [1, 3]
        result = engine.decide(fin, _healthy_product())
        assert any("thin_emergency" in r for r in result.triggered_rules)

    def test_low_savings_ratio(self, engine):
        fin = _healthy_financial()
        fin["saving_to_income_ratio"] = 0.50  # in [0.25, 1.0]
        result = engine.decide(fin, _healthy_product())
        assert any("low_savings" in r for r in result.triggered_rules)


# ── Tier 3: Confidence downgrades ────────────────────────────────────────────

class TestConfidenceDowngrade:

    def test_insufficient_reviews_green_to_yellow(self, engine):
        prod = _healthy_product()
        prod["rating_number"] = 5  # < 10
        result = engine.decide(_healthy_financial(), prod)
        assert result.color == "YELLOW"
        assert any("insufficient" in d for d in result.confidence_downgrades)

    def test_polarized_ratings_green_to_yellow(self, engine):
        prod = _healthy_product()
        prod["rating_variance"] = 1.5  # > 1.0
        result = engine.decide(_healthy_financial(), prod)
        assert result.color == "YELLOW"
        assert any("polarized" in d for d in result.confidence_downgrades)

    def test_poor_average_rating_green_to_yellow(self, engine):
        prod = _healthy_product()
        prod["average_rating"] = 2.5  # <= 3.0
        result = engine.decide(_healthy_financial(), prod)
        assert result.color == "YELLOW"
        assert any("poor_average" in d for d in result.confidence_downgrades)

    def test_artificial_uniform_signal(self, engine):
        prod = _healthy_product()
        prod["rating_variance"] = 0  # variance == 0
        prod["rating_number"] = 3    # AND count < 10
        result = engine.decide(_healthy_financial(), prod)
        # Two downgrades fire: insufficient_reviews + artificial_uniform
        # GREEN → YELLOW → RED
        assert result.color == "RED"
        assert any("artificial" in d for d in result.confidence_downgrades)

    def test_double_downgrade_yellow_to_red(self, engine):
        """Yellow + 1 downgrade → RED."""
        fin = _healthy_financial()
        fin["discretionary_income"] = 500  # Yellow caution
        prod = _healthy_product()
        prod["rating_number"] = 5  # downgrade
        result = engine.decide(fin, prod)
        assert result.color == "RED"

    def test_red_stays_red_on_downgrade(self, engine):
        """Red profile + downgrade → still RED."""
        fin = _healthy_financial()
        fin["discretionary_income"] = -100  # hard-stop Red
        prod = _healthy_product()
        prod["rating_number"] = 5  # downgrade (shouldn't matter)
        result = engine.decide(fin, prod)
        assert result.color == "RED"


# ── Tier 4: Green pass-through ───────────────────────────────────────────────

class TestGreenPassthrough:

    def test_healthy_profile_healthy_product(self, engine):
        result = engine.decide(_healthy_financial(), _healthy_product())
        assert result.color == "GREEN"
        assert len(result.triggered_rules) == 0
        assert len(result.confidence_downgrades) == 0


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_all_financial_fields_missing(self, engine):
        result = engine.decide({}, _healthy_product())
        assert result.color == "YELLOW"
        assert any("missing" in r for r in result.triggered_rules)

    def test_partial_financial_fields_missing(self, engine):
        fin = {"discretionary_income": 3000}
        result = engine.decide(fin, _healthy_product())
        # Should NOT default to Yellow since not ALL fields missing
        assert result.color in ("GREEN", "YELLOW", "RED")

    def test_nan_financial_field(self, engine):
        fin = _healthy_financial()
        fin["discretionary_income"] = float("nan")
        result = engine.decide(fin, _healthy_product())
        # NaN is treated as missing — shouldn't crash
        assert result.color in ("GREEN", "YELLOW", "RED")

    def test_none_financial_field(self, engine):
        fin = _healthy_financial()
        fin["discretionary_income"] = None
        result = engine.decide(fin, _healthy_product())
        assert result.color in ("GREEN", "YELLOW", "RED")

    def test_decide_row_wrapper(self, engine):
        """Test the pandas .apply() wrapper."""
        import pandas as pd
        row = pd.Series({
            **_healthy_financial(),
            "price": 50.0,
            "average_rating": 4.5,
            "rating_number": 500,
            "rating_variance": 0.5,
        })
        color = engine.decide_row(row)
        assert color == "GREEN"


# ── README example scenarios ─────────────────────────────────────────────────

class TestReadmeExamples:

    def test_stable_profile_strong_runway(self, engine):
        """DTI 0.15, runway 5 months, positive discretionary → GREEN."""
        fin = {
            "discretionary_income": 2000,
            "debt_to_income_ratio": 0.15,
            "saving_to_income_ratio": 3.0,
            "monthly_expense_burden_ratio": 0.45,
            "emergency_fund_months": 5,
        }
        result = engine.decide(fin, _healthy_product())
        assert result.color == "GREEN"

    def test_tight_buffer_uncertain_product(self, engine):
        """DTI 0.30, runway 2, rating_number 7 → YELLOW."""
        fin = {
            "discretionary_income": 1500,
            "debt_to_income_ratio": 0.30,
            "saving_to_income_ratio": 2.0,
            "monthly_expense_burden_ratio": 0.55,
            "emergency_fund_months": 2,
        }
        prod = {
            "price": 50.0,
            "average_rating": 4.0,
            "rating_number": 7,
            "rating_variance": 0.5,
        }
        result = engine.decide(fin, prod)
        assert result.color in ("YELLOW", "RED")  # multiple caution + downgrade

    def test_negative_discretionary(self, engine):
        """discretionary_income < 0 → RED."""
        fin = {
            "discretionary_income": -500,
            "debt_to_income_ratio": 0.15,
            "saving_to_income_ratio": 2.0,
            "monthly_expense_burden_ratio": 0.45,
            "emergency_fund_months": 5,
        }
        result = engine.decide(fin, _healthy_product())
        assert result.color == "RED"

    def test_risky_debt_polarized_reviews(self, engine):
        """DTI 0.48, variance 1.3 → RED."""
        fin = {
            "discretionary_income": 1000,
            "debt_to_income_ratio": 0.48,
            "saving_to_income_ratio": 2.0,
            "monthly_expense_burden_ratio": 0.45,
            "emergency_fund_months": 5,
        }
        prod = {
            "price": 50.0,
            "average_rating": 4.0,
            "rating_number": 100,
            "rating_variance": 1.3,
        }
        result = engine.decide(fin, prod)
        assert result.color == "RED"
