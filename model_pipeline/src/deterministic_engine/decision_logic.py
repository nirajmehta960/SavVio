"""
Deterministic Decision Engine for SavVio.

Rule-based labeling system that assigns GREEN / YELLOW / RED to
(user, product) pairs. This engine is AUTHORITATIVE — neither the
ML model nor the LLM wrapper may override its output.

The engine answers: "Should this user buy this product?"

    GREEN  — Purchase fits comfortably within the user's financial life.
    YELLOW — Genuine concern — the user should pause and think.
    RED    — Purchase would cause financial harm.

Design principle:
    Every rule combines signals from at least TWO independent correlation
    groups to avoid false triggers from a single underlying cause.

Correlation groups:
    Group 1 — Income capacity:   affordability_score, discretionary_income,
                                  price_to_income_ratio
    Group 2 — Savings depth:     saving_to_income_ratio, savings_to_price_ratio,
                                  emergency_fund_months, residual_utility_score
    Group 3 — Debt burden:       debt_to_income_ratio, monthly_expense_burden_ratio
    Group 4 — Independent:       credit_risk_indicator, net_worth_indicator

Features used — 11 total (5 DB + 6 computed), ALL financial:
    DB:       discretionary_income, debt_to_income_ratio, saving_to_income_ratio,
              monthly_expense_burden_ratio, emergency_fund_months
    Computed: affordability_score, price_to_income_ratio, residual_utility_score,
              savings_to_price_ratio, net_worth_indicator, credit_risk_indicator

Usage (batch — label generation for ML training):
    from deterministic_engine.decision_logic import DecisionEngine
    engine = DecisionEngine()
    df["label"] = df.apply(engine.decide_row, axis=1)

Usage (single pair — Decision API):
    result = engine.decide(financial_dict, product_dict)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Outcome of the deterministic engine for one (user, product) pair."""
    color: str
    triggered_rules: List[str] = field(default_factory=list)
    confidence_downgrades: List[str] = field(default_factory=list)
    explanation: str = ""


class DecisionEngine:
    """
    Pure-financial multi-condition labeling engine.

    RED:    4 compound AND rules — each crosses at least 2 correlation groups
            and includes a price_to_income_ratio escape so trivial purchases
            never trigger RED.
    YELLOW: 5 compound AND rules — each crosses at least 2 groups.
            YELLOW triggers when >= 2 rules fire.
    GREEN:  default when no RED fires and fewer than 2 YELLOW rules fire.
    """

    @staticmethod
    def _safe(val, default=0.0):
        """Return default if val is None or NaN."""
        if val is None:
            return default
        try:
            if math.isnan(val):
                return default
        except TypeError:
            pass
        return val

    def decide(self, financial: dict, product: dict) -> DecisionResult:
        """
        Core labeling logic.

        Evaluates a user-product pair and returns GREEN / YELLOW / RED
        with the rules that triggered the decision.

        Args:
            financial: Dict with per-pair computed features + per-entity DB features.
            product:   Dict with product price (only price is used by the engine).

        Returns:
            DecisionResult with color, triggered rules, and explanation.
        """
        # Pull all values with safe defaults.
        # --- Per-pair computed features (from affordability_features.py) ---
        affordability = self._safe(financial.get("affordability_score"))
        pir = self._safe(financial.get("price_to_income_ratio"))
        rus = self._safe(financial.get("residual_utility_score"))
        spr = self._safe(financial.get("savings_to_price_ratio"))
        nwi = self._safe(financial.get("net_worth_indicator"))
        credit = self._safe(financial.get("credit_risk_indicator"), 0.5)

        # --- Per-entity DB features (from financial_features.py) ---
        dti = self._safe(financial.get("debt_to_income_ratio"))
        stir = self._safe(financial.get("saving_to_income_ratio"))
        meb = self._safe(financial.get("monthly_expense_burden_ratio"))
        efm = self._safe(financial.get("emergency_fund_months"))

        triggered = []

        # =================================================================
        # RED: Purchase would cause financial harm.
        # All conditions within each rule must be true (AND logic).
        # Every rule crosses at least 2 independent groups and includes
        # a PIR escape hatch so trivial purchases never trigger RED.
        # =================================================================

        # RED Rule 1 — Groups 1 + 2 + price awareness
        # Income can't handle it AND savings can't absorb it AND
        # purchase isn't trivial.
        if (affordability < 0 and spr < 1.5
                and rus < 1.0 and pir > 0.10):
            triggered.append("red:cant_afford_from_any_angle")
            return DecisionResult("RED", triggered,
                explanation="Income can't handle it, savings barely cover "
                            "the price, and the purchase is non-trivial.")

        # RED Rule 2 — Groups 3 + 1 + 2
        # Budget is maxed AND purchase is significant AND savings can't
        # backstop.
        if (meb > 0.80 and pir > 0.20
                and efm < 3.0 and spr < 3.0):
            triggered.append("red:maxed_budget_significant_purchase")
            return DecisionResult("RED", triggered,
                explanation="Budget over 80% consumed, product is 20%+ of "
                            "income, thin emergency fund, and savings can't "
                            "backstop.")

        # RED Rule 3 — Groups 4 + 1 + 2
        # Deeply underwater AND no surplus AND no safety cushion.
        if (nwi < -2.0 and affordability < 0
                and pir > 0.15 and efm < 3.0):
            triggered.append("red:underwater_no_surplus_significant")
            return DecisionResult("RED", triggered,
                explanation="Net worth deeply negative, no surplus income, "
                            "significant purchase, and thin emergency fund.")

        # RED Rule 4 — Groups 2 + 3 + price awareness
        # No emergency runway AND purchase wipes remaining runway AND
        # heavily indebted AND purchase isn't trivial.
        if (efm < 1.0 and rus < 0.5
                and dti > 0.30 and pir > 0.10):
            triggered.append("red:paycheck_to_paycheck_wipes_net")
            return DecisionResult("RED", triggered,
                explanation="Less than 1 month emergency fund, purchase "
                            "leaves <0.5 months runway, DTI over 30%, "
                            "and purchase is non-trivial.")

        # =================================================================
        # YELLOW: Genuine concern — user should pause and think.
        #
        # 5 compound AND rules, each crossing at least 2 correlation groups.
        # YELLOW triggers when >= 2 rules fire.
        # =================================================================

        yellow_count = 0
        yellow_reasons = []

        # YELLOW Rule 1 — Groups 1 + 2
        # Income can't cover it AND purchase is a big portion of paycheck
        # AND savings don't easily absorb it.
        if affordability < 0 and pir > 0.25 and spr < 10.0:
            yellow_count += 1
            yellow_reasons.append("yellow:income_pressure")

        # YELLOW Rule 2 — Groups 2 + 1
        # Savings are thin for this purchase AND post-purchase runway is
        # uncomfortable AND purchase is meaningful relative to income.
        if spr < 5.0 and rus < 3.0 and pir > 0.10:
            yellow_count += 1
            yellow_reasons.append("yellow:savings_strain")

        # YELLOW Rule 3 — Groups 3 + 2 + price awareness
        # Heavy debt obligations AND thin emergency cushion AND
        # purchase is meaningful.
        if dti > 0.30 and efm < 4.0 and pir > 0.10:
            yellow_count += 1
            yellow_reasons.append("yellow:debt_stress")

        # YELLOW Rule 4 — Groups 2 + 1
        # Thin safety net AND low annual savings AND income can't
        # cover the purchase.
        if efm < 3.0 and stir < 0.25 and affordability < 0:
            yellow_count += 1
            yellow_reasons.append("yellow:low_resilience")

        # YELLOW Rule 5 — Groups 4 + 1 + 2
        # Weak credit AND poor net worth AND purchase is significant
        # AND savings can't easily absorb.
        if credit < 0.35 and nwi < 1.0 and pir > 0.15 and spr < 10.0:
            yellow_count += 1
            yellow_reasons.append("yellow:weak_profile")

        # YELLOW fires when 2+ rules fire.
        if yellow_count >= 2:
            return DecisionResult("YELLOW", yellow_reasons, [],
                explanation=f"{yellow_count} financial concern(s) triggered.")

        # =================================================================
        # GREEN: Purchase fits comfortably within the user's finances.
        # No RED rules fired, and fewer than 2 YELLOW rules accumulated.
        # =================================================================
        return DecisionResult("GREEN", ["green:all_clear"], [],
            explanation="Affordable purchase with healthy financial position.")

    def decide_row(self, row) -> str:
        """
        Wrapper for pandas .apply(). Extracts financial and product
        fields from a DataFrame row and returns the label string.
        """
        financial = {
            # Per-pair computed features.
            "affordability_score": row.get("affordability_score"),
            "price_to_income_ratio": row.get("price_to_income_ratio"),
            "residual_utility_score": row.get("residual_utility_score"),
            "savings_to_price_ratio": row.get("savings_to_price_ratio"),
            "net_worth_indicator": row.get("net_worth_indicator"),
            "credit_risk_indicator": row.get("credit_risk_indicator"),
            # Per-entity DB features.
            "debt_to_income_ratio": row.get("debt_to_income_ratio"),
            "saving_to_income_ratio": row.get("saving_to_income_ratio"),
            "monthly_expense_burden_ratio": row.get("monthly_expense_burden_ratio"),
            "emergency_fund_months": row.get("emergency_fund_months"),
        }
        product = {
            "price": row.get("price", row.get("product_price")),
        }
        return self.decide(financial, product).color
