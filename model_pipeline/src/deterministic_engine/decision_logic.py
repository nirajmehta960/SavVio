"""
Deterministic Decision Engine for SavVio.

Pure-Python rule system that assigns GREEN / YELLOW / RED labels
to (user, product) pairs based on financial safety and product
signal quality.  This engine is AUTHORITATIVE — neither the ML
model nor the LLM wrapper may override its output.

Rule evaluation order (most conservative first):
    Tier 1  Hard-stop safety checks     → RED  (immediate return)
    Tier 2  Caution checks              → YELLOW candidate
    Tier 3  Confidence downgrade checks → downgrade one level
    Tier 4  Final color assignment

Usage (batch — for label generation):
    from deterministic_engine.decision_logic import DecisionEngine
    engine = DecisionEngine()
    result = engine.decide(financial_row, product_row)

Usage (single row in a DataFrame via .apply):
    df["label"] = df.apply(engine.decide_row, axis=1)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    """Outcome of the deterministic engine for one (user, product) pair."""
    color: str                                  # GREEN | YELLOW | RED
    triggered_rules: List[str] = field(default_factory=list)
    confidence_downgrades: List[str] = field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Stateless rule engine.  All thresholds are injected at init so that
    the engine can be unit-tested without importing Config.
    """

    def __init__(
        self,
        hard_stop: Optional[dict] = None,
        caution: Optional[dict] = None,
        confidence_downgrade: Optional[dict] = None,
    ):
        # Lazy-import Config only if caller didn't supply thresholds.
        if hard_stop is None or caution is None or confidence_downgrade is None:
            from config import Config
            hard_stop = hard_stop or Config.HARD_STOP
            caution = caution or Config.CAUTION
            confidence_downgrade = confidence_downgrade or Config.CONFIDENCE_DOWNGRADE

        self.hs = hard_stop
        self.ct = caution
        self.cd = confidence_downgrade

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _safe(val, default=None):
        """Return default if val is None or NaN."""
        if val is None:
            return default
        try:
            if math.isnan(val):
                return default
        except TypeError:
            pass
        return val

    def _in_range(self, val, lo, hi):
        """Inclusive range check, NaN-safe."""
        v = self._safe(val)
        if v is None:
            return False
        return lo <= v <= hi

    # ---------------------------------------------------------------------------
    # Tier 1: Hard-stop safety checks → RED
    # ---------------------------------------------------------------------------
    # Any single hard-stop violation causes an immediate RED classification.
    # These represent financial situations too risky to recommend a purchase.
    # ---------------------------------------------------------------------------

    def _check_hard_stops(self, fin: dict, prod: dict) -> List[str]:
        """Return list of triggered hard-stop rule names."""
        triggered = []
        di = self._safe(fin.get("discretionary_income"))
        dti = self._safe(fin.get("debt_to_income_ratio"))
        meb = self._safe(fin.get("monthly_expense_burden_ratio"))
        efm = self._safe(fin.get("emergency_fund_months"))
        price = self._safe(prod.get("price"), 0)

        # Rule 1: Negative discretionary income — user cannot cover basic expenses.
        if di is not None and di < self.hs["discretionary_income_lt"]:
            triggered.append("hard_stop:negative_discretionary_income")

        # Rule 2: Debt-to-income ratio too high — over-leveraged.
        if dti is not None and dti > self.hs["debt_to_income_ratio_gt"]:
            triggered.append("hard_stop:high_debt_to_income")

        # Rule 3: Expense burden ratio too high — expenses consume too much income.
        if meb is not None and meb > self.hs["monthly_expense_burden_ratio_gt"]:
            triggered.append("hard_stop:high_expense_burden")

        # Rule 4: Emergency fund too thin — less than 1 month of runway.
        if efm is not None and efm < self.hs["emergency_fund_months_lt"]:
            triggered.append("hard_stop:low_emergency_fund")

        # Rule 5 (compound): Product price exceeds discretionary income AND thin runway.
        if (di is not None and efm is not None
                and price > di
                and efm < self.hs["price_exceeds_discretionary_emergency_lt"]):
            triggered.append("hard_stop:price_exceeds_discretionary_thin_runway")

        return triggered

    # ---------------------------------------------------------------------------
    # Tier 2: Caution checks → YELLOW candidate
    # ---------------------------------------------------------------------------
    # These represent borderline financial situations. Any single caution
    # violation marks the scenario as YELLOW (proceed with caution).
    # ---------------------------------------------------------------------------

    def _check_caution(self, fin: dict, prod: dict) -> List[str]:
        """Return list of triggered caution rule names."""
        triggered = []

        di = self._safe(fin.get("discretionary_income"))
        dti = self._safe(fin.get("debt_to_income_ratio"))
        meb = self._safe(fin.get("monthly_expense_burden_ratio"))
        efm = self._safe(fin.get("emergency_fund_months"))
        sti = self._safe(fin.get("saving_to_income_ratio"))

        # Caution 1: Tight discretionary budget (0–1000).
        lo, hi = self.ct["discretionary_income_range"]
        if di is not None and lo <= di <= hi:
            triggered.append("caution:tight_discretionary_income")

        # Caution 2: Moderate debt load (0.20–0.40).
        lo, hi = self.ct["debt_to_income_ratio_range"]
        if dti is not None and lo <= dti <= hi:
            triggered.append("caution:moderate_debt_to_income")

        # Caution 3: Moderate expense burden (0.50–0.80).
        lo, hi = self.ct["monthly_expense_burden_ratio_range"]
        if meb is not None and lo <= meb <= hi:
            triggered.append("caution:moderate_expense_burden")

        # Caution 4: Thin emergency fund (1–3 months).
        lo, hi = self.ct["emergency_fund_months_range"]
        if efm is not None and lo <= efm <= hi:
            triggered.append("caution:thin_emergency_fund")

        # Caution 5: Low savings ratio (0.25–1.0).
        lo, hi = self.ct["saving_to_income_ratio_range"]
        if sti is not None and lo <= sti <= hi:
            triggered.append("caution:low_savings_ratio")

        return triggered

    # ---------------------------------------------------------------------------
    # Tier 3: Confidence downgrade — product signal quality
    # ---------------------------------------------------------------------------
    # These check the reliability of product review data. Weak product signals
    # indicate we cannot trust the product quality assessment, so we downgrade
    # the recommendation by one level: GREEN → YELLOW, YELLOW → RED.
    # ---------------------------------------------------------------------------

    def _check_confidence_downgrades(self, prod: dict) -> List[str]:
        """Return list of triggered downgrade rule names."""
        triggered = []
        rn = self._safe(prod.get("rating_number"), 0)
        rv = self._safe(prod.get("rating_variance"), 0)
        ar = self._safe(prod.get("average_rating"), 0)

        # Downgrade 1: Too few reviews — insufficient data for reliable signal.
        if rn < self.cd["rating_number_lt"]:
            triggered.append("downgrade:insufficient_reviews")

        # Downgrade 2: Zero variance with few reviews — likely artificial/gamed ratings.
        if rv == 0 and rn < self.cd["rating_variance_zero_count_lt"]:
            triggered.append("downgrade:artificial_uniform_signal")

        # Downgrade 3: High variance — polarized opinions indicate inconsistent quality.
        if rv > self.cd["rating_variance_gt"]:
            triggered.append("downgrade:polarized_ratings")

        # Downgrade 4: Poor average rating — product itself has quality issues.
        if ar <= self.cd["average_rating_lte"]:
            triggered.append("downgrade:poor_average_rating")

        return triggered

    # ---------------------------------------------------------------------------
    # Tier 4: Final color assignment
    # ---------------------------------------------------------------------------
    # Combines results from all tiers into a single decision.
    # Conflicting signals always resolve to the MORE conservative label.
    # ---------------------------------------------------------------------------

    def decide(self, financial: dict, product: dict) -> DecisionResult:
        """
        Evaluate all tiers and return the final decision.

        Args:
            financial: Dict with keys matching Config.FINANCIAL_FEATURES.
            product:   Dict with keys matching Config.PRODUCT_FEATURES.

        Returns:
            DecisionResult with color, triggered rules, and explanation.
        """
        # Edge case: all financial fields missing → default to YELLOW.
        # We cannot assess safety without financial data, but RED is too harsh.
        critical_fields = [
            "discretionary_income", "debt_to_income_ratio",
            "monthly_expense_burden_ratio", "emergency_fund_months",
        ]
        missing_fields = [f for f in critical_fields
                          if self._safe(financial.get(f)) is None]
        if len(missing_fields) == len(critical_fields):
            return DecisionResult(
                color="YELLOW",
                triggered_rules=["edge_case:all_financial_fields_missing"],
                explanation="All financial fields are missing — defaulting to YELLOW.",
            )

        # Tier 1 — Hard-stop checks (any match → immediate RED).
        hard_stops = self._check_hard_stops(financial, product)
        if hard_stops:
            return DecisionResult(
                color="RED",
                triggered_rules=hard_stops,
                explanation=f"Hard-stop triggered: {', '.join(hard_stops)}",
            )

        # Tier 2 — Caution checks (any match → YELLOW base color).
        caution_rules = self._check_caution(financial, product)

        # Tier 3 — Confidence downgrades (each shifts color one level).
        downgrades = self._check_confidence_downgrades(product)

        # Determine base color from caution results.
        if caution_rules:
            base_color = "YELLOW"
        else:
            base_color = "GREEN"

        # Apply downgrades: each downgrade shifts one level.
        #   GREEN → YELLOW, YELLOW → RED, RED stays RED.
        final_color = base_color
        for _ in downgrades:
            if final_color == "GREEN":
                final_color = "YELLOW"
            elif final_color == "YELLOW":
                final_color = "RED"
            # RED stays RED — cannot downgrade past the worst label.

        # Build human-readable explanation.
        explanation_parts = []
        if caution_rules:
            explanation_parts.append(f"Caution: {', '.join(caution_rules)}")
        if downgrades:
            explanation_parts.append(f"Downgrades: {', '.join(downgrades)}")
        if not caution_rules and not downgrades:
            explanation_parts.append("All checks passed — GREEN.")

        return DecisionResult(
            color=final_color,
            triggered_rules=caution_rules,
            confidence_downgrades=downgrades,
            explanation=" | ".join(explanation_parts),
        )

    # ---------------------------------------------------------------------------
    # Convenience wrapper for DataFrame.apply()
    # ---------------------------------------------------------------------------

    def decide_row(self, row) -> str:
        """
        Wrapper for pandas .apply(). Expects a row containing both financial
        and product columns.

        Args:
            row: A pandas Series or dict-like with financial + product features.

        Returns:
            Label string: "GREEN", "YELLOW", or "RED".
        """
        financial = {
            "discretionary_income": row.get("discretionary_income"),
            "debt_to_income_ratio": row.get("debt_to_income_ratio"),
            "saving_to_income_ratio": row.get("saving_to_income_ratio"),
            "monthly_expense_burden_ratio": row.get("monthly_expense_burden_ratio"),
            "emergency_fund_months": row.get("emergency_fund_months"),
        }
        product = {
            "price": row.get("price", row.get("product_price")),
            "average_rating": row.get("average_rating"),
            "rating_number": row.get("rating_number"),
            "rating_variance": row.get("rating_variance"),
        }
        return self.decide(financial, product).color
