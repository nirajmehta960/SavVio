"""
Processed data validation for SavVio pipeline (Phase 9).

Validates data in data/processed/ to ensure preprocessing
transformations didn't break data or introduce errors.

Expected processed columns (based on preprocessing steps):
  Financial: *_usd renamed, monthly normalized, median imputed
  Products:  product_name, price_usd, appliance_type, description_length
  Reviews:   sentiment, review_length, cleaned text
"""

import logging
import pandas as pd
import great_expectations as gx
from great_expectations.dataset import PandasDataset

from validation_config import (
    CheckResult, Severity, ValidationReport, load_thresholds,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load(path: str) -> PandasDataset:
    """Load CSV or JSONL depending on extension."""
    if path.endswith(".jsonl"):
        return gx.from_pandas(pd.read_json(path, lines=True))
    return gx.from_pandas(pd.read_csv(path))


def _check(ge_result: dict, name: str, severity: Severity,
           dataset: str, details: str = "") -> CheckResult:
    passed = ge_result["success"]
    metric = ge_result.get("result", {})
    return CheckResult(
        check_name=name, passed=passed, severity=severity,
        dataset=dataset, stage="processed",
        details=details or str(metric.get("partial_unexpected_list", ""))[:200],
        metric_value=metric.get("unexpected_percent") or metric.get("observed_value"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Financial processed validation
# ═══════════════════════════════════════════════════════════════════════════

FINANCIAL_PROCESSED_COLS = [
    "user_id", "income_usd", "monthly_expenses", "monthly_savings",
    "total_fixed_expenses",
]
FINANCIAL_OPTIONAL_PROCESSED = [
    "income_missing", "has_loan", "loan_amount", "monthly_emi",
    "credit_score", "employment_status", "region",
]


def validate_financial_processed(path: str, raw_path: str,
                                  thresholds: dict) -> list[CheckResult]:
    """Validate processed financial data."""
    results: list[CheckResult] = []
    ds = "financial"

    gdf = _load(path)

    # ── 1. Required processed columns exist ───────────────────────────────
    for col in FINANCIAL_PROCESSED_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"fin_proc_col_{col}", Severity.CRITICAL, ds,
                              f"Processed column '{col}' must exist"))

    # ── 2. No new nulls in required fields ────────────────────────────────
    for col in ["income_usd", "monthly_expenses", "total_fixed_expenses"]:
        if col not in gdf.columns:
            continue
        # After median imputation, these should have ZERO nulls
        res = gdf.expect_column_values_to_not_be_null(col)
        results.append(_check(res, f"fin_proc_no_nulls_{col}", Severity.CRITICAL, ds,
                              f"'{col}' should have no nulls after imputation"))

    # ── 3. USD suffix columns are numeric and >= 0 ────────────────────────
    for col in ["income_usd", "monthly_expenses", "monthly_savings", "total_fixed_expenses"]:
        if col not in gdf.columns:
            continue
        res = gdf.expect_column_values_to_be_between(col, min_value=0)
        results.append(_check(res, f"fin_proc_{col}_non_negative", Severity.WARNING, ds,
                              f"'{col}' should be >= 0 after processing"))

    # ── 4. Values rounded to 2 decimals ───────────────────────────────────
    for col in ["income_usd", "monthly_expenses", "monthly_savings", "total_fixed_expenses"]:
        if col not in gdf.columns:
            continue
        # Check that values have at most 2 decimal places
        not_rounded = (gdf[col].dropna() * 100 % 1 > 0.001).sum()
        pct_bad = not_rounded / max(len(gdf), 1)
        results.append(CheckResult(
            check_name=f"fin_proc_{col}_rounded",
            passed=pct_bad < 0.01,
            severity=Severity.INFO, dataset=ds, stage="processed",
            details=f"{pct_bad:.2%} of values not rounded to 2 decimals",
            metric_value=round(pct_bad, 4),
        ))

    # ── 5. income_missing flag is boolean (if present) ────────────────────
    if "income_missing" in gdf.columns:
        res = gdf.expect_column_values_to_be_in_set("income_missing", [True, False, 0, 1])
        results.append(_check(res, "fin_proc_income_missing_bool", Severity.WARNING, ds,
                              "income_missing should be boolean"))

    # ── 6. Record count comparison (raw vs processed) ─────────────────────
    try:
        raw_df = pd.read_csv(raw_path)
        raw_count = len(raw_df)
        proc_count = len(gdf)
        loss_pct = 1 - proc_count / max(raw_count, 1)

        sev = Severity.CRITICAL if loss_pct > 0.20 else (
              Severity.WARNING if loss_pct > 0.05 else Severity.INFO)
        results.append(CheckResult(
            check_name="fin_proc_record_loss",
            passed=loss_pct <= 0.20,
            severity=sev, dataset=ds, stage="processed",
            details=f"Raw: {raw_count} → Processed: {proc_count} (loss: {loss_pct:.1%})",
            metric_value=round(loss_pct, 4),
        ))
    except Exception as e:
        logger.warning("Could not compare raw vs processed counts: %s", e)

    # ── 7. No duplicates after processing ─────────────────────────────────
    if "user_id" in gdf.columns:
        dup_count = gdf.duplicated(subset=["user_id"]).sum()
        results.append(CheckResult(
            check_name="fin_proc_no_duplicates",
            passed=dup_count == 0,
            severity=Severity.WARNING, dataset=ds, stage="processed",
            details=f"{dup_count} duplicate user_ids after processing",
            metric_value=dup_count,
        ))

    # ── 8. total_fixed_expenses <= monthly_expenses (sanity) ──────────────
    if "total_fixed_expenses" in gdf.columns and "monthly_expenses" in gdf.columns:
        violations = (gdf["total_fixed_expenses"] > gdf["monthly_expenses"]).sum()
        pct = violations / max(len(gdf), 1)
        results.append(CheckResult(
            check_name="fin_proc_fixed_lte_total",
            passed=pct < 0.05,
            severity=Severity.WARNING, dataset=ds, stage="processed",
            details=f"{violations} rows where fixed_expenses > total_expenses ({pct:.1%})",
            metric_value=round(pct, 4),
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Products processed validation
# ═══════════════════════════════════════════════════════════════════════════

PRODUCT_PROCESSED_COLS = [
    "product_id", "product_name", "price_usd",
]
PRODUCT_OPTIONAL_PROCESSED = [
    "average_rating", "rating_number", "description", "features",
    "details", "category", "appliance_type", "description_length",
]


def validate_products_processed(path: str, raw_path: str,
                                 thresholds: dict) -> list[CheckResult]:
    """Validate processed products data."""
    results: list[CheckResult] = []
    ds = "products"

    gdf = _load(path)

    # ── 1. Required columns ───────────────────────────────────────────────
    for col in PRODUCT_PROCESSED_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"prod_proc_col_{col}", Severity.CRITICAL, ds,
                              f"Processed column '{col}' must exist"))

    # ── 2. product_name not empty (cleaned title) ─────────────────────────
    if "product_name" in gdf.columns:
        res = gdf.expect_column_value_lengths_to_be_between("product_name", min_value=1, mostly=0.99)
        results.append(_check(res, "prod_proc_name_non_empty", Severity.WARNING, ds,
                              "product_name should not be empty after cleaning"))

    # ── 3. price_usd valid ────────────────────────────────────────────────
    if "price_usd" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("price_usd", min_value=0.01, max_value=100_000)
        results.append(_check(res, "prod_proc_price_range", Severity.WARNING, ds,
                              "price_usd should be $0.01–$100,000"))

        # No nulls in price after processing
        res = gdf.expect_column_values_to_not_be_null("price_usd")
        results.append(_check(res, "prod_proc_price_no_nulls", Severity.CRITICAL, ds,
                              "price_usd must not be null after processing"))

    # ── 4. main_category should be dropped (constant column) ──────────────
    if "main_category" in gdf.columns:
        n_unique = gdf["main_category"].nunique()
        results.append(CheckResult(
            check_name="prod_proc_main_cat_dropped",
            passed=False,  # It should have been dropped
            severity=Severity.INFO, dataset=ds, stage="processed",
            details=f"main_category still present ({n_unique} unique values) — expected to be dropped",
            metric_value=n_unique,
        ))

    # ── 5. description_length is non-negative integer ─────────────────────
    if "description_length" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("description_length", min_value=0)
        results.append(_check(res, "prod_proc_desc_len_valid", Severity.INFO, ds,
                              "description_length should be >= 0"))

    # ── 6. appliance_type populated ───────────────────────────────────────
    if "appliance_type" in gdf.columns:
        null_pct = gdf["appliance_type"].isnull().mean()
        results.append(CheckResult(
            check_name="prod_proc_appliance_type_populated",
            passed=null_pct < 0.30,
            severity=Severity.INFO, dataset=ds, stage="processed",
            details=f"appliance_type null rate = {null_pct:.1%}",
            metric_value=round(null_pct, 4),
        ))

    # ── 7. No duplicate products ──────────────────────────────────────────
    if "product_id" in gdf.columns:
        dup_count = gdf.duplicated(subset=["product_id"]).sum()
        results.append(CheckResult(
            check_name="prod_proc_no_duplicates",
            passed=dup_count == 0,
            severity=Severity.WARNING, dataset=ds, stage="processed",
            details=f"{dup_count} duplicate product_ids after deduplication",
            metric_value=dup_count,
        ))

    # ── 8. Record count comparison ────────────────────────────────────────
    try:
        raw_df = pd.read_json(raw_path, lines=True)
        raw_count = len(raw_df)
        proc_count = len(gdf)
        loss_pct = 1 - proc_count / max(raw_count, 1)
        sev = Severity.CRITICAL if loss_pct > 0.30 else (
              Severity.WARNING if loss_pct > 0.10 else Severity.INFO)
        results.append(CheckResult(
            check_name="prod_proc_record_loss",
            passed=loss_pct <= 0.30,
            severity=sev, dataset=ds, stage="processed",
            details=f"Raw: {raw_count} → Processed: {proc_count} (loss: {loss_pct:.1%})",
            metric_value=round(loss_pct, 4),
        ))
    except Exception as e:
        logger.warning("Could not compare raw vs processed product counts: %s", e)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Reviews processed validation
# ═══════════════════════════════════════════════════════════════════════════

REVIEW_PROCESSED_COLS = [
    "user_id", "asin", "rating",
]
REVIEW_OPTIONAL_PROCESSED = [
    "product_id", "review_title", "review_text", "sentiment",
    "review_length", "verified_purchase", "helpful_vote",
]


def validate_reviews_processed(path: str, raw_path: str,
                                thresholds: dict) -> list[CheckResult]:
    """Validate processed reviews data."""
    results: list[CheckResult] = []
    ds = "reviews"

    gdf = _load(path)

    # ── 1. Required columns ───────────────────────────────────────────────
    for col in REVIEW_PROCESSED_COLS:
        res = gdf.expect_column_to_exist(col)
        results.append(_check(res, f"rev_proc_col_{col}", Severity.CRITICAL, ds,
                              f"Processed column '{col}' must exist"))

    # ── 2. Rating still 1–5 ──────────────────────────────────────────────
    if "rating" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("rating", min_value=1.0, max_value=5.0)
        results.append(_check(res, "rev_proc_rating_range", Severity.CRITICAL, ds,
                              "rating must remain 1.0–5.0 after processing"))

    # ── 3. Sentiment mapping correct ──────────────────────────────────────
    if "sentiment" in gdf.columns:
        valid_sentiments = ["positive", "neutral", "negative"]
        res = gdf.expect_column_values_to_be_in_set("sentiment", valid_sentiments)
        results.append(_check(res, "rev_proc_sentiment_valid", Severity.WARNING, ds,
                              "sentiment must be positive/neutral/negative"))

        # Cross-check: sentiment matches rating
        if "rating" in gdf.columns:
            df_check = pd.DataFrame({"rating": gdf["rating"], "sentiment": gdf["sentiment"]})
            mismatches = (
                ((df_check["rating"] >= 4) & (df_check["sentiment"] != "positive")) |
                ((df_check["rating"] == 3) & (df_check["sentiment"] != "neutral")) |
                ((df_check["rating"] <= 2) & (df_check["sentiment"] != "negative"))
            ).sum()
            mismatch_pct = mismatches / max(len(gdf), 1)
            results.append(CheckResult(
                check_name="rev_proc_sentiment_rating_consistency",
                passed=mismatch_pct < 0.01,
                severity=Severity.WARNING, dataset=ds, stage="processed",
                details=f"{mismatches} sentiment-rating mismatches ({mismatch_pct:.1%})",
                metric_value=round(mismatch_pct, 4),
            ))

    # ── 4. review_length non-negative ─────────────────────────────────────
    if "review_length" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("review_length", min_value=0)
        results.append(_check(res, "rev_proc_review_length_valid", Severity.INFO, ds,
                              "review_length should be >= 0"))

    # ── 5. helpful_vote no negatives after fillna ─────────────────────────
    if "helpful_vote" in gdf.columns:
        res = gdf.expect_column_values_to_be_between("helpful_vote", min_value=0)
        results.append(_check(res, "rev_proc_helpful_vote_valid", Severity.INFO, ds,
                              "helpful_vote should be >= 0 after fillna"))

        # No nulls after processing
        null_pct = gdf["helpful_vote"].isnull().mean()
        results.append(CheckResult(
            check_name="rev_proc_helpful_vote_no_nulls",
            passed=null_pct == 0,
            severity=Severity.INFO, dataset=ds, stage="processed",
            details=f"helpful_vote null rate = {null_pct:.2%} (should be 0 after fillna)",
            metric_value=round(null_pct, 4),
        ))

    # ── 6. No duplicate reviews ───────────────────────────────────────────
    dedup_cols = ["user_id", "asin"]
    if all(c in gdf.columns for c in dedup_cols):
        dup_count = gdf.duplicated(subset=dedup_cols).sum()
        results.append(CheckResult(
            check_name="rev_proc_no_duplicates",
            passed=dup_count == 0,
            severity=Severity.WARNING, dataset=ds, stage="processed",
            details=f"{dup_count} duplicate (user_id, asin) after deduplication",
            metric_value=dup_count,
        ))

    # ── 7. Record count comparison ────────────────────────────────────────
    try:
        raw_df = pd.read_json(raw_path, lines=True)
        raw_count = len(raw_df)
        proc_count = len(gdf)
        loss_pct = 1 - proc_count / max(raw_count, 1)
        sev = Severity.CRITICAL if loss_pct > 0.30 else (
              Severity.WARNING if loss_pct > 0.10 else Severity.INFO)
        results.append(CheckResult(
            check_name="rev_proc_record_loss",
            passed=loss_pct <= 0.30,
            severity=sev, dataset=ds, stage="processed",
            details=f"Raw: {raw_count} → Processed: {proc_count} (loss: {loss_pct:.1%})",
            metric_value=round(loss_pct, 4),
        ))
    except Exception as e:
        logger.warning("Could not compare raw vs processed review counts: %s", e)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_processed_validation(
    financial_path: str = "data/processed/financial_processed.csv",
    products_path: str = "data/processed/products_processed.csv",
    reviews_path: str = "data/processed/reviews_processed.csv",
    raw_financial: str = "data/raw/financial.csv",
    raw_products: str = "data/raw/products.jsonl",
    raw_reviews: str = "data/raw/reviews.jsonl",
    threshold_config: str | None = "config/validation_thresholds.json",
) -> ValidationReport:
    """Run all processed data validations."""
    thresholds = load_thresholds(threshold_config)
    report = ValidationReport(stage="processed")

    logger.info("═" * 50)
    logger.info("  PROCESSED DATA VALIDATION")
    logger.info("═" * 50)

    logger.info("── Validating financial_processed ──")
    for r in validate_financial_processed(financial_path, raw_financial, thresholds):
        report.add(r)

    logger.info("── Validating products_processed ──")
    for r in validate_products_processed(products_path, raw_products, thresholds):
        report.add(r)

    logger.info("── Validating reviews_processed ──")
    for r in validate_reviews_processed(reviews_path, raw_reviews, thresholds):
        report.add(r)

    report.print_summary()
    report.save()

    if not report.passed:
        logger.critical("PROCESSED VALIDATION FAILED — pipeline should HALT")
    elif report.has_warnings:
        logger.warning("PROCESSED VALIDATION passed with warnings — sending alerts")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Validate processed SavVio data")
    parser.add_argument("--financial", default="data/processed/financial_processed.csv")
    parser.add_argument("--products", default="data/processed/products_processed.csv")
    parser.add_argument("--reviews", default="data/processed/reviews_processed.csv")
    parser.add_argument("--raw-financial", default="data/raw/financial.csv")
    parser.add_argument("--raw-products", default="data/raw/products.jsonl")
    parser.add_argument("--raw-reviews", default="data/raw/reviews.jsonl")
    parser.add_argument("--thresholds", default=None)
    args = parser.parse_args()

    report = run_processed_validation(
        args.financial, args.products, args.reviews,
        args.raw_financial, args.raw_products, args.raw_reviews,
        args.thresholds,
    )

    if not report.passed:
        exit(1)
