"""
Anomaly validation for SavVio pipeline (Phase 7).

Detects statistical outliers and anomalies in raw data using
Z-score, IQR, and rule-based methods from detectors.py.

Targets:
  - financial.csv (Income outliers, expense anomalies)
  - reviews.jsonl (Rating distribution anomalies)
"""

import logging
import pandas as pd
from pathlib import Path
from anomaly.detectors import AnomalyDetector
from validation_config import CheckResult, Severity, ValidationReport

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def _report_outliers(column: str, method: str, outliers: list, 
                     dataset: str, severity: Severity = Severity.WARNING) -> CheckResult:
    """Convert outlier indices into a CheckResult."""
    count = len(outliers)
    passed = count == 0
    
    details = f"Found {count} outliers in '{column}' using {method}"
    if not passed:
        details += f" (indices: {outliers[:5]}...)"
        
    return CheckResult(
        check_name=f"anomaly_{column}_{method}",
        passed=passed,
        severity=severity,
        dataset=dataset,
        stage="anomaly",
        details=details,
        metric_value=count,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Financial Anomalies
# ═══════════════════════════════════════════════════════════════════════════

def validate_financial_anomalies(path: str) -> list[CheckResult]:
    """Check for outliers in financial data."""
    results = []
    ds = "financial"
    
    if not Path(path).exists():
        logger.warning(f"Financial data not found at {path}")
        return [CheckResult(f"load_{ds}", False, Severity.CRITICAL, ds, "anomaly", "File not found", 0)]

    try:
        df = _load_csv(path)
        detector = AnomalyDetector(df)
        
        # 1. Monthly Income (IQR method is redundant if we assume log-normal, but useful for extreme outliers)
        # Use IQR with a higher multiplier for income to avoid flagging valid high earners too aggressively
        outliers_income = detector.check_iqr("monthly_income_usd", multiplier=3.0)
        results.append(_report_outliers("monthly_income_usd", "iqr_3.0", outliers_income, ds))

        # 2. Savings Balance (Z-score might be skewed by super-savers, try IQR)
        outliers_savings = detector.check_iqr("savings_usd", multiplier=3.0)
        results.append(_report_outliers("savings_usd", "iqr_3.0", outliers_savings, ds))
        
        # 3. Monthly Expenses (Z-score check)
        outliers_expenses = detector.check_z_score("monthly_expenses_usd", threshold=4.0)
        results.append(_report_outliers("monthly_expenses_usd", "zscore_4.0", outliers_expenses, ds))

        # 4. Zero Income Check (Rule-based)
        # Income = 0 might be valid (unemployed) but is suspicious for a purchase guardrail app
        outliers_zero_income = detector.check_rule(
            "monthly_income_usd", lambda x: x > 0, "income_positive"
        )
        results.append(_report_outliers("monthly_income_usd", "zero_check", outliers_zero_income, ds, Severity.WARNING))

        # 5. Expenses > 2x Income (Rule-based Cross-column)
        # This requires a custom check since detector.check_rule works on a single column
        if "monthly_income_usd" in df.columns and "monthly_expenses_usd" in df.columns:
            # We want to flag rows where expenses > 2 * income
            # Valid rows are those where expenses <= 2 * income OR income is 0 (to avoid double jeopardy with check 4)
            # Actually, if income is 0, expenses > 0 is essentially infinite ratio.
            
            # Let's find indices where expenses > 2 * income
            high_expense_mask = df["monthly_expenses_usd"] > (2 * df["monthly_income_usd"])
            # Filter out cases where income is 0, as that's handled by valid_zero_income check, 
            # or maybe we DO want to flag consistent overspending even if income is low.
            # Let's stick to the rule: Expenses > 2x Income
            outliers_ratio = df[high_expense_mask].index.tolist()
            
            results.append(_report_outliers(
                "expense_income_ratio", "expenses_2x_income", outliers_ratio, ds, Severity.WARNING
            ))
            
            # Combine all outliers for quarantine
            all_outliers = set(outliers_income + outliers_savings + outliers_expenses + outliers_zero_income + outliers_ratio)
            
            if all_outliers:
                _quarantine_records(df, list(all_outliers), ds)

    except Exception as e:
        logger.error(f"Failed to validate financial anomalies: {e}")
        results.append(CheckResult(f"anomaly_check_{ds}", False, Severity.CRITICAL, ds, "anomaly", str(e), 0))
        
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Review Anomalies
# ═══════════════════════════════════════════════════════════════════════════

def validate_review_anomalies(path: str) -> list[CheckResult]:
    """Check for outliers in review data."""
    results = []
    ds = "reviews"

    if not Path(path).exists():
        logger.warning(f"Review data not found at {path}")
        return [CheckResult(f"load_{ds}", False, Severity.CRITICAL, ds, "anomaly", "File not found", 0)]

    try:
        df = _load_jsonl(path)
        detector = AnomalyDetector(df)
        
        all_outliers = set()

        # 1. Rating (Should be 1-5, so 'outliers' statistically might clean up bad scraping)
        # However, ratings are categorical/ordinal. Z-score isn't great, but huge variance might indicate scraping errors?
        # Actually, for ratings, we care if they are outside 1-5 (Rule) or if the distribution is suspicious.
        # Let's check 'helpful_votes' if it exists for outliers (spam detection?)
        if "helpful_vote" in df.columns:
            # Check Z-score for helpful votes
            outliers_votes = detector.check_z_score("helpful_vote", threshold=5.0)
            results.append(_report_outliers("helpful_vote", "zscore_5.0", outliers_votes, ds, Severity.INFO))
            all_outliers.update(outliers_votes)
        elif "helpful_votes" in df.columns:
             # Handle alternative column name if present
            outliers_votes = detector.check_z_score("helpful_votes", threshold=5.0)
            results.append(_report_outliers("helpful_votes", "zscore_5.0", outliers_votes, ds, Severity.INFO))
            all_outliers.update(outliers_votes)

        if all_outliers:
             _quarantine_records(df, list(all_outliers), ds)
            
    except Exception as e:
        logger.error(f"Failed to validate review anomalies: {e}")
        results.append(CheckResult(f"anomaly_check_{ds}", False, Severity.CRITICAL, ds, "anomaly", str(e), 0))

    return results


def _quarantine_records(df: pd.DataFrame, indices: list, dataset_name: str, quarantine_dir: str = "data/quarantine"):
    """Save suspicious records to a quarantine file."""
    if not indices:
        return

    try:
        Path(quarantine_dir).mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_anomalies_{ts}.json"
        path = Path(quarantine_dir) / filename
        
        suspicious_df = df.loc[indices]
        suspicious_df.to_json(path, orient="records", lines=True)
        
        logger.warning(f"Quarantined {len(indices)} suspicious records to {path}")
    except Exception as e:
        logger.error(f"Failed to quarantine records: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_anomaly_validation(
    financial_path: str = "data/raw/financial_data.csv",
    reviews_path: str = "data/raw/review_data.jsonl"
) -> ValidationReport:
    """Run all anomaly detection checks."""
    report = ValidationReport(stage="anomaly")
    
    logger.info("═" * 50)
    logger.info("  ANOMALY DETECTION (Phase 7)")
    logger.info("═" * 50)
    
    logger.info(f"── Checking financial anomalies in {financial_path} ──")
    for r in validate_financial_anomalies(financial_path):
        report.add(r)
        
    logger.info(f"── Checking review anomalies in {reviews_path} ──")
    for r in validate_review_anomalies(reviews_path):
        report.add(r)
        
    report.print_summary()
    report.save()
    
    # Anomaly detection usually shouldn't HALT the pipeline unless specified.
    # We'll treat CRITICAL as HALT, WARNING as ALERT.
    if not report.passed:
        logger.critical("ANOMALY DETECTION FAILED — pipeline should HALT")
    elif report.has_warnings:
        logger.warning("ANOMALY DETECTION found statistical outliers — triggering alerts")
        
    return report

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Run Anomaly Detection")
    parser.add_argument("--financial", default="data/raw/financial_data.csv")
    parser.add_argument("--reviews", default="data/raw/review_data.jsonl")
    args = parser.parse_args()
    
    run_anomaly_validation(args.financial, args.reviews)
