"""
Bias Detection & Mitigation Analysis for SavVio data pipeline feature datasets.

Usage:
    python data-pipeline/dags/src/bias/bias_analysis.py
"""

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd


LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger(__name__)


SliceRule = Tuple[str, Callable[[pd.Series], pd.Series]]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure deterministic console logging."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


def get_repo_root() -> Path:
    """Return repository root based on this script location."""
    return Path(__file__).resolve().parents[4]


def validate_columns(df: pd.DataFrame, required_columns: List[str], dataset_name: str) -> None:
    """Ensure required columns exist before analysis."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def load_financial_features(path: Path) -> pd.DataFrame:
    """Load financial featured dataset."""
    logger.info("Loading financial features from %s", path)
    df = pd.read_csv(path)
    validate_columns(
        df,
        [
            "discretionary_income",
            "debt_to_income_ratio",
            "saving_to_income_ratio",
            "monthly_expense_burden_ratio",
            "financial_runway",
        ],
        dataset_name="Financial features",
    )
    logger.info("Loaded %s financial records.", len(df))
    return df


def load_review_features(path: Path) -> pd.DataFrame:
    """
    Load review features from expected path.

    The current artifact has a .jsonl extension but may be CSV-formatted.
    We attempt JSONL first, then fallback to CSV.
    """
    logger.info("Loading review features from %s", path)
    try:
        df = pd.read_json(path, lines=True)
        if {"product_id", "rating_variance"}.issubset(df.columns):
            logger.info("Loaded review features as JSONL with %s products.", len(df))
            return df
        logger.warning("JSONL parse succeeded but schema did not match. Falling back to CSV parse.")
    except ValueError:
        logger.warning("JSONL parse failed. Falling back to CSV parse.")

    df = pd.read_csv(path)
    validate_columns(df, ["product_id", "rating_variance"], dataset_name="Review features")
    logger.info("Loaded review features as CSV with %s products.", len(df))
    return df


def apply_slice_rules(series: pd.Series, rules: List[SliceRule]) -> pd.Series:
    """Assign each record to exactly one configured slice."""
    result = pd.Series("Unknown", index=series.index, dtype="object")
    for label, rule_fn in rules:
        result.loc[rule_fn(series)] = label
    return result


def build_distribution(
    sliced_series: pd.Series,
    ordered_labels: List[str],
    total_count: int,
) -> pd.DataFrame:
    """Build count + percentage distribution table in deterministic order."""
    counts = sliced_series.value_counts(dropna=False)
    rows: List[Dict[str, object]] = []
    for label in ordered_labels:
        count = int(counts.get(label, 0))
        percentage = (count / total_count) * 100 if total_count else 0.0
        rows.append({"group": label, "count": count, "percentage": round(percentage, 2)})

    # Include unknown values, if any.
    unknown_count = int(counts.get("Unknown", 0))
    if unknown_count:
        unknown_pct = (unknown_count / total_count) * 100 if total_count else 0.0
        rows.append({"group": "Unknown", "count": unknown_count, "percentage": round(unknown_pct, 2)})

    return pd.DataFrame(rows)


def detect_underrepresented_groups(
    distribution_df: pd.DataFrame,
    candidate_groups: List[str],
    threshold_pct: float,
) -> List[str]:
    """Return underrepresented groups from candidate list based on threshold."""
    flagged: List[str] = []
    for group_name in candidate_groups:
        row = distribution_df[distribution_df["group"] == group_name]
        if row.empty:
            continue
        pct = float(row.iloc[0]["percentage"])
        if pct < threshold_pct:
            flagged.append(f"{group_name} ({pct:.2f}%)")
    return flagged


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown without external dependencies."""
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = [
        "| " + " | ".join(str(row[col]) for col in headers) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header_line, separator, *body_lines])


def analyze_financial_bias(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """Analyze slice distributions for financial vulnerability dimensions."""
    total = len(df)
    logger.info("Starting financial bias detection on %s records.", total)

    slice_configs: Dict[str, Dict[str, object]] = {
        "Discretionary Income": {
            "series": df["discretionary_income"],
            "rules": [
                ("Negative", lambda s: s < 0),
                ("Tight", lambda s: (s >= 0) & (s <= 1000)),
                ("Comfortable", lambda s: s > 1000),
            ],
            "labels": ["Negative", "Tight", "Comfortable"],
            "vulnerable_groups": ["Negative"],
        },
        "Debt-to-Income Ratio": {
            "series": df["debt_to_income_ratio"],
            "rules": [
                ("safe", lambda s: s < 0.2),
                ("warning", lambda s: (s >= 0.2) & (s <= 0.4)),
                ("Risky", lambda s: s > 0.4),
            ],
            "labels": ["safe", "warning", "Risky"],
            "vulnerable_groups": ["Risky"],
        },
        "Saving-to-Income Ratio": {
            "series": df["saving_to_income_ratio"],
            "rules": [
                ("Fragile", lambda s: s < 0.25),
                ("Moderate", lambda s: (s >= 0.25) & (s <= 1.0)),
                ("strong", lambda s: s > 1.0),
            ],
            "labels": ["Fragile", "Moderate", "strong"],
            "vulnerable_groups": ["Fragile"],
        },
        "Monthly Expense Burden": {
            "series": df["monthly_expense_burden_ratio"],
            "rules": [
                ("comfortable", lambda s: s < 0.5),
                ("tight", lambda s: (s >= 0.5) & (s <= 0.8)),
                ("Overstretched", lambda s: s > 0.8),
            ],
            "labels": ["comfortable", "tight", "Overstretched"],
            "vulnerable_groups": ["Overstretched"],
        },
        "Financial Runway": {
            "series": df["financial_runway"],
            "rules": [
                ("Critical", lambda s: s < 1),
                ("Fragile", lambda s: (s >= 1) & (s <= 3)),
                ("Healthy", lambda s: s > 3),
            ],
            "labels": ["Critical", "Fragile", "Healthy"],
            "vulnerable_groups": ["Critical", "Fragile"],
        },
    }

    distributions: Dict[str, pd.DataFrame] = {}
    flags: Dict[str, List[str]] = {}

    for slice_name, config in slice_configs.items():
        sliced = apply_slice_rules(config["series"], config["rules"])  # type: ignore[arg-type]
        distribution = build_distribution(sliced, config["labels"], total)  # type: ignore[arg-type]
        distributions[slice_name] = distribution

        flagged = detect_underrepresented_groups(
            distribution_df=distribution,
            candidate_groups=config["vulnerable_groups"],  # type: ignore[arg-type]
            threshold_pct=10.0,
        )
        flags[slice_name] = flagged

    return distributions, flags


def analyze_review_bias(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], int]:
    """Analyze review rating variance distributions and low-confidence products."""
    total = len(df)
    logger.info("Starting review bias detection on %s products.", total)

    variance_rules: List[SliceRule] = [
        ("consensus", lambda s: s < 0.5),
        ("Mixed", lambda s: (s >= 0.5) & (s <= 1.0)),
        ("Polarized", lambda s: s > 1.0),
    ]
    variance_labels = ["consensus", "Mixed", "Polarized"]

    variance_slices = apply_slice_rules(df["rating_variance"], variance_rules)
    distribution = build_distribution(variance_slices, variance_labels, total)
    flagged_groups = detect_underrepresented_groups(distribution, variance_labels, threshold_pct=5.0)

    low_confidence_count = int((df["rating_variance"] == 0.0).sum())
    return distribution, flagged_groups, low_confidence_count


def log_distribution_table(title: str, distribution_df: pd.DataFrame) -> None:
    """Print analysis tables to console logs."""
    logger.info("=== %s ===", title)
    logger.info("\n%s", distribution_df.to_string(index=False))


def write_markdown_report(
    output_path: Path,
    financial_distributions: Dict[str, pd.DataFrame],
    financial_flags: Dict[str, List[str]],
    review_distribution: pd.DataFrame,
    review_flags: List[str],
    low_confidence_count: int,
) -> None:
    """Generate markdown bias report with distributions and mitigation documentation."""
    os.makedirs(output_path.parent, exist_ok=True)

    underrepresented_lines: List[str] = []
    for slice_name, flagged in financial_flags.items():
        if flagged:
            underrepresented_lines.append(f"- Financial / {slice_name}: {', '.join(flagged)}")
    if review_flags:
        underrepresented_lines.append(f"- Review / Rating Variance: {', '.join(review_flags)}")
    if not underrepresented_lines:
        underrepresented_lines.append("- No underrepresented configured groups detected.")

    financial_tables = []
    for slice_name, table in financial_distributions.items():
        financial_tables.append(f"### {slice_name}\n\n{dataframe_to_markdown(table)}")

    review_table_md = dataframe_to_markdown(review_distribution)

    report = f"""# Bias Analysis Report

## Overview
This report analyzes representation balance in SavVio's engineered financial and review feature datasets.
The goal is to surface underrepresented segments and confidence risks before downstream modeling decisions.

## Financial Slice Distributions

{chr(10).join(financial_tables)}

## Review Variance Distributions

{review_table_md}

## Underrepresented Groups
{chr(10).join(underrepresented_lines)}

## Low-Confidence Review Count
- Products with `rating_variance == 0.0` (single-review proxy): **{low_confidence_count}**

## Why This Matters for SavVio
- Underrepresented vulnerable users can lead to skewed affordability/risk behavior in downstream scoring.
- Imbalanced variance groups can over-weight consensus products while under-learning polarized outcomes.
- Low-confidence review products can introduce unreliable quality signals if treated as fully certain.

## Mitigation Steps Taken
### Financial
- Documented vulnerable-slice monitoring with explicit underrepresentation flags (`<10%`) to trigger data balancing.
- Mitigation policy: oversample vulnerable user groups during model training/evaluation splits.
- If vulnerable cohorts remain sparse, generate synthetic profiles under controlled governance and validation.

### Review
- Documented variance-group monitoring with explicit underrepresentation flags (`<5%`).
- Products with `rating_variance == 0.0` are flagged as low-confidence for downstream weighting or exclusion rules.
"""

    output_path.write_text(report, encoding="utf-8")
    logger.info("Wrote markdown report to %s", output_path)


def main() -> None:
    """Run complete bias analysis and report generation."""
    setup_logging()
    repo_root = get_repo_root()

    financial_path = repo_root / "data-pipeline/data/features/financial_featured.csv"
    review_path = repo_root / "data-pipeline/data/features/reviews_featured.jsonl"
    report_path = repo_root / "docs/bias_analysis_report.md"

    if not financial_path.exists():
        raise FileNotFoundError(f"Financial feature file not found: {financial_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Review feature file not found: {review_path}")

    financial_df = load_financial_features(financial_path)
    review_df = load_review_features(review_path)

    financial_distributions, financial_flags = analyze_financial_bias(financial_df)
    review_distribution, review_flags, low_confidence_count = analyze_review_bias(review_df)

    logger.info("----- Financial Slice Distributions -----")
    for slice_name, table in financial_distributions.items():
        log_distribution_table(slice_name, table)
        if financial_flags[slice_name]:
            logger.warning(
                "Flagged underrepresented vulnerable groups (%s): %s",
                slice_name,
                ", ".join(financial_flags[slice_name]),
            )
        else:
            logger.info("No underrepresented vulnerable groups flagged for %s.", slice_name)

    logger.info("----- Review Variance Distribution -----")
    log_distribution_table("Rating Variance", review_distribution)
    if review_flags:
        logger.warning("Flagged underrepresented review groups: %s", ", ".join(review_flags))
    else:
        logger.info("No underrepresented review groups flagged.")

    logger.info("Low-confidence review products (rating_variance == 0.0): %s", low_confidence_count)

    write_markdown_report(
        output_path=report_path,
        financial_distributions=financial_distributions,
        financial_flags=financial_flags,
        review_distribution=review_distribution,
        review_flags=review_flags,
        low_confidence_count=low_confidence_count,
    )
    logger.info("Bias analysis completed successfully.")


if __name__ == "__main__":
    main()
