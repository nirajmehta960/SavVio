"""
Bias analysis utilities for financial feature slices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


REQUIRED_COLUMNS = [
    "discretionary_income",
    "debt_to_income_ratio",
    "saving_to_income_ratio",
    "monthly_expense_burden_ratio",
    "financial_runway",
]

SLICE_ORDERS: Dict[str, List[str]] = {
    "Discretionary Income": ["Negative", "Tight", "Comfortable"],
    "Debt-to-Income Ratio": ["safe", "warning", "Risky"],
    "Saving-to-Income Ratio": ["Fragile", "Moderate", "strong"],
    "Monthly Expense Burden": ["comfortable", "tight", "Overstretched"],
    "Financial Runway": ["Critical", "Fragile", "Healthy"],
}

VULNERABLE_GROUPS: Dict[str, List[str]] = {
    "Discretionary Income": ["Negative"],
    "Debt-to-Income Ratio": ["Risky"],
    "Saving-to-Income Ratio": ["Fragile"],
    "Monthly Expense Burden": ["Overstretched"],
    "Financial Runway": ["Critical", "Fragile"],
}


def _default_financial_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "data-pipeline" / "dags" / "data" / "features" / "financial_featured.csv"


def load_financial_features(input_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the financial feature dataset from CSV.
    """
    path = Path(input_path) if input_path else _default_financial_path()
    if not path.exists():
        raise FileNotFoundError(f"Financial features file not found: {path}")
    return pd.read_csv(path)


def validate_financial_columns(df: pd.DataFrame) -> None:
    """
    Validate that required feature columns exist.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required financial columns: {missing}")


def _group_discretionary_income(value: float) -> Optional[str]:
    if pd.isna(value):
        return None
    if value < 0:
        return "Negative"
    if value <= 1000:
        return "Tight"
    return "Comfortable"


def _group_dti(value: float) -> Optional[str]:
    if pd.isna(value):
        return None
    if value < 0.2:
        return "safe"
    if value <= 0.4:
        return "warning"
    return "Risky"


def _group_saving_to_income(value: float) -> Optional[str]:
    if pd.isna(value):
        return None
    if value < 0.25:
        return "Fragile"
    if value <= 1.0:
        return "Moderate"
    return "strong"


def _group_expense_burden(value: float) -> Optional[str]:
    if pd.isna(value):
        return None
    if value < 0.5:
        return "comfortable"
    if value <= 0.8:
        return "tight"
    return "Overstretched"


def _group_financial_runway(value: float) -> Optional[str]:
    if pd.isna(value):
        return None
    if value < 1:
        return "Critical"
    if value <= 3:
        return "Fragile"
    return "Healthy"


def _distribution_df(groups: pd.Series, ordered_groups: List[str], total_count: int) -> pd.DataFrame:
    counts = groups.value_counts(dropna=False)
    ordered_counts = counts.reindex(ordered_groups, fill_value=0)
    percentages = (ordered_counts / total_count * 100).round(2) if total_count > 0 else ordered_counts * 0.0
    return pd.DataFrame(
        {
            "group": ordered_counts.index,
            "count": ordered_counts.values.astype(int),
            "percentage": percentages.values,
        }
    )


def _extract_flags(distribution_df: pd.DataFrame, vulnerable_groups: List[str], threshold: float = 10.0) -> List[str]:
    flags: List[str] = []
    for group in vulnerable_groups:
        row = distribution_df.loc[distribution_df["group"] == group]
        if row.empty:
            continue
        percentage = float(row.iloc[0]["percentage"])
        if percentage < threshold:
            flags.append(f"{group} ({percentage:.2f}%)")
    return flags


def analyze_financial_bias(
    df: Optional[pd.DataFrame] = None, input_path: Optional[str] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """
    Build financial bias slice distributions and vulnerable-group flags.

    Returns:
        distributions: Dict[slice_name -> DataFrame(group, count, percentage)]
        flags: Dict[slice_name -> List[str]]
    """
    financial_df = df.copy() if df is not None else load_financial_features(input_path=input_path)
    validate_financial_columns(financial_df)

    total_count = len(financial_df)
    grouped_columns: Dict[str, pd.Series] = {
        "Discretionary Income": financial_df["discretionary_income"].apply(_group_discretionary_income),
        "Debt-to-Income Ratio": financial_df["debt_to_income_ratio"].apply(_group_dti),
        "Saving-to-Income Ratio": financial_df["saving_to_income_ratio"].apply(_group_saving_to_income),
        "Monthly Expense Burden": financial_df["monthly_expense_burden_ratio"].apply(_group_expense_burden),
        "Financial Runway": financial_df["financial_runway"].apply(_group_financial_runway),
    }

    distributions: Dict[str, pd.DataFrame] = {}
    flags: Dict[str, List[str]] = {}

    for slice_name in SLICE_ORDERS:
        dist_df = _distribution_df(
            groups=grouped_columns[slice_name],
            ordered_groups=SLICE_ORDERS[slice_name],
            total_count=total_count,
        )
        distributions[slice_name] = dist_df
        flags[slice_name] = _extract_flags(
            distribution_df=dist_df,
            vulnerable_groups=VULNERABLE_GROUPS[slice_name],
            threshold=10.0,
        )

    return distributions, flags
