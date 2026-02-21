"""
Bias analysis utilities for review feature slices.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


REQUIRED_COLUMNS = ["rating_variance"]
RATING_GROUP_ORDER = ["consensus", "Mixed", "Polarized"]


def _default_reviews_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "data-pipeline" / "data" / "features" / "reviews_featured.jsonl"


def load_review_features(input_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load review features by attempting JSONL first, then CSV fallback.
    """
    path = Path(input_path) if input_path else _default_reviews_path()
    if not path.exists():
        raise FileNotFoundError(f"Review features file not found: {path}")

    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_csv(path)


def validate_review_columns(df: pd.DataFrame) -> None:
    """
    Validate required review feature columns.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required review columns: {missing}")


def _group_rating_variance(value: float) -> Optional[str]:
    if pd.isna(value):
        return None
    if value < 0.5:
        return "consensus"
    if value <= 1.0:
        return "Mixed"
    return "Polarized"


def analyze_review_bias(
    df: Optional[pd.DataFrame] = None, input_path: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str], int]:
    """
    Build review variance distribution, underrepresentation flags, and low-confidence count.

    Returns:
        distribution_df: DataFrame(group, count, percentage)
        flags: List[str] where each flagged group is under 5%
        low_confidence_count: number of rows with rating_variance == 0.0
    """
    review_df = df.copy() if df is not None else load_review_features(input_path=input_path)
    validate_review_columns(review_df)

    total_count = len(review_df)
    grouped = review_df["rating_variance"].apply(_group_rating_variance)
    counts = grouped.value_counts(dropna=False).reindex(RATING_GROUP_ORDER, fill_value=0)
    percentages = (counts / total_count * 100).round(2) if total_count > 0 else counts * 0.0

    distribution_df = pd.DataFrame(
        {
            "group": counts.index,
            "count": counts.values.astype(int),
            "percentage": percentages.values,
        }
    )

    flags: List[str] = []
    for _, row in distribution_df.iterrows():
        percentage = float(row["percentage"])
        if percentage < 5.0:
            flags.append(f"{row['group']} ({percentage:.2f}%)")

    low_confidence_count = int((review_df["rating_variance"] == 0.0).sum())
    return distribution_df, flags, low_confidence_count
