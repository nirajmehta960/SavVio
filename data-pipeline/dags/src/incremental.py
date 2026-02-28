"""
Incremental merge utilities for the SavVio data pipeline.

Provides reusable functions for merging new data with existing data
at every stage (preprocessing, features). When a record with the same
key exists in both old and new data, the new version wins. New records
are appended, and old records not present in the new data are kept.
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File checksum
# ---------------------------------------------------------------------------

def file_checksum(path: str) -> str:
    """
    Compute MD5 checksum of a file.

    Uses MD5 for consistency with GCS blob checksums.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded MD5 digest string.
    """
    return hashlib.md5(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------------------
# CSV merge
# ---------------------------------------------------------------------------

def merge_csv(
    new_path: str,
    existing_path: str,
    key_cols: List[str],
) -> Dict[str, int]:
    """
    Merge a newly-produced CSV with an existing CSV file.

    Logic:
        - Records whose key exists in both → replaced with the new version.
        - Records only in the new file → appended.
        - Records only in the existing file → kept unchanged.

    The merged result is written back to *existing_path*.
    If *existing_path* does not exist, the new file is simply
    copied (first-run behaviour).

    Args:
        new_path:      Path to the newly-produced CSV.
        existing_path: Path to the existing (accumulated) CSV.
        key_cols:      Column(s) that uniquely identify a record.

    Returns:
        Dict with stats: {updated, appended, unchanged, total}.
    """
    new_df = pd.read_csv(new_path)
    logger.info("New CSV records: %d", len(new_df))

    if not os.path.exists(existing_path) or os.path.getsize(existing_path) == 0:
        # First run — nothing to merge with.
        new_df.to_csv(existing_path, index=False)
        stats = {
            "updated": 0,
            "appended": len(new_df),
            "unchanged": 0,
            "total": len(new_df),
        }
        logger.info("First run — wrote %d records: %s", len(new_df), existing_path)
        return stats

    existing_df = pd.read_csv(existing_path)
    logger.info("Existing CSV records: %d", len(existing_df))

    merged_df, stats = _merge_dataframes(new_df, existing_df, key_cols)

    merged_df.to_csv(existing_path, index=False)
    logger.info("Merged CSV → %s | %s", existing_path, stats)
    return stats


# ---------------------------------------------------------------------------
# JSONL merge
# ---------------------------------------------------------------------------

def merge_jsonl(
    new_path: str,
    existing_path: str,
    key_cols: List[str],
) -> Dict[str, int]:
    """
    Merge a newly-produced JSONL with an existing JSONL file.

    Same logic as merge_csv but for JSONL files (one JSON object per line).

    Args:
        new_path:      Path to the newly-produced JSONL.
        existing_path: Path to the existing (accumulated) JSONL.
        key_cols:      Column(s) that uniquely identify a record.

    Returns:
        Dict with stats: {updated, appended, unchanged, total}.
    """
    new_df = pd.read_json(new_path, lines=True)
    logger.info("New JSONL records: %d", len(new_df))

    if not os.path.exists(existing_path) or os.path.getsize(existing_path) == 0:
        # First run — nothing to merge with.
        new_df.to_json(existing_path, orient="records", lines=True)
        stats = {
            "updated": 0,
            "appended": len(new_df),
            "unchanged": 0,
            "total": len(new_df),
        }
        logger.info("First run — wrote %d records: %s", len(new_df), existing_path)
        return stats

    existing_df = pd.read_json(existing_path, lines=True)
    logger.info("Existing JSONL records: %d", len(existing_df))

    merged_df, stats = _merge_dataframes(new_df, existing_df, key_cols)

    merged_df.to_json(existing_path, orient="records", lines=True)
    logger.info("Merged JSONL → %s | %s", existing_path, stats)
    return stats


# ---------------------------------------------------------------------------
# Core merge logic (shared by CSV and JSONL)
# ---------------------------------------------------------------------------

def _merge_dataframes(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    key_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Merge two DataFrames on key columns.

    - Rows present in both (by key) → new version replaces old.
    - Rows only in new_df          → appended.
    - Rows only in existing_df     → kept unchanged.

    Returns:
        (merged_df, stats_dict)
    """
    # Build a merge key for comparison.
    new_keys = new_df[key_cols].apply(tuple, axis=1)
    existing_keys = existing_df[key_cols].apply(tuple, axis=1)

    new_key_set = set(new_keys)
    existing_key_set = set(existing_keys)

    # Records in both → count as "updated", new version wins.
    updated_keys = new_key_set & existing_key_set
    # Records only in new → "appended".
    appended_keys = new_key_set - existing_key_set
    # Records only in existing → "unchanged".
    unchanged_keys = existing_key_set - new_key_set

    # Keep unchanged rows from existing.
    unchanged_mask = existing_keys.isin(unchanged_keys)
    unchanged_df = existing_df[unchanged_mask]

    # All new rows go in (they replace updated + add appended).
    merged_df = pd.concat([unchanged_df, new_df], ignore_index=True)

    stats = {
        "updated": len(updated_keys),
        "appended": len(appended_keys),
        "unchanged": len(unchanged_keys),
        "total": len(merged_df),
    }
    return merged_df, stats
