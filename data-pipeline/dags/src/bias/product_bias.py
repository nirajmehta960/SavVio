"""Phase 15 representation bias detection for SavVio product datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


BOOL_TRUE = {"1", "true", "yes", "y", "t"}
BOOL_FALSE = {"0", "false", "no", "n", "f"}
MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none", "missing"}

COMMON_DETAIL_KEYS = ["Brand", "Model", "Capacity", "Date First Available"]


@dataclass
class SliceStat:
    label: str
    count: int
    pct: float


@dataclass
class FlagItem:
    column: str
    group: str
    reason: str


def _norm(text: str) -> str:
    return text.strip().lower()


def _pct(count: int, total: int) -> float:
    return round((count / total * 100.0), 2) if total else 0.0


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _safe_json_parse(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if (stripped.startswith("[") and stripped.endswith("]")) or (
            stripped.startswith("{") and stripped.endswith("}")
        ):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None
    return None


def _first_existing(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lookup = {_norm(c): c for c in df.columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def _missing_mask_generic(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    lowered = series.astype(str).str.strip().str.lower()
    return series.isna() | lowered.isin(MISSING_STRINGS)


def _is_list_column(name: str, series: pd.Series) -> bool:
    if _norm(name) in {"features", "description", "categories", "images", "videos"}:
        return True
    parsed = series.dropna().head(200).apply(_safe_json_parse)
    return bool(parsed.apply(lambda x: isinstance(x, list)).mean() > 0.7)


def _is_dict_column(name: str, series: pd.Series) -> bool:
    if _norm(name) == "details":
        return True
    parsed = series.dropna().head(200).apply(_safe_json_parse)
    return bool(parsed.apply(lambda x: isinstance(x, dict)).mean() > 0.7)


def _infer_column_type(name: str, series: pd.Series) -> str:
    n = _norm(name)
    non_missing = series[~_missing_mask_generic(series)]
    if non_missing.empty:
        return "categorical"

    if n in {"product_id", "parent_asin"}:
        uniqueness = non_missing.nunique(dropna=True) / max(len(non_missing), 1)
        return "id" if uniqueness >= 0.95 else "categorical"

    if _is_list_column(name, series):
        return "list"
    if _is_dict_column(name, series):
        return "dict"

    lowered = non_missing.astype(str).str.strip().str.lower()
    if lowered.isin(BOOL_TRUE | BOOL_FALSE).all() or pd.api.types.is_bool_dtype(series):
        return "boolean"

    num = pd.to_numeric(non_missing, errors="coerce")
    if pd.api.types.is_numeric_dtype(series) or num.notna().mean() >= 0.95:
        return "numeric"

    if n in {"title", "product_name"}:
        return "text"
    if non_missing.astype(str).str.len().mean() > 80:
        return "text"
    return "categorical"


def _slice_counts(labels: pd.Series, total: int, sort_index: bool = False) -> List[SliceStat]:
    counts = labels.value_counts(dropna=False, sort=not sort_index)
    if sort_index:
        counts = counts.sort_index()
    return [SliceStat(str(k), int(v), _pct(int(v), total)) for k, v in counts.items()]


def _print_slice_stats(stats: List[SliceStat]) -> None:
    for s in stats:
        print(f"  - {s.label}: {s.count} ({s.pct}%)")


def _numeric_slices(column: str, numeric: pd.Series) -> Tuple[pd.Series, List[str]]:
    name = _norm(column)
    labels = pd.Series(index=numeric.index, dtype="object")
    risk_targets: List[str] = []

    if name == "price":
        labels[numeric.isna()] = "Missing"
        labels[numeric < 20] = "Budget"
        labels[(numeric >= 20) & (numeric <= 100)] = "Mid-range"
        labels[numeric > 100] = "Premium"
        risk_targets = ["Missing", "Budget"]
        return labels, risk_targets

    if name == "average_rating":
        labels[numeric.isna()] = "Missing"
        labels[numeric <= 3.0] = "Low"
        labels[(numeric > 3.0) & (numeric <= 4.0)] = "Medium"
        labels[numeric > 4.0] = "High"
        risk_targets = ["Low"]
        return labels, risk_targets

    if name == "rating_number":
        labels[numeric.isna()] = "Missing"
        labels[numeric < 10] = "Low-confidence"
        labels[(numeric >= 10) & (numeric <= 100)] = "Medium-confidence"
        labels[numeric > 100] = "High-confidence"
        risk_targets = ["Low-confidence"]
        return labels, risk_targets

    if name == "rating_variance":
        labels[numeric.isna()] = "Missing"
        labels[numeric == 0.0] = "Single-review proxy"
        labels[(numeric > 0.0) & (numeric < 0.5)] = "Consensus"
        labels[(numeric >= 0.5) & (numeric <= 1.0)] = "Mixed"
        labels[numeric > 1.0] = "Polarized"
        risk_targets = ["Single-review proxy", "Polarized"]
        return labels, risk_targets

    labels[numeric.isna()] = "Missing"
    valid = numeric.dropna()
    if valid.empty:
        return labels, []
    try:
        q = pd.qcut(valid, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        labels.loc[q.index] = q.astype(str)
    except ValueError:
        labels.loc[valid.index] = "Q2"
    labels[valid >= valid.quantile(0.99)] = "Outlier (Top 1%)"
    return labels, []


def _categorical_profile(column: str, series: pd.Series, total: int) -> Tuple[List[SliceStat], List[FlagItem]]:
    labels = series.astype(str).str.strip().where(~_missing_mask_generic(series), "Missing")
    stats = _slice_counts(labels, total)
    flags: List[FlagItem] = []

    for row in stats:
        if row.label != "Missing" and row.pct < 5.0:
            flags.append(FlagItem(column, row.label, f"Category underrepresented (<5%): {row.pct}%"))
        if row.label != "Missing" and row.pct > 70.0:
            flags.append(FlagItem(column, row.label, f"Category domination risk (>70%): {row.pct}%"))
    return stats, flags


def _list_profile(column: str, series: pd.Series, total: int) -> Tuple[List[SliceStat], List[FlagItem]]:
    parsed = series.apply(_safe_json_parse)
    lengths = pd.Series(index=series.index, dtype="float")

    for idx, value in parsed.items():
        if isinstance(value, list):
            lengths.at[idx] = len(value)
        elif _missing_mask_generic(series.loc[[idx]]).iloc[0]:
            lengths.at[idx] = 0
        else:
            lengths.at[idx] = math.nan

    labels = pd.Series(index=series.index, dtype="object")
    labels[lengths.isna()] = "Invalid"
    labels[lengths == 0] = "0"
    labels[(lengths >= 1) & (lengths <= 2)] = "1-2"
    labels[(lengths >= 3) & (lengths <= 5)] = "3-5"
    labels[lengths >= 6] = "6+"

    stats = _slice_counts(labels, total)
    flags: List[FlagItem] = []
    empty_share = _pct(int((labels == "0").sum()), total)

    n = _norm(column)
    if n in {"features", "categories"} and empty_share > 20.0:
        flags.append(FlagItem(column, "0", f"Empty list share >20% ({empty_share}%)."))
    if n == "images" and empty_share > 10.0:
        flags.append(FlagItem(column, "0", f"Images empty share >10% ({empty_share}%)."))
    # videos empty is reported but intentionally not flagged unless required.
    return stats, flags


def _details_profile(column: str, series: pd.Series, total: int) -> Tuple[List[SliceStat], List[FlagItem]]:
    parsed = series.apply(_safe_json_parse)
    has_dict = parsed.apply(lambda x: isinstance(x, dict))
    empty_or_missing = (~has_dict) | parsed.apply(lambda x: isinstance(x, dict) and len(x) == 0)

    labels = pd.Series(index=series.index, dtype="object")
    labels[empty_or_missing] = "Missing/Empty"
    labels[~empty_or_missing] = "Present"
    stats = _slice_counts(labels, total)

    flags: List[FlagItem] = []
    empty_share = _pct(int(empty_or_missing.sum()), total)
    if empty_share > 20.0:
        flags.append(FlagItem(column, "Missing/Empty", f"Details missing/empty >20% ({empty_share}%)."))

    # Key-level completeness + brand distribution
    key_rows: Dict[str, int] = {k: 0 for k in COMMON_DETAIL_KEYS}
    brand_values: List[str] = []
    for value in parsed:
        if isinstance(value, dict):
            for key in COMMON_DETAIL_KEYS:
                if key in value and str(value[key]).strip() != "":
                    key_rows[key] += 1
            if "Brand" in value and str(value["Brand"]).strip() != "":
                brand_values.append(str(value["Brand"]).strip())

    print("  - details key missingness:")
    for key in COMMON_DETAIL_KEYS:
        miss_pct = _pct(total - key_rows[key], total)
        print(f"    - {key}: {miss_pct}% missing")

    if brand_values:
        brand_counts = pd.Series(brand_values).value_counts(normalize=True) * 100.0
        print("  - Brand distribution (top 10):")
        for brand, pct in brand_counts.head(10).items():
            print(f"    - {brand}: {round(float(pct), 2)}%")
        rare_brands = brand_counts[brand_counts < 5.0]
        if not rare_brands.empty:
            sample_rare = ", ".join([f"{b} ({round(float(p), 2)}%)" for b, p in rare_brands.head(5).items()])
            flags.append(FlagItem(column, "Brand", f"Rare brands (<5%) present: {sample_rare}"))

    return stats, flags


def _text_profile(column: str, series: pd.Series, total: int) -> Tuple[List[SliceStat], List[FlagItem]]:
    txt = series.astype(str).where(~_missing_mask_generic(series), "")
    lengths = txt.str.len()
    labels = pd.Series(index=series.index, dtype="object")
    labels[lengths == 0] = "Missing/Empty"
    labels[(lengths > 0) & (lengths < 40)] = "Short"
    labels[(lengths >= 40) & (lengths <= 120)] = "Medium"
    labels[lengths > 120] = "Long"
    stats = _slice_counts(labels, total)

    flags: List[FlagItem] = []
    missing_share = _pct(int((labels == "Missing/Empty").sum()), total)
    if _norm(column) in {"title", "product_name"} and missing_share > 5.0:
        flags.append(FlagItem(column, "Missing/Empty", f"Title missing/empty >5% ({missing_share}%)."))
    return stats, flags


def _profile_column(column: str, series: pd.Series, total: int) -> List[FlagItem]:
    inferred = _infer_column_type(column, series)
    missing_pct = _pct(int(_missing_mask_generic(series).sum()), total)
    flags: List[FlagItem] = []

    print(f"\nColumn: {column}")
    print(f"- inferred type: {inferred}")
    print(f"- missing rate (%): {missing_pct}")
    print("- representation slices:")

    if missing_pct > 20.0:
        flags.append(FlagItem(column, "Missingness", f"Column missingness >20% ({missing_pct}%)."))
    if _norm(column) == "price" and missing_pct > 5.0:
        flags.append(FlagItem(column, "Missingness", f"Price missingness is high ({missing_pct}%)."))

    if inferred == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        labels, risk_targets = _numeric_slices(column, numeric)
        stats = _slice_counts(labels, total)
        _print_slice_stats(stats)
        for target in risk_targets:
            share = _pct(int((labels == target).sum()), total)
            if share < 5.0:
                flags.append(FlagItem(column, target, f"Uncertainty/negative slice <5% or missing ({share}%)."))
    elif inferred == "categorical":
        stats, f = _categorical_profile(column, series, total)
        _print_slice_stats(stats)
        flags.extend(f)
    elif inferred == "boolean":
        lowered = series.astype(str).str.strip().str.lower()
        labels = pd.Series(index=series.index, dtype="object")
        labels[lowered.isin(BOOL_TRUE)] = "True"
        labels[lowered.isin(BOOL_FALSE)] = "False"
        labels[_missing_mask_generic(series)] = "Missing"
        labels[labels.isna()] = "Invalid"
        stats = _slice_counts(labels, total)
        _print_slice_stats(stats)
    elif inferred == "text":
        stats, f = _text_profile(column, series, total)
        _print_slice_stats(stats)
        flags.extend(f)
    elif inferred == "list":
        stats, f = _list_profile(column, series, total)
        _print_slice_stats(stats)
        flags.extend(f)
    elif inferred == "dict":
        stats, f = _details_profile(column, series, total)
        _print_slice_stats(stats)
        flags.extend(f)
    else:  # id
        non_missing = series[~_missing_mask_generic(series)]
        uniq_rate = _pct(int(non_missing.nunique(dropna=True)), max(len(non_missing), 1))
        _print_slice_stats([SliceStat("Unique non-missing rate", int(non_missing.nunique(dropna=True)), uniq_rate)])
        if uniq_rate < 95.0:
            flags.append(FlagItem(column, "Uniqueness", f"Identifier uniqueness below 95% ({uniq_rate}%)."))

    if flags:
        print("- flagged risks:")
        for f in flags:
            print(f"  - {f.group}: {f.reason}")
    return flags


def _load_products(repo_root: str, preprocessed_path: Optional[str], featured_path: Optional[str]) -> pd.DataFrame:
    pre_path = preprocessed_path or os.path.join(
        repo_root, "data-pipeline", "dags", "data", "processed", "product_preprocessed.jsonl"
    )
    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"Preprocessed file not found: {pre_path}")
    df = pd.read_json(pre_path, lines=True)

    feat_path = featured_path or os.path.join(
        repo_root, "data-pipeline", "dags", "data", "features", "product_featured.jsonl"
    )
    if os.path.exists(feat_path):
        feat = pd.read_json(feat_path, lines=True)
        left_key = _first_existing(df, ["product_id", "parent_asin"])
        right_key = _first_existing(feat, ["product_id", "parent_asin"])
        if left_key and right_key:
            dedup_feat = feat.drop(columns=[c for c in feat.columns if c in df.columns and c != right_key], errors="ignore")
            df = df.merge(dedup_feat, left_on=left_key, right_on=right_key, how="left")
            if left_key != right_key and right_key in df.columns:
                df = df.drop(columns=[right_key])
    return df


def run_phase15_product_bias(
    root_dir: Optional[str] = None,
    preprocessed_path: Optional[str] = None,
    featured_path: Optional[str] = None,
) -> int:
    repo_root = root_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    df = _load_products(repo_root, preprocessed_path, featured_path)
    total_rows = len(df)

    _print_header("1) Dataset Summary")
    print(f"total records: {total_rows}, total columns: {len(df.columns)}")
    overall_missing = round(df.isna().sum().sum() / max(df.size, 1) * 100.0, 2)
    print(f"overall missingness rate: {overall_missing}%")
    missing_by_col = {c: _pct(int(_missing_mask_generic(df[c]).sum()), total_rows) for c in df.columns}
    top5 = sorted(missing_by_col.items(), key=lambda x: x[1], reverse=True)[:5]
    print("top 5 columns by missingness:")
    for col, pct in top5:
        print(f"- {col}: {pct}%")

    _print_header("2) Column-by-column analysis")
    all_flags: List[FlagItem] = []
    for column in df.columns:
        all_flags.extend(_profile_column(column, df[column], total_rows))

    _print_header("3) Final Summary")
    print("Flagged Representation Risks:")
    if all_flags:
        for f in all_flags:
            print(f"- {f.column} | {f.group}: {f.reason}")
    else:
        print("- None")

    print("\nTraining-time-only mitigation recommendations:")
    if all_flags:
        print("- Stratified sampling by (price band x rating confidence band).")
        print("- Controlled oversampling of underrepresented slices (<5%).")
        print("- Down-weight low-confidence products (rating_variance == 0 or rating_number < 10).")
        print("- Stress-test evaluation on budget, low-rated, and polarized-variance products.")
    else:
        print("- No flagged risks; continue standard monitoring and periodic re-checks.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 15 product representation bias detector (terminal only).")
    parser.add_argument("--root-dir", default=None, help="Repository root override.")
    parser.add_argument("--preprocessed-path", default=None, help="Path override for product_preprocessed.jsonl.")
    parser.add_argument("--featured-path", default=None, help="Path override for product_featured.jsonl.")
    args = parser.parse_args()
    return run_phase15_product_bias(args.root_dir, args.preprocessed_path, args.featured_path)


if __name__ == "__main__":
    raise SystemExit(main())
