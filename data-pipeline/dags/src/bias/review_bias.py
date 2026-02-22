"""Phase 15 representation bias detection for SavVio review datasets."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd


BOOL_TRUE = {"1", "true", "yes", "y", "t"}
BOOL_FALSE = {"0", "false", "no", "n", "f"}
MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none", "missing"}


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


def _norm(name: str) -> str:
    return name.strip().lower()


def _pct(count: int, total: int) -> float:
    return round((count / total * 100.0), 2) if total else 0.0


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    lowered = series.astype(str).str.strip().str.lower()
    return series.isna() | lowered.isin(MISSING_STRINGS)


def _counts(series: pd.Series, total: int, sort_index: bool = False) -> List[SliceStat]:
    counts = series.value_counts(dropna=False, sort=not sort_index)
    if sort_index:
        counts = counts.sort_index()
    return [SliceStat(str(k), int(v), _pct(int(v), total)) for k, v in counts.items()]


def _print_slice_stats(stats: List[SliceStat], indent: str = "  ") -> None:
    for row in stats:
        print(f"{indent}- {row.label}: {row.count} ({row.pct}%)")


def _print_header(title: str) -> None:
    print(title)
    print("-" * len(title))


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def _infer_type(col: str, series: pd.Series) -> str:
    name = _norm(col)
    if name in {"user_id", "asin", "product_id"}:
        return "id"
    if name == "verified_purchase":
        return "boolean"
    if name in {"review_title", "review_text"}:
        return "text"

    non_missing = series[~_missing_mask(series)]
    if non_missing.empty:
        return "categorical"

    lowered = non_missing.astype(str).str.strip().str.lower()
    if lowered.isin(BOOL_TRUE | BOOL_FALSE).all() or pd.api.types.is_bool_dtype(series):
        return "boolean"

    numeric = pd.to_numeric(non_missing, errors="coerce")
    if pd.api.types.is_numeric_dtype(series) or numeric.notna().mean() >= 0.95:
        return "numeric"

    uniq_ratio = non_missing.nunique(dropna=True) / max(len(non_missing), 1)
    if uniq_ratio > 0.98:
        return "id"
    if non_missing.astype(str).str.len().mean() > 60:
        return "text"
    return "categorical"


def _profile_user_id(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    missing_rate = _pct(int(_missing_mask(series).sum()), total)
    non_missing = series[~_missing_mask(series)]
    unique_count = int(non_missing.nunique(dropna=True))
    unique_rate = _pct(unique_count, max(len(non_missing), 1))

    print(f"column name: {col}")
    print("inferred type: id")
    print(f"missing rate (%): {missing_rate}")
    print("representation slices (bands):")
    _print_slice_stats([SliceStat("uniqueness rate (non-missing)", unique_count, unique_rate)])

    flags: List[FlagItem] = []
    if missing_rate > 0:
        flags.append(FlagItem(col, "missing", f"Missing > 0% ({missing_rate}%)."))
    if unique_rate < 95.0:
        flags.append(FlagItem(col, "uniqueness", f"Uniqueness < 95% ({unique_rate}%)."))

    if flags:
        print("flagged representation risks:")
        for flag in flags:
            print(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_product_identifier(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    missing_rate = _pct(int(_missing_mask(series).sum()), total)
    valid = series[~_missing_mask(series)].astype(str)
    unique_products = int(valid.nunique(dropna=True))
    review_counts = valid.value_counts()

    band_labels = pd.Series(index=review_counts.index, dtype="object")
    band_labels[review_counts == 1] = "1 review"
    band_labels[(review_counts >= 2) & (review_counts <= 5)] = "2-5 reviews"
    band_labels[(review_counts >= 6) & (review_counts <= 20)] = "6-20 reviews"
    band_labels[review_counts >= 21] = "21+ reviews"
    band_stats = _counts(band_labels, max(unique_products, 1))

    print(f"column name: {col}")
    print("inferred type: id")
    print(f"missing rate (%): {missing_rate}")
    print("representation slices (bands):")
    print(f"  - unique products: {unique_products}")
    _print_slice_stats(band_stats)

    flags: List[FlagItem] = []
    single_review_share = _pct(int((band_labels == "1 review").sum()), max(unique_products, 1))
    if single_review_share > 60.0:
        flags.append(
            FlagItem(col, "1 review", f"Single-review products dominate >60% ({single_review_share}%).")
        )

    if flags:
        print("flagged representation risks:")
        for flag in flags:
            print(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_rating(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    num = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(index=num.index, dtype="object")
    labels[num.isna()] = "Missing"
    labels[(num >= 1) & (num <= 2)] = "Negative"
    labels[num == 3] = "Neutral"
    labels[(num >= 4) & (num <= 5)] = "Positive"
    labels[labels.isna()] = "Invalid"
    stats = _counts(labels, total)

    print(f"column name: {col}")
    print("inferred type: numeric")
    print(f"missing rate (%): {_pct(int((labels == 'Missing').sum()), total)}")
    print("representation slices (bands):")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    neg_share = _pct(int((labels == "Negative").sum()), total)
    neu_share = _pct(int((labels == "Neutral").sum()), total)
    pos_count = int((labels == "Positive").sum())
    if neg_share < 5.0:
        flags.append(FlagItem(col, "Negative", f"Negative <5% or missing ({neg_share}%)."))
    if neu_share < 5.0:
        flags.append(FlagItem(col, "Neutral", f"Neutral <5% or missing ({neu_share}%)."))
    if pos_count == 0 or int((labels == "Negative").sum()) == 0 or int((labels == "Neutral").sum()) == 0:
        flags.append(FlagItem(col, "bucket coverage", "At least one rating bucket has 0%."))

    if flags:
        print("flagged representation risks:")
        for flag in flags:
            print(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_helpful_vote(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    num = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(index=num.index, dtype="object")
    labels[num.isna()] = "Missing"
    labels[num == 0] = "None"
    labels[(num >= 1) & (num <= 2)] = "Low"
    labels[(num >= 3) & (num <= 10)] = "Medium"
    labels[num > 10] = "High"
    labels[labels.isna()] = "Invalid"
    stats = _counts(labels, total)

    print(f"column name: {col}")
    print("inferred type: numeric")
    print(f"missing rate (%): {_pct(int((labels == 'Missing').sum()), total)}")
    print("representation slices (bands):")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    high_share = _pct(int((labels == "High").sum()), total)
    none_share = _pct(int((labels == "None").sum()), total)
    if high_share < 5.0:
        flags.append(FlagItem(col, "High", f"High helpful votes <5% ({high_share}%)."))
    if none_share > 90.0:
        flags.append(
            FlagItem(
                col,
                "None",
                f"Low usefulness signal coverage: helpful_vote==0 exceeds 90% ({none_share}%).",
            )
        )

    if flags:
        print("flagged representation risks:")
        for flag in flags:
            print(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_verified_purchase(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    lowered = series.astype(str).str.strip().str.lower()
    labels = pd.Series(index=series.index, dtype="object")
    labels[lowered.isin(BOOL_TRUE)] = "True"
    labels[lowered.isin(BOOL_FALSE)] = "False"
    labels[_missing_mask(series)] = "Missing"
    labels[labels.isna()] = "Invalid"
    stats = _counts(labels, total)

    print(f"column name: {col}")
    print("inferred type: boolean")
    print(f"missing rate (%): {_pct(int((labels == 'Missing').sum()), total)}")
    print("representation slices (bands):")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    valid = labels[labels.isin(["True", "False"])]
    if not valid.empty:
        dist = valid.value_counts(normalize=True)
        minority = float(dist.min() * 100.0)
        minority_label = str(dist.idxmin())
        if minority < 5.0:
            flags.append(
                FlagItem(col, minority_label, f"Minority class <5% ({round(minority, 2)}%).")
            )

    if flags:
        print("flagged representation risks:")
        for flag in flags:
            print(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_review_title(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    txt = series.astype(str).where(~_missing_mask(series), "")
    length = txt.str.len()
    labels = pd.Series(index=series.index, dtype="object")
    labels[length == 0] = "Empty"
    labels[(length > 0) & (length < 15)] = "Short"
    labels[(length >= 15) & (length <= 60)] = "Medium"
    labels[length > 60] = "Long"
    stats = _counts(labels, total)

    missing_share = _pct(int((labels == "Empty").sum()), total)
    print(f"column name: {col}")
    print("inferred type: text")
    print(f"missing rate (%): {missing_share}")
    print("representation slices (bands):")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    if missing_share > 20.0:
        flags.append(FlagItem(col, "Empty", f"Empty/missing >20% ({missing_share}%)."))
    if flags:
        print("flagged representation risks:")
        for flag in flags:
            print(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_review_text(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    txt = series.astype(str).where(~_missing_mask(series), "")
    length = txt.str.len()
    labels = pd.Series(index=series.index, dtype="object")
    labels[length == 0] = "Empty"
    labels[(length > 0) & (length < 50)] = "Short"
    labels[(length >= 50) & (length <= 200)] = "Medium"
    labels[length > 200] = "Long"
    stats = _counts(labels, total)

    empty_share = _pct(int((labels == "Empty").sum()), total)
    short_share = _pct(int((labels == "Short").sum()), total)

    print(f"column name: {col}")
    print("inferred type: text")
    print(f"missing rate (%): {empty_share}")
    print("representation slices (bands):")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    if empty_share > 10.0:
        flags.append(FlagItem(col, "Empty", f"Empty/missing >10% ({empty_share}%)."))
    if short_share < 5.0:
        flags.append(FlagItem(col, "Short", f"Short reviews <5% ({short_share}%)."))
    if flags:
        print("flagged representation risks:")
        for flag in flags:
            print(f"- {flag.group}: {flag.reason}")
    return flags


def _profile_generic(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    inferred = _infer_type(col, series)
    missing_rate = _pct(int(_missing_mask(series).sum()), total)
    flags: List[FlagItem] = []

    print(f"column name: {col}")
    print(f"inferred type: {inferred}")
    print(f"missing rate (%): {missing_rate}")
    print("representation slices (bands):")

    if inferred == "numeric":
        num = pd.to_numeric(series, errors="coerce")
        labels = pd.Series(index=num.index, dtype="object")
        labels[num.isna()] = "Missing"
        valid = num.dropna()
        if not valid.empty:
            try:
                q = pd.qcut(valid, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
                labels.loc[q.index] = q.astype(str)
            except ValueError:
                labels.loc[valid.index] = "Q2"
            labels[valid >= valid.quantile(0.99)] = "Outlier (Top 1%)"
        _print_slice_stats(_counts(labels, total))
    elif inferred == "boolean":
        lowered = series.astype(str).str.strip().str.lower()
        labels = pd.Series(index=series.index, dtype="object")
        labels[lowered.isin(BOOL_TRUE)] = "True"
        labels[lowered.isin(BOOL_FALSE)] = "False"
        labels[_missing_mask(series)] = "Missing"
        labels[labels.isna()] = "Invalid"
        _print_slice_stats(_counts(labels, total))
    elif inferred == "text":
        txt = series.astype(str).where(~_missing_mask(series), "")
        labels = pd.Series(index=series.index, dtype="object")
        length = txt.str.len()
        labels[length == 0] = "Empty"
        labels[(length > 0) & (length < 50)] = "Short"
        labels[(length >= 50) & (length <= 200)] = "Medium"
        labels[length > 200] = "Long"
        _print_slice_stats(_counts(labels, total))
    else:
        labels = series.astype(str).str.strip().where(~_missing_mask(series), "Missing")
        stats = _counts(labels, total)
        _print_slice_stats(stats)
        for row in stats:
            if row.label != "Missing" and row.pct < 5.0:
                flags.append(FlagItem(col, row.label, f"Underrepresented slice <5% ({row.pct}%)."))

    if flags:
        print("flagged representation risks:")
        for flag in flags:
            print(f"- {flag.group}: {flag.reason}")
    return flags


def _load_review_data(root_dir: str, preprocessed_path: Optional[str], featured_path: Optional[str]) -> pd.DataFrame:
    pre_path = preprocessed_path or os.path.join(
        root_dir, "data-pipeline", "dags", "data", "processed", "review_preprocessed.jsonl"
    )
    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"Preprocessed file not found: {pre_path}")
    df = pd.read_json(pre_path, lines=True)

    feat_path = featured_path or os.path.join(
        root_dir, "data-pipeline", "dags", "data", "features", "review_featured.jsonl"
    )
    if os.path.exists(feat_path):
        feat = pd.read_json(feat_path, lines=True)
        missing_cols = [c for c in feat.columns if c not in df.columns]
        if missing_cols:
            # Align by row index when featured is pass-through plus additional columns.
            df = df.join(feat[missing_cols])
    return df


def run_phase15_review_bias(
    root_dir: Optional[str] = None,
    preprocessed_path: Optional[str] = None,
    featured_path: Optional[str] = None,
) -> int:
    repo_root = root_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    df = _load_review_data(repo_root, preprocessed_path, featured_path)

    total_rows = len(df)
    total_cols = len(df.columns)

    _print_header("1) Dataset Summary")
    print(f"total number of reviews: {total_rows}")
    print(f"number of columns: {total_cols}")
    overall_missing = round(df.isna().sum().sum() / max(df.size, 1) * 100.0, 2)
    print(f"overall missingness rate (%): {overall_missing}")
    col_missing = {c: _pct(int(_missing_mask(df[c]).sum()), total_rows) for c in df.columns}
    top5 = sorted(col_missing.items(), key=lambda x: x[1], reverse=True)[:5]
    print("top 5 columns by missingness:")
    for col, pct in top5:
        print(f"- {col}: {pct}%")

    _print_header("2) Column-by-column analysis")
    all_flags: List[FlagItem] = []
    for col in df.columns:
        name = _norm(col)
        print("")
        if name == "user_id":
            all_flags.extend(_profile_user_id(col, df[col], total_rows))
        elif name in {"asin", "product_id"}:
            all_flags.extend(_profile_product_identifier(col, df[col], total_rows))
        elif name == "rating":
            all_flags.extend(_profile_rating(col, df[col], total_rows))
        elif name == "helpful_vote":
            all_flags.extend(_profile_helpful_vote(col, df[col], total_rows))
        elif name == "verified_purchase":
            all_flags.extend(_profile_verified_purchase(col, df[col], total_rows))
        elif name == "review_title":
            all_flags.extend(_profile_review_title(col, df[col], total_rows))
        elif name == "review_text":
            all_flags.extend(_profile_review_text(col, df[col], total_rows))
        else:
            all_flags.extend(_profile_generic(col, df[col], total_rows))

    _print_header("3) Final Summary")
    print("all flagged representation risks:")
    if all_flags:
        for flag in all_flags:
            print(f"- {flag.column} | {flag.group}: {flag.reason}")
    else:
        print("- None")

    print("\nrecommended training-time-only mitigations:")
    if all_flags:
        print("- Stratified sampling by rating bucket (negative/neutral/positive), verified_purchase, and per-product review-count band.")
        print("- Controlled oversampling of underrepresented slices (<5%).")
        print("- Evaluation stress tests on negative reviews, neutral reviews, and cold-start products (single-review band).")
        print("- Optional weighting: down-weight helpful_vote==0 if dominance causes overfitting to low-signal reviews.")
    else:
        print("- No flagged representation risks; continue periodic bias monitoring.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 15 representation bias detection for review datasets (terminal only).")
    parser.add_argument("--root-dir", default=None, help="Repository root override.")
    parser.add_argument("--preprocessed-path", default=None, help="Path override for review_preprocessed.jsonl.")
    parser.add_argument("--featured-path", default=None, help="Path override for review_featured.jsonl.")
    args = parser.parse_args()
    return run_phase15_review_bias(args.root_dir, args.preprocessed_path, args.featured_path)


if __name__ == "__main__":
    raise SystemExit(main())
