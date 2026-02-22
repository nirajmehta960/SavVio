"""Phase 15 representation bias detection for SavVio financial dataset."""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd


BOOL_TRUE = {"1", "true", "yes", "y", "t"}
BOOL_FALSE = {"0", "false", "no", "n", "f"}


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


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    return series.isna() | series.astype(str).str.strip().eq("")


def _pct(count: int, total: int) -> float:
    return round((count / total * 100.0), 2) if total else 0.0


def _lower(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _norm(name: str) -> str:
    return name.strip().lower()


def _first_existing(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lookup = {_norm(c): c for c in df.columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def _infer_type(col: str, series: pd.Series) -> str:
    n = _norm(col)
    non_missing = series[~_missing_mask(series)]
    if non_missing.empty:
        return "datetime" if "date" in n or "time" in n else "categorical"
    if n == "user_id" or (n.endswith("_id") and "credit" not in n):
        return "id"
    if n == "record_date":
        return "datetime"
    lowered = _lower(non_missing)
    if pd.api.types.is_bool_dtype(series) or lowered.isin(BOOL_TRUE | BOOL_FALSE).all():
        return "boolean"
    if (("date" in n or "time" in n) and pd.to_datetime(non_missing, errors="coerce").notna().mean() > 0.95):
        return "datetime"
    num = _to_num(non_missing)
    if pd.api.types.is_numeric_dtype(series) or num.notna().mean() >= 0.95:
        return "numeric"
    if n == "job_title":
        return "text"
    uniq_ratio = non_missing.nunique(dropna=True) / max(len(non_missing), 1)
    return "id" if uniq_ratio > 0.98 else "categorical"


def _print_header(text: str) -> None:
    print(f"\n{text}")
    print("-" * len(text))


def _print_slices(stats: List[SliceStat]) -> None:
    for s in stats:
        print(f"  - {s.label}: {s.count} ({s.pct}%)")


def _counts(series: pd.Series, total: int, sort_index: bool = False) -> List[SliceStat]:
    counts = series.value_counts(dropna=False, sort=not sort_index)
    if sort_index:
        counts = counts.sort_index()
    return [SliceStat(str(k), int(v), _pct(int(v), total)) for k, v in counts.items()]


def _bands_for_numeric(col: str, values: pd.Series) -> Tuple[pd.Series, Optional[str]]:
    n = _norm(col)
    if n == "age":
        labels = pd.Series(index=values.index, dtype="object")
        labels[(values >= 18) & (values <= 24)] = "Young (18-24)"
        labels[(values >= 25) & (values <= 34)] = "Early-career (25-34)"
        labels[(values >= 35) & (values <= 49)] = "Mid-career (35-49)"
        labels[(values >= 50) & (values <= 64)] = "Late-career (50-64)"
        labels[values >= 65] = "Senior (65+)"
        labels[labels.isna()] = "Out-of-range"
        return labels, None
    if n in {"monthly_income_usd", "monthly_income"}:
        return pd.cut(values, [-math.inf, 3000, 7000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str), "Low"
    if n in {"monthly_expenses_usd", "monthly_expenses"}:
        return pd.cut(values, [-math.inf, 1000, 3000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str), None
    if n in {"savings_usd", "savings_balance"}:
        return pd.cut(values, [-math.inf, 500, 3000, 15000, math.inf], labels=["Near-zero", "Low", "Moderate", "High"], right=False).astype(str), "Near-zero"
    if n in {"monthly_emi_usd", "monthly_emi"}:
        labels = pd.Series(index=values.index, dtype="object")
        labels[values == 0] = "None"
        labels[(values > 0) & (values < 500)] = "Low"
        labels[(values >= 500) & (values <= 1500)] = "Moderate"
        labels[values > 1500] = "High"
        return labels, None
    if n in {"loan_amount_usd", "loan_amount"}:
        return pd.cut(values, [-math.inf, 5000, 25000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str), None
    if n == "loan_term_months":
        return pd.cut(values, [-math.inf, 24, 60, math.inf], labels=["Short", "Medium", "Long"], right=False).astype(str), None
    if n in {"loan_interest_rate_pct", "loan_interest_rate"}:
        return pd.cut(values, [-math.inf, 5, 12, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str), None
    if n == "debt_to_income_ratio":
        return pd.cut(values, [-math.inf, 0.2, 0.4, math.inf], labels=["Safe", "Warning", "Risky"], right=False).astype(str), "Risky"
    if n == "credit_score":
        labels = pd.Series(index=values.index, dtype="object")
        labels[values < 580] = "Poor"
        labels[(values >= 580) & (values <= 669)] = "Fair"
        labels[(values >= 670) & (values <= 739)] = "Good"
        labels[(values >= 740) & (values <= 799)] = "Very Good"
        labels[values >= 800] = "Excellent"
        return labels, None
    if n in {"savings_to_income_ratio", "saving_to_income_ratio"}:
        return pd.cut(values, [-math.inf, 0.25, 1.0, math.inf], labels=["Fragile", "Moderate", "Strong"], right=False).astype(str), None

    labels = pd.Series(index=values.index, dtype="object")
    try:
        q = pd.qcut(values, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        labels.loc[q.index] = q.astype(str)
    except ValueError:
        labels.loc[values.index] = "Q2"
    labels[values >= values.quantile(0.99)] = "Outlier (Top 1%)"
    return labels, None


def _profile_column(col: str, series: pd.Series, total: int) -> List[FlagItem]:
    inferred = _infer_type(col, series)
    flags: List[FlagItem] = []
    n = _norm(col)

    print(f"Column: {col}")
    print(f"  - inferred type: {inferred}")
    print(f"  - missing rate: {_pct(int(_missing_mask(series).sum()), total)}%")
    print("  - representation slices:")

    if n == "user_id" or inferred == "id":
        non_missing = series[~_missing_mask(series)]
        uniq = non_missing.nunique(dropna=True)
        uniq_rate = round((uniq / max(len(non_missing), 1) * 100.0), 2) if len(non_missing) else 0.0
        _print_slices([SliceStat("Unique (non-missing)", int(uniq), uniq_rate)])
        if _missing_mask(series).any():
            flags.append(FlagItem(col, "Missing", "Identifier has missing values (>0%)."))
        if uniq_rate < 95.0:
            flags.append(FlagItem(col, "Uniqueness", f"Uniqueness below 95% ({uniq_rate}%)."))
    elif n == "record_date" or inferred == "datetime":
        dt = pd.to_datetime(series, errors="coerce")
        month = dt.dt.to_period("M").astype(str).where(dt.notna(), "Missing")
        _print_slices(_counts(month, total, sort_index=True))
        valid = month[month != "Missing"]
        if not valid.empty:
            top = valid.value_counts(normalize=True)
            top_share = float(top.max() * 100.0)
            if top_share > 60.0:
                flags.append(FlagItem(col, str(top.idxmax()), f"One month has >60% of records ({round(top_share, 2)}%)."))
    elif n == "job_title":
        valid = series[~_missing_mask(series)].astype(str).str.strip()
        top = valid.value_counts().head(10)
        print(f"  - unique titles: {valid.nunique(dropna=True)}")
        for title, count in top.items():
            print(f"  - top title '{title}': {count} ({_pct(int(count), total)}%)")
        other = max(len(valid) - int(top.sum()), 0)
        print(f"  - Other: {other} ({_pct(other, total)}%)")
        miss = _pct(int(_missing_mask(series).sum()), total)
        if miss > 20.0:
            flags.append(FlagItem(col, "Missing", f"Missing rate above 20% ({miss}%)."))
    elif n == "has_loan" or inferred == "boolean":
        lowered = _lower(series)
        labels = pd.Series(index=series.index, dtype="object")
        labels[lowered.isin(BOOL_TRUE)] = "True"
        labels[lowered.isin(BOOL_FALSE)] = "False"
        labels[_missing_mask(series)] = "Missing"
        labels[labels.isna()] = "Invalid"
        _print_slices(_counts(labels, total))
        valid = labels[labels.isin(["True", "False"])]
        if not valid.empty:
            dist = valid.value_counts(normalize=True)
            if dist.min() * 100.0 < 10.0:
                flags.append(FlagItem(col, str(dist.idxmin()), f"Minority class below 10% ({round(float(dist.min()*100.0), 2)}%)."))
    elif inferred == "numeric":
        num = _to_num(series)
        valid = num.dropna()
        labels = pd.Series(index=num.index, dtype="object")
        vuln_band: Optional[str] = None
        if not valid.empty:
            banded, vuln_band = _bands_for_numeric(col, valid)
            labels.loc[valid.index] = banded.astype(str)
        labels.loc[num.isna()] = "Missing"
        _print_slices(_counts(labels, total))
        if vuln_band is not None:
            vuln_count = int((labels == vuln_band).sum())
            vuln_pct = _pct(vuln_count, total)
            if vuln_count == 0 or vuln_pct < 10.0:
                flags.append(FlagItem(col, vuln_band, f"Vulnerable/high-risk band <10% or missing ({vuln_pct}%)."))
    else:
        labels = series.astype(str).str.strip().where(~_missing_mask(series), "Missing")
        stats = _counts(labels, total)
        _print_slices(stats)
        for s in stats:
            if s.label != "Missing" and s.pct < 10.0:
                flags.append(FlagItem(col, s.label, f"Category share below 10% ({s.pct}%)."))

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _missingness_bias_flags(df: pd.DataFrame) -> List[FlagItem]:
    key_numeric_candidates = [
        "monthly_income_usd", "monthly_expenses_usd", "savings_usd", "monthly_emi_usd", "loan_amount_usd",
        "debt_to_income_ratio", "credit_score", "savings_to_income_ratio",
        "monthly_income", "monthly_expenses", "savings_balance", "monthly_emi", "loan_amount", "saving_to_income_ratio",
    ]
    cat_targets = ["gender", "education_level", "employment_status", "loan_type", "region"]

    num_cols: List[str] = []
    for c in key_numeric_candidates:
        found = _first_existing(df, [c])
        if found and found not in num_cols:
            num_cols.append(found)
    cat_cols = [c for c in [_first_existing(df, [n]) for n in cat_targets] if c]

    flags: List[FlagItem] = []
    for num_col in num_cols:
        overall = _missing_mask(df[num_col]).mean()
        if overall <= 0:
            continue
        for cat_col in cat_cols:
            groups = df[cat_col].astype(str).str.strip().where(~_missing_mask(df[cat_col]), "Missing")
            group_missing = _missing_mask(df[num_col]).groupby(groups).mean()
            for g, rate in group_missing.items():
                if rate > 2.0 * overall:
                    flags.append(
                        FlagItem(
                            num_col,
                            f"{cat_col}={g}",
                            f"Missingness bias: subgroup {round(rate*100.0, 2)}% >2x overall {round(overall*100.0, 2)}%.",
                        )
                    )
    return flags


def _load_dataset(root_dir: str, processed_path: Optional[str], featured_path: Optional[str]) -> pd.DataFrame:
    processed = processed_path or os.path.join(root_dir, "data-pipeline", "dags", "data", "processed", "financial_preprocessed.csv")
    if not os.path.exists(processed):
        raise FileNotFoundError(f"Required file not found: {processed}")
    df = pd.read_csv(processed)

    featured = featured_path or os.path.join(root_dir, "data-pipeline", "dags", "data", "features", "financial_featured.csv")
    if os.path.exists(featured):
        feat = pd.read_csv(featured)
        key_left = _first_existing(df, ["user_id"])
        key_right = _first_existing(feat, ["user_id"])
        if key_left and key_right:
            feat = feat.drop(columns=[c for c in feat.columns if c in df.columns and c != key_right], errors="ignore")
            df = df.merge(feat, left_on=key_left, right_on=key_right, how="left")
            if key_right != key_left and key_right in df.columns:
                df = df.drop(columns=[key_right])
    return df


def run_phase15_financial_bias(root_dir: Optional[str] = None, processed_path: Optional[str] = None, featured_path: Optional[str] = None) -> int:
    repo_root = root_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    df = _load_dataset(repo_root, processed_path, featured_path)
    rows, cols = len(df), len(df.columns)

    _print_header("Dataset summary:")
    overall_missing = round(df.isna().sum().sum() / max(df.size, 1) * 100.0, 2)
    print(f"rows: {rows}, columns: {cols}")
    print(f"overall missingness (%): {overall_missing}")
    print("top 5 columns by missingness:")
    top5 = sorted(((c, _pct(int(_missing_mask(df[c]).sum()), rows)) for c in df.columns), key=lambda x: x[1], reverse=True)[:5]
    for c, p in top5:
        print(f"- {c}: {p}%")

    _print_header("Column-level representation analysis:")
    flags: List[FlagItem] = []
    for col in df.columns:
        flags.extend(_profile_column(col, df[col], rows))
    flags.extend(_missingness_bias_flags(df))

    _print_header("Flagged Representation Risks")
    if flags:
        for f in flags:
            print(f"- {f.column} | {f.group}: {f.reason}")
    else:
        print("- None")

    _print_header("Recommended Training-Time Mitigations")
    if flags:
        print("- Stratified sampling by flagged bands/groups.")
        print("- Controlled oversampling of underrepresented high-risk bands (training only).")
        if any(_norm(f.column) == "record_date" for f in flags):
            print("- Time-stratified split due to record_date temporal skew.")
    else:
        print("- No flagged representation risks; keep routine monitoring.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 15 representation bias detection (terminal output only).")
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--processed-path", default=None)
    parser.add_argument("--featured-path", default=None)
    args = parser.parse_args()
    return run_phase15_financial_bias(args.root_dir, args.processed_path, args.featured_path)


if __name__ == "__main__":
    raise SystemExit(main())
"""
Phase 15 representation bias detection for SavVio financial data.

CLI behavior:
- Loads data/processed/financial_preprocessed.csv
- Optionally merges data/features/financial_featured.csv if present
- Prints terminal-only representation summary and risk flags
- Does not write any output files
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd


BOOL_TRUE = {"1", "true", "yes", "y", "t"}
BOOL_FALSE = {"0", "false", "no", "n", "f"}


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


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    stripped = series.astype(str).str.strip()
    return series.isna() | stripped.eq("")


def _pct(count: int, total: int) -> float:
    return round((count / total * 100.0), 2) if total else 0.0


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _series_lower(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _value_counts_pct(series: pd.Series, total: int, sort_index: bool = False) -> List[SliceStat]:
    counts = series.value_counts(dropna=False, sort=not sort_index)
    if sort_index:
        counts = counts.sort_index()
    return [SliceStat(label=str(k), count=int(v), pct=_pct(int(v), total)) for k, v in counts.items()]


def _print_header(text: str) -> None:
    print(f"\n{text}")
    print("-" * len(text))


def _normalize_column_name(name: str) -> str:
    return name.strip().lower()


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {_normalize_column_name(c): c for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def _infer_type(name: str, series: pd.Series) -> str:
    norm_name = _normalize_column_name(name)
    missing = _missing_mask(series)
    non_missing = series[~missing]
    if non_missing.empty:
        if "date" in norm_name or "time" in norm_name:
            return "datetime"
        return "categorical"

    if norm_name == "user_id" or (norm_name.endswith("_id") and "credit" not in norm_name):
        return "id"

    if norm_name == "record_date":
        return "datetime"

    lowered = _series_lower(non_missing)
    if pd.api.types.is_bool_dtype(series) or lowered.isin(BOOL_TRUE | BOOL_FALSE).all():
        return "boolean"

    dt_try = pd.to_datetime(non_missing, errors="coerce")
    if dt_try.notna().mean() > 0.95 and ("date" in norm_name or "time" in norm_name):
        return "datetime"

    numeric_try = _as_numeric(non_missing)
    if pd.api.types.is_numeric_dtype(series) or numeric_try.notna().mean() >= 0.95:
        return "numeric"

    if norm_name == "job_title":
        return "text"

    uniq_ratio = non_missing.nunique(dropna=True) / max(len(non_missing), 1)
    avg_len = non_missing.astype(str).str.len().mean()
    if uniq_ratio > 0.98:
        return "id"
    if avg_len > 40:
        return "text"
    return "categorical"


def _band_age(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    labels[(values >= 18) & (values <= 24)] = "Young (18-24)"
    labels[(values >= 25) & (values <= 34)] = "Early-career (25-34)"
    labels[(values >= 35) & (values <= 49)] = "Mid-career (35-49)"
    labels[(values >= 50) & (values <= 64)] = "Late-career (50-64)"
    labels[values >= 65] = "Senior (65+)"
    labels[labels.isna()] = "Out-of-range"
    return labels


def _band_monthly_income(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 3000, 7000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_monthly_expenses(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 1000, 3000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_savings(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        [-math.inf, 500, 3000, 15000, math.inf],
        labels=["Near-zero", "Low", "Moderate", "High"],
        right=False,
    ).astype(str)


def _band_emi(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    labels[values == 0] = "None"
    labels[(values > 0) & (values < 500)] = "Low"
    labels[(values >= 500) & (values <= 1500)] = "Moderate"
    labels[values > 1500] = "High"
    return labels


def _band_loan_amount(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 5000, 25000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_loan_term(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 24, 60, math.inf], labels=["Short", "Medium", "Long"], right=False).astype(str)


def _band_interest(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 5, 12, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_dti(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 0.2, 0.4, math.inf], labels=["Safe", "Warning", "Risky"], right=False).astype(str)


def _band_credit_score(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    labels[values < 580] = "Poor"
    labels[(values >= 580) & (values <= 669)] = "Fair"
    labels[(values >= 670) & (values <= 739)] = "Good"
    labels[(values >= 740) & (values <= 799)] = "Very Good"
    labels[values >= 800] = "Excellent"
    return labels


def _band_savings_income_ratio(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        [-math.inf, 0.25, 1.0, math.inf],
        labels=["Fragile", "Moderate", "Strong"],
        right=False,
    ).astype(str)


def _band_unknown_numeric(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    if values.empty:
        return labels
    try:
        q = pd.qcut(values, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        labels.loc[q.index] = q.astype(str)
    except ValueError:
        labels.loc[values.index] = "Q2"
    top_1 = values.quantile(0.99)
    labels[values >= top_1] = "Outlier (Top 1%)"
    return labels


def _resolve_banding(column: str, values: pd.Series) -> Tuple[pd.Series, Optional[str]]:
    name = _normalize_column_name(column)

    if name == "age":
        return _band_age(values), None
    if name in {"monthly_income_usd", "monthly_income"}:
        return _band_monthly_income(values), "Low"
    if name in {"monthly_expenses_usd", "monthly_expenses"}:
        return _band_monthly_expenses(values), None
    if name in {"savings_usd", "savings_balance"}:
        return _band_savings(values), "Near-zero"
    if name in {"monthly_emi_usd", "monthly_emi"}:
        return _band_emi(values), None
    if name in {"loan_amount_usd", "loan_amount"}:
        return _band_loan_amount(values), None
    if name == "loan_term_months":
        return _band_loan_term(values), None
    if name in {"loan_interest_rate_pct", "loan_interest_rate"}:
        return _band_interest(values), None
    if name == "debt_to_income_ratio":
        return _band_dti(values), "Risky"
    if name == "credit_score":
        return _band_credit_score(values), None
    if name in {"savings_to_income_ratio", "saving_to_income_ratio"}:
        return _band_savings_income_ratio(values), None

    return _band_unknown_numeric(values), None


def _print_slice_stats(stats: List[SliceStat], indent: str = "  ") -> None:
    for row in stats:
        print(f"{indent}- {row.label}: {row.count} ({row.pct}%)")


def _profile_id(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    missing = int(_missing_mask(series).sum())
    non_missing = series[~_missing_mask(series)]
    uniqueness = (non_missing.nunique(dropna=True) / max(len(non_missing), 1) * 100.0) if len(non_missing) else 0.0

    print(f"Column: {column}")
    print("  - inferred type: id")
    print(f"  - missing rate: {_pct(missing, total_rows)}%")
    print("  - representation slices:")
    _print_slice_stats([SliceStat("Unique (non-missing)", int(non_missing.nunique(dropna=True)), round(uniqueness, 2))])

    flags: List[FlagItem] = []
    if missing > 0:
        flags.append(FlagItem(column, "Missing", f"Identifier has missing values ({_pct(missing, total_rows)}%)."))
    if uniqueness < 95.0:
        flags.append(FlagItem(column, "Uniqueness", f"Uniqueness below 95% ({round(uniqueness, 2)}%)."))

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _profile_boolean(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    lowered = _series_lower(series)
    labels = pd.Series(index=series.index, dtype="object")
    labels[lowered.isin(BOOL_TRUE)] = "True"
    labels[lowered.isin(BOOL_FALSE)] = "False"
    labels[_missing_mask(series)] = "Missing"
    labels[labels.isna()] = "Invalid"

    missing = int((labels == "Missing").sum())
    stats = _value_counts_pct(labels, total_rows)

    print(f"Column: {column}")
    print("  - inferred type: boolean")
    print(f"  - missing rate: {_pct(missing, total_rows)}%")
    print("  - representation slices:")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    valid = labels[labels.isin(["True", "False"])]
    if not valid.empty:
        valid_dist = valid.value_counts(normalize=True)
        minority_pct = float(valid_dist.min() * 100.0)
        minority_label = str(valid_dist.idxmin())
        if minority_pct < 10.0:
            flags.append(
                FlagItem(column, minority_label, f"Minority class below 10% ({round(minority_pct, 2)}%).")
            )
    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _profile_datetime(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    parsed = pd.to_datetime(series, errors="coerce")
    missing = int(parsed.isna().sum())
    month = parsed.dt.to_period("M").astype(str)
    month = month.where(parsed.notna(), "Missing")
    stats = _value_counts_pct(month, total_rows, sort_index=True)

    print(f"Column: {column}")
    print("  - inferred type: datetime")
    print(f"  - missing rate: {_pct(missing, total_rows)}%")
    print("  - representation slices:")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    valid_month = month[month != "Missing"]
    if not valid_month.empty:
        top_share = valid_month.value_counts(normalize=True).max() * 100.0
        top_month = str(valid_month.value_counts(normalize=True).idxmax())
        if top_share > 60.0:
            flags.append(
                FlagItem(column, top_month, f"One month has >60% of records ({round(top_share, 2)}%).")
            )

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _profile_job_title(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    missing = _missing_mask(series)
    valid = series[~missing].astype(str).str.strip()
    total_valid = len(valid)
    unique_titles = int(valid.nunique(dropna=True))
    top = valid.value_counts().head(10)
    top_total = int(top.sum())
    other_count = max(total_valid - top_total, 0)

    print(f"Column: {column}")
    print("  - inferred type: text")
    print(f"  - missing rate: {_pct(int(missing.sum()), total_rows)}%")
    print("  - representation slices:")
    print(f"  - unique titles: {unique_titles}")
    for title, count in top.items():
        print(f"  - top title '{title}': {count} ({_pct(int(count), total_rows)}%)")
    print(f"  - Other: {other_count} ({_pct(other_count, total_rows)}%)")

    flags: List[FlagItem] = []
    miss_pct = _pct(int(missing.sum()), total_rows)
    if miss_pct > 20.0:
        flags.append(
            FlagItem(column, "Missing", f"Missing rate above 20% ({miss_pct}%), representation quality risk.")
        )

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _profile_categorical(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    labels = series.astype(str).str.strip()
    labels = labels.where(~_missing_mask(series), "Missing")
    stats = _value_counts_pct(labels, total_rows)

    print(f"Column: {column}")
    print("  - inferred type: categorical")
    print(f"  - missing rate: {_pct(int((labels == 'Missing').sum()), total_rows)}%")
    print("  - representation slices:")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    for row in stats:
        if row.label != "Missing" and row.pct < 10.0:
            flags.append(FlagItem(column, row.label, f"Category share below 10% ({row.pct}%)."))

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _profile_numeric(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    numeric = _as_numeric(series)
    missing = int(numeric.isna().sum())
    valid = numeric.dropna()
    bands = pd.Series(index=numeric.index, dtype="object")
    vulnerable_band: Optional[str]
    if not valid.empty:
        assigned, vulnerable_band = _resolve_banding(column, valid)
        bands.loc[valid.index] = assigned.astype(str)
    else:
        vulnerable_band = None

    bands.loc[numeric.isna()] = "Missing"
    stats = _value_counts_pct(bands, total_rows)

    print(f"Column: {column}")
    print("  - inferred type: numeric")
    print(f"  - missing rate: {_pct(missing, total_rows)}%")
    print("  - representation slices:")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    if vulnerable_band is not None:
        vuln_count = int((bands == vulnerable_band).sum())
        vuln_pct = _pct(vuln_count, total_rows)
        if vuln_count == 0 or vuln_pct < 10.0:
            flags.append(
                FlagItem(
                    column,
                    vulnerable_band,
                    f"Vulnerable/high-risk band underrepresented ({vuln_pct}%; threshold 10%).",
                )
            )

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _apply_missingness_bias_checks(df: pd.DataFrame) -> List[FlagItem]:
    flags: List[FlagItem] = []
    key_numeric_candidates = [
        "monthly_income_usd",
        "monthly_expenses_usd",
        "savings_usd",
        "monthly_emi_usd",
        "loan_amount_usd",
        "debt_to_income_ratio",
        "credit_score",
        "savings_to_income_ratio",
        "monthly_income",
        "monthly_expenses",
        "savings_balance",
        "monthly_emi",
        "loan_amount",
        "saving_to_income_ratio",
    ]
    numeric_cols: List[str] = []
    for name in key_numeric_candidates:
        resolved = _first_existing(df, [name])
        if resolved and resolved not in numeric_cols:
            numeric_cols.append(resolved)

    categorical_targets = ["gender", "education_level", "employment_status", "loan_type", "region"]
    cat_cols = [c for c in [_first_existing(df, [n]) for n in categorical_targets] if c is not None]

    for num_col in numeric_cols:
        overall_missing = _missing_mask(df[num_col]).mean()
        if overall_missing <= 0:
            continue
        for cat_col in cat_cols:
            groups = df[cat_col].astype(str).str.strip().where(~_missing_mask(df[cat_col]), "Missing")
            by_group = _missing_mask(df[num_col]).groupby(groups).mean()
            for grp, rate in by_group.items():
                if rate > (2.0 * overall_missing):
                    flags.append(
                        FlagItem(
                            num_col,
                            f"{cat_col}={grp}",
                            (
                                f"Missingness bias: subgroup missing {round(rate * 100.0, 2)}% "
                                f">2x overall {round(overall_missing * 100.0, 2)}%."
                            ),
                        )
                    )
    return flags


def _load_dataset(root_dir: str, processed_override: Optional[str], featured_override: Optional[str]) -> pd.DataFrame:
    processed_path = processed_override or os.path.join(
        root_dir,
        "data-pipeline",
        "dags",
        "data",
        "processed",
        "financial_preprocessed.csv",
    )
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Required file not found: {processed_path}")

    df = pd.read_csv(processed_path)

    featured_path = featured_override or os.path.join(
        root_dir,
        "data-pipeline",
        "dags",
        "data",
        "features",
        "financial_featured.csv",
    )
    if os.path.exists(featured_path):
        feat = pd.read_csv(featured_path)
        join_key = _first_existing(df, ["user_id"])
        feat_key = _first_existing(feat, ["user_id"])
        if join_key and feat_key:
            feat_no_overlap = feat.drop(columns=[c for c in feat.columns if c in df.columns and c != feat_key], errors="ignore")
            df = df.merge(feat_no_overlap, left_on=join_key, right_on=feat_key, how="left")
            if feat_key != join_key and feat_key in df.columns:
                df = df.drop(columns=[feat_key])
    return df


def run_phase15_financial_bias(root_dir: Optional[str] = None, processed_path: Optional[str] = None, featured_path: Optional[str] = None) -> int:
    repo_root = root_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    df = _load_dataset(repo_root, processed_path, featured_path)
    total_rows = len(df)
    total_cols = len(df.columns)

    _print_header("Dataset summary:")
    overall_missing = float(df.isna().sum().sum() / max(df.size, 1) * 100.0)
    print(f"rows: {total_rows}, columns: {total_cols}")
    print(f"overall missingness (%): {round(overall_missing, 2)}")

    col_missing = {col: _pct(int(_missing_mask(df[col]).sum()), total_rows) for col in df.columns}
    top5 = sorted(col_missing.items(), key=lambda x: x[1], reverse=True)[:5]
    print("top 5 columns by missingness:")
    for col, pct in top5:
        print(f"- {col}: {pct}%")

    _print_header("Column-level representation analysis:")
    all_flags: List[FlagItem] = []
    for column in df.columns:
        series = df[column]
        inferred = _infer_type(column, series)
        norm_col = _normalize_column_name(column)

        if norm_col == "user_id":
            all_flags.extend(_profile_id(column, series, total_rows))
            continue
        if norm_col == "record_date" or inferred == "datetime":
            all_flags.extend(_profile_datetime(column, series, total_rows))
            continue
        if norm_col == "job_title":
            all_flags.extend(_profile_job_title(column, series, total_rows))
            continue
        if norm_col == "has_loan" or inferred == "boolean":
            all_flags.extend(_profile_boolean(column, series, total_rows))
            continue
        if inferred == "numeric":
            all_flags.extend(_profile_numeric(column, series, total_rows))
            continue
        if inferred == "id":
            all_flags.extend(_profile_id(column, series, total_rows))
            continue
        all_flags.extend(_profile_categorical(column, series, total_rows))

    all_flags.extend(_apply_missingness_bias_checks(df))

    _print_header("Flagged Representation Risks")
    if not all_flags:
        print("- None")
    else:
        for f in all_flags:
            print(f"- {f.column} | {f.group}: {f.reason}")

    _print_header("Recommended Training-Time Mitigations")
    if not all_flags:
        print("- No representation risks were flagged; continue standard monitoring.")
    else:
        print("- Stratified sampling by flagged bands/groups.")
        print("- Controlled oversampling of underrepresented high-risk bands (training only).")
        if any(_normalize_column_name(f.column) == "record_date" for f in all_flags):
            print("- Time-stratified train/validation split due to temporal skew.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 15 representation bias detection for financial dataset.")
    parser.add_argument("--root-dir", default=None, help="Repository root path override.")
    parser.add_argument("--processed-path", default=None, help="Path to financial_preprocessed.csv override.")
    parser.add_argument("--featured-path", default=None, help="Path to financial_featured.csv override.")
    args = parser.parse_args()
    return run_phase15_financial_bias(args.root_dir, args.processed_path, args.featured_path)


if __name__ == "__main__":
    raise SystemExit(main())
"""
Phase 15 representation bias detection for SavVio financial data.

CLI behavior:
- Loads data/processed/financial_preprocessed.csv
- Optionally merges data/features/financial_featured.csv if present
- Prints terminal-only representation summary and risk flags
- Does not write any output files
"""

# removed duplicate future import

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none", "unknown", "missing", "-"}
BOOL_TRUE = {"1", "true", "yes", "y", "t"}
BOOL_FALSE = {"0", "false", "no", "n", "f"}


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


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    stripped = series.astype(str).str.strip()
    return series.isna() | stripped.eq("")


def _pct(count: int, total: int) -> float:
    return round((count / total * 100.0), 2) if total else 0.0


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _series_lower(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _value_counts_pct(series: pd.Series, total: int, sort_index: bool = False) -> List[SliceStat]:
    counts = series.value_counts(dropna=False, sort=not sort_index)
    if sort_index:
        counts = counts.sort_index()
    return [SliceStat(label=str(k), count=int(v), pct=_pct(int(v), total)) for k, v in counts.items()]


def _print_header(text: str) -> None:
    print(f"\n{text}")
    print("-" * len(text))


def _normalize_column_name(name: str) -> str:
    return name.strip().lower()


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {_normalize_column_name(c): c for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def _infer_type(name: str, series: pd.Series) -> str:
    norm_name = _normalize_column_name(name)
    missing = _missing_mask(series)
    non_missing = series[~missing]
    if non_missing.empty:
        if "date" in norm_name or "time" in norm_name:
            return "datetime"
        return "categorical"

    if norm_name == "user_id" or (norm_name.endswith("_id") and "credit" not in norm_name):
        return "id"

    if norm_name == "record_date":
        return "datetime"

    lowered = _series_lower(non_missing)
    if pd.api.types.is_bool_dtype(series) or lowered.isin(BOOL_TRUE | BOOL_FALSE).all():
        return "boolean"

    dt_try = pd.to_datetime(non_missing, errors="coerce")
    if dt_try.notna().mean() > 0.95 and ("date" in norm_name or "time" in norm_name):
        return "datetime"

    numeric_try = _as_numeric(non_missing)
    if pd.api.types.is_numeric_dtype(series) or numeric_try.notna().mean() >= 0.95:
        return "numeric"

    if norm_name == "job_title":
        return "text"

    uniq_ratio = non_missing.nunique(dropna=True) / max(len(non_missing), 1)
    avg_len = non_missing.astype(str).str.len().mean()
    if uniq_ratio > 0.98:
        return "id"
    if avg_len > 40:
        return "text"
    return "categorical"


def _band_age(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    labels[(values >= 18) & (values <= 24)] = "Young (18-24)"
    labels[(values >= 25) & (values <= 34)] = "Early-career (25-34)"
    labels[(values >= 35) & (values <= 49)] = "Mid-career (35-49)"
    labels[(values >= 50) & (values <= 64)] = "Late-career (50-64)"
    labels[values >= 65] = "Senior (65+)"
    labels[labels.isna()] = "Out-of-range"
    return labels


def _band_monthly_income(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 3000, 7000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_monthly_expenses(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 1000, 3000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_savings(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        [-math.inf, 500, 3000, 15000, math.inf],
        labels=["Near-zero", "Low", "Moderate", "High"],
        right=False,
    ).astype(str)


def _band_emi(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    labels[values == 0] = "None"
    labels[(values > 0) & (values < 500)] = "Low"
    labels[(values >= 500) & (values <= 1500)] = "Moderate"
    labels[values > 1500] = "High"
    return labels


def _band_loan_amount(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 5000, 25000, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_loan_term(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 24, 60, math.inf], labels=["Short", "Medium", "Long"], right=False).astype(str)


def _band_interest(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 5, 12, math.inf], labels=["Low", "Medium", "High"], right=False).astype(str)


def _band_dti(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-math.inf, 0.2, 0.4, math.inf], labels=["Safe", "Warning", "Risky"], right=False).astype(str)


def _band_credit_score(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    labels[values < 580] = "Poor"
    labels[(values >= 580) & (values <= 669)] = "Fair"
    labels[(values >= 670) & (values <= 739)] = "Good"
    labels[(values >= 740) & (values <= 799)] = "Very Good"
    labels[values >= 800] = "Excellent"
    return labels


def _band_savings_income_ratio(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        [-math.inf, 0.25, 1.0, math.inf],
        labels=["Fragile", "Moderate", "Strong"],
        right=False,
    ).astype(str)


def _band_unknown_numeric(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    if values.empty:
        return labels
    try:
        q = pd.qcut(values, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        labels.loc[q.index] = q.astype(str)
    except ValueError:
        labels.loc[values.index] = "Q2"
    top_1 = values.quantile(0.99)
    labels[values >= top_1] = "Outlier (Top 1%)"
    return labels


def _resolve_banding(column: str, values: pd.Series) -> Tuple[pd.Series, Optional[str]]:
    name = _normalize_column_name(column)

    if name == "age":
        return _band_age(values), None
    if name in {"monthly_income_usd", "monthly_income"}:
        return _band_monthly_income(values), "Low"
    if name in {"monthly_expenses_usd", "monthly_expenses"}:
        return _band_monthly_expenses(values), None
    if name in {"savings_usd", "savings_balance"}:
        return _band_savings(values), "Near-zero"
    if name in {"monthly_emi_usd", "monthly_emi"}:
        return _band_emi(values), None
    if name in {"loan_amount_usd", "loan_amount"}:
        return _band_loan_amount(values), None
    if name == "loan_term_months":
        return _band_loan_term(values), None
    if name in {"loan_interest_rate_pct", "loan_interest_rate"}:
        return _band_interest(values), None
    if name == "debt_to_income_ratio":
        return _band_dti(values), "Risky"
    if name == "credit_score":
        return _band_credit_score(values), None
    if name in {"savings_to_income_ratio", "saving_to_income_ratio"}:
        return _band_savings_income_ratio(values), None

    return _band_unknown_numeric(values), None


def _print_slice_stats(stats: List[SliceStat], indent: str = "  ") -> None:
    for row in stats:
        print(f"{indent}- {row.label}: {row.count} ({row.pct}%)")


def _profile_id(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    missing = int(_missing_mask(series).sum())
    non_missing = series[~_missing_mask(series)]
    uniqueness = (non_missing.nunique(dropna=True) / max(len(non_missing), 1) * 100.0) if len(non_missing) else 0.0

    print(f"Column: {column}")
    print("  - inferred type: id")
    print(f"  - missing rate: {_pct(missing, total_rows)}%")
    print("  - representation slices:")
    _print_slice_stats([SliceStat("Unique (non-missing)", int(non_missing.nunique(dropna=True)), round(uniqueness, 2))])

    flags: List[FlagItem] = []
    if missing > 0:
        flags.append(FlagItem(column, "Missing", f"Identifier has missing values ({_pct(missing, total_rows)}%)."))
    if uniqueness < 95.0:
        flags.append(FlagItem(column, "Uniqueness", f"Uniqueness below 95% ({round(uniqueness, 2)}%)."))

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _profile_boolean(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    lowered = _series_lower(series)
    labels = pd.Series(index=series.index, dtype="object")
    labels[lowered.isin(BOOL_TRUE)] = "True"
    labels[lowered.isin(BOOL_FALSE)] = "False"
    labels[_missing_mask(series)] = "Missing"
    labels[labels.isna()] = "Invalid"

    missing = int((labels == "Missing").sum())
    stats = _value_counts_pct(labels, total_rows)

    print(f"Column: {column}")
    print("  - inferred type: boolean")
    print(f"  - missing rate: {_pct(missing, total_rows)}%")
    print("  - representation slices:")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    valid = labels[labels.isin(["True", "False"])]
    if not valid.empty:
        valid_dist = valid.value_counts(normalize=True)
        minority_pct = float(valid_dist.min() * 100.0)
        minority_label = str(valid_dist.idxmin())
        if minority_pct < 10.0:
            flags.append(
                FlagItem(column, minority_label, f"Minority class below 10% ({round(minority_pct, 2)}%).")
            )
    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _profile_datetime(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    parsed = pd.to_datetime(series, errors="coerce")
    missing = int(parsed.isna().sum())
    month = parsed.dt.to_period("M").astype(str)
    month = month.where(parsed.notna(), "Missing")
    stats = _value_counts_pct(month, total_rows, sort_index=True)

    print(f"Column: {column}")
    print("  - inferred type: datetime")
    print(f"  - missing rate: {_pct(missing, total_rows)}%")
    print("  - representation slices:")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    valid_month = month[month != "Missing"]
    if not valid_month.empty:
        top_share = valid_month.value_counts(normalize=True).max() * 100.0
        top_month = str(valid_month.value_counts(normalize=True).idxmax())
        if top_share > 60.0:
            flags.append(
                FlagItem(column, top_month, f"One month has >60% of records ({round(top_share, 2)}%).")
            )

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _profile_job_title(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    missing = _missing_mask(series)
    valid = series[~missing].astype(str).str.strip()
    total_valid = len(valid)
    unique_titles = int(valid.nunique(dropna=True))
    top = valid.value_counts().head(10)
    top_total = int(top.sum())
    other_count = max(total_valid - top_total, 0)

    print(f"Column: {column}")
    print("  - inferred type: text")
    print(f"  - missing rate: {_pct(int(missing.sum()), total_rows)}%")
    print("  - representation slices:")
    print(f"  - unique titles: {unique_titles}")
    for title, count in top.items():
        print(f"  - top title '{title}': {count} ({_pct(int(count), total_rows)}%)")
    print(f"  - Other: {other_count} ({_pct(other_count, total_rows)}%)")

    flags: List[FlagItem] = []
    miss_pct = _pct(int(missing.sum()), total_rows)
    if miss_pct > 20.0:
        flags.append(
            FlagItem(column, "Missing", f"Missing rate above 20% ({miss_pct}%), representation quality risk.")
        )

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _profile_categorical(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    labels = series.astype(str).str.strip()
    labels = labels.where(~_missing_mask(series), "Missing")
    stats = _value_counts_pct(labels, total_rows)

    print(f"Column: {column}")
    print("  - inferred type: categorical")
    print(f"  - missing rate: {_pct(int((labels == 'Missing').sum()), total_rows)}%")
    print("  - representation slices:")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    for row in stats:
        if row.label != "Missing" and row.pct < 10.0:
            flags.append(FlagItem(column, row.label, f"Category share below 10% ({row.pct}%)."))

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _profile_numeric(column: str, series: pd.Series, total_rows: int) -> List[FlagItem]:
    numeric = _as_numeric(series)
    missing = int(numeric.isna().sum())
    valid = numeric.dropna()
    bands = pd.Series(index=numeric.index, dtype="object")
    vulnerable_band: Optional[str]
    if not valid.empty:
        assigned, vulnerable_band = _resolve_banding(column, valid)
        bands.loc[valid.index] = assigned.astype(str)
    else:
        vulnerable_band = None

    bands.loc[numeric.isna()] = "Missing"
    stats = _value_counts_pct(bands, total_rows)

    print(f"Column: {column}")
    print("  - inferred type: numeric")
    print(f"  - missing rate: {_pct(missing, total_rows)}%")
    print("  - representation slices:")
    _print_slice_stats(stats)

    flags: List[FlagItem] = []
    if vulnerable_band is not None:
        vuln_count = int((bands == vulnerable_band).sum())
        vuln_pct = _pct(vuln_count, total_rows)
        if vuln_count == 0 or vuln_pct < 10.0:
            flags.append(
                FlagItem(
                    column,
                    vulnerable_band,
                    f"Vulnerable/high-risk band underrepresented ({vuln_pct}%; threshold 10%).",
                )
            )

    if flags:
        print("  - flagged groups:")
        for f in flags:
            print(f"    - {f.group}: {f.reason}")
    print("")
    return flags


def _apply_missingness_bias_checks(df: pd.DataFrame) -> List[FlagItem]:
    flags: List[FlagItem] = []
    key_numeric_candidates = [
        "monthly_income_usd",
        "monthly_expenses_usd",
        "savings_usd",
        "monthly_emi_usd",
        "loan_amount_usd",
        "debt_to_income_ratio",
        "credit_score",
        "savings_to_income_ratio",
        # legacy aliases
        "monthly_income",
        "monthly_expenses",
        "savings_balance",
        "monthly_emi",
        "loan_amount",
        "saving_to_income_ratio",
    ]
    numeric_cols = [c for c in key_numeric_candidates if _first_existing(df, [c]) is not None]
    numeric_cols = list(dict.fromkeys([_first_existing(df, [c]) for c in numeric_cols if _first_existing(df, [c])]))

    categorical_targets = [
        "gender",
        "education_level",
        "employment_status",
        "loan_type",
        "region",
    ]
    cat_cols = [c for c in [_first_existing(df, [n]) for n in categorical_targets] if c is not None]

    for num_col in numeric_cols:
        overall_missing = _missing_mask(df[num_col]).mean()
        if overall_missing <= 0:
            continue
        for cat_col in cat_cols:
            groups = df[cat_col].astype(str).str.strip().where(~_missing_mask(df[cat_col]), "Missing")
            by_group = _missing_mask(df[num_col]).groupby(groups).mean()
            for grp, rate in by_group.items():
                if rate > (2.0 * overall_missing):
                    flags.append(
                        FlagItem(
                            num_col,
                            f"{cat_col}={grp}",
                            (
                                f"Missingness bias: subgroup missing {round(rate * 100.0, 2)}% "
                                f">2x overall {round(overall_missing * 100.0, 2)}%."
                            ),
                        )
                    )
    return flags


def _load_dataset(root_dir: str, processed_override: Optional[str], featured_override: Optional[str]) -> pd.DataFrame:
    processed_path = processed_override or os.path.join(
        root_dir,
        "data-pipeline",
        "dags",
        "data",
        "processed",
        "financial_preprocessed.csv",
    )
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Required file not found: {processed_path}")

    df = pd.read_csv(processed_path)

    featured_path = featured_override or os.path.join(
        root_dir,
        "data-pipeline",
        "dags",
        "data",
        "features",
        "financial_featured.csv",
    )
    if os.path.exists(featured_path):
        feat = pd.read_csv(featured_path)
        join_key = _first_existing(df, ["user_id"])
        feat_key = _first_existing(feat, ["user_id"])
        if join_key and feat_key:
            feat_no_overlap = feat.drop(columns=[c for c in feat.columns if c in df.columns and c != feat_key], errors="ignore")
            df = df.merge(feat_no_overlap, left_on=join_key, right_on=feat_key, how="left")
            if feat_key != join_key and feat_key in df.columns:
                df = df.drop(columns=[feat_key])
    return df


def run_phase15_financial_bias(root_dir: Optional[str] = None, processed_path: Optional[str] = None, featured_path: Optional[str] = None) -> int:
    repo_root = root_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    df = _load_dataset(repo_root, processed_path, featured_path)
    total_rows = len(df)
    total_cols = len(df.columns)

    _print_header("Dataset summary:")
    overall_missing = float(df.isna().sum().sum() / max(df.size, 1) * 100.0)
    print(f"rows: {total_rows}, columns: {total_cols}")
    print(f"overall missingness (%): {round(overall_missing, 2)}")

    col_missing = {col: _pct(int(_missing_mask(df[col]).sum()), total_rows) for col in df.columns}
    top5 = sorted(col_missing.items(), key=lambda x: x[1], reverse=True)[:5]
    print("top 5 columns by missingness:")
    for col, pct in top5:
        print(f"- {col}: {pct}%")

    _print_header("Column-level representation analysis:")
    all_flags: List[FlagItem] = []
    for column in df.columns:
        series = df[column]
        inferred = _infer_type(column, series)
        norm_col = _normalize_column_name(column)

        if norm_col == "user_id":
            all_flags.extend(_profile_id(column, series, total_rows))
            continue
        if norm_col == "record_date" or inferred == "datetime":
            all_flags.extend(_profile_datetime(column, series, total_rows))
            continue
        if norm_col == "job_title":
            all_flags.extend(_profile_job_title(column, series, total_rows))
            continue
        if norm_col == "has_loan" or inferred == "boolean":
            all_flags.extend(_profile_boolean(column, series, total_rows))
            continue
        if inferred == "numeric":
            all_flags.extend(_profile_numeric(column, series, total_rows))
            continue
        if inferred == "id":
            all_flags.extend(_profile_id(column, series, total_rows))
            continue
        all_flags.extend(_profile_categorical(column, series, total_rows))

    all_flags.extend(_apply_missingness_bias_checks(df))

    _print_header("Flagged Representation Risks")
    if not all_flags:
        print("- None")
    else:
        for f in all_flags:
            print(f"- {f.column} | {f.group}: {f.reason}")

    _print_header("Recommended Training-Time Mitigations")
    if not all_flags:
        print("- No representation risks were flagged; continue standard monitoring.")
    else:
        print("- Stratified sampling by flagged bands/groups.")
        print("- Controlled oversampling of underrepresented high-risk bands (training only).")
        if any(_normalize_column_name(f.column) == "record_date" for f in all_flags):
            print("- Time-stratified train/validation split due to temporal skew.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 15 representation bias detection for financial dataset.")
    parser.add_argument("--root-dir", default=None, help="Repository root path override.")
    parser.add_argument("--processed-path", default=None, help="Path to financial_preprocessed.csv override.")
    parser.add_argument("--featured-path", default=None, help="Path to financial_featured.csv override.")
    args = parser.parse_args()
    return run_phase15_financial_bias(args.root_dir, args.processed_path, args.featured_path)


if __name__ == "__main__":
    raise SystemExit(main())
"""
Phase 15 representation bias detection for financial featured datasets.

This module analyzes all columns in financial_featured.csv and produces
a column-level bias report with:
- Column type inference
- Missingness and invalid value diagnostics
- Representation slices per type/domain
- Flagged columns with reasons
- Mitigation recommendations
"""

# removed duplicate future import

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "unknown", "missing", "-"}

BOOLEAN_TRUE = {"1", "true", "yes", "y", "t"}
BOOLEAN_FALSE = {"0", "false", "no", "n", "f"}
BOOLEAN_ALLOWED = BOOLEAN_TRUE | BOOLEAN_FALSE

VULNERABLE_BANDS = {
    "discretionary_income": "Negative",
    "dti": "Risky",
    "runway": "Fragile",
    "savings": "Near-zero",
}

KEY_NUMERIC_COLUMNS = [
    "monthly_income",
    "monthly_expenses",
    "monthly_emi",
    "savings_balance",
    "discretionary_income",
    "debt_to_income_ratio",
    "saving_to_income_ratio",
    "monthly_expense_burden_ratio",
    "emergency_fund_months",
]


@dataclass
class SliceStat:
    label: str
    count: int
    pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "count": self.count, "pct": self.pct}


def _normalize_text_series(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip()
    normalized = as_str.str.lower()
    return normalized.where(~series.isna(), other=float("nan"))


def _missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    stripped = series.astype(str).str.strip()
    return series.isna() | stripped.eq("")


def _financial_domain(column_name: str) -> Optional[str]:
    name = column_name.lower()

    if "discretionary_income" in name:
        return "discretionary_income"
    if "debt_to_income" in name or name in {"dti", "debt_income_ratio"}:
        return "dti"
    if "saving_to_income" in name or "savings_to_income" in name:
        return "savings_to_income"
    if "expense_burden" in name:
        return "expense_burden"
    if "emergency_fund_months" in name or "runway" in name:
        return "runway"
    if "emi" in name:
        return "emi"
    if "savings" in name:
        return "savings"
    if "expense" in name:
        return "expenses"
    if "income" in name:
        return "income"
    return None


def _infer_column_type(series: pd.Series, column_name: str) -> str:
    name = column_name.lower()
    non_missing = series[~_missing_mask(series)]
    non_missing_count = int(non_missing.shape[0])

    if "id" in name and not any(token in name for token in ["credit"]):
        return "id"

    if non_missing_count == 0:
        return "categorical"

    normalized = _normalize_text_series(non_missing)
    if normalized.isin(BOOLEAN_ALLOWED).all():
        return "boolean"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    numeric_cast = pd.to_numeric(non_missing, errors="coerce")
    numeric_ratio = float(numeric_cast.notna().mean()) if non_missing_count else 0.0
    if pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.95:
        return "numeric"

    unique_ratio = float(non_missing.nunique(dropna=True) / max(non_missing_count, 1))
    if unique_ratio > 0.98:
        return "id"

    avg_len = float(non_missing.astype(str).str.len().mean())
    if avg_len > 40:
        return "text"

    return "categorical"


def _invalid_count(series: pd.Series, inferred_type: str) -> int:
    missing = _missing_mask(series)
    values = series[~missing]
    if values.empty:
        return 0

    if inferred_type == "numeric":
        return int(pd.to_numeric(values, errors="coerce").isna().sum())

    if inferred_type == "boolean":
        normalized = _normalize_text_series(values)
        return int((~normalized.isin(BOOLEAN_ALLOWED)).sum())

    if inferred_type in {"categorical", "text"}:
        normalized = _normalize_text_series(values)
        return int(normalized.isin(MISSING_TOKENS - {""}).sum())

    return 0


def _slice_from_labels(labels: pd.Series, total_rows: int) -> List[SliceStat]:
    counts = labels.value_counts(dropna=False)
    stats: List[SliceStat] = []
    for label, count in counts.items():
        clean_label = "Unassigned" if pd.isna(label) else str(label)
        pct = (float(count) / total_rows * 100.0) if total_rows else 0.0
        stats.append(SliceStat(label=clean_label, count=int(count), pct=round(pct, 2)))
    return stats


def _apply_financial_bands(values: pd.Series, domain: str) -> pd.Series:
    if domain == "income":
        return pd.cut(values, [-math.inf, 3000, 7000, math.inf], labels=["Low", "Medium", "High"], right=False)
    if domain == "expenses":
        return pd.cut(values, [-math.inf, 1000, 3000, math.inf], labels=["Low", "Medium", "High"], right=False)
    if domain == "emi":
        labels = pd.Series(index=values.index, dtype="object")
        labels[values == 0] = "None"
        labels[(values > 0) & (values < 500)] = "Low"
        labels[(values >= 500) & (values <= 1500)] = "Moderate"
        labels[values > 1500] = "High"
        return labels
    if domain == "savings":
        return pd.cut(
            values,
            [-math.inf, 500, 3000, 15000, math.inf],
            labels=["Near-zero", "Low", "Moderate", "High"],
            right=False,
        )
    if domain == "discretionary_income":
        return pd.cut(
            values,
            [-math.inf, 0, 1000, math.inf],
            labels=["Negative", "Tight", "Comfortable"],
            right=False,
        )
    if domain == "dti":
        return pd.cut(values, [-math.inf, 0.2, 0.4, math.inf], labels=["Safe", "Warning", "Risky"], right=False)
    if domain == "savings_to_income":
        return pd.cut(
            values,
            [-math.inf, 0.25, 1.0, math.inf],
            labels=["Fragile", "Moderate", "Strong"],
            right=False,
        )
    if domain == "expense_burden":
        return pd.cut(
            values,
            [-math.inf, 0.5, 0.8, math.inf],
            labels=["Comfortable", "Tight", "Overstretched"],
            right=False,
        )
    if domain == "runway":
        return pd.cut(values, [-math.inf, 1, 3, math.inf], labels=["Critical", "Fragile", "Stable"], right=False)

    return pd.Series(index=values.index, dtype="object")


def _numeric_unknown_slices(values: pd.Series) -> pd.Series:
    labels = pd.Series(index=values.index, dtype="object")
    if values.empty:
        return labels

    try:
        qlabels = pd.qcut(values, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        labels.loc[qlabels.index] = qlabels.astype(str)
    except ValueError:
        labels.loc[values.index] = "Q2"

    outlier_threshold = values.quantile(0.99)
    labels[values >= outlier_threshold] = "Outlier (Top 1%)"
    return labels


def _bool_distribution(series: pd.Series) -> pd.Series:
    normalized = _normalize_text_series(series)
    mapped = pd.Series(index=series.index, dtype="object")
    mapped[normalized.isin(BOOLEAN_TRUE)] = "True"
    mapped[normalized.isin(BOOLEAN_FALSE)] = "False"
    mapped[_missing_mask(series)] = "Missing"
    mapped[mapped.isna()] = "Invalid"
    return mapped


def _text_length_slices(series: pd.Series) -> pd.Series:
    labels = pd.Series(index=series.index, dtype="object")
    lengths = series.astype(str).str.len()
    missing = _missing_mask(series)
    labels[missing] = "Missing"
    labels[(~missing) & (lengths < 20)] = "Short"
    labels[(~missing) & (lengths >= 20) & (lengths <= 100)] = "Medium"
    labels[(~missing) & (lengths > 100)] = "Long"
    return labels


def _column_profile(
    df: pd.DataFrame,
    column: str,
    total_rows: int,
    key_numeric_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    series = df[column]
    inferred_type = _infer_column_type(series, column)
    missing_count = int(_missing_mask(series).sum())
    invalid_count = _invalid_count(series, inferred_type)

    profile: Dict[str, Any] = {
        "column": column,
        "inferred_type": inferred_type,
        "missing_count": missing_count,
        "missing_pct": round((missing_count / total_rows * 100.0) if total_rows else 0.0, 2),
        "invalid_count": invalid_count,
        "invalid_pct": round((invalid_count / total_rows * 100.0) if total_rows else 0.0, 2),
        "slice_definition": "",
        "slices": [],
        "flags": [],
    }

    missing_mask = _missing_mask(series)

    if inferred_type == "id":
        non_missing = series[~missing_mask]
        unique_count = int(non_missing.nunique(dropna=True))
        uniqueness_pct = round((unique_count / max(len(non_missing), 1) * 100.0), 2)
        profile["slice_definition"] = "Identifier column: no slicing."
        profile["id_stats"] = {
            "unique_non_missing_count": unique_count,
            "uniqueness_pct_non_missing": uniqueness_pct,
        }
        return profile

    if inferred_type == "numeric":
        numeric_values = pd.to_numeric(series, errors="coerce")
        valid_values = numeric_values[~missing_mask & numeric_values.notna()]
        domain = _financial_domain(column)

        if domain:
            labels = pd.Series(index=series.index, dtype="object")
            labels.loc[valid_values.index] = _apply_financial_bands(valid_values, domain)
            labels.loc[missing_mask] = "Missing"
            labels.loc[(~missing_mask) & numeric_values.isna()] = "Invalid"
            profile["slice_definition"] = f"Financial-domain numeric bands ({domain})."
            profile["slices"] = [x.to_dict() for x in _slice_from_labels(labels, total_rows)]
            profile["financial_domain"] = domain

            # Vulnerable band checks.
            if domain in VULNERABLE_BANDS:
                vulnerable = VULNERABLE_BANDS[domain]
                vulnerable_count = int((labels == vulnerable).sum())
                vulnerable_pct = round((vulnerable_count / total_rows * 100.0) if total_rows else 0.0, 2)
                if vulnerable_count == 0 or vulnerable_pct < 10.0:
                    profile["flags"].append(
                        f"Vulnerable band '{vulnerable}' underrepresented ({vulnerable_pct}%)."
                    )
        else:
            labels = pd.Series(index=series.index, dtype="object")
            labels.loc[valid_values.index] = _numeric_unknown_slices(valid_values)
            labels.loc[missing_mask] = "Missing"
            labels.loc[(~missing_mask) & numeric_values.isna()] = "Invalid"
            profile["slice_definition"] = "Unknown-domain numeric bins (Q1/Q2/Q3/Q4 + top 1% outlier band)."
            profile["slices"] = [x.to_dict() for x in _slice_from_labels(labels, total_rows)]

        return profile

    if inferred_type == "categorical":
        normalized = series.astype(str).str.strip()
        labels = normalized.where(~missing_mask, other="Missing")
        counts = labels.value_counts(dropna=False)
        profile["slice_definition"] = "Category distribution with minority-category detection (<10%)."
        profile["slices"] = [
            SliceStat(label=str(k), count=int(v), pct=round((v / total_rows * 100.0), 2)).to_dict()
            for k, v in counts.items()
        ]

        minority_categories = []
        for k, v in counts.items():
            if str(k) != "Missing":
                pct = (float(v) / total_rows * 100.0) if total_rows else 0.0
                if pct < 10.0:
                    minority_categories.append(f"{k} ({pct:.2f}%)")
        if minority_categories:
            profile["flags"].append(
                "Minority categories below 10%: " + ", ".join(minority_categories[:8])
            )

        # Missingness by category for key numeric columns.
        if key_numeric_columns:
            missingness_by_category: Dict[str, Any] = {}
            for key_num in key_numeric_columns:
                if key_num not in df.columns:
                    continue
                key_missing = _missing_mask(df[key_num])
                by_cat = key_missing.groupby(labels).mean()
                missingness_by_category[key_num] = {
                    str(cat): round(float(rate) * 100.0, 2) for cat, rate in by_cat.items()
                }
            if missingness_by_category:
                profile["missingness_by_category_for_key_numeric"] = missingness_by_category
        return profile

    if inferred_type == "boolean":
        labels = _bool_distribution(series)
        profile["slice_definition"] = "Boolean class balance with minority-class detection (<10%)."
        profile["slices"] = [x.to_dict() for x in _slice_from_labels(labels, total_rows)]

        valid = labels[labels.isin(["True", "False"])]
        if not valid.empty:
            dist = valid.value_counts(normalize=True)
            minority = dist.min() * 100.0
            if minority < 10.0:
                profile["flags"].append(f"Boolean minority class below 10% ({minority:.2f}%).")
        return profile

    labels = _text_length_slices(series)
    profile["slice_definition"] = "Text length bands (Short/Medium/Long) + Missing."
    profile["slices"] = [x.to_dict() for x in _slice_from_labels(labels, total_rows)]
    return profile


def _missingness_by_subgroup(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for num_col in numeric_columns:
        if num_col not in df.columns:
            continue
        target_missing = _missing_mask(df[num_col])
        overall = float(target_missing.mean())
        if overall <= 0:
            continue

        for cat_col in categorical_columns:
            if cat_col not in df.columns:
                continue
            cat_values = df[cat_col].astype(str).str.strip().where(~_missing_mask(df[cat_col]), "Missing")
            grouped = target_missing.groupby(cat_values).mean()
            for category, rate in grouped.items():
                if rate > (2.0 * overall):
                    findings.append(
                        {
                            "numeric_column": num_col,
                            "group_column": cat_col,
                            "group_value": str(category),
                            "group_missing_pct": round(float(rate) * 100.0, 2),
                            "overall_missing_pct": round(overall * 100.0, 2),
                            "reason": (
                                f"Missingness in subgroup '{category}' for '{num_col}' "
                                f"is {round(float(rate) * 100.0, 2)}% "
                                f"(>2x overall {round(overall * 100.0, 2)}%)."
                            ),
                        }
                    )
    return findings


def analyze_financial_bias(input_path: str) -> Dict[str, Any]:
    df = pd.read_csv(input_path)
    total_rows = len(df)
    key_numeric_present = [col for col in KEY_NUMERIC_COLUMNS if col in df.columns]

    profiles: List[Dict[str, Any]] = []
    for col in df.columns:
        profiles.append(
            _column_profile(
                df,
                col,
                total_rows,
                key_numeric_columns=key_numeric_present,
            )
        )

    categorical_cols = [p["column"] for p in profiles if p["inferred_type"] == "categorical"]
    subgroup_findings = _missingness_by_subgroup(df, categorical_cols, key_numeric_present)

    flags_by_column: Dict[str, List[str]] = {}
    for profile in profiles:
        if profile.get("flags"):
            flags_by_column.setdefault(profile["column"], []).extend(profile["flags"])

    for finding in subgroup_findings:
        flags_by_column.setdefault(finding["numeric_column"], []).append(finding["reason"])

    flagged_columns: List[Dict[str, Any]] = []
    for column, reasons in flags_by_column.items():
        unique_reasons = list(dict.fromkeys(reasons))
        flagged_columns.append({"column": column, "reasons": unique_reasons})

    report = {
        "dataset": os.path.basename(input_path),
        "row_count": total_rows,
        "column_count": len(df.columns),
        "columns": profiles,
        "subgroup_missingness_findings": subgroup_findings,
        "flagged_columns": flagged_columns,
        "mitigation_recommendations": [
            "Use stratified train split by (income band x risk band).",
            "Oversample vulnerable bands at training time only.",
            "Stress-test on vulnerable bands to ensure the classifier does not become optimistic.",
        ],
    }
    return report


def _render_markdown_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Phase 15 Financial Representation Bias Report")
    lines.append("")
    lines.append(f"- Dataset: `{report['dataset']}`")
    lines.append(f"- Rows: {report['row_count']}")
    lines.append(f"- Columns: {report['column_count']}")
    lines.append("")
    lines.append("## Column-by-Column Analysis")
    lines.append("")

    for col in report["columns"]:
        lines.append(f"### `{col['column']}`")
        lines.append(f"- Inferred type: {col['inferred_type']}")
        lines.append(f"- Missingness: {col['missing_count']} ({col['missing_pct']}%)")
        lines.append(f"- Invalid values: {col['invalid_count']} ({col['invalid_pct']}%)")
        lines.append(f"- Slice logic: {col['slice_definition']}")
        if "id_stats" in col:
            lines.append(
                "- Identifier stats: "
                f"{col['id_stats']['unique_non_missing_count']} unique non-missing "
                f"({col['id_stats']['uniqueness_pct_non_missing']}% uniqueness)"
            )
        if col.get("slices"):
            lines.append("- Slices:")
            for s in col["slices"]:
                lines.append(f"  - {s['label']}: {s['count']} ({s['pct']}%)")
        if col.get("flags"):
            lines.append("- Flags:")
            for flag in col["flags"]:
                lines.append(f"  - {flag}")
        if col.get("missingness_by_category_for_key_numeric"):
            lines.append("- Missingness by category for key numeric columns:")
            for key_num, category_rates in col["missingness_by_category_for_key_numeric"].items():
                rendered_rates = ", ".join(
                    [f"{cat}: {rate}%" for cat, rate in list(category_rates.items())[:12]]
                )
                if len(category_rates) > 12:
                    rendered_rates += ", ..."
                lines.append(f"  - `{key_num}` -> {rendered_rates}")
        lines.append("")

    lines.append("## Missingness by Subgroup (>2x overall)")
    lines.append("")
    if report["subgroup_missingness_findings"]:
        for finding in report["subgroup_missingness_findings"]:
            lines.append(
                f"- `{finding['numeric_column']}` by `{finding['group_column']}` = "
                f"`{finding['group_value']}`: {finding['group_missing_pct']}% "
                f"(overall {finding['overall_missing_pct']}%)"
            )
    else:
        lines.append("- No subgroup missingness disparities exceeded the 2x threshold.")
    lines.append("")

    lines.append("## Flagged Columns and Reasons")
    lines.append("")
    if report["flagged_columns"]:
        for entry in report["flagged_columns"]:
            lines.append(f"- `{entry['column']}`")
            for reason in entry["reasons"]:
                lines.append(f"  - {reason}")
    else:
        lines.append("- No columns were flagged by configured thresholds.")
    lines.append("")

    lines.append("## Mitigation Recommendations")
    lines.append("")
    for rec in report["mitigation_recommendations"]:
        lines.append(f"- {rec}")
    lines.append("")
    return "\n".join(lines)


def save_report(report: Dict[str, Any], markdown_output_path: str, json_output_path: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(markdown_output_path), exist_ok=True)
    with open(markdown_output_path, "w", encoding="utf-8") as md_file:
        md_file.write(_render_markdown_report(report))

    if json_output_path:
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(report, json_file, indent=2)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, "data/features/financial_featured.csv")
    output_md = os.path.join(base_dir, "data/reports/financial_bias_report.md")
    output_json = os.path.join(base_dir, "data/reports/financial_bias_report.json")

    bias_report = analyze_financial_bias(input_file)
    save_report(bias_report, output_md, output_json)
    print(f"Bias report saved to: {output_md}")
    print(f"Bias report JSON saved to: {output_json}")
