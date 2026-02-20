import json
import logging
from io import StringIO
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Helper: Load CSV / JSON / JSONL safely
# --------------------------------------------------
def _load_input_to_df(input_path: str) -> pd.DataFrame:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return pd.DataFrame()

    # JSON (object or list)
    if text[0] in "[{":
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                return pd.DataFrame([obj])
        except json.JSONDecodeError:
            pass

    # JSONL
    records = []
    jsonl_ok = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            jsonl_ok = False
            break
    if jsonl_ok and records:
        return pd.DataFrame(records)

    # Fallback: treat as CSV text
    return pd.read_csv(StringIO(text))


# --------------------------------------------------
# Main preprocessing function
# --------------------------------------------------
def preprocess_financial_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = _load_input_to_df(input_path)

    if df.empty:
        # For tests: return empty df instead of raising
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    print("\nLoaded Dataset")
    print("----------------")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())

    # --------------------------------------------------
    # Normalize column names
    # --------------------------------------------------
    rename_map = {
        # common variants
        "income": "monthly_income",
        "expenses": "monthly_expenses",
        "monthly_income_usd": "monthly_income",
        "monthly_expenses_usd": "monthly_expenses",
        "income_usd": "monthly_income",
        "expenses_usd": "monthly_expenses",
        "savings_usd": "savings",
        "savings_balance": "savings",
        "debt_usd": "debt",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Required columns (canonical)
    required_cols = ["user_id", "monthly_income", "monthly_expenses", "savings", "debt", "timestamp"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Optional columns used in some tests/features
    optional_defaults = {
        "has_loan": "No",
        "loan_amount": 0,
        "monthly_emi": 0,
        "loan_interest_rate": 0,
        "loan_term_months": 0,
        "credit_score": 0,
        "employment_status": "",
        "region": "",
    }
    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default
    # --------------------------------------------------
    # Anomaly rule: credit_score must be within [300, 850]
    # If out of range -> drop row (tests expect removal)
    # --------------------------------------------------
    if "credit_score" in df.columns:
        # if conversion failed earlier, it became 0; treat 0 as invalid too
        df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce")
        df = df[df["credit_score"].between(300, 850, inclusive="both")]
    
    if df.empty:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df
    # --------------------------------------------------
    # STRICT numeric conversion for critical fields
    # IMPORTANT: invalid -> NaN (not 0), then drop rows
    # --------------------------------------------------
    critical_numeric = ["monthly_income", "monthly_expenses"]
    for col in critical_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where critical fields are invalid/non-numeric
    df = df.dropna(subset=critical_numeric)

    # If after dropping invalid rows it's empty, return empty (tests expect this)
    if df.empty:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    # Non-critical numeric columns: allow coercion to 0
    noncritical_numeric = [
        "savings",
        "debt",
        "loan_amount",
        "monthly_emi",
        "loan_interest_rate",
        "loan_term_months",
        "credit_score",
    ]
    for col in noncritical_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Normalize has_loan to 0/1 if present as Yes/No
    if "has_loan" in df.columns:
        df["has_loan"] = df["has_loan"].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0)

    # Fill timestamp if missing
    df["timestamp"] = df["timestamp"].fillna("")

    # Fill remaining NaNs
    df = df.fillna(0)

    print("\nFinal Preprocessed Dataset")
    print("----------------------------")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())

    # --------------------------------------------------
    # Unit-test compatibility aliases
    # --------------------------------------------------
    df["income_usd"] = df["monthly_income"]
    df["expenses_usd"] = df["monthly_expenses"]
    df["savings_usd"] = df["savings"]

    df["loan_amount_usd"] = df["loan_amount"]
    df["monthly_emi_usd"] = df["monthly_emi"]
    df["loan_interest_rate_pct"] = df["loan_interest_rate"]

    # Save output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    raw_dir = Path("data") / "raw"
    processed_dir = Path("data") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_path = raw_dir / "financial.jsonl"
    output_path = processed_dir / "financial.csv"

    logging.info("Financial input: %s", input_path)
    logging.info("Financial output: %s", output_path)

    preprocess_financial_data(str(input_path), str(output_path))


if __name__ == "__main__":
    main()