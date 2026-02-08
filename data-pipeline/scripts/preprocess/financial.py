import pandas as pd

from .utils import fill_numeric_median, load_csv, save_csv


MONETARY_COLUMNS = [
    "income_usd",
    "expenses_usd",
    "savings_usd",
    "monthly_income",
    "monthly_expenses",
    "monthly_savings",
    "rent_expenses_usd",
    "subscriptions_expenses_usd",
    "loans_expenses_usd",
    "utilities_expenses_usd",
    "other_expenses_usd",
    "total_fixed_expenses",
]

OPTIONAL_NUMERIC_COLUMNS = ["credit_score", "age"]


def preprocess_financial(input_path, output_path):
    df = load_csv(input_path)

    df = df.drop_duplicates()

    df = df.rename(
        columns={
            "income": "income_usd",
            "expenses": "expenses_usd",
            "savings": "savings_usd",
        }
    )

    if "income_usd" in df.columns:
        df["income_usd"] = pd.to_numeric(df["income_usd"], errors="coerce")
        df["income_missing"] = df["income_usd"].isna()
        df["income_usd"] = fill_numeric_median(df["income_usd"], default=0).clip(lower=0)

    for column in ["expenses_usd", "savings_usd"]:
        if column in df.columns:
            df[column] = fill_numeric_median(df[column], default=0).clip(lower=0)

    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = fill_numeric_median(df[column], default=0).clip(lower=0)

    if "income_usd" in df.columns:
        df["monthly_income"] = (df["income_usd"] / 12).round(2)
    if "expenses_usd" in df.columns:
        df["monthly_expenses"] = (df["expenses_usd"] / 12).round(2)
    elif "weekly_expenses" in df.columns:
        weekly_expenses = fill_numeric_median(df["weekly_expenses"], default=0).clip(
            lower=0
        )
        df["monthly_expenses"] = (weekly_expenses * 4.33).round(2)
    if "savings_usd" in df.columns:
        df["monthly_savings"] = (df["savings_usd"] / 12).round(2)

    if "monthly_expenses" in df.columns:
        df["total_fixed_expenses"] = df["monthly_expenses"].round(2)

        # Deterministic split of monthly expenses into categories.
        rent_ratio = 0.35
        subscriptions_ratio = 0.1
        loans_ratio = 0.3
        utilities_ratio = 0.2
        other_ratio = 0.05

        df["rent_expenses_usd"] = (df["monthly_expenses"] * rent_ratio).round(2)
        df["subscriptions_expenses_usd"] = (
            df["monthly_expenses"] * subscriptions_ratio
        ).round(2)
        df["loans_expenses_usd"] = (df["monthly_expenses"] * loans_ratio).round(2)
        df["utilities_expenses_usd"] = (
            df["monthly_expenses"] * utilities_ratio
        ).round(2)
        df["other_expenses_usd"] = (df["monthly_expenses"] * other_ratio).round(2)

    for column in MONETARY_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").round(2)

    if "loan_status" in df.columns:
        df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce").fillna(0)
        df["loan_status"] = df["loan_status"].clip(lower=0, upper=1).astype(int)

    if "transaction_category" in df.columns:
        df["transaction_category"] = (
            df["transaction_category"].fillna("").astype(str).str.strip().str.lower()
        )
        df["transaction_category"] = df["transaction_category"].replace(
            {
                "netflix": "subscription",
                "spotify": "subscription",
                "prime": "subscription",
                "electric": "utilities",
                "power": "utilities",
                "water": "utilities",
                "rent payment": "rent",
            }
        )

    df = df.drop(
        columns=["gender", "region", "employment_status", "timestamp"], errors="ignore"
    )

    ordered_columns = [
        "id",
        "age",
        "income_usd",
        "expenses_usd",
        "savings_usd",
        "credit_score",
        "loan_status",
        "income_missing",
        "monthly_income",
        "monthly_expenses",
        "monthly_savings",
        "rent_expenses_usd",
        "subscriptions_expenses_usd",
        "loans_expenses_usd",
        "utilities_expenses_usd",
        "other_expenses_usd",
        "total_fixed_expenses",
    ]
    remaining_columns = [col for col in df.columns if col not in ordered_columns]
    df = df[[col for col in ordered_columns if col in df.columns] + remaining_columns]

    monetary_df = df[[col for col in MONETARY_COLUMNS if col in df.columns]]
    if (monetary_df < 0).any().any():
        raise AssertionError("Negative values found in monetary columns.")

    if df.isna().any().any():
        raise AssertionError("NaN values remain after preprocessing.")

    if df.duplicated().any():
        raise AssertionError("Duplicate rows found after preprocessing.")

    save_csv(df, output_path)
    return df
