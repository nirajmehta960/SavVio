import os

import pandas as pd


def check_shapes_and_nulls(files):
    for name, path in files.items():
        df = pd.read_csv(path)
        print(f"\n{name.upper()} -> {path}")
        print("shape:", df.shape)
        print("nulls (top 10):")
        print(df.isna().sum().sort_values(ascending=False).head(10))


def check_financial(path):
    df = pd.read_csv(path)
    cols = ["income", "monthly_income", "monthly_expenses", "monthly_savings"]
    existing = [col for col in cols if col in df.columns]
    print("\nFINANCIAL SAMPLE:")
    print(df[existing].head())
    if "income" in df.columns:
        print("Any missing income?", df["income"].isna().any())
    expected = {"monthly_income", "monthly_expenses", "monthly_savings"}
    print("Monthly columns present?", expected.issubset(df.columns))


def check_products(path):
    df = pd.read_csv(path)
    cols = ["title", "main_category", "price", "categories"]
    existing = [col for col in cols if col in df.columns]
    print("\nPRODUCT SAMPLE:")
    print(df[existing].head())
    if "price" in df.columns:
        print("Any missing price?", df["price"].isna().any())
    if {"title", "price"}.issubset(df.columns):
        print("Any duplicate title+price?", df.duplicated(subset=["title", "price"]).any())


def check_reviews(path):
    df = pd.read_csv(path)
    cols = ["rating", "helpful_vote", "verified_purchase", "title"]
    existing = [col for col in cols if col in df.columns]
    print("\nREVIEW SAMPLE:")
    print(df[existing].head())
    if "verified_purchase" in df.columns:
        print("verified_purchase unique:", df["verified_purchase"].unique()[:5])
    if {"asin", "user_id", "timestamp"}.issubset(df.columns):
        print(
            "Any duplicate asin+user_id+timestamp?",
            df.duplicated(subset=["asin", "user_id", "timestamp"]).any(),
        )


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")

    files = {
        "financial": os.path.join(processed_dir, "financial_processed.csv"),
        "products": os.path.join(processed_dir, "products_processed.csv"),
        "reviews": os.path.join(processed_dir, "review_processed.csv"),
    }

    missing = [path for path in files.values() if not os.path.exists(path)]
    if missing:
        print("Missing processed files:")
        for path in missing:
            print(f"- {path}")
        return

    check_shapes_and_nulls(files)
    check_financial(files["financial"])
    check_products(files["products"])
    check_reviews(files["reviews"])


if __name__ == "__main__":
    main()
