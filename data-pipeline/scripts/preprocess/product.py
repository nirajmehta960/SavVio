import re

import pandas as pd

from .utils import (
    fill_numeric_median,
    load_csv,
    normalize_pipe_list,
    normalize_whitespace,
    save_csv,
)


def _clean_product_name(series):
    cleaned = series.fillna("").astype(str).str.lower()
    cleaned = cleaned.str.strip()
    cleaned = cleaned.str.replace(r"[^a-z0-9 ]", "", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    return cleaned


def _clean_price(series):
    cleaned = series.fillna("").astype(str)
    cleaned = cleaned.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _extract_specs_from_description(series):
    pattern = re.compile(
        r"(\d+(?:\.\d+)?\s*(?:cu\s*ft|l|liter|litre|w|kw|v|hz|rpm|in|inch|cm|mm|ft))"
    )
    specs = []
    for value in series.fillna("").astype(str):
        matches = pattern.findall(value.lower())
        unique = list(dict.fromkeys(match.strip() for match in matches if match.strip()))
        specs.append(" | ".join(unique))
    return pd.Series(specs, index=series.index)


def preprocess_product(input_path, output_path):
    df = load_csv(input_path)

    if "title" in df.columns:
        df["product_name"] = _clean_product_name(df["title"])
    else:
        df["product_name"] = ""

    if "price" in df.columns:
        df["price_usd"] = _clean_price(df["price"])
    else:
        df["price_usd"] = pd.NA

    df = df.dropna(subset=["price_usd"])
    df = df[df["price_usd"] > 0]
    df["price_usd"] = df["price_usd"].round(2)

    if "description" in df.columns:
        description_clean = (
            df["description"].fillna("").astype(str).apply(normalize_whitespace)
        )
        df["description_length"] = description_clean.str.len()
    else:
        df["description_length"] = 0

    if "categories" in df.columns:
        normalized_categories = df["categories"].fillna("").apply(normalize_pipe_list)
        category_parts = normalized_categories.str.split(" | ", n=1)
        df["appliance_type"] = category_parts.str[1].fillna(category_parts.str[0])
        df["appliance_type"] = df["appliance_type"].str.strip().str.lower()
    else:
        df["appliance_type"] = ""

    if "features" in df.columns:
        df["key_specs"] = df["features"].fillna("").apply(normalize_pipe_list)
    elif "description" in df.columns:
        df["key_specs"] = _extract_specs_from_description(df["description"]).fillna("")
    else:
        df["key_specs"] = ""

    if "average_rating" in df.columns:
        df["average_rating"] = fill_numeric_median(df["average_rating"], default=0)
        df["average_rating"] = df["average_rating"].clip(lower=0, upper=5).round(2)

    if "rating_number" in df.columns:
        df["rating_number"] = fill_numeric_median(df["rating_number"], default=0)
        df["rating_number"] = df["rating_number"].clip(lower=0).round(0).astype(int)

    df = df.drop_duplicates(subset=["product_name", "price_usd"], keep="first")

    df = df.drop(
        columns=["title", "price", "main_category", "features", "store", "categories"],
        errors="ignore",
    )

    ordered_columns = [
        "asin",
        "product_name",
        "price_usd",
        "appliance_type",
        "key_specs",
        "description_length",
        "average_rating",
        "rating_number",
    ]
    remaining_columns = [col for col in df.columns if col not in ordered_columns]
    df = df[[col for col in ordered_columns if col in df.columns] + remaining_columns]

    if (df["price_usd"] <= 0).any():
        raise AssertionError("price_usd must be greater than 0.")

    if df.isna().any().any():
        raise AssertionError("NaN values remain after preprocessing.")

    if df.duplicated().any():
        raise AssertionError("Duplicate rows found after preprocessing.")

    save_csv(df, output_path)
    return df
