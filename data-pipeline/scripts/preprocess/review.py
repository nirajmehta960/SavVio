import pandas as pd

from .utils import load_csv, save_csv, to_bool


def _clean_review_text(series):
    cleaned = series.fillna("").astype(str).str.lower()
    cleaned = cleaned.str.strip()
    cleaned = cleaned.str.replace(r"[^a-z0-9 ]", "", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    return cleaned


def _sentiment_from_rating(rating):
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def preprocess_review(input_path, output_path):
    df = load_csv(input_path)

    if "rating" not in df.columns:
        df["rating"] = pd.NA
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].clip(lower=1, upper=5)

    if "text" in df.columns:
        df["review_text_clean"] = _clean_review_text(df["text"])
    else:
        df["review_text_clean"] = ""

    df["sentiment_label"] = df["rating"].apply(_sentiment_from_rating)
    df["review_length"] = df["review_text_clean"].str.len()

    if "verified_purchase" in df.columns:
        df["verified_purchase"] = df["verified_purchase"].apply(to_bool)
    else:
        df["verified_purchase"] = False

    if "helpful_vote" in df.columns:
        df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0)
        df["helpful_vote"] = df["helpful_vote"].clip(lower=0).astype(int)
    else:
        df["helpful_vote"] = 0

    if {"user_id", "asin", "review_text_clean"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["user_id", "asin", "review_text_clean"], keep="first")
    else:
        df = df.drop_duplicates()

    df = df.drop(columns=["title", "parent_asin", "user_id", "timestamp", "text"], errors="ignore")

    ordered_columns = [
        "asin",
        "rating",
        "sentiment_label",
        "review_length",
        "verified_purchase",
        "helpful_vote",
    ]
    remaining_columns = [col for col in df.columns if col not in ordered_columns]
    df = df[[col for col in ordered_columns if col in df.columns] + remaining_columns]
    df = df.drop_duplicates()

    if ((df["rating"] < 1) | (df["rating"] > 5)).any():
        raise AssertionError("Ratings must be between 1 and 5.")

    if df.isna().any().any():
        raise AssertionError("NaN values remain after preprocessing.")

    if df.duplicated().any():
        raise AssertionError("Duplicate rows found after preprocessing.")

    save_csv(df, output_path)
    return df
