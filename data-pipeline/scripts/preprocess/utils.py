import os

import pandas as pd


def load_csv(path):
    return pd.read_csv(path)


def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def normalize_whitespace(value):
    if pd.isna(value):
        return value
    return " ".join(str(value).split())


def to_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def normalize_pipe_list(value):
    if pd.isna(value):
        return value
    parts = [part.strip().lower() for part in str(value).split("|") if part.strip()]
    return " | ".join(parts)


def fill_numeric_median(series, default=0):
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.median()
    if pd.isna(median):
        median = default
    return numeric.fillna(median)
