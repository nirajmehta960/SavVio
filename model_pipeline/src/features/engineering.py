"""
Feature Engineering — Model Pipeline (Phase 2).

Transforms raw DB tables into a model-ready feature matrix with
deterministic GREEN/YELLOW/RED labels.

Pipeline:
    1. Load financial_profiles + products from PostgreSQL
    2. Generate (user, product) scenarios via random pairing
    3. Label each scenario with the DecisionEngine
    4. Handle missing values & encode categoricals
    5. Scale numeric features
    6. Return (X, y, metadata) ready for ML training

Input:  PostgreSQL tables — financial_profiles, products
Output: Feature matrix (X), label vector (y), raw scenarios CSV
"""

import os
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values following the project conventions.

    Strategy:
        - Financial numeric fields: fill with column median.
        - Product numeric fields: fill 0 for rating_variance, median for others.
        - Affordability metrics: fill with 0 (safe default for computed ratios).
        - Categorical fields: fill with 'Unknown'.
    """
    from config import Config

    df = df.copy()

    # Financial numerics — median imputation (preserves distribution center).
    for col in Config.FINANCIAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Filled %d nulls in '%s' with median %.4f",
                        df[col].isnull().sum(), col, median_val)

    # Product numerics — rating_variance gets 0 (absence of variance = uniform signal).
    for col in Config.PRODUCT_FEATURES:
        if col in df.columns and df[col].isnull().any():
            if col == "rating_variance":
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(df[col].median())

    # Affordability metrics — fill with 0 (neutral default).
    for col in ["affordability_score", "price_to_income_ratio", "residual_utility_score"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Categorical features — fill with 'Unknown' for OrdinalEncoder compatibility.
    for col in Config.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(
    df: pd.DataFrame,
    is_training: bool = True,
) -> pd.DataFrame:
    """
    Ordinal-encode categorical features.

    During training: fits a new OrdinalEncoder and saves it as a pickle artifact.
    During inference: loads the saved encoder to keep encoding consistent.
    Unknown categories at inference time are mapped to -1.

    Args:
        df: DataFrame containing categorical columns from Config.CATEGORICAL_FEATURES.
        is_training: If True, fit and save encoder; if False, load saved encoder.

    Returns:
        DataFrame with categorical columns replaced by ordinal integer codes.
    """
    from config import Config

    df = df.copy()
    existing_cat_cols = [c for c in Config.CATEGORICAL_FEATURES if c in df.columns]

    if not existing_cat_cols:
        return df

    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    encoder_path = os.path.join(Config.MODEL_SAVE_DIR, "categorical_encoder.pkl")

    if is_training:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[existing_cat_cols] = encoder.fit_transform(df[existing_cat_cols])
        joblib.dump(encoder, encoder_path)
        logger.info("OrdinalEncoder fitted and saved to %s", encoder_path)
    else:
        encoder = joblib.load(encoder_path)
        df[existing_cat_cols] = encoder.transform(df[existing_cat_cols])
        logger.info("OrdinalEncoder loaded from %s", encoder_path)

    return df


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

def scale_features(
    df: pd.DataFrame,
    is_training: bool = True,
) -> pd.DataFrame:
    """
    Apply StandardScaler on numeric features (zero mean, unit variance).

    During training: fits and saves the scaler.
    During inference: loads saved scaler to ensure consistent transformations.

    Args:
        df: DataFrame containing numeric columns from Config.NUMERIC_FEATURES.
        is_training: If True, fit and save scaler; if False, load saved scaler.

    Returns:
        DataFrame with numeric columns scaled to zero mean and unit variance.
    """
    from config import Config

    df = df.copy()
    numeric_cols = [c for c in Config.NUMERIC_FEATURES if c in df.columns]

    if not numeric_cols:
        return df

    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    scaler_path = os.path.join(Config.MODEL_SAVE_DIR, "feature_scaler.pkl")

    if is_training:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        joblib.dump(scaler, scaler_path)
        logger.info("StandardScaler fitted and saved to %s", scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        logger.info("StandardScaler loaded from %s", scaler_path)

    return df


# ---------------------------------------------------------------------------
# Main entry point — build_feature_matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    financial_df: pd.DataFrame = None,
    products_df: pd.DataFrame = None,
    n_scenarios: int = None,
    is_training: bool = True,
):
    """
    End-to-end feature engineering pipeline.

    Orchestrates the full flow from raw data to model-ready features:
        1. Generate (user, product) scenarios via random pairing.
        2. Label each scenario with the deterministic engine.
        3. Handle missing values via imputation.
        4. Encode categorical features (OrdinalEncoder).
        5. Scale numeric features (StandardScaler).
        6. Drop non-feature columns (IDs, text, labels).

    Args:
        financial_df: Financial profiles DataFrame. If None, loads from DB.
        products_df:  Products DataFrame. If None, loads from DB.
        n_scenarios:  Number of scenarios to generate. Defaults to Config.N_SCENARIOS.
        is_training:  True = fit encoders/scalers; False = load saved artifacts.

    Returns:
        Tuple of (X, y, scenarios_raw):
            X: Feature matrix (pd.DataFrame) — all numeric, no IDs, no labels.
            y: Label Series (GREEN/YELLOW/RED strings).
            scenarios_raw: Unscaled scenario DataFrame for analysis/debugging.
    """
    from config import Config
    from features.affordability_features import generate_scenarios

    if n_scenarios is None:
        n_scenarios = Config.N_SCENARIOS

    # Load data from PostgreSQL if not provided by caller.
    if financial_df is None or products_df is None:
        from data.db_loader import load_financial_profiles, load_products
        logger.info("Loading data from PostgreSQL...")
        financial_df = financial_df if financial_df is not None else load_financial_profiles()
        products_df = products_df if products_df is not None else load_products()

    # Step 1-2: Generate user-product scenarios and label with DecisionEngine.
    logger.info("Generating %d scenarios...", n_scenarios)
    scenarios = generate_scenarios(
        financial_df, products_df,
        n_scenarios=n_scenarios,
        random_state=Config.RANDOM_STATE,
    )

    # Keep a raw (unscaled) copy for analysis and MLflow artifact logging.
    scenarios_raw = scenarios.copy()

    # Save raw scenarios as a versioned artifact.
    os.makedirs(os.path.dirname(Config.SCENARIO_OUTPUT_PATH), exist_ok=True)
    scenarios_raw.to_csv(Config.SCENARIO_OUTPUT_PATH, index=False)
    logger.info("Saved raw scenarios to %s", Config.SCENARIO_OUTPUT_PATH)

    # Extract labels before applying transformations.
    y = scenarios[Config.LABEL_COL].copy()

    # Step 3: Handle missing values.
    scenarios = handle_missing_values(scenarios)

    # Step 4: Encode categorical features.
    scenarios = encode_categoricals(scenarios, is_training=is_training)

    # Step 5: Scale numeric features.
    scenarios = scale_features(scenarios, is_training=is_training)

    # Step 6: Drop non-feature columns (IDs, text blobs, labels).
    cols_to_drop = [c for c in Config.COLUMNS_TO_DROP + [Config.LABEL_COL]
                    if c in scenarios.columns]
    # Drop product_price since it duplicates 'price' from product features.
    if "product_price" in scenarios.columns:
        cols_to_drop.append("product_price")

    X = scenarios.drop(columns=cols_to_drop, errors="ignore")

    logger.info(
        "Feature matrix built — X: %s, y distribution:\n%s",
        X.shape, y.value_counts().to_string(),
    )

    return X, y, scenarios_raw


# ---------------------------------------------------------------------------
# Backward-compatible wrapper (legacy)
# ---------------------------------------------------------------------------

def preprocess_features(df: pd.DataFrame, is_training: bool = True):
    """
    Applies feature engineering (Ordinal Encoding, Dropping leaked columns).
    If `is_training` is True, it fits a new OrdinalEncoder and saves it to disk
    so it can be uploaded to MLflow as an artifact later.

    Note: This is a legacy wrapper kept for backward compatibility with older run_pipeline.py.
    New code should call build_feature_matrix() directly.
    """
    from config import Config # localized import to avoid circular dependencies

    # Drop columns that are used to compute the Target to prevent data leakage, plus IDs/Dates
    cols_to_drop = [c for c in Config.COLUMNS_TO_DROP if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Filter numerical/categorical features
    existing_cat_cols = [c for c in Config.CATEGORICAL_FEATURES if c in X.columns]

    if is_training:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[existing_cat_cols] = encoder.fit_transform(X[existing_cat_cols])

        # Save the encoder so MLflow can register it as an artifact later
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        encoder_path = os.path.join(Config.MODEL_SAVE_DIR, "categorical_encoder.pkl")
        joblib.dump(encoder, encoder_path)
        print(f"OrdinalEncoder fitted and saved to {encoder_path}")
    else:
        # NOTE: At inference time, you would load the encoder downloaded from MLflow
        # This branch is just a placeholder for the future FastAPI integration.
        encoder_path = os.path.join(Config.MODEL_SAVE_DIR, "categorical_encoder.pkl")
        encoder = joblib.load(encoder_path)
        X[existing_cat_cols] = encoder.transform(X[existing_cat_cols])

    return X
