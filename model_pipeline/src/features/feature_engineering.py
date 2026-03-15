"""
Feature Engineering — Model Pipeline (Phase 2).

Transforms raw DB tables into a model-ready feature matrix with
deterministic GREEN/YELLOW/RED labels.

Pipeline:
    1. generate_training_data() — Load data, create scenarios, label them
    2. transform_features()     — Impute, encode, scale, drop non-features
    3. build_feature_matrix()   — Orchestrator calls 1 then 2 and returns (X, y, scenarios_raw)

Input:  PostgreSQL tables — financial_profiles, products
Output: Feature matrix (X), label vector (y), raw scenarios CSV
"""

import os
import logging

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import Config
from features.training_data_generator import generate_scenarios
from data.db_loader import load_financial_profiles, load_products, load_reviews


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scikit-Learn Compatible Transformers
# ---------------------------------------------------------------------------

class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values following the project conventions.

    Strategy:
        - Financial numeric fields: fill with column median.
        - Product numeric fields: fill 0 for rating_variance, median for others.
        - Computed financial features: fill with column median to avoid
          injecting false signals (0.0 has real semantic meaning for ratios).
        - Categorical fields: fill with 'Unknown'.
    """
    def fit(self, X: pd.DataFrame, y=None):
        # In a real-world scenario, you would learn medians here during fit.
        # For simplicity and matching the old functional logic, we compute medians at transform.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Financial numerics
        for col in Config.FINANCIAL_FEATURES:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Product numerics
        for col in Config.PRODUCT_FEATURES:
            if col in df.columns and df[col].isnull().any():
                if col == "rating_variance":
                    df[col] = df[col].fillna(0.0)
                else:
                    df[col] = df[col].fillna(df[col].median())

        # Computed features — use median to avoid injecting false signals.
        computed_features = [
            "affordability_score", "price_to_income_ratio", "residual_utility_score",
            "savings_to_price_ratio", "net_worth_indicator", "credit_risk_indicator",
        ] + Config.PRODUCT_COMPUTED_FEATURES + Config.REVIEW_COMPUTED_FEATURES
        for col in computed_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Categorical features
        for col in Config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encode categorical features. Unknown categories mapped to -1."""
    
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.existing_cat_cols = []

    def fit(self, X: pd.DataFrame, y=None):
        self.existing_cat_cols = [c for c in Config.CATEGORICAL_FEATURES if c in X.columns]
        if self.existing_cat_cols:
            self.encoder.fit(X[self.existing_cat_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cols = [c for c in self.existing_cat_cols if c in df.columns]
        if cols:
            df[cols] = self.encoder.transform(df[cols])
        return df


class NumericScaler(BaseEstimator, TransformerMixin):
    """Scale numeric features to zero mean and unit variance."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols = []

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_cols = [c for c in Config.NUMERIC_FEATURES if c in X.columns]
        if self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cols = [c for c in self.numeric_cols if c in df.columns]
        if cols:
            df[cols] = self.scaler.transform(df[cols])
        return df


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drop non-feature columns (IDs, text blobs, labels)."""
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cols_to_drop = [c for c in Config.COLUMNS_TO_DROP + [Config.LABEL_COL]
                        if c in df.columns]
        if "product_price" in df.columns:
            cols_to_drop.append("product_price")
        if "financial_label" in df.columns:
            cols_to_drop.append("financial_label")
        return df.drop(columns=cols_to_drop, errors="ignore")


class FeaturePipeline:
    """
    Main transformation pipeline wrapping strict scikit-learn transformers.
    Saves/loads pipeline state to disk to ensure training and inference match perfectly.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ('imputer', MissingValueImputer()),
            ('encoder', CategoricalEncoder()),
            ('scaler', NumericScaler()),
            ('dropper', FeatureDropper()),
        ])
        
    def save(self, path=None):
        path = path or os.path.join(Config.MODEL_SAVE_DIR, "feature_pipeline.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Feature pipeline saved to %s", path)

    def load(self, path=None):
        path = path or os.path.join(Config.MODEL_SAVE_DIR, "feature_pipeline.pkl")
        self.pipeline = joblib.load(path)
        logger.info("Feature pipeline loaded from %s", path)

    def fit_transform(self, X: pd.DataFrame, save_pipeline: bool = True) -> pd.DataFrame:
        transformed = self.pipeline.fit_transform(X)
        if save_pipeline:
            self.save()
        return transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.load()
        return self.pipeline.transform(X)


# ---------------------------------------------------------------------------
# Functions for Downstream Use
# ---------------------------------------------------------------------------

def generate_training_data(
    financial_df: pd.DataFrame = None,
    products_df: pd.DataFrame = None,
    reviews_df: pd.DataFrame = None,
    n_scenarios: int = None,
):
    """
    Create raw labeled scenarios — no transformations applied.
    """
    if n_scenarios is None:
        n_scenarios = Config.N_SCENARIOS

    if financial_df is None or products_df is None or reviews_df is None:
        logger.info("Loading data from PostgreSQL...")
        financial_df = financial_df if financial_df is not None else load_financial_profiles()
        products_df = products_df if products_df is not None else load_products()
        reviews_df = reviews_df if reviews_df is not None else load_reviews()

    logger.info("Generating %d scenarios...", n_scenarios)
    scenarios_raw = generate_scenarios(
        financial_df,
        products_df,
        reviews_df=reviews_df,
        n_scenarios=n_scenarios,
        random_state=Config.RANDOM_STATE,
    )

    os.makedirs(os.path.dirname(Config.SCENARIO_OUTPUT_PATH), exist_ok=True)
    scenarios_raw.to_csv(Config.SCENARIO_OUTPUT_PATH, index=False)
    logger.info("Saved raw scenarios to %s", Config.SCENARIO_OUTPUT_PATH)

    y = scenarios_raw[Config.LABEL_COL].copy()
    logger.info("Training data generated — %d scenarios", len(scenarios_raw))

    return scenarios_raw, y


def transform_features(
    scenarios: pd.DataFrame,
    y: pd.Series,
    is_training: bool = True,
):
    """
    Transform raw scenarios into a model-ready feature matrix using the pipeline.
    """
    pipeline = FeaturePipeline()
    if is_training:
        X = pipeline.fit_transform(scenarios)
    else:
        X = pipeline.transform(scenarios)

    logger.info("Feature matrix built — X shape: %s", X.shape)
    return X, y


def build_feature_matrix(
    financial_df: pd.DataFrame = None,
    products_df: pd.DataFrame = None,
    reviews_df: pd.DataFrame = None,
    n_scenarios: int = None,
    is_training: bool = True,
):
    """
    End-to-end feature engineering pipeline. Orchestrates loading then transformation.
    """
    scenarios_raw, y = generate_training_data(
        financial_df=financial_df,
        products_df=products_df,
        reviews_df=reviews_df,
        n_scenarios=n_scenarios,
    )
    X, y = transform_features(scenarios_raw, y, is_training=is_training)

    logger.info("build_feature_matrix complete — X: %s", X.shape)
    return X, y, scenarios_raw