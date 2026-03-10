"""
Model Pipeline Configuration.

Centralized settings for the ML training pipeline — paths, MLflow,
feature lists, and scenario generation. All constants used by
engineering.py and run_pipeline.py live here.

"""

import os


class Config:
    # ---------------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "artifacts")
    SCENARIO_OUTPUT_PATH = os.path.join(BASE_DIR, "artifacts", "training_scenarios.csv")

    # ---------------------------------------------------------------------------
    # MLflow Configuration
    # ---------------------------------------------------------------------------
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    EXPERIMENT_NAME = "Financial_Wellbeing_Prediction"

    # ---------------------------------------------------------------------------
    # Feature Lists (used by engineering.py for imputation, scaling, encoding)
    # ---------------------------------------------------------------------------

    # Raw financial columns from the financial_profiles table.
    # Used by handle_missing_values() for median imputation.
    FINANCIAL_FEATURES = [
        "discretionary_income",
        "debt_to_income_ratio",
        "saving_to_income_ratio",
        "monthly_expense_burden_ratio",
        "emergency_fund_months",
    ]

    # Raw product columns from the products table.
    # Used by handle_missing_values() — rating_variance fills with 0, rest with median.
    PRODUCT_FEATURES = [
        "price",
        "average_rating",
        "rating_number",
        "rating_variance",
    ]

    # Computed product features (Layer 2).
    PRODUCT_COMPUTED_FEATURES = [
        "value_density",
        "review_confidence",
        "rating_polarization",
        "quality_risk_score",
        "cold_start_flag",
        "price_category_rank",
        "category_rating_deviation",
    ]

    # Computed review features (Layer 2).
    REVIEW_COMPUTED_FEATURES = [
        "verified_purchase_ratio",
        "helpful_concentration",
        "sentiment_spread",
        "review_depth_score",
        "reviewer_diversity",
        "extreme_rating_ratio",
    ]

    # Categorical columns for OrdinalEncoding.
    CATEGORICAL_FEATURES = ["employment_status", "has_loan", "region"]

    # Columns dropped before model training (IDs, text blobs, metadata).
    COLUMNS_TO_DROP = ["user_id", "product_id", "product_name",
                       "description", "features", "details", "category"]

    # All numeric columns passed to StandardScaler.
    # Combines raw DB columns with the 6 computed financial features
    # from financial_features.py and the 13 Layer 2 product/review features.
    NUMERIC_FEATURES = FINANCIAL_FEATURES + PRODUCT_FEATURES + PRODUCT_COMPUTED_FEATURES + REVIEW_COMPUTED_FEATURES + [
        "monthly_income", "monthly_expenses", "savings_balance", "monthly_emi",
        # Computed financial features (6).
        "affordability_score", "price_to_income_ratio", "residual_utility_score",
        "savings_to_price_ratio", "net_worth_indicator", "credit_risk_indicator",
    ]

    # ---------------------------------------------------------------------------
    # Sensitive Features for Bias Tracking (Fairlearn)
    # ---------------------------------------------------------------------------
    SENSITIVE_FEATURES = ["region", "employment_status"]

    # ---------------------------------------------------------------------------
    # Label Configuration
    # ---------------------------------------------------------------------------
    LABEL_COL = "label"
    LABELS = ["GREEN", "YELLOW", "RED"]

    # ---------------------------------------------------------------------------
    # Scenario Generation Settings
    # ---------------------------------------------------------------------------
    N_SCENARIOS = 50_000
    RANDOM_STATE = 42
