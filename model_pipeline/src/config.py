"""
Model Pipeline Configuration.

Centralized configuration for feature engineering, deterministic engine
thresholds, and ML training settings. All constants used across the
pipeline are defined here for easy maintenance.
"""

import os


class Config:
    # ---------------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "temp_data", "financial_featured.csv")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "artifacts")
    SCENARIO_OUTPUT_PATH = os.path.join(BASE_DIR, "artifacts", "training_scenarios.csv")

    # ---------------------------------------------------------------------------
    # MLflow Configuration
    # ---------------------------------------------------------------------------
    # These match the docker-compose environment variables you established
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    EXPERIMENT_NAME = "Financial_Wellbeing_Prediction"

    # ---------------------------------------------------------------------------
    # Target Definition (Legacy — kept for backward compatibility)
    # ---------------------------------------------------------------------------
    # The columns we use to dynamically calculate our target "Good Financial Condition" Y variable
    # Replaced by deterministic engine labels (GREEN/YELLOW/RED) in Phase 3.
    TARGET_CALC_COLS = ["credit_score", "saving_to_income_ratio", "debt_to_income_ratio"]

    # ---------------------------------------------------------------------------
    # Feature Lists
    # ---------------------------------------------------------------------------

    # Financial features used by the deterministic engine.
    # Source: financial_profiles table in PostgreSQL.
    # Pre-computed by data_pipeline/dags/src/features/financial_features.py.
    FINANCIAL_FEATURES = [
        "discretionary_income",
        "debt_to_income_ratio",
        "saving_to_income_ratio",       # NOTE: DB column has no trailing 's'.
        "monthly_expense_burden_ratio",
        "emergency_fund_months",
    ]

    # Product features used by the deterministic engine.
    # Source: products table in PostgreSQL.
    # rating_variance is pre-computed by data_pipeline/dags/src/features/product_review_features.py.
    PRODUCT_FEATURES = [
        "price",
        "average_rating",
        "rating_number",
        "rating_variance",
    ]

    # ---------------------------------------------------------------------------
    # Feature Engineering Settings
    # ---------------------------------------------------------------------------

    # Categorical features for OrdinalEncoding.
    CATEGORICAL_FEATURES = ["employment_status", "has_loan", "region"]

    # Columns to drop before model training (IDs, text blobs, metadata).
    COLUMNS_TO_DROP = ["user_id", "product_id", "product_name",
                       "description", "features", "details", "category"]

    # Numeric features for StandardScaler (engine inputs + computed affordability metrics).
    NUMERIC_FEATURES = FINANCIAL_FEATURES + PRODUCT_FEATURES + [
        "monthly_income", "monthly_expenses", "savings_balance", "monthly_emi",
        "affordability_score", "price_to_income_ratio", "residual_utility_score",
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

    # =========================================================================
    # Deterministic Engine Thresholds
    # =========================================================================
    # These thresholds define the 4-tier rule system used to assign labels.
    # Adjust these values to tune the label distribution (currently ~82% RED,
    # ~14% YELLOW, ~4% GREEN with the dataset's financial profile distribution).
    # =========================================================================

    # Tier 1: Hard-stop → RED.
    # Any single violation triggers an immediate RED classification.
    HARD_STOP = {
        "discretionary_income_lt": 0,                       # Negative = cannot cover expenses.
        "debt_to_income_ratio_gt": 0.40,                    # Over 40% income goes to debt.
        "monthly_expense_burden_ratio_gt": 0.80,            # Over 80% income consumed by costs.
        "emergency_fund_months_lt": 1,                      # Less than 1 month of runway.
        "price_exceeds_discretionary_emergency_lt": 3,      # Compound: price > discretionary AND thin runway.
    }

    # Tier 2: Caution → YELLOW candidate.
    # Borderline financial situations — any one marks the scenario as YELLOW.
    CAUTION = {
        "discretionary_income_range": (0, 1000),            # Tight but positive discretionary budget.
        "debt_to_income_ratio_range": (0.20, 0.40),         # Moderate debt load.
        "monthly_expense_burden_ratio_range": (0.50, 0.80), # Moderate expense burden.
        "emergency_fund_months_range": (1, 3),              # Thin emergency runway.
        "saving_to_income_ratio_range": (0.25, 1.0),        # Low savings relative to income.
    }

    # Tier 3: Confidence downgrade (product signal quality).
    # Weak product signals downgrade the label by one level (GREEN → YELLOW, YELLOW → RED).
    CONFIDENCE_DOWNGRADE = {
        "rating_number_lt": 10,                # Too few reviews for reliable signal.
        "rating_variance_zero_count_lt": 10,   # Zero variance with few reviews → artificial.
        "rating_variance_gt": 1.0,             # High variance → polarized opinions.
        "average_rating_lte": 3.0,             # Poor average rating → product quality concern.
    }

    # ---------------------------------------------------------------------------
    # Scenario Generation Settings
    # ---------------------------------------------------------------------------
    N_SCENARIOS = 50_000        # Number of random (user, product) pairs for training.
    RANDOM_STATE = 42           # Seed for reproducibility.
