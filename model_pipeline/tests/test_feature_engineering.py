"""
Unit tests for Feature Engineering pipeline.

Tests missing-value handling, encoding, scaling, and build_feature_matrix().
Uses synthetic data so tests run without a live PostgreSQL connection.
"""

import pytest
import numpy as np
import pandas as pd


# ── Synthetic data helpers ───────────────────────────────────────────────────

def _make_financial_df(n=50):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "user_id": [f"U{i}" for i in range(n)],
        "monthly_income": rng.uniform(2000, 10000, n),
        "monthly_expenses": rng.uniform(1000, 5000, n),
        "savings_balance": rng.uniform(500, 50000, n),
        "has_loan": rng.choice([True, False], n),
        "loan_amount": rng.uniform(0, 200000, n),
        "monthly_emi": rng.uniform(0, 2000, n),
        "loan_interest_rate": rng.uniform(0, 20, n),
        "loan_term_months": rng.choice([0, 60, 120, 240], n),
        "credit_score": rng.integers(300, 850, n),
        "employment_status": rng.choice(["employed", "self-employed", "unemployed"], n),
        "region": rng.choice(["east", "west", "south", "north"], n),
        "discretionary_income": rng.uniform(-2000, 5000, n),
        "debt_to_income_ratio": rng.uniform(0, 0.6, n),
        "saving_to_income_ratio": rng.uniform(0.1, 5.0, n),
        "monthly_expense_burden_ratio": rng.uniform(0.3, 1.2, n),
        "emergency_fund_months": rng.uniform(0.5, 12, n),
    })


def _make_products_df(n=30):
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "product_id": [f"P{i}" for i in range(n)],
        "product_name": [f"Product {i}" for i in range(n)],
        "price": rng.uniform(5, 2000, n),
        "average_rating": rng.uniform(1, 5, n),
        "rating_number": rng.integers(1, 5000, n),
        "rating_variance": rng.uniform(0, 2, n),
        "description": ["desc"] * n,
        "features": ["feat"] * n,
        "details": ["det"] * n,
        "category": ["cat"] * n,
    })


# ── Missing-value tests ─────────────────────────────────────────────────────

class TestMissingValues:

    def test_financial_nulls_filled(self):
        from features.engineering import handle_missing_values
        df = _make_financial_df(20)
        df.loc[0, "discretionary_income"] = np.nan
        df.loc[1, "debt_to_income_ratio"] = np.nan
        result = handle_missing_values(df)
        assert result["discretionary_income"].isnull().sum() == 0
        assert result["debt_to_income_ratio"].isnull().sum() == 0

    def test_product_variance_null_filled_with_zero(self):
        from features.engineering import handle_missing_values
        df = pd.DataFrame({
            "rating_variance": [1.0, np.nan, 0.5],
            "price": [10, 20, 30],
        })
        result = handle_missing_values(df)
        assert result["rating_variance"].iloc[1] == 0.0

    def test_categorical_nulls_filled_with_unknown(self):
        from features.engineering import handle_missing_values
        df = pd.DataFrame({
            "employment_status": ["employed", None, "self-employed"],
            "has_loan": [True, None, False],
            "region": ["east", "west", None],
        })
        result = handle_missing_values(df)
        assert result["employment_status"].iloc[1] == "Unknown"
        assert result["region"].iloc[2] == "Unknown"

    def test_computed_features_nulls_filled_with_zero(self):
        """All 6 computed financial features should have NaN filled with 0.0."""
        from features.engineering import handle_missing_values
        computed_cols = [
            "affordability_score", "price_to_income_ratio", "residual_utility_score",
            "savings_to_price_ratio", "net_worth_indicator", "credit_risk_indicator",
        ]
        df = pd.DataFrame({col: [np.nan, 1.0, np.nan] for col in computed_cols})
        result = handle_missing_values(df)
        for col in computed_cols:
            assert result[col].isnull().sum() == 0, f"NaN not filled in {col}"
            assert result[col].iloc[0] == 0.0


# ── Encoding tests ───────────────────────────────────────────────────────────

class TestEncoding:

    def test_ordinal_encoder_produces_numeric(self, tmp_path):
        from features.engineering import encode_categoricals
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)

        df = pd.DataFrame({
            "employment_status": ["employed", "self-employed", "unemployed"],
            "has_loan": [True, False, True],
            "region": ["east", "west", "south"],
        })
        result = encode_categoricals(df, is_training=True)
        assert result["employment_status"].dtype in (np.float64, np.int64)
        assert result["region"].dtype in (np.float64, np.int64)

        # Verify inference with saved encoder produces same values.
        result2 = encode_categoricals(df, is_training=False)
        assert (result["employment_status"] == result2["employment_status"]).all()

        config.Config.MODEL_SAVE_DIR = original

    def test_unknown_category_at_inference(self, tmp_path):
        from features.engineering import encode_categoricals
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)

        train_df = pd.DataFrame({
            "employment_status": ["employed", "self-employed"],
            "has_loan": [True, False],
            "region": ["east", "west"],
        })
        encode_categoricals(train_df, is_training=True)

        test_df = pd.DataFrame({
            "employment_status": ["NEVER_SEEN_BEFORE"],
            "has_loan": [True],
            "region": ["UNKNOWN_REGION"],
        })
        result = encode_categoricals(test_df, is_training=False)
        assert result["employment_status"].iloc[0] == -1
        assert result["region"].iloc[0] == -1

        config.Config.MODEL_SAVE_DIR = original


# ── Scaling tests ────────────────────────────────────────────────────────────

class TestScaling:

    def test_scaled_features_near_zero_mean(self, tmp_path):
        from features.engineering import scale_features
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "discretionary_income": rng.uniform(-1000, 5000, 100),
            "price": rng.uniform(10, 1000, 100),
            "savings_to_price_ratio": rng.uniform(0.1, 50, 100),
            "credit_risk_indicator": rng.uniform(0, 1, 100),
        })
        result = scale_features(df, is_training=True)

        for col in ["discretionary_income", "price", "savings_to_price_ratio"]:
            if col in result.columns:
                assert abs(result[col].mean()) < 0.1

        config.Config.MODEL_SAVE_DIR = original


# ── build_feature_matrix integration ─────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_returns_correct_shapes(self, tmp_path):
        from features.engineering import build_feature_matrix
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)
        original_scenario = config.Config.SCENARIO_OUTPUT_PATH
        config.Config.SCENARIO_OUTPUT_PATH = str(tmp_path / "scenarios.csv")

        fin = _make_financial_df(50)
        prod = _make_products_df(30)

        X, y, raw = build_feature_matrix(
            financial_df=fin, products_df=prod, n_scenarios=200
        )

        assert len(X) == 200
        assert len(y) == 200
        assert len(raw) == 200
        assert X.isnull().any().any() == False
        assert set(y.unique()).issubset({"GREEN", "YELLOW", "RED"})

        config.Config.MODEL_SAVE_DIR = original
        config.Config.SCENARIO_OUTPUT_PATH = original_scenario

    def test_label_column_not_in_X(self, tmp_path):
        from features.engineering import build_feature_matrix
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)
        original_scenario = config.Config.SCENARIO_OUTPUT_PATH
        config.Config.SCENARIO_OUTPUT_PATH = str(tmp_path / "scenarios.csv")

        fin = _make_financial_df(50)
        prod = _make_products_df(30)

        X, y, _ = build_feature_matrix(
            financial_df=fin, products_df=prod, n_scenarios=100
        )
        assert "label" not in X.columns

        config.Config.MODEL_SAVE_DIR = original
        config.Config.SCENARIO_OUTPUT_PATH = original_scenario

    def test_no_id_columns_in_X(self, tmp_path):
        from features.engineering import build_feature_matrix
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)
        original_scenario = config.Config.SCENARIO_OUTPUT_PATH
        config.Config.SCENARIO_OUTPUT_PATH = str(tmp_path / "scenarios.csv")

        fin = _make_financial_df(50)
        prod = _make_products_df(30)

        X, _, _ = build_feature_matrix(
            financial_df=fin, products_df=prod, n_scenarios=100
        )
        assert "user_id" not in X.columns
        assert "product_id" not in X.columns

        config.Config.MODEL_SAVE_DIR = original
        config.Config.SCENARIO_OUTPUT_PATH = original_scenario

    def test_computed_features_present_in_X(self, tmp_path):
        """Verify all 6 computed financial features survive the pipeline into X."""
        from features.engineering import build_feature_matrix
        import config
        original = config.Config.MODEL_SAVE_DIR
        config.Config.MODEL_SAVE_DIR = str(tmp_path)
        original_scenario = config.Config.SCENARIO_OUTPUT_PATH
        config.Config.SCENARIO_OUTPUT_PATH = str(tmp_path / "scenarios.csv")

        fin = _make_financial_df(50)
        prod = _make_products_df(30)

        X, _, _ = build_feature_matrix(
            financial_df=fin, products_df=prod, n_scenarios=100
        )

        feature_cols = [
            "affordability_score", "price_to_income_ratio",
            "residual_utility_score", "savings_to_price_ratio",
            "net_worth_indicator", "credit_risk_indicator",
        ]
        for col in feature_cols:
            assert col in X.columns, f"Missing computed feature in X: {col}"

        # Verify review features are NOT in X (engine is purely financial).
        assert "review_confidence_score" not in X.columns
        assert "review_polarization_index" not in X.columns

        config.Config.MODEL_SAVE_DIR = original
        config.Config.SCENARIO_OUTPUT_PATH = original_scenario

