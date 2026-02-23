import pytest
import os
import sys
from sqlalchemy import create_engine, inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dags.src.database.db_schema import (  # noqa: E402
    Base,
    FinancialProfile,
    Product,
    Review,
    create_tables,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def engine():
    """Use in-memory SQLite DB for schema tests"""
    return create_engine("sqlite:///:memory:")


@pytest.fixture()
def inspector(engine):
    """SQLAlchemy inspector after tables are created"""
    create_tables(engine)
    return inspect(engine)


# =============================================================================
# Tests for create_tables
# =============================================================================

def test_create_tables_success(inspector):
    tables = set(inspector.get_table_names())
    assert "financial_profiles" in tables
    assert "products" in tables
    assert "reviews" in tables


# =============================================================================
# Tests for Products table
# =============================================================================

def test_products_table_has_expected_columns(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("products")}
    expected = {
        "id", "product_id", "product_name", "price",
        "average_rating", "rating_number", "description",
        "features", "details", "category", "created_at",
    }
    assert expected.issubset(set(cols.keys()))

def test_products_table_not_null_constraints(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("products")}
    assert cols["product_id"]["nullable"] is False
    assert cols["product_name"]["nullable"] is False
    assert cols["price"]["nullable"] is False


# =============================================================================
# Tests for FinancialProfile table
# =============================================================================

def test_financial_profiles_table_has_expected_columns(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("financial_profiles")}
    expected = {
        "id",
        "user_id",
        "monthly_income",
        "monthly_expenses",
        "savings_balance",
        "has_loan",
        "loan_amount",
        "monthly_emi",
        "loan_interest_rate",
        "loan_term_months",
        "credit_score",
        "employment_status",
        "region",
        "discretionary_income",
        "debt_to_income_ratio",
        "saving_to_income_ratio",       # 修正：原本是 savings_rate
        "monthly_expense_burden_ratio", # 修正：原本是 expense_burden_ratio
        "emergency_fund_months",
        "created_at",
    }
    assert expected.issubset(set(cols.keys()))

def test_financial_profiles_not_null_constraints(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("financial_profiles")}
    assert cols["user_id"]["nullable"] is False
    assert cols["monthly_income"]["nullable"] is False
    assert cols["monthly_expenses"]["nullable"] is False


# =============================================================================
# Tests for Reviews table
# =============================================================================

def test_reviews_table_has_expected_columns(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("reviews")}
    expected = {
        "id", "user_id", "asin", "product_id", "rating",
        "review_title", "review_text", "verified_purchase",
        "helpful_vote", "created_at",
    }
    assert expected.issubset(set(cols.keys()))

def test_reviews_rating_not_null(inspector):
    cols = {c["name"]: c for c in inspector.get_columns("reviews")}
    assert cols["rating"]["nullable"] is False

def test_reviews_foreign_key_to_products(inspector):
    fks = inspector.get_foreign_keys("reviews")
    assert len(fks) >= 1
    matches = [
        fk for fk in fks
        if fk.get("referred_table") == "products"
        and fk.get("constrained_columns") == ["product_id"]
        and fk.get("referred_columns") == ["product_id"]
    ]
    assert matches, f"Expected FK reviews.product_id -> products.product_id, got: {fks}"


# =============================================================================
# Tests for model metadata
# =============================================================================

def test_model_tablenames_and_metadata():
    assert FinancialProfile.__tablename__ == "financial_profiles"
    assert Product.__tablename__ == "products"
    assert Review.__tablename__ == "reviews"
    table_names = set(Base.metadata.tables.keys())
    assert {"financial_profiles", "products", "reviews"}.issubset(table_names)