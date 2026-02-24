import sys
from pathlib import Path
import pytest
from sqlalchemy import create_engine, inspect

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DAGS_SRC = PROJECT_ROOT / "dags" / "src"
sys.path.insert(0, str(DAGS_SRC))

from database.db_schema import (  # noqa: E402
    Base,
    FinancialProfile,
    Product,
    Review,
    create_tables,
)

@pytest.fixture()
def engine():
    return create_engine("sqlite:///:memory:")

def test_create_tables_creates_expected_tables(engine):
    create_tables(engine)
    tables = set(inspect(engine).get_table_names())
    assert "financial_profiles" in tables
    assert "products" in tables
    assert "reviews" in tables

def test_products_table_has_expected_columns(engine):
    create_tables(engine)
    cols = {c["name"]: c for c in inspect(engine).get_columns("products")}
    expected = {
        "id", "product_id", "product_name", "price",
        "average_rating", "rating_number", "description",
        "features", "details", "category", "created_at",
    }
    assert expected.issubset(set(cols.keys()))
    assert cols["product_id"]["nullable"] is False
    assert cols["product_name"]["nullable"] is False
    assert cols["price"]["nullable"] is False

def test_financial_profiles_table_has_expected_columns(engine):
    create_tables(engine)
    cols = {c["name"]: c for c in inspect(engine).get_columns("financial_profiles")}
    expected = {
        "id", "user_id", "monthly_income", "monthly_expenses",
        "savings_balance", "has_loan", "loan_amount", "monthly_emi",
        "loan_interest_rate", "loan_term_months", "credit_score",
        "employment_status", "region", "discretionary_income",
        "debt_to_income_ratio",
        "saving_to_income_ratio",
        "monthly_expense_burden_ratio",
        "emergency_fund_months", "created_at",
    }
    assert expected.issubset(set(cols.keys()))
    assert cols["user_id"]["nullable"] is False
    assert cols["monthly_income"]["nullable"] is False
    assert cols["monthly_expenses"]["nullable"] is False

def test_reviews_table_has_expected_columns_and_fk(engine):
    create_tables(engine)
    inspector = inspect(engine)
    cols = {c["name"]: c for c in inspector.get_columns("reviews")}
    expected = {
        "id", "user_id", "asin", "product_id", "rating",
        "review_title", "review_text", "verified_purchase",
        "helpful_vote", "created_at",
    }
    assert expected.issubset(set(cols.keys()))
    assert cols["rating"]["nullable"] is False
    fks = inspector.get_foreign_keys("reviews")
    assert len(fks) >= 1
    matches = [
        fk for fk in fks
        if fk.get("referred_table") == "products"
        and fk.get("constrained_columns") == ["product_id"]
        and fk.get("referred_columns") == ["product_id"]
    ]
    assert matches, f"Expected FK reviews.product_id -> products.product_id, got: {fks}"

def test_sqlalchemy_models_have_expected_tablenames():
    assert FinancialProfile.__tablename__ == "financial_profiles"
    assert Product.__tablename__ == "products"
    assert Review.__tablename__ == "reviews"
    table_names = set(Base.metadata.tables.keys())
    assert {"financial_profiles", "products", "reviews"}.issubset(table_names)