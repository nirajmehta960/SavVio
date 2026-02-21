"""
SQLAlchemy table definitions for SavVio.
Matches the actual columns from preprocessing (see Data Pipeline Plan).
"""

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, Text, DateTime, ForeignKey,
    func, JSON
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# ---------------------------------------------------------------------------
# Financial profiles
# ---------------------------------------------------------------------------

class FinancialProfile(Base):
    __tablename__ = "financial_profiles"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    user_id             = Column(String(100), unique=True, nullable=False)
    monthly_income      = Column(Float, nullable=False)
    monthly_expenses    = Column(Float, nullable=False)
    savings_balance     = Column(Float)
    has_loan            = Column(Integer)          # 1 or 0
    loan_amount         = Column(Float, default=0)
    monthly_emi         = Column(Float, default=0)
    loan_interest_rate  = Column(Float, default=0)
    loan_term_months    = Column(Float, default=0)
    credit_score        = Column(Integer)
    employment_status   = Column(String(50))
    region              = Column(String(50))
    # --- feature-engineered columns (optional, loaded if present) ---
    discretionary_income  = Column(Float)
    debt_to_income_ratio  = Column(Float)
    saving_to_income_ratio       = Column(Float)
    monthly_expense_burden_ratio = Column(Float)
    emergency_fund_months = Column(Float)
    created_at            = Column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

class Product(Base):
    __tablename__ = "products"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    product_id      = Column(String(100), unique=True, nullable=False)  # parent_asin
    product_name    = Column(String(500), nullable=False)
    price           = Column(Float, nullable=False)
    average_rating  = Column(Float)
    rating_number   = Column(Integer)
    description     = Column(Text)
    features        = Column(Text)
    details         = Column(JSON)        # stored as JSONB in PostgreSQL
    category        = Column(String(200))
    created_at      = Column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------

class Review(Base):
    __tablename__ = "reviews"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    user_id           = Column(String(255))
    asin              = Column(String(100))
    product_id        = Column(String(100), ForeignKey("products.product_id"))
    rating            = Column(Float, nullable=False)
    review_title      = Column(Text)
    review_text       = Column(Text)
    verified_purchase = Column(Boolean)
    helpful_vote      = Column(Integer, default=0)
    created_at        = Column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Helper to create all tables
# ---------------------------------------------------------------------------

def create_tables(engine):
    """Create all tables that don't already exist."""
    Base.metadata.create_all(engine, checkfirst=True)
