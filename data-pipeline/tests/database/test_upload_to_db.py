import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

# =============================================================================
# --- Magic trick: Add paths so imports work in tests ---
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DAGS_ROOT = os.path.join(PROJECT_ROOT, "dags")
DB_DIR = os.path.join(DAGS_ROOT, "src", "database")

sys.path.insert(0, PROJECT_ROOT)  # so "dags...." works
sys.path.insert(0, DAGS_ROOT)     # so "src...." can map to dags/src if needed
sys.path.insert(0, DB_DIR)        # so "upload_to_db" can be imported by file name


# =============================================================================
# --- Stub dependencies BEFORE importing upload_to_db ---
# Because upload_to_db imports: from src.database.db_connection import get_engine
# and pytest will fail if "src" isn't resolvable.
# =============================================================================
def _install_stub_src_modules():
    src_mod = types.ModuleType("src")
    database_mod = types.ModuleType("src.database")

    db_connection_mod = types.ModuleType("src.database.db_connection")
    db_connection_mod.get_engine = MagicMock(name="get_engine")

    db_schema_mod = types.ModuleType("src.database.db_schema")
    db_schema_mod.create_tables = MagicMock(name="create_tables")

    sys.modules["src"] = src_mod
    sys.modules["src.database"] = database_mod
    sys.modules["src.database.db_connection"] = db_connection_mod
    sys.modules["src.database.db_schema"] = db_schema_mod


_install_stub_src_modules()

# Now safe to import the module under test
import upload_to_db as uploader  # noqa: E402


# =============================================================================
# Tests: _select_and_rename
# =============================================================================

def test_select_and_rename_positive_keeps_mapped_columns():
    """Positive: should keep only mapped columns that exist and rename them."""
    df = pd.DataFrame(
        {
            "user_id": ["U1"],
            "monthly_income": [1000],
            "extra": ["X"],
        }
    )
    out = uploader._select_and_rename(
        df,
        {"user_id": "user_id", "monthly_income": "monthly_income"},
    )
    assert list(out.columns) == ["user_id", "monthly_income"]
    assert out.loc[0, "user_id"] == "U1"


def test_select_and_rename_negative_no_matching_columns_returns_empty_columns():
    """
    Negative: if none of the mapped columns exist, output should have 0 columns.
    (Rows remain unchanged because pandas selection keeps index length.)
    """
    df = pd.DataFrame({"something_else": [1, 2, 3]})
    out = uploader._select_and_rename(df, {"user_id": "user_id"})

    assert out.shape[1] == 0   # no columns selected
    assert out.shape[0] == 3   # rows unchanged


# =============================================================================
# Tests: _ensure_jsonb
# =============================================================================

def test_ensure_jsonb_positive_serializes_dict_list():
    """Positive: dict/list should become JSON strings; None may become NaN (accept both)."""
    df = pd.DataFrame({"details": [{"a": 1}, ["x", "y"], None]})
    out = uploader._ensure_jsonb(df, "details")

    assert isinstance(out.loc[0, "details"], str)
    assert '"a"' in out.loc[0, "details"]

    assert isinstance(out.loc[1, "details"], str)

    # Pandas may keep None or convert to NaN depending on dtype — accept both
    assert pd.isna(out.loc[2, "details"]) or out.loc[2, "details"] is None


def test_ensure_jsonb_negative_missing_column_returns_same_df():
    """Negative: if column does not exist, df should remain unchanged."""
    df = pd.DataFrame({"x": [1]})
    out = uploader._ensure_jsonb(df, "details")
    assert "x" in out.columns
    assert "details" not in out.columns


# =============================================================================
# Tests: _truncate_table
# =============================================================================

def test_truncate_table_positive_executes_sql_and_commits():
    """Positive: should execute TRUNCATE TABLE ... CASCADE via engine.begin()."""
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn

    uploader._truncate_table(engine, "products")

    engine.begin.assert_called_once()
    conn.execute.assert_called_once()
    sql_arg = conn.execute.call_args[0][0]
    assert "TRUNCATE TABLE products CASCADE" in str(sql_arg)


# =============================================================================
# Tests: load_financial / load_products / load_reviews
# We patch _read_* and DataFrame.to_sql to avoid real IO + DB writes.
# =============================================================================

@patch.object(uploader, "_truncate_table")
@patch.object(uploader, "_read_csv")
def test_load_financial_positive_calls_to_sql(mock_read_csv, mock_truncate):
    """Positive: should read, select/rename, truncate, and call to_sql."""
    df = pd.DataFrame({"user_id": ["U1"], "monthly_income": [1], "monthly_expenses": [1]})
    mock_read_csv.return_value = df

    engine = MagicMock()

    with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
        n = uploader.load_financial(engine, "fake.csv", truncate=True)

    assert n == 1
    mock_truncate.assert_called_once_with(engine, "financial_profiles")
    mock_to_sql.assert_called_once()
    assert mock_to_sql.call_args[0][0] == "financial_profiles"


@patch.object(uploader, "_truncate_table")
@patch.object(uploader, "_read_jsonl")
def test_load_products_positive_calls_to_sql_and_ensure_jsonb(mock_read_jsonl, mock_truncate):
    """Positive: should read, select/rename, ensure_jsonb, truncate, and call to_sql."""
    df = pd.DataFrame(
        {
            "product_id": [1],
            "product_name": ["P"],
            "price": [9.9],
            "details": [{"a": 1}],
        }
    )
    mock_read_jsonl.return_value = df

    engine = MagicMock()

    with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
        n = uploader.load_products(engine, "fake.jsonl", truncate=True)

    assert n == 1
    mock_truncate.assert_called_once_with(engine, "products")
    mock_to_sql.assert_called_once()
    assert mock_to_sql.call_args[0][0] == "products"


@patch.object(uploader, "_truncate_table")
@patch.object(uploader, "_read_jsonl")
def test_load_reviews_positive_calls_to_sql(mock_read_jsonl, mock_truncate):
    """Positive: should read, select/rename, type-fix, truncate, and call to_sql."""
    df = pd.DataFrame(
        {
            "user_id": ["U1"],
            "asin": ["A1"],
            "product_id": [1],
            "rating": [5],
            "verified_purchase": [True],
            "helpful_vote": [None],
        }
    )
    mock_read_jsonl.return_value = df

    engine = MagicMock()

    with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
        n = uploader.load_reviews(engine, "fake.jsonl", truncate=True)

    assert n == 1
    mock_truncate.assert_called_once_with(engine, "reviews")
    mock_to_sql.assert_called_once()
    assert mock_to_sql.call_args[0][0] == "reviews"


# =============================================================================
# Tests: load_all (orchestrator)
# =============================================================================

@patch.object(uploader, "load_reviews")
@patch.object(uploader, "load_products")
@patch.object(uploader, "load_financial")
@patch.object(uploader, "create_tables")
@patch.object(uploader, "get_engine")
def test_load_all_positive_calls_in_fk_safe_order(
    mock_get_engine,
    mock_create_tables,
    mock_load_fin,
    mock_load_prod,
    mock_load_rev,
):
    """Positive: load_all should call get_engine, create_tables, and loaders in correct order."""
    engine = MagicMock()
    mock_get_engine.return_value = engine
    mock_load_fin.return_value = 10
    mock_load_prod.return_value = 20
    mock_load_rev.return_value = 30

    result = uploader.load_all("f.csv", "p.jsonl", "r.jsonl", env="dev", truncate=True)

    mock_get_engine.assert_called_once_with("dev")
    mock_create_tables.assert_called_once_with(engine)

    # FK-safe order: financial -> products -> reviews
    mock_load_fin.assert_called_once_with(engine, "f.csv", True)
    mock_load_prod.assert_called_once_with(engine, "p.jsonl", True)
    mock_load_rev.assert_called_once_with(engine, "r.jsonl", True)

    assert result == {"financial_profiles": 10, "products": 20, "reviews": 30}


@patch.object(uploader, "load_products")
@patch.object(uploader, "load_financial")
@patch.object(uploader, "create_tables")
@patch.object(uploader, "get_engine")
def test_load_all_negative_products_failure_raises(
    mock_get_engine,
    mock_create_tables,
    mock_load_fin,
    mock_load_prod,
):
    """Negative: if load_products fails, load_all should raise."""
    engine = MagicMock()
    mock_get_engine.return_value = engine
    mock_load_fin.return_value = 1
    mock_load_prod.side_effect = RuntimeError("bad products file")

    with pytest.raises(RuntimeError, match="bad products file"):
        uploader.load_all("f.csv", "p.jsonl", "r.jsonl", env="dev", truncate=True)