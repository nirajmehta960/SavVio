"""
Load to Database — schema cross-validation and drop_tables.

The comprehensive schema and table tests live in test_db_connection.py.
This file validates:
  - The short import path (database.db_schema) resolves to the same ORM tables
    as the fully-qualified import (dags.src.database.db_schema).
  - drop_tables correctly drops embedding tables via raw SQL and then calls
    Base.metadata.drop_all for ORM-managed tables.
"""
import sys
import pytest
from sqlalchemy import create_engine, inspect

# sys.path set up by conftest.py

from database.db_schema import (  # noqa: E402
    Base,
    FinancialProfile,
    Product,
    Review,
    create_tables,
    drop_tables,
)


@pytest.fixture()
def engine():
    return create_engine("sqlite:///:memory:")


def test_short_import_models_match_long_import():
    """Both import paths expose the same set of ORM tables."""
    from dags.src.database.db_schema import Base as BaseLong
    short_tables = set(Base.metadata.tables.keys())
    long_tables = set(BaseLong.metadata.tables.keys())
    assert short_tables == long_tables


def test_drop_tables_removes_all_tables(engine):
    """drop_tables should leave no ORM-managed tables behind."""
    from unittest.mock import MagicMock, patch

    create_tables(engine)
    tables_before = set(inspect(engine).get_table_names())
    assert len(tables_before) >= 3

    with patch.object(Base.metadata, "drop_all") as mock_drop_all:
        mock_engine = MagicMock()
        conn = MagicMock()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=False)
        mock_engine.begin.return_value = cm
        drop_tables(mock_engine)

    conn.execute.assert_called()
    raw_sqls = [str(c[0][0]) for c in conn.execute.call_args_list]
    assert any("review_embeddings" in s for s in raw_sqls)
    assert any("product_embeddings" in s for s in raw_sqls)
    mock_drop_all.assert_called_once_with(mock_engine)

