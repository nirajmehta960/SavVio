import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# --- Magic trick: Add paths so imports work in tests ---
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DAGS_ROOT = os.path.join(PROJECT_ROOT, "dags")
DB_DIR = os.path.join(DAGS_ROOT, "src", "database")  # where run_database.py lives

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DAGS_ROOT)
sys.path.insert(0, DB_DIR)  # IMPORTANT: supports "from upload_to_db import ..." style imports


# =============================================================================
# --- Stub dependencies BEFORE importing run_database ---
# This prevents import-time failures like: ModuleNotFoundError: db_config
# =============================================================================

def _install_stub_modules():
    """
    Create fake modules that run_database imports:
      - upload_to_db
      - db_connection
      - db_schema
      - vector_embed

    Insert them into sys.modules so Python uses these stubs during import.
    """
    # --- upload_to_db stub ---
    upload_to_db = types.ModuleType("upload_to_db")
    upload_to_db.load_financial = MagicMock(name="load_financial")
    upload_to_db.load_products = MagicMock(name="load_products")
    upload_to_db.load_reviews = MagicMock(name="load_reviews")
    upload_to_db.load_all = MagicMock(name="load_all")

    # --- db_connection stub ---
    db_connection = types.ModuleType("db_connection")
    db_connection.get_engine = MagicMock(name="get_engine")
    db_connection.ensure_pgvector = MagicMock(name="ensure_pgvector")

    # --- db_schema stub ---
    db_schema = types.ModuleType("db_schema")
    db_schema.create_tables = MagicMock(name="create_tables")

    # --- vector_embed stub (this avoids importing real vector_embed.py which imports db_config) ---
    vector_embed = types.ModuleType("vector_embed")
    vector_embed.load_model = MagicMock(name="load_model")
    vector_embed.embed_products = MagicMock(name="embed_products")
    vector_embed.embed_reviews = MagicMock(name="embed_reviews")
    vector_embed._ensure_embedding_tables = MagicMock(name="_ensure_embedding_tables")

    sys.modules["upload_to_db"] = upload_to_db
    sys.modules["db_connection"] = db_connection
    sys.modules["db_schema"] = db_schema
    sys.modules["vector_embed"] = vector_embed


_install_stub_modules()

# Now import the module under test (safe now)
import dags.src.database.run_database as orchestrator  # noqa: E402


# =============================================================================
# Helpers
# =============================================================================

def _fake_setup(tmp_path):
    """
    Return (data_dir, engine) like _setup().
    Creates a fake folder structure:
      data/features/...
    """
    data_dir = tmp_path / "data"
    (data_dir / "features").mkdir(parents=True, exist_ok=True)
    engine = MagicMock(name="engine")
    return str(data_dir), engine


# =============================================================================
# Tests for _setup
# =============================================================================

@patch.object(orchestrator, "create_tables")
@patch.object(orchestrator, "get_engine")
def test_setup_positive_calls_engine_and_create_tables(mock_get_engine, mock_create_tables):
    """Positive: _setup() should call get_engine() and create_tables(engine)."""
    mock_engine = MagicMock(name="engine")
    mock_get_engine.return_value = mock_engine

    data_dir, engine = orchestrator._setup()

    assert isinstance(data_dir, str)
    assert engine is mock_engine
    mock_get_engine.assert_called_once()
    mock_create_tables.assert_called_once_with(mock_engine)


@patch.object(orchestrator, "get_engine")
def test_setup_negative_engine_failure_propagates(mock_get_engine):
    """Negative: if get_engine() fails, _setup() should raise."""
    mock_get_engine.side_effect = RuntimeError("DB down")

    with pytest.raises(RuntimeError, match="DB down"):
        orchestrator._setup()


# =============================================================================
# Tests for load_*_task wrappers
# =============================================================================

@patch.object(orchestrator, "load_financial")
@patch.object(orchestrator, "_setup")
def test_load_financial_task_positive(mock_setup, mock_load_financial, tmp_path):
    """Positive: load_financial_task should call load_financial(engine, correct_path)."""
    data_dir, engine = _fake_setup(tmp_path)
    mock_setup.return_value = (data_dir, engine)

    orchestrator.load_financial_task()

    mock_setup.assert_called_once()
    mock_load_financial.assert_called_once_with(
        engine,
        os.path.join(data_dir, "features/financial_featured.csv"),
    )


@patch.object(orchestrator, "load_financial")
@patch.object(orchestrator, "_setup")
def test_load_financial_task_negative_loader_failure(mock_setup, mock_load_financial, tmp_path):
    """Negative: if load_financial raises, task should raise (Airflow should mark task failed)."""
    data_dir, engine = _fake_setup(tmp_path)
    mock_setup.return_value = (data_dir, engine)
    mock_load_financial.side_effect = ValueError("bad csv")

    with pytest.raises(ValueError, match="bad csv"):
        orchestrator.load_financial_task()


@patch.object(orchestrator, "load_products")
@patch.object(orchestrator, "_setup")
def test_load_products_task_positive(mock_setup, mock_load_products, tmp_path):
    """Positive: load_products_task should call load_products(engine, correct_path)."""
    data_dir, engine = _fake_setup(tmp_path)
    mock_setup.return_value = (data_dir, engine)

    orchestrator.load_products_task()

    mock_setup.assert_called_once()
    mock_load_products.assert_called_once_with(
        engine,
        os.path.join(data_dir, "features/products_featured.jsonl"),
    )


@patch.object(orchestrator, "load_reviews")
@patch.object(orchestrator, "_setup")
def test_load_reviews_task_positive(mock_setup, mock_load_reviews, tmp_path):
    """Positive: load_reviews_task should call load_reviews(engine, correct_path)."""
    data_dir, engine = _fake_setup(tmp_path)
    mock_setup.return_value = (data_dir, engine)

    orchestrator.load_reviews_task()

    mock_setup.assert_called_once()
    mock_load_reviews.assert_called_once_with(
        engine,
        os.path.join(data_dir, "features/reviews_featured.jsonl"),
    )


# =============================================================================
# Tests for generate_and_load_embedding_task
# =============================================================================

@patch.object(orchestrator, "embed_reviews")
@patch.object(orchestrator, "embed_products")
@patch.object(orchestrator, "load_model")
@patch.object(orchestrator, "_ensure_embedding_tables")
@patch.object(orchestrator, "ensure_pgvector")
@patch.object(orchestrator, "_setup")
def test_generate_and_load_embedding_task_positive(
    mock_setup,
    mock_ensure_pgvector,
    mock_ensure_embed_tables,
    mock_load_model,
    mock_embed_products,
    mock_embed_reviews,
    tmp_path,
):
    """Positive: should setup db, ensure pgvector/tables, load model, then embed products and reviews."""
    data_dir, engine = _fake_setup(tmp_path)
    mock_setup.return_value = (data_dir, engine)

    model = MagicMock(name="model")
    mock_load_model.return_value = model
    mock_embed_products.return_value = 123
    mock_embed_reviews.return_value = 456

    orchestrator.generate_and_load_embedding_task()

    mock_setup.assert_called_once()
    mock_ensure_pgvector.assert_called_once_with(engine)
    mock_ensure_embed_tables.assert_called_once_with(engine)
    mock_load_model.assert_called_once()

    mock_embed_products.assert_called_once_with(
        engine,
        os.path.join(data_dir, "features/products_featured.jsonl"),
        model,
    )
    mock_embed_reviews.assert_called_once_with(
        engine,
        os.path.join(data_dir, "features/reviews_featured.jsonl"),
        model,
    )


@patch.object(orchestrator, "ensure_pgvector")
@patch.object(orchestrator, "_setup")
def test_generate_and_load_embedding_task_negative_pgvector_failure(mock_setup, mock_ensure_pgvector, tmp_path):
    """Negative: if ensure_pgvector fails, task should raise."""
    data_dir, engine = _fake_setup(tmp_path)
    mock_setup.return_value = (data_dir, engine)
    mock_ensure_pgvector.side_effect = RuntimeError("no extension rights")

    with pytest.raises(RuntimeError, match="no extension rights"):
        orchestrator.generate_and_load_embedding_task()