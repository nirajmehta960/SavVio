"""
Database Loading Orchestrator – Airflow task wrappers.

Wraps the upload_to_db and vector_embed modules so Airflow
PythonOperator can call them with a consistent *_task(**context) interface.
"""

import os
import logging
import sys

# Add current script directory to import path (mirrors run_features.py pattern).
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from upload_to_db import load_financial, load_products, load_reviews, load_all
from db_connection import get_engine, ensure_pgvector
from db_schema import create_tables
from vector_embed import (
    load_model,
    embed_products,
    embed_reviews,
    _ensure_embedding_tables,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared setup — engine, tables, data path (called once per task)
# ---------------------------------------------------------------------------

def _setup():
    """Create engine, ensure tables exist, return (data_dir, engine)."""
    # __file__ = .../dags/src/database/run_database.py
    # We need 3 dirname() calls to reach .../dags/
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base, "data")
    engine = get_engine()
    try:
        create_tables(engine)
    except Exception as exc:
        # Parallel tasks may race to CREATE TABLE; ignore "already exists"
        if "already exists" in str(exc):
            logger.warning("Tables already exist (parallel race) — continuing")
        else:
            raise
    return data_dir, engine


# ---------------------------------------------------------------------------
# Airflow task wrappers
# ---------------------------------------------------------------------------

#TODO: data added to DB might change after bias detection and mitigation

def load_financial_task(**context):
    """Airflow task: load financial profiles into PostgreSQL."""
    logger.info(">>> Loading Financial Profiles into PostgreSQL...")
    data, engine = _setup()
    load_financial(engine, os.path.join(data, "features/financial_featured.csv"))
    logger.info(">>> Financial Profiles Load: SUCCESS")


def load_products_task(**context):
    """Airflow task: load products into PostgreSQL."""
    logger.info(">>> Loading Products into PostgreSQL...")
    data, engine = _setup()
    load_products(engine, os.path.join(data, "features/products_featured.jsonl"))
    logger.info(">>> Products Load: SUCCESS")


def load_reviews_task(**context):
    """Airflow task: load reviews into PostgreSQL."""
    logger.info(">>> Loading Reviews into PostgreSQL...")
    data, engine = _setup()
    load_reviews(engine, os.path.join(data, "features/reviews_featured.jsonl"))
    logger.info(">>> Reviews Load: SUCCESS")


def generate_and_load_embedding_task(**context):
    """Airflow task: generate embeddings for products & reviews, store in pgvector."""
    logger.info(">>> Generating Embeddings and Loading into pgvector...")
    data, engine = _setup()
    ensure_pgvector(engine)
    _ensure_embedding_tables(engine)

    model = load_model()

    n_prod = embed_products(
        engine,
        os.path.join(data, "features/products_featured.jsonl"),
        model,
    )
    logger.info("Embedded %d products", n_prod)

    n_rev = embed_reviews(
        engine,
        os.path.join(data, "features/reviews_featured.jsonl"),
        model,
    )
    logger.info("Embedded %d reviews", n_rev)
    logger.info(">>> Embedding Generation & Load: SUCCESS")


# ---------------------------------------------------------------------------
# CLI – optional, for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Running full database loading pipeline...")
    load_financial_task()
    load_products_task()
    load_reviews_task()
    generate_and_load_embedding_task()
    logger.info("All done.")
