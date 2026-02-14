"""
Generate vector embeddings and store in pgvector.

Two embedding targets:
  1. Products  — name + category + description + features + details
                 → product_embeddings table (for product-level RAG retrieval)
  2. Reviews   — review_title + review_text
                 → review_embeddings table (for contextual utility analysis)

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim) by default.
Processes in batches to keep memory usage reasonable.
"""

import json
import logging
import pandas as pd
import numpy as np
from sqlalchemy import text

from db_config import get_engine, ensure_pgvector
from models import create_tables

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load the sentence-transformer model once."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def build_product_text(row: pd.Series) -> str:
    """
    Combine product text fields into a single string for embedding.
    Concatenates: name | category | description | features | details
    """
    parts = []
    for col in ["product_name", "category", "description", "features"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())

    # Flatten details (dict/JSON) into readable key-value pairs
    details = row.get("details", None)
    if details:
        details_str = _flatten_details(details)
        if details_str:
            parts.append(details_str)

    return " | ".join(parts)


def _flatten_details(details) -> str:
    """Convert details dict/JSON into a readable string for embedding."""
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except (json.JSONDecodeError, TypeError):
            return details.strip()
    if isinstance(details, dict):
        return ", ".join(f"{k}: {v}" for k, v in details.items() if v)
    return ""


def build_review_text(row: pd.Series) -> str:
    """
    Combine review title and body into a single string for embedding.
    Title provides a concise signal; body provides detail.
    """
    parts = []
    for col in ["review_title", "review_text"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Embedding generation (shared)
# ---------------------------------------------------------------------------

def generate_embeddings(texts: list[str], model) -> np.ndarray:
    """
    Encode a list of texts into embeddings, batched.
    Returns numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    logger.info("Generating embeddings for %d texts (batch_size=%d)", len(texts), BATCH_SIZE)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def _ensure_embedding_tables(engine):
    """Create product_embeddings and review_embeddings tables."""
    product_ddl = f"""
    CREATE TABLE IF NOT EXISTS product_embeddings (
        id          SERIAL PRIMARY KEY,
        product_id  VARCHAR(100) UNIQUE NOT NULL REFERENCES products(product_id),
        embedding   vector({EMBEDDING_DIM}),
        created_at  TIMESTAMP DEFAULT NOW()
    );
    """
    review_ddl = f"""
    CREATE TABLE IF NOT EXISTS review_embeddings (
        id          SERIAL PRIMARY KEY,
        product_id  VARCHAR(100) NOT NULL,
        user_id     VARCHAR(255) NOT NULL,
        embedding   vector({EMBEDDING_DIM}),
        created_at  TIMESTAMP DEFAULT NOW(),
        UNIQUE (product_id, user_id)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(product_ddl))
        conn.execute(text(review_ddl))
    logger.info("Embedding tables ensured (product_embeddings, review_embeddings)")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_product_embeddings(engine, product_ids: list[str], embeddings: np.ndarray):
    """Upsert product embeddings into product_embeddings table."""
    logger.info("Storing %d product embeddings", len(product_ids))
    sql = text("""
        INSERT INTO product_embeddings (product_id, embedding)
        VALUES (:pid, :emb::vector)
        ON CONFLICT (product_id) DO UPDATE SET embedding = EXCLUDED.embedding
    """)
    with engine.begin() as conn:
        for pid, emb in zip(product_ids, embeddings):
            conn.execute(sql, {"pid": str(pid), "emb": str(emb.tolist())})
    logger.info("Product embeddings stored")


def store_review_embeddings(engine, review_ids: list[int], product_ids: list[str], embeddings: np.ndarray):
    """Upsert review embeddings into review_embeddings table."""
    logger.info("Storing %d review embeddings", len(review_ids))
    sql = text("""
        INSERT INTO review_embeddings (review_id, product_id, embedding)
        VALUES (:rid, :pid, :emb::vector)
        ON CONFLICT (review_id) DO UPDATE SET embedding = EXCLUDED.embedding
    """)
    with engine.begin() as conn:
        for rid, pid, emb in zip(review_ids, product_ids, embeddings):
            conn.execute(sql, {
                "rid": int(rid),
                "pid": str(pid),
                "emb": str(emb.tolist()),
            })
    logger.info("Review embeddings stored")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def embed_products(engine, products_path: str, model):
    """Read processed products → embed → store."""
    df = _read_file(products_path)
    logger.info("Loaded %d products from %s", len(df), products_path)

    if "product_id" not in df.columns:
        raise ValueError("Products file must contain a 'product_id' column")

    df["_embed_text"] = df.apply(build_product_text, axis=1)
    df = df[df["_embed_text"].str.strip().astype(bool)].reset_index(drop=True)
    logger.info("Products with non-empty embedding text: %d", len(df))

    embeddings = generate_embeddings(df["_embed_text"].tolist(), model)
    store_product_embeddings(engine, df["product_id"].tolist(), embeddings)
    return len(df)


def embed_reviews(engine, reviews_path: str, model):
    """
    Read processed reviews → embed → store.
    Only embeds reviews that have text content (title or body).
    Reviews are linked to products via product_id for filtered retrieval.
    """
    df = _read_file(reviews_path)
    logger.info("Loaded %d reviews from %s", len(df), reviews_path)

    for col in ["product_id", "rating"]:
        if col not in df.columns:
            raise ValueError(f"Reviews file must contain a '{col}' column")

    df["_embed_text"] = df.apply(build_review_text, axis=1)

    # Only embed reviews that have actual text
    df = df[df["_embed_text"].str.strip().astype(bool)].reset_index(drop=True)
    logger.info("Reviews with text content to embed: %d", len(df))

    if df.empty:
        logger.warning("No reviews with text found — skipping review embeddings")
        return 0

    # We need the reviews table `id` (primary key) as foreign key.
    # Fetch id mapping from DB based on unique identifiers.
    review_ids = _get_review_ids(engine, df)
    if review_ids is None:
        logger.warning("Could not map reviews to DB ids — skipping")
        return 0

    embeddings = generate_embeddings(df["_embed_text"].tolist(), model)
    store_review_embeddings(engine, review_ids, df["product_id"].tolist(), embeddings)
    return len(df)


def _get_review_ids(engine, df: pd.DataFrame) -> list[int] | None:
    """
    Fetch the auto-generated review `id` from the reviews table.
    Matches on (user_id, product_id, rating) as a composite key.
    Returns list of DB ids in same order as df, or None on failure.
    """
    ids = []
    sql = text("""
        SELECT id FROM reviews
        WHERE user_id = :uid AND product_id = :pid AND rating = :rating
        LIMIT 1
    """)
    with engine.connect() as conn:
        for _, row in df.iterrows():
            result = conn.execute(sql, {
                "uid": str(row.get("user_id", "")),
                "pid": str(row["product_id"]),
                "rating": float(row["rating"]),
            })
            db_row = result.fetchone()
            if db_row:
                ids.append(db_row[0])
            else:
                ids.append(None)

    # Drop any that couldn't be matched
    matched = sum(1 for i in ids if i is not None)
    logger.info("Matched %d / %d reviews to DB ids", matched, len(ids))

    if matched == 0:
        return None
    return ids


# ---------------------------------------------------------------------------
# File reader helper
# ---------------------------------------------------------------------------

def _read_file(path: str) -> pd.DataFrame:
    """Read CSV or JSONL based on file extension."""
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_embed(
    products_path: str,
    reviews_path: str | None = None,
    env: str = "dev",
):
    """
    End-to-end embedding pipeline.

    Args:
        products_path: Path to processed products file (CSV or JSONL)
        reviews_path:  Path to processed reviews file (CSV or JSONL).
                       If None, only product embeddings are generated.
        env: "dev" or "prod"
    """
    engine = get_engine(env)
    ensure_pgvector(engine)
    create_tables(engine)
    _ensure_embedding_tables(engine)

    model = load_model()

    n_products = embed_products(engine, products_path, model)
    logger.info("Embedded %d products", n_products)

    n_reviews = 0
    if reviews_path:
        n_reviews = embed_reviews(engine, reviews_path, model)
        logger.info("Embedded %d reviews", n_reviews)
    else:
        logger.info("No reviews path provided — skipping review embeddings")

    summary = {"products_embedded": n_products, "reviews_embedded": n_reviews}
    logger.info("Embedding pipeline complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Generate & store embeddings (products + reviews)")
    parser.add_argument("--products", required=True, help="Path to processed products file")
    parser.add_argument("--reviews", default=None, help="Path to processed reviews file (optional)")
    parser.add_argument("--env", default="dev", choices=["dev", "prod"])
    args = parser.parse_args()

    result = run_embed(args.products, args.reviews, args.env)
    print("Done:", result)
