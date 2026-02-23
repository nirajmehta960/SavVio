# tests/database/test_vector_embed.py
import os
import sys
import json
import types
import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

DAGS_SRC_DB = os.path.join(PROJECT_ROOT, "dags", "src", "database")
DAGS_SRC    = os.path.join(PROJECT_ROOT, "dags", "src")
for _p in (DAGS_SRC_DB, DAGS_SRC):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

if "db_config" not in sys.modules:
    _db = types.ModuleType("db_config")
    _db.get_engine = lambda env="dev": MagicMock(name=f"engine_{env}")
    _db.ensure_pgvector = lambda engine: None
    sys.modules["db_config"] = _db

if "models" not in sys.modules:
    _mo = types.ModuleType("models")
    _mo.create_tables = lambda engine: None
    sys.modules["models"] = _mo

if "sentence_transformers" not in sys.modules:
    _st = types.ModuleType("sentence_transformers")
    class _FakeST:
        def __init__(self, *a, **kw): pass
        def encode(self, sentences, **kw):
            return np.zeros((len(sentences), 384), dtype=np.float32)
    _st.SentenceTransformer = _FakeST
    sys.modules["sentence_transformers"] = _st

for _mod in ("sqlalchemy", "sqlalchemy.orm", "sqlalchemy.engine",
             "sqlalchemy.dialects", "sqlalchemy.dialects.postgresql"):
    if _mod not in sys.modules:
        _s = types.ModuleType(_mod)
        _s.text = lambda s: s
        _s.Column = MagicMock()
        _s.String = MagicMock()
        _s.Integer = MagicMock()
        _s.Float = MagicMock()
        sys.modules[_mod] = _s

for _mod in ("pgvector", "pgvector.sqlalchemy"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

def _load():
    fpath = os.path.join(DAGS_SRC_DB, "vector_embed.py")
    if not os.path.isfile(fpath):
        raise ImportError(f"Could not find vector_embed.py at {fpath}")
    spec = importlib.util.spec_from_file_location("vector_embed", fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vector_embed"] = mod
    spec.loader.exec_module(mod)
    return mod

M = _load()


# =============================================================================
# 1) _flatten_details tests
# =============================================================================

@pytest.mark.parametrize("input_val, expected_contains", [
    (None, ""),
    (np.nan, ""),
    ("", ""),
    ("  hello  ", "hello"),
    ('{"a": "1", "b": "x"}', "a: 1"),
    ('{"a": "1", "b": "x"}', "b: x"),
    ("{bad json", "{bad json"),
    ({"color": "red", "size": "M"}, "color: red"),
    ({"color": "red", "size": "M"}, "size: M"),
    (["a", "b"], ""),
    ([{"k": "v"}], ""),
])
def test_flatten_details_various_inputs(input_val, expected_contains):
    out = M._flatten_details(input_val)
    assert isinstance(out, str)
    if expected_contains == "":
        assert out.strip() == ""
    else:
        assert expected_contains in out


def test_flatten_details_dict_ignores_empty_values():
    out = M._flatten_details({"a": "", "b": None, "c": "ok"})
    assert "c: ok" in out
    assert "a:" not in out
    assert "b:" not in out


# =============================================================================
# 2) build_product_text tests
# =============================================================================

def test_build_product_text_concatenates_fields_and_details():
    row = pd.Series({
        "product_name": "Widget", "category": "Tools",
        "description": "A useful widget", "features": "Lightweight",
        "details": {"color": "blue"},
    })
    out = M.build_product_text(row)
    assert "Widget" in out
    assert "Tools" in out
    assert "color: blue" in out
    assert " | " in out


def test_build_product_text_skips_empty_fields():
    row = pd.Series({
        "product_name": "Widget", "category": "",
        "description": np.nan, "features": "   ", "details": None,
    })
    assert M.build_product_text(row) == "Widget"


# =============================================================================
# 3) build_review_text tests
# =============================================================================

def test_build_review_text_title_and_body_joined():
    row = pd.Series({"review_title": "Great", "review_text": "Loved it"})
    assert M.build_review_text(row) == "Great | Loved it"


def test_build_review_text_skips_empty_values():
    row = pd.Series({"review_title": "  ", "review_text": "Body"})
    assert M.build_review_text(row) == "Body"


# =============================================================================
# 4) _read_file tests
# =============================================================================

def test_read_file_csv(tmp_path):
    p = tmp_path / "products.csv"
    pd.DataFrame([{"a": 1, "b": "x"}]).to_csv(p, index=False)
    df = M._read_file(str(p))
    assert df.shape == (1, 2)
    assert df.loc[0, "a"] == 1


def test_read_file_jsonl(tmp_path):
    p = tmp_path / "reviews.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"a": 1, "b": "x"}) + "\n")
        f.write(json.dumps({"a": 2, "b": "y"}) + "\n")
    df = M._read_file(str(p))
    assert df.shape == (2, 2)
    assert df.loc[1, "b"] == "y"


# =============================================================================
# 5) generate_embeddings tests
# =============================================================================

def test_generate_embeddings_calls_model_encode_with_expected_args():
    fake_model = MagicMock()
    dim = getattr(M, "EMBEDDING_DIM", 384)
    fake_model.encode.return_value = np.zeros((2, dim), dtype=np.float32)

    out = M.generate_embeddings(["a", "b"], fake_model)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, dim)
    fake_model.encode.assert_called_once()


# =============================================================================
# 6) store_* embeddings tests
# =============================================================================

def _mock_engine_with_conn():
    engine, conn, cm = MagicMock(), MagicMock(), MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    engine.begin.return_value = cm
    return engine, conn


def test_store_product_embeddings_executes_db_calls():
    engine, conn = _mock_engine_with_conn()
    dim = getattr(M, "EMBEDDING_DIM", 384)
    M.store_product_embeddings(engine, ["p1","p2","p3"], np.zeros((3, dim), dtype=np.float32))
    assert conn.execute.call_count >= 1


def test_store_review_embeddings_executes_db_calls():
    engine, conn = _mock_engine_with_conn()
    dim = getattr(M, "EMBEDDING_DIM", 384)
    M.store_review_embeddings(engine, ["p1","p2"], ["u1","u2"], np.zeros((2, dim), dtype=np.float32))
    assert conn.execute.call_count >= 1


# =============================================================================
# 7) embed_products / embed_reviews tests
# =============================================================================

def test_embed_products_raises_without_product_id(tmp_path):
    p = tmp_path / "products.csv"
    pd.DataFrame([{"product_name": "x"}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="product_id"):
        M.embed_products(MagicMock(), str(p), MagicMock())


def test_embed_products_filters_empty_text_and_stores(tmp_path):
    p = tmp_path / "products.csv"
    pd.DataFrame([
        {"product_id": "p1", "product_name": "Widget", "category": "Tools"},
        {"product_id": "p2", "product_name": "   ", "category": ""},
    ]).to_csv(p, index=False)
    dim = getattr(M, "EMBEDDING_DIM", 384)
    with patch.object(M, "generate_embeddings", return_value=np.zeros((1, dim), dtype=np.float32)) as mg, \
         patch.object(M, "store_product_embeddings") as ms:
        n = M.embed_products(MagicMock(), str(p), MagicMock())
        assert n == 1
        mg.assert_called_once()
        assert ms.call_args[0][1] == ["p1"]


def test_embed_reviews_raises_without_required_columns(tmp_path):
    p = tmp_path / "reviews.csv"
    pd.DataFrame([{"review_text": "x"}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="product_id"):
        M.embed_reviews(MagicMock(), str(p), MagicMock())


def test_embed_reviews_skips_when_no_text(tmp_path):
    p = tmp_path / "reviews.csv"
    pd.DataFrame([
        {"product_id": "p1", "user_id": "u1", "review_title": "   ", "review_text": ""},
        {"product_id": "p2", "user_id": "u2", "review_title": None,  "review_text": "   "},
    ]).to_csv(p, index=False)
    with patch.object(M, "generate_embeddings") as mg, \
         patch.object(M, "store_review_embeddings") as ms:
        assert M.embed_reviews(MagicMock(), str(p), MagicMock()) == 0
        mg.assert_not_called()
        ms.assert_not_called()


def test_embed_reviews_filters_and_stores(tmp_path):
    p = tmp_path / "reviews.csv"
    pd.DataFrame([
        {"product_id": "p1", "user_id": "u1", "review_title": "Nice", "review_text": ""},
        {"product_id": "p2", "user_id": "u2", "review_title": "   ", "review_text": "   "},
    ]).to_csv(p, index=False)
    dim = getattr(M, "EMBEDDING_DIM", 384)
    with patch.object(M, "generate_embeddings", return_value=np.zeros((1, dim), dtype=np.float32)) as mg, \
         patch.object(M, "store_review_embeddings") as ms:
        n = M.embed_reviews(MagicMock(), str(p), MagicMock())
        assert n == 1
        mg.assert_called_once()
        assert ms.call_args[0][1] == ["p1"]
        assert ms.call_args[0][2] == ["u1"]