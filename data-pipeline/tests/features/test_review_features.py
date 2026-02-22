# tests/features/test_review_features.py
import os
import sys
import types
import json
import importlib.util

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

for _p in [
    os.path.join(PROJECT_ROOT, "dags", "src", "features"),
    os.path.join(PROJECT_ROOT, "dags", "src"),
]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Stub dependencies
# ---------------------------------------------------------------------------
if "utils" not in sys.modules:
    _utils = types.ModuleType("utils")
    _utils.setup_logging = lambda *a, **kw: None
    _utils.ensure_output_dir = lambda path: os.makedirs(os.path.dirname(path), exist_ok=True)
    sys.modules["utils"] = _utils

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "features", "review_features.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "review_features.py"),
        os.path.join(PROJECT_ROOT, "src", "features", "review_features.py"),
        os.path.join(PROJECT_ROOT, "review_features.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        try:
            spec = importlib.util.spec_from_file_location("review_features", fpath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["review_features"] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            raise ImportError(f"Found {fpath} but failed to load: {e}")
    raise ImportError("Could not find review_features.py. Searched:\n" + "\n".join(candidates))

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reviews_df(*rows):
    """Build a DataFrame from list of (product_id, rating) tuples."""
    return pd.DataFrame(rows, columns=["product_id", "rating"])

def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# =============================================================================
# 1) compute_rating_variance
# =============================================================================

def test_compute_rating_variance_basic():
    """Single product with multiple ratings → correct std dev."""
    df = _reviews_df(("p1", 1), ("p1", 3), ("p1", 5))
    out = M.compute_rating_variance(df)
    assert list(out.columns) == ["product_id", "rating_variance"]
    assert len(out) == 1
    expected = pd.Series([1, 3, 5], dtype=float).std()
    assert out.loc[0, "rating_variance"] == pytest.approx(expected)


def test_compute_rating_variance_multiple_products():
    """Multiple products → one row per product."""
    df = _reviews_df(("p1", 4), ("p1", 5), ("p2", 1), ("p2", 3), ("p2", 5))
    out = M.compute_rating_variance(df)
    assert len(out) == 2
    p1_var = out.loc[out["product_id"] == "p1", "rating_variance"].values[0]
    p2_var = out.loc[out["product_id"] == "p2", "rating_variance"].values[0]
    assert p1_var == pytest.approx(pd.Series([4, 5], dtype=float).std())
    assert p2_var == pytest.approx(pd.Series([1, 3, 5], dtype=float).std())


def test_compute_rating_variance_single_review_is_zero():
    """Product with only one review → variance defaults to 0.0 (not NaN)."""
    df = _reviews_df(("p1", 4))
    out = M.compute_rating_variance(df)
    assert out.loc[0, "rating_variance"] == pytest.approx(0.0)
    assert not out["rating_variance"].isna().any()


def test_compute_rating_variance_identical_ratings():
    """All ratings identical → variance is 0.0."""
    df = _reviews_df(("p1", 3), ("p1", 3), ("p1", 3))
    out = M.compute_rating_variance(df)
    assert out.loc[0, "rating_variance"] == pytest.approx(0.0)


def test_compute_rating_variance_missing_columns_raises():
    """Missing required columns → ValueError."""
    df = pd.DataFrame([{"product_id": "p1", "review_text": "great"}])
    with pytest.raises(ValueError, match="product_id"):
        M.compute_rating_variance(df)


def test_compute_rating_variance_missing_rating_raises():
    """Missing 'rating' column → ValueError."""
    df = pd.DataFrame([{"product_id": "p1"}])
    with pytest.raises(ValueError):
        M.compute_rating_variance(df)


def test_compute_rating_variance_output_columns():
    """Output must have exactly product_id and rating_variance columns."""
    df = _reviews_df(("p1", 3), ("p2", 5))
    out = M.compute_rating_variance(df)
    assert set(out.columns) == {"product_id", "rating_variance"}


def test_compute_rating_variance_no_nan_in_output():
    """No NaN values in rating_variance column after computation."""
    df = _reviews_df(("p1", 5), ("p2", 3), ("p3", 1))  # all single-review products
    out = M.compute_rating_variance(df)
    assert not out["rating_variance"].isna().any()


# =============================================================================
# 2) run_review_features (integration)
# =============================================================================

def test_run_review_features_missing_input(tmp_path):
    """Missing input file → logs error, no output file created, no exception."""
    bad = str(tmp_path / "nonexistent.jsonl")
    out = str(tmp_path / "out.csv")
    M.run_review_features(bad, out)  # must not raise
    assert not os.path.exists(out)


def test_run_review_features_creates_output(tmp_path):
    """Valid input → output CSV created with correct columns."""
    inp = tmp_path / "reviews.jsonl"
    out = tmp_path / "features" / "product_rating_variance.csv"

    _write_jsonl(inp, [
        {"product_id": "p1", "rating": 4},
        {"product_id": "p1", "rating": 5},
        {"product_id": "p2", "rating": 2},
    ])

    M.run_review_features(str(inp), str(out))

    assert out.exists()
    df_out = pd.read_csv(out)
    assert "product_id" in df_out.columns
    assert "rating_variance" in df_out.columns


def test_run_review_features_correct_product_count(tmp_path):
    """Output has one row per unique product."""
    inp = tmp_path / "reviews.jsonl"
    out = tmp_path / "features" / "out.csv"

    _write_jsonl(inp, [
        {"product_id": "p1", "rating": 3},
        {"product_id": "p1", "rating": 5},
        {"product_id": "p2", "rating": 1},
        {"product_id": "p3", "rating": 4},
    ])

    M.run_review_features(str(inp), str(out))
    df_out = pd.read_csv(out)
    assert len(df_out) == 3


def test_run_review_features_single_review_products_no_nan(tmp_path):
    """Single-review products in output must have 0.0 variance, not NaN."""
    inp = tmp_path / "reviews.jsonl"
    out = tmp_path / "features" / "out.csv"

    _write_jsonl(inp, [
        {"product_id": "p1", "rating": 5},
        {"product_id": "p2", "rating": 3},
    ])

    M.run_review_features(str(inp), str(out))
    df_out = pd.read_csv(out)
    assert not df_out["rating_variance"].isna().any()
    assert (df_out["rating_variance"] == 0.0).all()