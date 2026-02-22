# tests/features/test_run_features.py
import os
import sys
import types
import importlib.util
from unittest.mock import patch, MagicMock

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
# Stub ALL dependencies before loading module
# ---------------------------------------------------------------------------
for _name, _attrs in {
    "utils": {"setup_logging": lambda *a, **kw: None, "ensure_output_dir": lambda p: None},
    "financial_features": {"run_financial_features": MagicMock()},
    "review_features": {"run_review_features": MagicMock()},
}.items():
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        for k, v in _attrs.items():
            setattr(_m, k, v)
        sys.modules[_name] = _m

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "features", "run_features.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "run_features.py"),
        os.path.join(PROJECT_ROOT, "src", "features", "run_features.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("run_features", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run_features"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find run_features.py. Searched:\n" + "\n".join(candidates))

M = _load()


# =============================================================================
# 1) feature_financial_task
# =============================================================================

def test_feature_financial_task_calls_run_financial_features():
    """feature_financial_task must call run_financial_features exactly once."""
    with patch.object(M, "run_financial_features") as mock_fn:
        M.feature_financial_task()
        mock_fn.assert_called_once()


def test_feature_financial_task_passes_correct_paths():
    """feature_financial_task passes input_path and output_path kwargs."""
    with patch.object(M, "run_financial_features") as mock_fn:
        M.feature_financial_task()
        _, kwargs = mock_fn.call_args
        assert "input_path" in kwargs
        assert "output_path" in kwargs
        assert kwargs["input_path"].endswith("financial_preprocessed.csv")
        assert kwargs["output_path"].endswith("financial_featured.csv")


# =============================================================================
# 2) feature_review_task
# =============================================================================

def test_feature_review_task_calls_run_review_features():
    """feature_review_task must call run_review_features exactly once."""
    with patch.object(M, "run_review_features") as mock_fn:
        M.feature_review_task()
        mock_fn.assert_called_once()


def test_feature_review_task_passes_correct_paths():
    """feature_review_task passes reviews_path and output_path kwargs."""
    with patch.object(M, "run_review_features") as mock_fn:
        M.feature_review_task()
        _, kwargs = mock_fn.call_args
        assert "reviews_path" in kwargs
        assert "output_path" in kwargs
        assert kwargs["reviews_path"].endswith("review_preprocessed.jsonl")
        assert kwargs["output_path"].endswith("product_rating_variance.csv")


# =============================================================================
# 3) main()
# =============================================================================

def test_main_calls_both_features_by_default():
    """main() with no args calls both financial and review features."""
    with patch.object(M, "run_financial_features") as mock_fin, \
         patch.object(M, "run_review_features") as mock_rev, \
         patch("sys.argv", ["run_features.py"]):
        M.main()
        mock_fin.assert_called_once()
        mock_rev.assert_called_once()


def test_main_skip_financial():
    """--skip-financial skips financial features but runs review features."""
    with patch.object(M, "run_financial_features") as mock_fin, \
         patch.object(M, "run_review_features") as mock_rev, \
         patch("sys.argv", ["run_features.py", "--skip-financial"]):
        M.main()
        mock_fin.assert_not_called()
        mock_rev.assert_called_once()


def test_main_skip_reviews():
    """--skip-reviews skips review features but runs financial features."""
    with patch.object(M, "run_financial_features") as mock_fin, \
         patch.object(M, "run_review_features") as mock_rev, \
         patch("sys.argv", ["run_features.py", "--skip-reviews"]):
        M.main()
        mock_fin.assert_called_once()
        mock_rev.assert_not_called()


def test_main_skip_both():
    """--skip-financial --skip-reviews skips both."""
    with patch.object(M, "run_financial_features") as mock_fin, \
         patch.object(M, "run_review_features") as mock_rev, \
         patch("sys.argv", ["run_features.py", "--skip-financial", "--skip-reviews"]):
        M.main()
        mock_fin.assert_not_called()
        mock_rev.assert_not_called()


def test_main_financial_exception_does_not_crash():
    """If financial features raise, main() should catch and continue (not raise)."""
    with patch.object(M, "run_financial_features", side_effect=Exception("boom")), \
         patch.object(M, "run_review_features") as mock_rev, \
         patch("sys.argv", ["run_features.py"]):
        M.main()  # must not raise
        mock_rev.assert_called_once()


def test_main_review_exception_does_not_crash():
    """If review features raise, main() should catch and continue (not raise)."""
    with patch.object(M, "run_financial_features") as mock_fin, \
         patch.object(M, "run_review_features", side_effect=Exception("boom")), \
         patch("sys.argv", ["run_features.py"]):
        M.main()  # must not raise
        mock_fin.assert_called_once()