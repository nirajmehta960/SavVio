# tests/preprocess/test_run_preprocessing.py
import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

for _p in [
    os.path.join(PROJECT_ROOT, "dags", "src", "preprocess"),
    os.path.join(PROJECT_ROOT, "dags", "src"),
]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Stub preprocess sub-modules BEFORE loading run_preprocessing
# ---------------------------------------------------------------------------
def _stub_preprocess():
    for name in ("preprocess", "preprocess.financial", "preprocess.product", "preprocess.review"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    sys.modules["preprocess.financial"].main = MagicMock()
    sys.modules["preprocess.product"].main  = MagicMock()
    sys.modules["preprocess.review"].main   = MagicMock()

    # Make attributes accessible via the parent module
    sys.modules["preprocess"].financial = sys.modules["preprocess.financial"]
    sys.modules["preprocess"].product   = sys.modules["preprocess.product"]
    sys.modules["preprocess"].review    = sys.modules["preprocess.review"]

_stub_preprocess()

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "src", "preprocess", "run_preprocessing.py"),
        os.path.join(PROJECT_ROOT, "src", "preprocess", "run_preprocessing.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        spec = importlib.util.spec_from_file_location("run_preprocessing", fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run_preprocessing"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError("Could not find run_preprocessing.py. Searched:\n" + "\n".join(candidates))

M = _load()

# Shortcuts to the stubbed main() functions
_fin  = sys.modules["preprocess.financial"]
_prod = sys.modules["preprocess.product"]
_rev  = sys.modules["preprocess.review"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reset():
    """Reset all main() mocks before each test."""
    _fin.main.reset_mock()
    _prod.main.reset_mock()
    _rev.main.reset_mock()


# =============================================================================
# 1) run_pipeline
# =============================================================================

def test_run_pipeline_calls_all_three_mains():
    _reset()
    M.run_pipeline()
    _fin.main.assert_called_once()
    _prod.main.assert_called_once()
    _rev.main.assert_called_once()


def test_run_pipeline_calls_in_order():
    _reset()
    call_order = []
    _fin.main.side_effect  = lambda: call_order.append("financial")
    _prod.main.side_effect = lambda: call_order.append("product")
    _rev.main.side_effect  = lambda: call_order.append("review")

    M.run_pipeline()
    assert call_order == ["financial", "product", "review"]

    # restore
    _fin.main.side_effect  = None
    _prod.main.side_effect = None
    _rev.main.side_effect  = None


def test_run_pipeline_exits_on_financial_failure():
    _reset()
    _fin.main.side_effect = Exception("financial boom")
    with pytest.raises(SystemExit):
        M.run_pipeline()
    _fin.main.side_effect = None


def test_run_pipeline_exits_on_product_failure():
    _reset()
    _prod.main.side_effect = Exception("product boom")
    with pytest.raises(SystemExit):
        M.run_pipeline()
    _prod.main.side_effect = None


def test_run_pipeline_exits_on_review_failure():
    _reset()
    _rev.main.side_effect = Exception("review boom")
    with pytest.raises(SystemExit):
        M.run_pipeline()
    _rev.main.side_effect = None


def test_run_pipeline_product_not_called_if_financial_fails():
    _reset()
    _fin.main.side_effect = Exception("boom")
    with pytest.raises(SystemExit):
        M.run_pipeline()
    _prod.main.assert_not_called()
    _rev.main.assert_not_called()
    _fin.main.side_effect = None


def test_run_pipeline_review_not_called_if_product_fails():
    _reset()
    _prod.main.side_effect = Exception("boom")
    with pytest.raises(SystemExit):
        M.run_pipeline()
    _rev.main.assert_not_called()
    _prod.main.side_effect = None


# =============================================================================
# 2) preprocess_financial_task
# =============================================================================

def test_preprocess_financial_task_calls_main():
    _reset()
    M.preprocess_financial_task()
    _fin.main.assert_called_once()


def test_preprocess_financial_task_accepts_context():
    _reset()
    M.preprocess_financial_task(ti=MagicMock(), dag_run=MagicMock())
    _fin.main.assert_called_once()


def test_preprocess_financial_task_propagates_exception():
    _reset()
    _fin.main.side_effect = RuntimeError("oops")
    with pytest.raises(RuntimeError, match="oops"):
        M.preprocess_financial_task()
    _fin.main.side_effect = None


# =============================================================================
# 3) preprocess_product_task
# =============================================================================

def test_preprocess_product_task_calls_main():
    _reset()
    M.preprocess_product_task()
    _prod.main.assert_called_once()


def test_preprocess_product_task_accepts_context():
    _reset()
    M.preprocess_product_task(ti=MagicMock())
    _prod.main.assert_called_once()


def test_preprocess_product_task_propagates_exception():
    _reset()
    _prod.main.side_effect = RuntimeError("oops")
    with pytest.raises(RuntimeError, match="oops"):
        M.preprocess_product_task()
    _prod.main.side_effect = None


# =============================================================================
# 4) preprocess_review_task
# =============================================================================

def test_preprocess_review_task_calls_main():
    _reset()
    M.preprocess_review_task()
    _rev.main.assert_called_once()


def test_preprocess_review_task_accepts_context():
    _reset()
    M.preprocess_review_task(ti=MagicMock())
    _rev.main.assert_called_once()


def test_preprocess_review_task_propagates_exception():
    _reset()
    _rev.main.side_effect = RuntimeError("oops")
    with pytest.raises(RuntimeError, match="oops"):
        M.preprocess_review_task()
    _rev.main.side_effect = None