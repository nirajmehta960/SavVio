"""Shared test configuration for the data_pipeline test suite.

Centralizes sys.path setup and common module stubs so that individual
test files do not need to duplicate this boilerplate.  pytest loads
conftest.py before collecting tests, so the side-effects here
(sys.path inserts, sys.modules stubs) are available to every test module.
"""
import os
import sys
import types


# ---------------------------------------------------------------------------
# Path constants (importable by test files that need them for _load())
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAGS_SRC = os.path.join(PROJECT_ROOT, "dags", "src")


# ---------------------------------------------------------------------------
# sys.path — add the dags/ directory (package root) so "from src.xxx" works
# ---------------------------------------------------------------------------
_dags_dir = os.path.join(PROJECT_ROOT, "dags")
if _dags_dir not in sys.path:
    sys.path.insert(0, _dags_dir)


# ---------------------------------------------------------------------------
# Shared ``utils`` module stub — attributes needed across bias, features,
# preprocess, and ingestion test suites.  Always create-or-extend so that
# collection order never causes missing-attribute ImportErrors.
# ---------------------------------------------------------------------------
_src_utils = sys.modules.get("src.utils", types.ModuleType("src.utils"))
_src_utils.setup_logging = lambda *a, **kw: None
_src_utils.get_processed_path = lambda f, **kw: f
_src_utils.get_features_path = lambda f, **kw: f
_src_utils.ensure_output_dir = lambda path: os.makedirs(
    os.path.dirname(path), exist_ok=True
)
sys.modules["src.utils"] = _src_utils

# Keep a bare "utils" stub for any tests that import "utils" directly
_utils = sys.modules.get("utils", types.ModuleType("utils"))
_utils.setup_logging = _src_utils.setup_logging
_utils.get_processed_path = _src_utils.get_processed_path
_utils.get_features_path = _src_utils.get_features_path
_utils.ensure_output_dir = _src_utils.ensure_output_dir
sys.modules["utils"] = _utils

_src = sys.modules.get("src", types.ModuleType("src"))
_src.utils = _src_utils
sys.modules["src"] = _src
