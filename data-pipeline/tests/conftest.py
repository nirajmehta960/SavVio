"""Shared test configuration for the data-pipeline test suite.

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
# sys.path — add all source directories so modules can resolve imports
# ---------------------------------------------------------------------------
_source_dirs = [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "dags"),
    DAGS_SRC,
    os.path.join(DAGS_SRC, "bias"),
    os.path.join(DAGS_SRC, "database"),
    os.path.join(DAGS_SRC, "features"),
    os.path.join(DAGS_SRC, "ingestion"),
    os.path.join(DAGS_SRC, "preprocess"),
    os.path.join(DAGS_SRC, "validation"),
    os.path.join(DAGS_SRC, "validation", "validate"),
    os.path.join(DAGS_SRC, "validation", "anomaly"),
]

for _d in _source_dirs:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)


# ---------------------------------------------------------------------------
# Shared ``utils`` module stub — attributes needed across bias, features,
# preprocess, and ingestion test suites.  Always create-or-extend so that
# collection order never causes missing-attribute ImportErrors.
# ---------------------------------------------------------------------------
_utils = sys.modules.get("utils", types.ModuleType("utils"))
_utils.setup_logging = lambda *a, **kw: None
_utils.get_processed_path = lambda f, **kw: f
_utils.get_features_path = lambda f, **kw: f
_utils.ensure_output_dir = lambda path: os.makedirs(
    os.path.dirname(path), exist_ok=True
)
sys.modules["utils"] = _utils

_src_utils = sys.modules.get("src.utils", types.ModuleType("src.utils"))
_src_utils.setup_logging = _utils.setup_logging
sys.modules["src.utils"] = _src_utils

_src = sys.modules.get("src", types.ModuleType("src"))
_src.utils = _src_utils
sys.modules["src"] = _src

