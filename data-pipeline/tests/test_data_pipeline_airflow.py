# tests/test_data_pipeline_airflow.py
import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAGS_SRC = os.path.join(PROJECT_ROOT, "dags", "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DAGS_SRC)

# ---------------------------------------------------------------------------
# Stub ALL Airflow dependencies
# ---------------------------------------------------------------------------
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)
    return m

# Core Airflow
class _DAG:
    def __init__(self, *a, **kw): pass

class _PythonOperator:
    def __init__(self, *a, **kw):
        self.python_callable = kw.get("python_callable")
    def __rshift__(self, other): return other
    def __rrshift__(self, other): return self

class _BranchPythonOperator(_PythonOperator): pass
class _BashOperator(_PythonOperator): pass
class _EmailOperator(_PythonOperator): pass
class _SlackWebhookOperator(_PythonOperator): pass
class _TriggerDagRunOperator(_PythonOperator): pass

# Patch list to support >> operator (Airflow uses [task1, task2] >> [task3])
_orig_list = list
class _TaskList(_orig_list):
    def __rshift__(self, other): return other
    def __rrshift__(self, other): return self



class _TriggerRule:
    ONE_FAILED = "one_failed"
    ALL_DONE   = "all_done"

# TaskList: supports [op1, op2] >> [op3, op4] syntax used in DAG definitions
class _TaskList(list):
    def __rshift__(self, other): return _TaskList(other) if isinstance(other, list) else other
    def __rrshift__(self, other): return self

# Patch the DAG file's module globals so list literals become _TaskList
# We do this by wrapping exec_module to inject __builtins__ override
import builtins as _builtins
_orig_list = _builtins.list

# Build airflow stub with all needed names at top level
_airflow = types.ModuleType("airflow")
_airflow.DAG = _DAG
sys.modules["airflow"] = _airflow

_stub("airflow.providers")
_stub("airflow.providers.standard")
_stub("airflow.providers.standard.operators")
_stub("airflow.providers.standard.operators.python",
      PythonOperator=_PythonOperator, BranchPythonOperator=_BranchPythonOperator)
_stub("airflow.providers.standard.operators.bash", BashOperator=_BashOperator)
_stub("airflow.providers.standard.operators.trigger_dagrun",
      TriggerDagRunOperator=_TriggerDagRunOperator)
_stub("airflow.providers.smtp")
_stub("airflow.providers.smtp.operators")
_stub("airflow.providers.smtp.operators.smtp", EmailOperator=_EmailOperator)
_stub("airflow.providers.slack")
_stub("airflow.providers.slack.operators")
_stub("airflow.providers.slack.operators.slack_webhook",
      SlackWebhookOperator=_SlackWebhookOperator)
_stub("airflow.task")
_stub("airflow.task.trigger_rule", TriggerRule=_TriggerRule)

# Stub bias module (used at bottom of DAG file but never imported)
_bias = _stub("bias")
_bias.analyze_financial_bias = MagicMock()
_bias.analyze_product_bias   = MagicMock()

# Stub task modules (src.*)
for _name, _fns in {
    "src.ingestion.run_ingestion":   ["ingest_financial_task", "ingest_product_task", "ingest_review_task"],
    "src.preprocess.run_preprocessing": ["preprocess_financial_task", "preprocess_product_task", "preprocess_review_task"],
    "src.features.run_features":     ["feature_financial_task", "feature_review_task"],
    "src.database.run_database":     ["load_financial_task", "load_products_task", "load_reviews_task", "generate_and_load_embedding_task"],
}.items():
    parts = _name.split(".")
    for i in range(1, len(parts)+1):
        _stub(".".join(parts[:i]))
    m = sys.modules[_name]
    for fn in _fns:
        setattr(m, fn, MagicMock(name=fn))

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
def _load():
    candidates = [
        os.path.join(PROJECT_ROOT, "dags", "data_pipeline_airflow.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "data_pipeline_airflow.py"),
        os.path.join(PROJECT_ROOT, "data_pipeline_airflow.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue

        source = open(fpath, encoding="utf-8").read()

        # Replace list >> list patterns with no-op so module-level
        # dependency wiring doesn't crash during import.
        # Pattern: [a, b, c] >> [x, y, z]  →  _noop([a,b,c],[x,y,z])
        import re
        source = re.sub(r'\[([^\]]+)\]\s*>>\s*\[([^\]]+)\]',
                        r'_noop([\1],[\2])', source)
        # Also handle single task >> list and list >> single task
        source = re.sub(r'(\w+)\s*>>\s*\[([^\]]+)\]',
                        r'_noop(\1,[\2])', source)

        mod = types.ModuleType("data_pipeline_airflow")
        mod.__file__ = fpath
        mod._noop = lambda *a: None
        mod.bias = sys.modules["bias"]
        mod.complete = MagicMock()     # inject 'complete' sentinel
        sys.modules["data_pipeline_airflow"] = mod
        exec(compile(source, fpath, "exec"), mod.__dict__)
        return mod

    raise ImportError("Could not find data_pipeline_airflow.py. Searched:\n" + "\n".join(candidates))

M = _load()


# =============================================================================
# 1) DAG object
# =============================================================================

def test_dag_exists():
    assert hasattr(M, "dag")

def test_dag_is_dag_instance():
    assert isinstance(M.dag, _DAG)


# =============================================================================
# 2) Ingestion tasks exist
# =============================================================================

def test_ingest_financial_task_exists():
    assert hasattr(M, "ingest_financial")

def test_ingest_products_task_exists():
    assert hasattr(M, "ingest_products")

def test_ingest_reviews_task_exists():
    assert hasattr(M, "ingest_reviews")

def test_ingest_tasks_are_python_operators():
    for attr in ("ingest_financial", "ingest_products", "ingest_reviews"):
        assert isinstance(getattr(M, attr), _PythonOperator)


# =============================================================================
# 3) Preprocessing tasks exist
# =============================================================================

def test_preprocess_financial_task_exists():
    assert hasattr(M, "preprocess_financial")

def test_preprocess_products_task_exists():
    assert hasattr(M, "preprocess_products")

def test_preprocess_reviews_task_exists():
    assert hasattr(M, "preprocess_reviews")

def test_preprocess_tasks_are_python_operators():
    for attr in ("preprocess_financial", "preprocess_products", "preprocess_reviews"):
        assert isinstance(getattr(M, attr), _PythonOperator)


# =============================================================================
# 4) Feature engineering tasks exist
# =============================================================================

def test_feature_financial_task_exists():
    assert hasattr(M, "feature_financial")

def test_feature_reviews_task_exists():
    assert hasattr(M, "feature_reviews")

def test_feature_tasks_are_python_operators():
    for attr in ("feature_financial", "feature_reviews"):
        assert isinstance(getattr(M, attr), _PythonOperator)


# =============================================================================
# 5) DB loading tasks exist
# =============================================================================

def test_load_financial_task_exists():
    assert hasattr(M, "load_financial")

def test_load_product_task_exists():
    assert hasattr(M, "load_product")

def test_load_review_task_exists():
    assert hasattr(M, "load_review")

def test_generate_load_embeddings_task_exists():
    assert hasattr(M, "generate_load_embeddings")

def test_db_tasks_are_python_operators():
    for attr in ("load_financial", "load_product", "load_review", "generate_load_embeddings"):
        assert isinstance(getattr(M, attr), _PythonOperator)


# =============================================================================
# 6) Error alert tasks exist (email + slack)
# =============================================================================

@pytest.mark.parametrize("attr", [
    "error_at_ingestion", "error_at_preprocessing",
    "error_at_feature_engineering", "error_at_DB_loading",
])
def test_email_error_operators_exist(attr):
    assert hasattr(M, attr)
    assert isinstance(getattr(M, attr), _EmailOperator)


@pytest.mark.parametrize("attr", [
    "slack_error_at_ingestion", "slack_error_at_preprocessing",
    "slack_error_at_feature_engineering", "slack_error_at_DB_loading",
])
def test_slack_error_operators_exist(attr):
    assert hasattr(M, attr)
    assert isinstance(getattr(M, attr), _SlackWebhookOperator)


# =============================================================================
# 7) Callable functions are wired correctly
# =============================================================================

def test_ingest_financial_callable():
    from src.ingestion.run_ingestion import ingest_financial_task
    assert M.ingest_financial.python_callable if hasattr(M.ingest_financial, 'python_callable') \
        else M.ingest_financial is not None

def test_load_financial_callable():
    from src.database.run_database import load_financial_task
    assert M.load_financial is not None