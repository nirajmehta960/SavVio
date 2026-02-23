# tests/features/test_financial_features.py
import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

DAGS_SRC = os.path.join(PROJECT_ROOT, "dags", "src")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "dags", "src", "features")
for _p in (DAGS_SRC, FEATURES_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Stub dependencies at module level
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
        os.path.join(PROJECT_ROOT, "dags", "src", "features", "financial_features.py"),
        os.path.join(PROJECT_ROOT, "dags", "src", "financial_features.py"),
        os.path.join(PROJECT_ROOT, "src", "features", "financial_features.py"),
        os.path.join(PROJECT_ROOT, "financial_features.py"),
    ]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        try:
            spec = importlib.util.spec_from_file_location("financial_features", fpath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["financial_features"] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            raise ImportError(f"Found {fpath} but failed to load: {e}")
    raise ImportError(f"Could not find financial_features.py. Searched:\n" + "\n".join(candidates))

M = _load()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_df(**overrides):
    """Return a minimal valid DataFrame row as a dict, with optional overrides."""
    row = {
        "monthly_income": 5000.0,
        "monthly_expenses": 2000.0,
        "monthly_emi": 500.0,
        "savings_balance": 30000.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# =============================================================================
# 1) calculate_discretionary_income
# =============================================================================

def test_discretionary_income_basic():
    """income - (expenses + emi) = 5000 - 2500 = 2500"""
    df = _base_df()
    out = M.calculate_discretionary_income(df)
    assert "discretionary_income" in out.columns
    assert out.loc[0, "discretionary_income"] == pytest.approx(2500.0)


def test_discretionary_income_negative():
    """Expenses exceed income → negative discretionary income."""
    df = _base_df(monthly_income=1000.0, monthly_expenses=1200.0, monthly_emi=300.0)
    out = M.calculate_discretionary_income(df)
    assert out.loc[0, "discretionary_income"] == pytest.approx(-500.0)


def test_discretionary_income_zero_income():
    """Zero income → discretionary income equals -(expenses + emi)."""
    df = _base_df(monthly_income=0.0, monthly_expenses=500.0, monthly_emi=100.0)
    out = M.calculate_discretionary_income(df)
    assert out.loc[0, "discretionary_income"] == pytest.approx(-600.0)


# =============================================================================
# 2) calculate_ratios
# =============================================================================

def test_debt_to_income_ratio_basic():
    """emi / income = 500 / 5000 = 0.1"""
    df = _base_df()
    out = M.calculate_ratios(df)
    assert out.loc[0, "debt_to_income_ratio"] == pytest.approx(0.1)


def test_debt_to_income_ratio_zero_income():
    """Zero income → ratio is NaN."""
    df = _base_df(monthly_income=0.0)
    out = M.calculate_ratios(df)
    assert np.isnan(out.loc[0, "debt_to_income_ratio"])


def test_saving_to_income_ratio_basic():
    """savings / (income * 12) = 30000 / 60000 = 0.5"""
    df = _base_df()
    out = M.calculate_ratios(df)
    assert out.loc[0, "saving_to_income_ratio"] == pytest.approx(0.5)


def test_saving_to_income_ratio_zero_income():
    """Zero income → ratio is NaN."""
    df = _base_df(monthly_income=0.0)
    out = M.calculate_ratios(df)
    assert np.isnan(out.loc[0, "saving_to_income_ratio"])


def test_monthly_expense_burden_ratio_basic():
    """(expenses + emi) / income = 2500 / 5000 = 0.5"""
    df = _base_df()
    out = M.calculate_ratios(df)
    assert out.loc[0, "monthly_expense_burden_ratio"] == pytest.approx(0.5)


def test_monthly_expense_burden_ratio_zero_income():
    """Zero income → ratio is NaN."""
    df = _base_df(monthly_income=0.0)
    out = M.calculate_ratios(df)
    assert np.isnan(out.loc[0, "monthly_expense_burden_ratio"])


def test_financial_runway_basic():
    """savings / (expenses + emi) = 30000 / 2500 = 12.0"""
    df = _base_df()
    out = M.calculate_ratios(df)
    assert out.loc[0, "financial_runway"] == pytest.approx(12.0)


def test_financial_runway_zero_expenses():
    """Zero expenses + emi → runway is NaN (no outflows to divide by)."""
    df = _base_df(monthly_expenses=0.0, monthly_emi=0.0)
    out = M.calculate_ratios(df)
    assert np.isnan(out.loc[0, "financial_runway"])


def test_calculate_ratios_all_columns_present():
    """All four ratio columns must be created."""
    df = _base_df()
    out = M.calculate_ratios(df)
    for col in ["debt_to_income_ratio", "saving_to_income_ratio",
                "monthly_expense_burden_ratio", "financial_runway"]:
        assert col in out.columns


def test_calculate_ratios_multiple_rows():
    """Ratios computed correctly across multiple rows."""
    df = pd.DataFrame([
        {"monthly_income": 4000.0, "monthly_expenses": 1000.0, "monthly_emi": 200.0, "savings_balance": 24000.0},
        {"monthly_income": 0.0,    "monthly_expenses": 500.0,  "monthly_emi": 100.0, "savings_balance": 5000.0},
    ])
    out = M.calculate_ratios(df)
    assert out.loc[0, "debt_to_income_ratio"] == pytest.approx(200 / 4000)
    assert np.isnan(out.loc[1, "debt_to_income_ratio"])


# =============================================================================
# 3) run_financial_features (integration)
# =============================================================================

def test_run_financial_features_missing_input(tmp_path, caplog):
    """Missing input file should log an error and return without raising."""
    bad_path = str(tmp_path / "nonexistent.csv")
    out_path = str(tmp_path / "out.csv")
    M.run_financial_features(bad_path, out_path)  # should not raise
    assert not os.path.exists(out_path)


def test_run_financial_features_creates_output(tmp_path):
    """Valid input should produce an output CSV with expected columns."""
    inp = tmp_path / "financial_preprocessed.csv"
    out = tmp_path / "features" / "financial_featured.csv"

    pd.DataFrame([{
        "monthly_income": 6000.0,
        "monthly_expenses": 2000.0,
        "monthly_emi": 500.0,
        "savings_balance": 36000.0,
    }]).to_csv(inp, index=False)

    M.run_financial_features(str(inp), str(out))

    assert out.exists()
    df_out = pd.read_csv(out)
    for col in ["discretionary_income", "debt_to_income_ratio",
                "saving_to_income_ratio", "monthly_expense_burden_ratio", "financial_runway"]:
        assert col in df_out.columns


def test_run_financial_features_inf_replaced(tmp_path):
    """Inf values in ratio columns should be replaced with NaN in output."""
    inp = tmp_path / "fin.csv"
    out = tmp_path / "features" / "out.csv"

    # Zero income triggers NaN (not inf) via np.where, but zero expenses+emi
    # with nonzero savings could produce inf in financial_runway if not guarded.
    pd.DataFrame([{
        "monthly_income": 5000.0,
        "monthly_expenses": 0.0,
        "monthly_emi": 0.0,
        "savings_balance": 10000.0,
    }]).to_csv(inp, index=False)

    M.run_financial_features(str(inp), str(out))
    df_out = pd.read_csv(out)

    ratio_cols = ["debt_to_income_ratio", "saving_to_income_ratio",
                  "monthly_expense_burden_ratio", "financial_runway"]
    for col in ratio_cols:
        val = df_out.loc[0, col]
        assert not np.isinf(val) if not pd.isna(val) else True


def test_run_financial_features_discretionary_income_no_nan(tmp_path):
    """discretionary_income NaN values should be filled with 0.0 in output."""
    inp = tmp_path / "fin.csv"
    out = tmp_path / "features" / "out.csv"

    pd.DataFrame([{
        "monthly_income": float("nan"),
        "monthly_expenses": float("nan"),
        "monthly_emi": float("nan"),
        "savings_balance": 0.0,
    }]).to_csv(inp, index=False)

    M.run_financial_features(str(inp), str(out))
    df_out = pd.read_csv(out)
    assert not df_out["discretionary_income"].isna().any()