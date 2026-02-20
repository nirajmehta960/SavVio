import sys
import types
from unittest.mock import Mock
import pytest

from dags.src.connections.postgres import get_postgres_connection


def test_postgres_connection_success(monkeypatch):
    fake_psycopg2 = types.SimpleNamespace()
    fake_psycopg2.connect = Mock(return_value=object())

    # Inject fake psycopg2 so "import psycopg2" works
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)

    conn = get_postgres_connection()
    fake_psycopg2.connect.assert_called_once()
    assert conn is not None


def test_postgres_connection_failure(monkeypatch):
    fake_psycopg2 = types.SimpleNamespace()
    fake_psycopg2.connect = Mock(side_effect=Exception("DB error"))

    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)

    with pytest.raises(Exception) as e:
        get_postgres_connection()

    assert str(e.value) == "DB error"