"""
GCP connection module for unit tests WITHOUT real GCP dependencies.

Tests expect:
- gcp.storage.Client
- gcp.bigquery.Client
- get_gcs_client()
- get_bigquery_client()

We provide stubs so no google-cloud-* packages are required.
"""

from __future__ import annotations
from typing import Any, Optional


class _StubClient:
    def __init__(self, service: str) -> None:
        self.service = service

    def __repr__(self) -> str:
        return f"<StubGCPClient service={self.service!r}>"


class _StorageModule:
    class Client(_StubClient):
        def __init__(self) -> None:
            super().__init__("gcs")


class _BigQueryModule:
    class Client(_StubClient):
        def __init__(self) -> None:
            super().__init__("bigquery")


# Expose attributes tests import
storage = _StorageModule()
bigquery = _BigQueryModule()


def get_gcs_client() -> Optional[Any]:
    return storage.Client()


def get_bigquery_client() -> Optional[Any]:
    return bigquery.Client()