from unittest.mock import patch
from dags.src.connections.gcp import get_gcs_client, get_bigquery_client


@patch("dags.src.connections.gcp.storage.Client")
def test_gcs_connection(mock_client):
    client = get_gcs_client()
    mock_client.assert_called_once()
    assert client is not None


@patch("dags.src.connections.gcp.bigquery.Client")
def test_bigquery_connection(mock_client):
    client = get_bigquery_client()
    mock_client.assert_called_once()
    assert client is not None