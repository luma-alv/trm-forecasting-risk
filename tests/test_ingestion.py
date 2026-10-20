import json
from urllib.error import HTTPError

import pytest

from trm_forecast.data.ingestion import EmptyResponseError, IngestionHTTPError, TRMSourceConfig, download_trm, normalize_download


def fake_get(payload):
    def getter(url, timeout):
        getter.url = url
        getter.timeout = timeout
        return json.dumps(payload).encode("utf-8")

    return getter


def test_download_uses_date_filters_and_normalizes_separately():
    getter = fake_get([{"vigenciadesde": "2024-01-01T00:00:00.000", "valor": 4000}])
    config = TRMSourceConfig(endpoint="https://example.test/resource", timeout_seconds=7)
    raw = download_trm("2024-01-01", "2024-01-10", config, http_get=getter)
    assert "vigenciadesde" in getter.url
    assert "2024-01-01" in getter.url and "2024-01-10" in getter.url
    assert getter.timeout == 7
    assert raw.records[0]["valor"] == 4000
    normalized = normalize_download(raw, config)
    assert list(normalized.columns) == ["date", "trm", "source", "retrieved_at"]


def test_empty_response_is_an_explicit_error():
    with pytest.raises(EmptyResponseError):
        download_trm(config=TRMSourceConfig(endpoint="https://example.test"), http_get=fake_get([]))


def test_http_error_is_an_explicit_error():
    def failing_get(url, timeout):
        raise HTTPError(url, 503, "unavailable", {}, None)

    with pytest.raises(IngestionHTTPError, match="503"):
        download_trm(config=TRMSourceConfig(endpoint="https://example.test"), http_get=failing_get)
