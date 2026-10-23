import json
from urllib.error import HTTPError

import pytest

from trm_forecast.data.ingestion import (
    EmptyResponseError,
    IngestionError,
    IngestionHTTPError,
    PaginationError,
    TRMSourceConfig,
    download_trm,
    normalize_download,
)


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


def test_download_combines_multiple_pages_and_terminates_on_short_page():
    pages = [
        [
            {"vigenciadesde": "2024-01-01", "valor": 4000},
            {"vigenciadesde": "2024-01-02", "valor": 4001},
        ],
        [
            {"vigenciadesde": "2024-01-03", "valor": 4002},
            {"vigenciadesde": "2024-01-04", "valor": 4003},
        ],
        [{"vigenciadesde": "2024-01-05", "valor": 4004}],
    ]
    requests = []

    def paged_get(url, timeout):
        requests.append(url)
        return json.dumps(pages[len(requests) - 1]).encode("utf-8")

    result = download_trm(
        "2024-01-01",
        "2024-01-05",
        TRMSourceConfig(endpoint="https://example.test", page_size=2),
        http_get=paged_get,
    )

    assert len(result.records) == 5
    assert len(requests) == 3
    assert "%24offset=0" in requests[0]
    assert "%24offset=2" in requests[1]
    assert "%24offset=4" in requests[2]
    assert all("2024-01-01" in url and "2024-01-05" in url for url in requests)
    assert len(normalize_download(result, TRMSourceConfig(endpoint="https://example.test"))) == 5


def test_download_stops_when_a_later_page_is_empty():
    calls = []

    def empty_after_first(url, timeout):
        calls.append(url)
        records = [{"vigenciadesde": "2024-01-01", "valor": 4000}] if len(calls) == 1 else []
        return json.dumps(records).encode("utf-8")

    result = download_trm(
        config=TRMSourceConfig(endpoint="https://example.test", page_size=1),
        http_get=empty_after_first,
    )

    assert len(result.records) == 1
    assert len(calls) == 2


def test_download_raises_on_timeout():
    def timing_out(url, timeout):
        raise TimeoutError("timed out")

    with pytest.raises(IngestionError, match="request failed"):
        download_trm(config=TRMSourceConfig(endpoint="https://example.test"), http_get=timing_out)


def test_download_prevents_unbounded_pagination():
    def always_full_page(url, timeout):
        return json.dumps([{"vigenciadesde": "2024-01-01", "valor": 4000}]).encode("utf-8")

    with pytest.raises(PaginationError, match="did not terminate"):
        download_trm(
            config=TRMSourceConfig(endpoint="https://example.test", page_size=1, max_pages=2),
            http_get=always_full_page,
        )
