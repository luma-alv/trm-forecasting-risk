"""TRM ingestion from Datos Abiertos Colombia and local raw files.

Downloading and normalization are deliberately separate. The downloader returns
the unmodified JSON records plus request metadata; ``validate_trm`` produces the
canonical four-column table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .validation import validate_trm

DEFAULT_ENDPOINT = "https://www.datos.gov.co/resource/ceyp-9c7c.json"
DEFAULT_SOURCE = "Datos Abiertos Colombia: ceyp-9c7c"


class IngestionError(RuntimeError):
    """Base error for transport and source-response failures."""


class IngestionHTTPError(IngestionError):
    """Raised when the Socrata endpoint responds with an HTTP error."""


class EmptyResponseError(IngestionError):
    """Raised when a successful response contains no records."""


@dataclass(frozen=True)
class TRMSourceConfig:
    """Configurable Socrata source and request behavior."""

    endpoint: str = DEFAULT_ENDPOINT
    source_name: str = DEFAULT_SOURCE
    timeout_seconds: float = 30.0
    date_field: str = "vigenciadesde"
    value_field: str = "valor"
    page_size: int = 50000
    extra_params: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RawDownload:
    """Raw source payload and provenance metadata."""

    records: list[dict[str, Any]]
    retrieved_at: datetime
    endpoint: str
    params: Mapping[str, str]
    source: str


def _iso_date(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid date filter: {value!r}")
    return parsed.date().isoformat()


def _build_params(config: TRMSourceConfig, start_date=None, end_date=None) -> dict[str, str]:
    params = dict(config.extra_params)
    params.setdefault("$select", f"{config.date_field},{config.value_field}")
    start, end = _iso_date(start_date), _iso_date(end_date)
    if start and end and start > end:
        raise ValueError("start_date cannot be after end_date")
    conditions = []
    if start:
        conditions.append(f"{config.date_field} >= '{start}T00:00:00'")
    if end:
        conditions.append(f"{config.date_field} <= '{end}T23:59:59'")
    if conditions:
        params["$where"] = " AND ".join(conditions)
    params.setdefault("$order", config.date_field)
    params.setdefault("$limit", str(config.page_size))
    return params


def _default_http_get(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "trm-forecasting-risk/0.1"})
    with urlopen(request, timeout=timeout) as response:  # nosec B310: endpoint is configurable by design
        return response.read()


def download_trm(
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    config: TRMSourceConfig | None = None,
    *,
    http_get: Callable[[str, float], bytes] | None = None,
) -> RawDownload:
    """Download raw TRM records, optionally restricting the date range.

    ``http_get`` is injectable so tests never depend on the real API. HTTP,
    timeout, malformed JSON, and empty-response errors are surfaced explicitly.
    """
    config = config or TRMSourceConfig()
    params = _build_params(config, start_date, end_date)
    url = f"{config.endpoint}?{urlencode(params)}"
    getter = http_get or _default_http_get
    retrieved_at = datetime.now(timezone.utc)
    try:
        payload = getter(url, config.timeout_seconds)
    except HTTPError as exc:
        raise IngestionHTTPError(f"TRM source returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise IngestionError(f"TRM source request failed: {exc}") from exc
    try:
        decoded = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise IngestionError("TRM source returned invalid JSON") from exc
    if not isinstance(decoded, list) or not decoded:
        raise EmptyResponseError("TRM source returned no records")
    if not all(isinstance(record, dict) for record in decoded):
        raise IngestionError("TRM source returned a JSON structure other than records")
    return RawDownload(decoded, retrieved_at, config.endpoint, params, config.source_name)


def normalize_download(download: RawDownload, config: TRMSourceConfig | None = None) -> pd.DataFrame:
    """Transform a raw download into the canonical validated schema."""
    config = config or TRMSourceConfig()
    return validate_trm(
        pd.DataFrame(download.records),
        date_column=config.date_field,
        value_column=config.value_field,
        source=download.source,
        retrieved_at=download.retrieved_at,
    )


def load_trm(path: str | Path, date_column: str = "date", value_column: str = "trm") -> pd.DataFrame:
    """Load a local raw CSV/Parquet file through the same strict validator."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(file_path)
    elif file_path.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(file_path)
    else:
        raise ValueError("supported input formats are CSV and Parquet")
    return validate_trm(raw, date_column=date_column, value_column=value_column, source="local_file")
