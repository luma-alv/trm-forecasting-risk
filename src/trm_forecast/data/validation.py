"""Canonical schema and strict validation for TRM observations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

import pandas as pd

CANONICAL_COLUMNS = ("date", "trm", "source", "retrieved_at")


class DataValidationError(ValueError):
    """Base error for invalid or ambiguous TRM data."""


class DuplicateDateError(DataValidationError):
    """Raised when more than one record has the same observation date."""


class MissingValueError(DataValidationError):
    """Raised when a required canonical field is missing or null."""


@dataclass(frozen=True)
class ValidationReport:
    """Counts returned alongside a normalized frame when requested."""

    input_rows: int
    output_rows: int


def _column_lookup(columns) -> dict[str, str]:
    return {str(column).strip().lower(): column for column in columns}


def _resolve_column(frame: pd.DataFrame, requested: str, aliases: tuple[str, ...]) -> str:
    lookup = _column_lookup(frame.columns)
    for candidate in (requested, *aliases):
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise DataValidationError(f"missing required column '{requested}'; available columns: {list(frame.columns)}")


def _parse_datetime(values: pd.Series, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        bad_rows = list(parsed.index[parsed.isna()][:5])
        raise DataValidationError(f"invalid dates in '{column}' at rows {bad_rows}")
    return parsed.dt.tz_convert(None).dt.normalize()


def _parse_trm(values: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna() | ~numeric.map(math.isfinite)
    if invalid.any():
        bad_rows = list(invalid.index[invalid][:5])
        raise DataValidationError(f"non-numeric or non-finite TRM values in '{column}' at rows {bad_rows}")
    if (numeric <= 0).any():
        bad_rows = list(numeric.index[numeric <= 0][:5])
        raise DataValidationError(f"TRM values must be positive; invalid rows: {bad_rows}")
    return numeric.astype(float)


def validate_trm(
    data: pd.DataFrame,
    date_column: str = "date",
    value_column: str = "trm",
    source: str | None = None,
    retrieved_at: datetime | str | None = None,
    *,
    allow_duplicates: bool = False,
) -> pd.DataFrame:
    """Normalize a raw frame into the four-column canonical TRM schema.

    Invalid rows, nulls, and duplicate dates raise ``DataValidationError``. No
    row is silently discarded. Duplicate dates can only be accepted explicitly
    with ``allow_duplicates=True``; even then they remain in the output.
    """
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("data must be a pandas DataFrame")
    if data.empty:
        raise DataValidationError("response contains no records")
    date_col = _resolve_column(data, date_column, ("date", "fecha", "vigenciadesde"))
    value_col = _resolve_column(data, value_column, ("trm", "valor", "value"))
    frame = pd.DataFrame({"date": _parse_datetime(data[date_col], date_col), "trm": _parse_trm(data[value_col], value_col)})
    source_value = source if source is not None else "unknown"
    if not isinstance(source_value, str) or not source_value.strip():
        raise DataValidationError("source must be a non-empty string")
    frame["source"] = source_value
    retrieved = retrieved_at or datetime.now(timezone.utc)
    retrieved_series = pd.to_datetime(pd.Series([retrieved] * len(frame)), errors="coerce", utc=True)
    if retrieved_series.isna().any():
        raise DataValidationError("retrieved_at must be a valid datetime")
    frame["retrieved_at"] = retrieved_series.dt.tz_convert(None).iloc[: len(frame)].to_numpy()
    duplicates = frame["date"].duplicated(keep=False)
    if duplicates.any() and not allow_duplicates:
        dates = frame.loc[duplicates, "date"].dt.strftime("%Y-%m-%d").unique().tolist()
        raise DuplicateDateError(f"duplicate observation dates found: {dates}")
    if frame["date"].isna().any() or frame["trm"].isna().any():
        raise MissingValueError("canonical date and trm fields cannot be missing")
    return frame.loc[:, CANONICAL_COLUMNS].sort_values("date").reset_index(drop=True)
