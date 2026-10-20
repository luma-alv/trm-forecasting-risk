"""Validation and normalization for TRM time series."""
from __future__ import annotations
import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input series cannot be used safely."""


def validate_trm(data: pd.DataFrame, date_column: str = "date", value_column: str = "trm") -> pd.DataFrame:
    """Return a sorted, de-duplicated TRM frame with a date index."""
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("data must be a pandas DataFrame")
    aliases = {}
    for target, candidates in ((date_column, [date_column, "fecha", "Date"]), (value_column, [value_column, "valor", "value"])):
        found = next((candidate for candidate in candidates if candidate in data), None)
        if found is None:
            raise DataValidationError(f"missing required column: {target}")
        aliases[found] = target
    frame = data.rename(columns=aliases)[[date_column, value_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        raise DataValidationError("no valid date/value observations found")
    if (frame[value_column] <= 0).any():
        raise DataValidationError("TRM values must be positive")
    frame = frame.drop_duplicates(date_column, keep="last").sort_values(date_column).set_index(date_column)
    frame.index.name = "date"
    return frame
