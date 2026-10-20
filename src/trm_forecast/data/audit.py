"""Non-destructive quality audit for normalized TRM observations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    """Return a stable, JSON-friendly representation for audit evidence."""
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _evidence(frame: pd.DataFrame, mask: pd.Series, columns: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in frame.loc[mask, columns].iterrows():
        item = {"row": _json_value(index)}
        item.update({column: _json_value(row[column]) for column in columns})
        result.append(item)
    return result


@dataclass(frozen=True)
class AuditReport:
    """Structured, reproducible results from :func:`audit_trm`.

    The report contains evidence from the input frame and derived intervals. It
    never imputes, drops, sorts, or otherwise changes the supplied data.
    """

    input_rows: int
    columns: tuple[str, ...]
    observation_count: int
    earliest_date: str | None
    latest_date: str | None
    unique_date_count: int
    duplicated_date_count: int
    rows_in_duplicated_dates: int
    missing_date_count: int
    invalid_date_count: int
    missing_value_count: int
    non_numeric_value_count: int
    non_finite_value_count: int
    zero_value_count: int
    negative_value_count: int
    is_sorted_chronologically: bool
    multiple_observations_per_date: bool
    calendar_day_gaps: tuple[dict[str, Any], ...]
    business_day_gaps: tuple[dict[str, Any], ...]
    candidate_missing_periods: tuple[dict[str, Any], ...]
    invalid_date_rows: tuple[dict[str, Any], ...]
    invalid_value_rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for JSON or a notebook table."""
        return asdict(self)


def audit_trm(
    data: pd.DataFrame,
    date_column: str = "date",
    value_column: str = "trm",
) -> AuditReport:
    """Audit normalized TRM data without silently cleaning it.

    Date and value validity are assessed independently. Gap calculations use
    rows with both a parseable date and a finite numeric TRM, sorted only in a
    temporary derived frame. ``business_day_gap`` means elapsed Monday-Friday
    days according to NumPy's generic calendar; Colombian holidays and source
    publication rules are intentionally not inferred.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if date_column not in data.columns or value_column not in data.columns:
        raise ValueError(f"required columns are missing: {date_column!r}, {value_column!r}")

    frame = data[[date_column, value_column]].copy()
    parsed_dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
    parsed_dates = parsed_dates.dt.tz_convert(None).dt.normalize()
    raw_values = frame[value_column]
    numeric_values = pd.to_numeric(raw_values, errors="coerce")

    missing_dates = frame[date_column].isna()
    invalid_dates = parsed_dates.isna() & ~missing_dates
    missing_values = raw_values.isna()
    non_numeric_values = numeric_values.isna() & ~missing_values
    non_finite_values = numeric_values.notna() & ~numeric_values.map(np.isfinite)
    zero_values = numeric_values.notna() & numeric_values.map(np.isfinite) & numeric_values.eq(0)
    negative_values = numeric_values.notna() & numeric_values.map(np.isfinite) & numeric_values.lt(0)

    valid = ~(
        parsed_dates.isna()
        | numeric_values.isna()
        | ~numeric_values.map(np.isfinite)
        | numeric_values.le(0)
    )
    valid_frame = pd.DataFrame({"date": parsed_dates[valid], "trm": numeric_values[valid]})
    valid_frame = valid_frame.sort_values("date", kind="mergesort")
    dates = valid_frame["date"]
    duplicated = dates.duplicated(keep=False)
    duplicate_dates = int(dates[duplicated].nunique())

    intervals: list[dict[str, Any]] = []
    for previous, current in zip(dates.iloc[:-1], dates.iloc[1:]):
        calendar_gap = int((current - previous).days)
        business_gap = int(np.busday_count(previous.date(), current.date()))
        interval = {
            "start": previous.date().isoformat(),
            "end": current.date().isoformat(),
            "calendar_days": calendar_gap,
            "business_days_monday_friday": business_gap,
        }
        intervals.append(interval)

    valid_dates_in_input = parsed_dates.dropna()
    is_sorted = bool(valid_dates_in_input.is_monotonic_increasing)
    candidate_periods = tuple(interval for interval in intervals if interval["calendar_days"] > 1)

    earliest = dates.min().date().isoformat() if not dates.empty else None
    latest = dates.max().date().isoformat() if not dates.empty else None
    invalid_date_mask = missing_dates | invalid_dates
    invalid_value_mask = missing_values | non_numeric_values | non_finite_values | zero_values | negative_values
    return AuditReport(
        input_rows=len(frame),
        columns=tuple(str(column) for column in data.columns),
        observation_count=len(valid_frame),
        earliest_date=earliest,
        latest_date=latest,
        unique_date_count=int(dates.nunique()),
        duplicated_date_count=duplicate_dates,
        rows_in_duplicated_dates=int(duplicated.sum()),
        missing_date_count=int(missing_dates.sum()),
        invalid_date_count=int(invalid_dates.sum()),
        missing_value_count=int(missing_values.sum()),
        non_numeric_value_count=int(non_numeric_values.sum()),
        non_finite_value_count=int(non_finite_values.sum()),
        zero_value_count=int(zero_values.sum()),
        negative_value_count=int(negative_values.sum()),
        is_sorted_chronologically=is_sorted,
        multiple_observations_per_date=duplicate_dates > 0,
        calendar_day_gaps=tuple(intervals),
        business_day_gaps=tuple(intervals),
        candidate_missing_periods=candidate_periods,
        invalid_date_rows=tuple(_evidence(frame, invalid_date_mask, [date_column])),
        invalid_value_rows=tuple(_evidence(frame, invalid_value_mask, [value_column])),
    )
