"""Date-based, leakage-aware temporal evaluation protocol."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd


def _as_date(value: date | str | pd.Timestamp) -> date:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"invalid date: {value!r}")
    return parsed.date()


@dataclass(frozen=True)
class PartitionSpec:
    """Inclusive calendar boundaries for one chronological partition."""

    name: str
    start: date | str
    end: date | str | None

    def resolved(self, as_of_date: date) -> tuple[date, date]:
        start = _as_date(self.start)
        end = as_of_date if self.end is None else _as_date(self.end)
        if start > end:
            raise ValueError(f"partition {self.name!r} has start after end")
        return start, end


@dataclass(frozen=True)
class PartitionSummary:
    name: str
    configured_start: str
    configured_end: str | None
    effective_start: str
    effective_end: str
    observation_count: int
    earliest_observation: str | None
    latest_observation: str | None
    enough_for_requested_horizons: bool


@dataclass(frozen=True)
class ProtocolResult:
    """Partitioned series and reproducible summaries."""

    series: pd.Series
    partitions: Mapping[str, pd.Series]
    summaries: tuple[PartitionSummary, ...]
    as_of_date: str
    excluded_after_cutoff: int


@dataclass(frozen=True)
class TemporalEvaluationProtocol:
    """Configurable protocol for observation-horizon forecasting experiments.

    Horizons count subsequent observations in the chronological TRM index, not
    calendar days. The ``as_of_date`` cutoff is applied before partitioning;
    ``retrieved_at`` is deliberately not used as a historical publication date.
    """

    as_of_date: date | str
    horizons: tuple[int, ...] = tuple(range(1, 11))
    partitions: tuple[PartitionSpec, ...] = (
        PartitionSpec("train", "2008-01-01", "2022-12-31"),
        PartitionSpec("validation", "2023-01-01", "2024-12-31"),
        PartitionSpec("test", "2025-01-01", None),
    )

    def split(self, series: pd.Series) -> ProtocolResult:
        """Validate and split a dated TRM series without imputing observations."""
        self._validate_configuration()
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("series must be a pandas Series with a DatetimeIndex")
        if series.empty:
            raise ValueError("series must not be empty")
        if series.index.has_duplicates:
            raise ValueError("series index must contain unique observation dates")
        if not series.index.is_monotonic_increasing:
            raise ValueError("series must be sorted chronologically")
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any() or not values.map(pd.notna).all():
            raise ValueError("series must contain numeric, non-missing values")
        cutoff = _as_date(self.as_of_date)
        cutoff_timestamp = pd.Timestamp(cutoff)
        eligible = values.loc[values.index.normalize() <= cutoff_timestamp].copy()
        if eligible.empty:
            raise ValueError("as_of_date excludes all observations")
        partitions: dict[str, pd.Series] = {}
        summaries: list[PartitionSummary] = []
        previous_end: date | None = None
        for spec in self.partitions:
            start, end = spec.resolved(cutoff)
            if end > cutoff:
                raise ValueError(f"partition {spec.name!r} ends after as_of_date")
            if previous_end is not None and start <= previous_end:
                raise ValueError("partition date boundaries overlap or are not chronological")
            previous_end = end
            selected = eligible.loc[(eligible.index.normalize() >= pd.Timestamp(start)) & (eligible.index.normalize() <= pd.Timestamp(end))]
            if len(selected) <= max(self.horizons):
                raise ValueError(
                    f"partition {spec.name!r} has {len(selected)} observations; "
                    f"at least {max(self.horizons) + 1} are required"
                )
            partitions[spec.name] = selected
            summaries.append(
                PartitionSummary(
                    name=spec.name,
                    configured_start=_as_date(spec.start).isoformat(),
                    configured_end=None if spec.end is None else _as_date(spec.end).isoformat(),
                    effective_start=start.isoformat(),
                    effective_end=end.isoformat(),
                    observation_count=len(selected),
                    earliest_observation=selected.index.min().date().isoformat(),
                    latest_observation=selected.index.max().date().isoformat(),
                    enough_for_requested_horizons=True,
                )
            )
        return ProtocolResult(
            series=eligible,
            partitions=partitions,
            summaries=tuple(summaries),
            as_of_date=cutoff.isoformat(),
            excluded_after_cutoff=len(values) - len(eligible),
        )

    def _validate_configuration(self) -> None:
        if not self.horizons or any(horizon < 1 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if not self.partitions:
            raise ValueError("at least one partition is required")
        names = [partition.name for partition in self.partitions]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("partition names must be non-empty and unique")
