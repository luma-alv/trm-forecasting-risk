"""Observation-index persistence baseline evaluation."""
from __future__ import annotations

import pandas as pd

from .metrics import mae, rmse
from .protocol import ProtocolResult


def evaluation_targets(
    result: ProtocolResult,
    horizons: tuple[int, ...] | None = None,
    partitions: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return the common origin/target grid used by all forecast models."""
    requested = tuple(horizons or tuple(range(1, 11)))
    if not requested or any(horizon < 1 for horizon in requested):
        raise ValueError("horizons must contain positive integers")
    names = partitions or tuple(result.partitions)
    rows: list[dict[str, object]] = []
    for partition_name in names:
        series = result.partitions[partition_name]
        for origin_position in range(len(series)):
            for horizon in requested:
                target_position = origin_position + horizon
                if target_position >= len(series):
                    continue
                rows.append(
                    {
                        "partition": partition_name,
                        "origin_position": origin_position,
                        "forecast_origin": series.index[origin_position].date().isoformat(),
                        "target_date": series.index[target_position].date().isoformat(),
                        "horizon": horizon,
                        "actual": float(series.iloc[target_position]),
                    }
                )
    targets = pd.DataFrame(rows)
    if targets.empty:
        raise ValueError("protocol partitions are too short for requested horizons")
    return targets


def evaluate_persistence(
    result: ProtocolResult,
    horizons: tuple[int, ...] | None = None,
    model_name: str = "persistence_last",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate last-observation persistence within every protocol partition.

    Each horizon is an offset in observed TRM records. An origin's value is the
    only feature used; target values are subsequent observations in the same
    partition. Metrics are returned by partition and horizon.
    """
    targets = evaluation_targets(result, horizons)
    predictions = targets.copy()
    predictions["prediction"] = predictions.apply(
        lambda row: float(result.partitions[row["partition"]].iloc[int(row["origin_position"])]), axis=1
    )
    predictions["model"] = model_name
    predictions["error"] = predictions["actual"] - predictions["prediction"]
    metric_rows: list[dict[str, object]] = []
    for (partition, horizon), group in predictions.groupby(["partition", "horizon"], sort=True):
        metric_rows.append(
            {
                "partition": partition,
                "horizon": int(horizon),
                "observations": len(group),
                "mae": mae(group["actual"], group["prediction"]),
                "rmse": rmse(group["actual"], group["prediction"]),
            }
        )
    return predictions, pd.DataFrame(metric_rows)
