"""Leakage-aware direct-horizon Ridge evaluation using TRM-derived features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trm_forecast.features.build_features import build_features
from trm_forecast.models.ridge import RidgeRegressor

from .metrics import mae, rmse
from .persistence import evaluation_targets
from .protocol import ProtocolResult


def _feature_matrix(result: ProtocolResult, lags: tuple[int, ...], rolling_windows: tuple[int, ...]):
    frame = build_features(result.series, lags=lags, rolling_windows=rolling_windows)
    # The current origin level is available at decision time and makes the
    # feature model comparable with the persistence baseline.
    columns = list(frame.columns)
    return frame, columns


def _training_data(
    result: ProtocolResult,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    partitions: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    targets = evaluation_targets(result, horizons=(horizon,), partitions=partitions)
    rows = targets.copy()
    rows["origin_timestamp"] = pd.to_datetime(rows["forecast_origin"])
    rows = rows.join(feature_frame[feature_columns], on="origin_timestamp")
    valid = rows[feature_columns + ["actual"]].replace([np.inf, -np.inf], np.nan).dropna()
    return valid[feature_columns].to_numpy(dtype=float), valid["actual"].to_numpy(dtype=float)


def evaluate_ridge(
    result: ProtocolResult,
    alpha: float = 1.0,
    horizons: tuple[int, ...] | None = None,
    partitions: tuple[str, ...] = ("validation", "test"),
    lags: tuple[int, ...] = (1, 2, 5, 10),
    rolling_windows: tuple[int, ...] = (5, 10, 20),
    model_name: str = "ridge_trm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate direct horizon-specific Ridge models without future leakage.

    Validation origins are fitted using training data only. Test origins are
    fitted using train plus validation data; the caller must select ``alpha``
    before making this test call. Feature scaling is fitted independently for
    each horizon on that horizon's training rows.
    """
    requested = tuple(horizons or tuple(range(1, 11)))
    if not requested or any(horizon < 1 for horizon in requested):
        raise ValueError("horizons must contain positive integers")
    if len(set(requested)) != len(requested):
        raise ValueError("horizons must be unique")
    if not lags or any(lag < 1 for lag in lags) or len(set(lags)) != len(lags):
        raise ValueError("lags must contain unique positive integers")
    if not rolling_windows or any(window < 2 for window in rolling_windows):
        raise ValueError("rolling_windows must contain windows of at least 2")
    feature_frame, feature_columns = _feature_matrix(result, tuple(lags), tuple(rolling_windows))
    partition_order = tuple(result.partitions)
    rows: list[dict[str, object]] = []
    for partition in partitions:
        if partition not in partition_order:
            raise ValueError(f"unknown partition: {partition}")
        partition_index = partition_order.index(partition)
        training_partitions = partition_order[:partition_index]
        if not training_partitions:
            raise ValueError(f"partition {partition!r} has no earlier training partition")
        targets = evaluation_targets(result, requested, (partition,))
        fitted: dict[int, RidgeRegressor] = {}
        statuses: dict[int, tuple[str, str | None]] = {}
        for horizon in requested:
            model = RidgeRegressor(alpha=float(alpha))
            try:
                x_train, y_train = _training_data(
                    result, feature_frame, feature_columns, horizon, training_partitions
                )
                if len(x_train) == 0:
                    raise ValueError("no complete training feature rows")
                model.fit(x_train, y_train)
                fitted[horizon] = model
                statuses[horizon] = ("converged", None)
            except (ValueError, np.linalg.LinAlgError) as exc:
                statuses[horizon] = ("fit_failed", str(exc))
        for target in targets.to_dict("records"):
            horizon = int(target["horizon"])
            status, fit_error = statuses[horizon]
            prediction = np.nan
            if status == "converged":
                try:
                    origin = pd.Timestamp(target["forecast_origin"])
                    values = feature_frame.loc[[origin], feature_columns].replace(
                        [np.inf, -np.inf], np.nan
                    )
                    if values.isna().any(axis=None):
                        raise ValueError("origin has incomplete features")
                    prediction = float(fitted[horizon].predict(values.to_numpy())[0])
                except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
                    status = "forecast_failed"
                    fit_error = str(exc)
            rows.append(
                {
                    **target,
                    "prediction": prediction,
                    "model": model_name,
                    "alpha": float(alpha),
                    "status": status,
                    "fit_error": fit_error,
                    "error": target["actual"] - prediction if np.isfinite(prediction) else np.nan,
                }
            )
    predictions = pd.DataFrame(rows)
    metrics: list[dict[str, object]] = []
    for (partition, horizon), group in predictions.groupby(["partition", "horizon"], sort=True):
        successful = group.loc[group["status"] == "converged"].loc[group["prediction"].map(np.isfinite)]
        metrics.append(
            {
                "partition": partition,
                "horizon": int(horizon),
                "observations": len(successful),
                "attempted_predictions": len(group),
                "failed_predictions": len(group) - len(successful),
                "mae": np.nan if successful.empty else mae(successful["actual"], successful["prediction"]),
                "rmse": np.nan if successful.empty else rmse(successful["actual"], successful["prediction"]),
            }
        )
    return predictions, pd.DataFrame(metrics)
