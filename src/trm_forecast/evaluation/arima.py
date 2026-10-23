"""Leakage-aware expanding-window ARIMA evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trm_forecast.models.arima import ARIMAForecast

from .metrics import mae, rmse
from .persistence import evaluation_targets
from .protocol import ProtocolResult


def evaluate_arima(
    result: ProtocolResult,
    order: tuple[int, int, int] = (1, 1, 1),
    horizons: tuple[int, ...] | None = None,
    partitions: tuple[str, ...] = ("validation", "test"),
    min_history: int = 60,
    maxiter: int = 100,
    refit_frequency: int = 10,
    trend: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate expanding-window ARIMA with a deterministic refit schedule.

    Parameters are refit at the configured frequency. Between refits, the
    fitted parameters are reused while the state is updated from observations
    available at the current origin.
    """
    if min_history < 1:
        raise ValueError("min_history must be positive")
    if refit_frequency < 1:
        raise ValueError("refit_frequency must be positive")
    targets = evaluation_targets(result, horizons, partitions)
    rows: list[dict[str, object]] = []
    last_model: ARIMAForecast | None = None
    last_status = "not_fitted"
    last_error: str | None = None
    for origin_number, (origin_date, group) in enumerate(targets.groupby("forecast_origin", sort=True)):
        origin = pd.Timestamp(origin_date)
        history = result.series.loc[result.series.index <= origin]
        status = "converged"
        error_message: str | None = None
        forecast_values: np.ndarray | None = None
        origin_position = int(result.series.index.get_loc(origin))
        needs_refit = last_model is None or origin_number % refit_frequency == 0
        if needs_refit:
            if len(history) < min_history:
                status = "insufficient_history"
            else:
                model = ARIMAForecast(order=tuple(order), maxiter=maxiter, trend=trend)
                try:
                    model.fit(history)
                    last_model = model
                    status = model.fit_status
                except (ValueError, RuntimeError, np.linalg.LinAlgError, ImportError) as exc:
                    status = model.fit_status
                    error_message = str(exc)
            last_status, last_error = status, error_message
        else:
            status, error_message = last_status, last_error
        if last_model is not None and status in {"converged", "converged_with_warning"}:
            try:
                max_target_position = max(
                    int(result.series.index.get_loc(pd.Timestamp(row["target_date"])))
                    for row in group.to_dict("records")
                )
                forecast_horizon = max_target_position - origin_position
                forecast_values = last_model.predict_from_history(history, forecast_horizon)
            except (ValueError, RuntimeError, np.linalg.LinAlgError, ImportError) as exc:
                status = "forecast_failed"
                error_message = str(exc)
        for row in group.to_dict("records"):
            target_position = int(result.series.index.get_loc(pd.Timestamp(row["target_date"])))
            current_offset = target_position - origin_position - 1
            prediction = (
                np.nan
                if forecast_values is None or current_offset >= len(forecast_values)
                else float(forecast_values[current_offset])
            )
            rows.append(
                {
                    **row,
                    "prediction": prediction,
                    "model": f"arima{tuple(order)}" + ("_drift" if trend == "t" else ""),
                    "status": status,
                    "fit_error": error_message,
                    "error": row["actual"] - prediction if np.isfinite(prediction) else np.nan,
                }
            )
    predictions = pd.DataFrame(rows)
    metric_rows: list[dict[str, object]] = []
    for (partition, horizon), group in predictions.groupby(["partition", "horizon"], sort=True):
        successful = group.loc[
            group["status"].isin(["converged", "converged_with_warning"])
            & group["prediction"].map(np.isfinite)
        ]
        metric_rows.append(
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
    return predictions, pd.DataFrame(metric_rows)
