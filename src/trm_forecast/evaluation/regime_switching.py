"""Leakage-aware Markov regime-switching volatility evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trm_forecast.models.garch import log_returns
from trm_forecast.models.regime_switching import MarkovSwitchingVolatility

from .garch import qlike
from .persistence import evaluation_targets
from .protocol import ProtocolResult


def evaluate_regime_switching(
    result: ProtocolResult,
    horizons: tuple[int, ...] | None = None,
    partitions: tuple[str, ...] = ("validation", "test"),
    min_history: int = 60,
    scale: float = 100.0,
    maxiter: int = 500,
    refit_frequency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate two-state filtered regime-switching cumulative variance."""
    if min_history < 31:
        raise ValueError("min_history must be at least 31 observations")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if refit_frequency < 1:
        raise ValueError("refit_frequency must be positive")
    targets = evaluation_targets(result, horizons, partitions)
    rows: list[dict[str, object]] = []
    model: MarkovSwitchingVolatility | None = None
    last_status = "not_fitted"
    last_error: str | None = None
    for origin_number, (origin_date, group) in enumerate(targets.groupby("forecast_origin", sort=True)):
        origin = pd.Timestamp(origin_date)
        history = result.series.loc[result.series.index <= origin]
        returns = log_returns(history, scale=scale) if len(history) >= 2 else np.array([])
        status = "converged"
        error_message: str | None = None
        forecast_mean: np.ndarray | None = None
        forecast_variance: np.ndarray | None = None
        filtered_probabilities: np.ndarray | None = None
        needs_refit = model is None or origin_number % refit_frequency == 0
        if needs_refit:
            if len(history) < min_history:
                status = "insufficient_history"
            else:
                candidate = MarkovSwitchingVolatility(maxiter=maxiter)
                try:
                    candidate.fit(returns)
                    model = candidate
                    status = candidate.fit_status
                except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as exc:
                    status = candidate.fit_status
                    error_message = str(exc)
            last_status, last_error = status, error_message
        else:
            status, error_message = last_status, last_error
        if model is not None and status in {"converged", "converged_with_warning"}:
            try:
                max_horizon = max(int(row["horizon"]) for row in group.to_dict("records"))
                forecast_mean, forecast_variance, filtered_probabilities = model.forecast_from_history(
                    returns, max_horizon
                )
            except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as exc:
                status = "forecast_failed"
                error_message = str(exc)
        origin_level = float(history.iloc[-1])
        for row in group.to_dict("records"):
            horizon = int(row["horizon"])
            cumulative_return = scale * np.log(float(row["actual"]) / origin_level)
            if forecast_mean is None or forecast_variance is None:
                mean = variance = np.nan
            else:
                mean = float(np.sum(forecast_mean[:horizon]))
                variance = float(np.sum(forecast_variance[:horizon]))
            rows.append(
                {
                    **row,
                    "origin_level": origin_level,
                    "realized_cumulative_return": cumulative_return,
                    "realized_variance_proxy": cumulative_return**2,
                    "forecast_mean_return": mean,
                    "forecast_variance": variance,
                    "forecast_volatility": np.sqrt(variance) if np.isfinite(variance) else np.nan,
                    "filtered_regime_0": np.nan
                    if filtered_probabilities is None
                    else float(filtered_probabilities[0]),
                    "filtered_regime_1": np.nan
                    if filtered_probabilities is None
                    else float(filtered_probabilities[1]),
                    "model": "markov_switching_2_state",
                    "status": status,
                    "fit_error": error_message,
                }
            )
    predictions = pd.DataFrame(rows)
    metrics: list[dict[str, object]] = []
    for (partition, horizon), group in predictions.groupby(["partition", "horizon"], sort=True):
        successful = group.loc[
            group["status"].isin(["converged", "converged_with_warning"])
            & group["forecast_variance"].map(np.isfinite)
        ]
        metrics.append(
            {
                "partition": partition,
                "horizon": int(horizon),
                "observations": len(successful),
                "attempted_predictions": len(group),
                "failed_predictions": len(group) - len(successful),
                "qlike": np.nan
                if successful.empty
                else qlike(successful["realized_variance_proxy"], successful["forecast_variance"]),
            }
        )
    return predictions, pd.DataFrame(metrics)
