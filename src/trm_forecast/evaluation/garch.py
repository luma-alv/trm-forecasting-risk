"""Leakage-aware GARCH volatility evaluation."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from trm_forecast.models.garch import GARCHVolatility, log_returns

from .persistence import evaluation_targets
from .protocol import ProtocolResult


def qlike(realized_variance, forecast_variance) -> float:
    """QLIKE loss for positive realized and forecast variances."""
    realized = np.asarray(realized_variance, dtype=float)
    forecast = np.asarray(forecast_variance, dtype=float)
    if realized.shape != forecast.shape or realized.size == 0:
        raise ValueError("variance arrays must have the same non-empty shape")
    if not np.isfinite(realized).all() or not np.isfinite(forecast).all() or (forecast <= 0).any():
        raise ValueError("variances must be finite and forecasts must be positive")
    return float(np.mean(realized / forecast + np.log(forecast)))


def evaluate_variance_benchmark(
    result: ProtocolResult,
    horizons: tuple[int, ...] | None = None,
    partitions: tuple[str, ...] = ("validation", "test"),
    scale: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate an expanding historical-variance benchmark.

    It forecasts h-step cumulative variance as h times the sample variance of
    all scaled returns available at the origin. The benchmark has no tuned
    parameters and is used only to put QLIKE values in context.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    targets = evaluation_targets(result, horizons, partitions)
    rows: list[dict[str, object]] = []
    for origin_date, group in targets.groupby("forecast_origin", sort=True):
        origin = pd.Timestamp(origin_date)
        history = result.series.loc[result.series.index <= origin]
        returns = log_returns(history, scale=scale)
        one_step_variance = float(np.var(returns, ddof=1))
        origin_level = float(history.iloc[-1])
        for row in group.to_dict("records"):
            cumulative_return = scale * np.log(float(row["actual"]) / origin_level)
            variance = int(row["horizon"]) * one_step_variance
            rows.append(
                {
                    **row,
                    "realized_variance_proxy": cumulative_return**2,
                    "forecast_variance": variance,
                    "model": "expanding_historical_variance",
                }
            )
    predictions = pd.DataFrame(rows)
    metrics: list[dict[str, object]] = []
    for (partition, horizon), group in predictions.groupby(["partition", "horizon"], sort=True):
        metrics.append(
            {
                "partition": partition,
                "horizon": int(horizon),
                "observations": len(group),
                "qlike": qlike(group["realized_variance_proxy"], group["forecast_variance"]),
            }
        )
    return predictions, pd.DataFrame(metrics)


def block_bootstrap_coverage(
    covered, block_length: int = 10, samples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Return a deterministic moving-block-bootstrap interval for coverage."""
    values = np.asarray(covered, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("covered must be a non-empty binary one-dimensional array")
    if block_length < 1 or samples < 1:
        raise ValueError("block_length and samples must be positive")
    n_blocks = int(np.ceil(len(values) / block_length))
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    starts = np.arange(len(values))
    for sample in range(samples):
        selected = rng.choice(starts, size=n_blocks, replace=True)
        indices = (selected[:, None] + np.arange(block_length)) % len(values)
        estimates[sample] = values[indices.ravel()[: len(values)]].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def evaluate_garch(
    result: ProtocolResult,
    horizons: tuple[int, ...] | None = None,
    partitions: tuple[str, ...] = ("validation", "test"),
    p: int = 1,
    q: int = 1,
    min_history: int = 60,
    scale: float = 100.0,
    maxiter: int = 500,
    refit_frequency: int = 10,
    confidence_level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate cumulative-return variance and TRM level intervals by origin."""
    if min_history < 21:
        raise ValueError("min_history must be at least 21 observations")
    if refit_frequency < 1:
        raise ValueError("refit_frequency must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    targets = evaluation_targets(result, horizons, partitions)
    z_value = NormalDist().inv_cdf((1 + confidence_level) / 2)
    rows: list[dict[str, object]] = []
    model: GARCHVolatility | None = None
    last_status = "not_fitted"
    last_error: str | None = None
    for origin_number, (origin_date, group) in enumerate(targets.groupby("forecast_origin", sort=True)):
        origin = pd.Timestamp(origin_date)
        history = result.series.loc[result.series.index <= origin]
        origin_level = float(history.iloc[-1])
        returns = log_returns(history, scale=scale) if len(history) >= 2 else np.array([])
        status = "converged"
        error_message: str | None = None
        forecast_mean: np.ndarray | None = None
        forecast_variance: np.ndarray | None = None
        needs_refit = model is None or origin_number % refit_frequency == 0
        if needs_refit:
            if len(history) < min_history:
                status = "insufficient_history"
            else:
                candidate = GARCHVolatility(p=p, q=q, maxiter=maxiter)
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
                forecast_mean, forecast_variance = model.forecast_from_history(returns, max_horizon)
            except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as exc:
                status = "forecast_failed"
                error_message = str(exc)
        for row in group.to_dict("records"):
            horizon = int(row["horizon"])
            target_level = float(row["actual"])
            cumulative_return = scale * np.log(target_level / origin_level)
            if forecast_mean is None or forecast_variance is None:
                mean = variance = lower = upper = np.nan
            else:
                mean = float(np.sum(forecast_mean[:horizon]))
                variance = float(np.sum(forecast_variance[:horizon]))
                lower = origin_level * np.exp((mean - z_value * np.sqrt(variance)) / scale)
                upper = origin_level * np.exp((mean + z_value * np.sqrt(variance)) / scale)
            rows.append(
                {
                    **row,
                    "origin_level": origin_level,
                    "realized_cumulative_return": cumulative_return,
                    "realized_variance_proxy": cumulative_return**2,
                    "forecast_mean_return": mean,
                    "forecast_variance": variance,
                    "forecast_volatility": np.sqrt(variance) if np.isfinite(variance) else np.nan,
                    "lower_level": lower,
                    "upper_level": upper,
                    "covered": bool(lower <= target_level <= upper) if np.isfinite(lower) else False,
                    "model": f"garch({p},{q})",
                    "status": status,
                    "fit_error": error_message,
                }
            )
    predictions = pd.DataFrame(rows)
    metric_rows: list[dict[str, object]] = []
    for (partition, horizon), group in predictions.groupby(["partition", "horizon"], sort=True):
        successful = group.loc[
            group["status"].isin(["converged", "converged_with_warning"])
            & group["forecast_variance"].map(np.isfinite)
        ]
        metric_rows.append(
            {
                "partition": partition,
                "horizon": int(horizon),
                "observations": len(successful),
                "attempted_predictions": len(group),
                "failed_predictions": len(group) - len(successful),
                "qlike": np.nan if successful.empty else qlike(
                    successful["realized_variance_proxy"], successful["forecast_variance"]
                ),
                "coverage": np.nan if successful.empty else float(successful["covered"].mean()),
                "mean_interval_width": np.nan if successful.empty else float(
                    (successful["upper_level"] - successful["lower_level"]).mean()
                ),
            }
        )
    return predictions, pd.DataFrame(metric_rows)
