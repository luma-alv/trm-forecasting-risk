"""Configurable GARCH conditional-volatility model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GARCHVolatility:
    """GARCH model for percentage log returns."""

    p: int = 1
    q: int = 1
    mean: str = "Constant"
    distribution: str = "normal"
    maxiter: int = 500
    fitted_model: object | None = None
    fit_status: str = "not_fitted"
    fit_warning: str | None = None

    def _validate(self, values: np.ndarray) -> None:
        if self.p < 1 or self.q < 1:
            raise ValueError("p and q must be positive")
        if values.ndim != 1 or len(values) < 20 or not np.isfinite(values).all():
            raise ValueError("returns must contain at least 20 finite one-dimensional values")

    def _arch_model(self, values: np.ndarray):
        try:
            from arch import arch_model
        except ImportError as exc:
            raise ImportError("install the 'models' extra to use GARCH") from exc
        return arch_model(
            values, mean=self.mean, vol="GARCH", p=self.p, q=self.q,
            dist=self.distribution, rescale=False
        )

    def fit(self, returns: np.ndarray) -> GARCHVolatility:
        values = np.asarray(returns, dtype=float)
        self._validate(values)
        model = self._arch_model(values)
        try:
            self.fitted_model = model.fit(disp="off", options={"maxiter": self.maxiter})
        except Exception:
            self.fit_status = "fit_failed"
            raise
        convergence_flag = getattr(self.fitted_model, "convergence_flag", 0)
        self.fit_status = "converged" if convergence_flag == 0 else "converged_with_warning"
        self.fit_warning = getattr(self.fitted_model, "message", None)
        return self

    def _forecast_result(self, fitted, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        result = fitted.forecast(horizon=horizon, reindex=False)
        mean = np.asarray(result.mean.iloc[-1], dtype=float)
        variance = np.asarray(result.variance.iloc[-1], dtype=float)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError("GARCH produced a non-finite or non-positive variance forecast")
        return mean, variance

    def forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        if self.fitted_model is None:
            raise RuntimeError("fit must be called before forecast")
        return self._forecast_result(self.fitted_model, horizon)

    def forecast_from_history(
        self, returns: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast from current returns while holding fitted parameters fixed."""
        if self.fitted_model is None:
            raise RuntimeError("fit must be called before forecast_from_history")
        values = np.asarray(returns, dtype=float)
        self._validate(values)
        fixed = self._arch_model(values).fix(self.fitted_model.params)
        return self._forecast_result(fixed, horizon)


def log_returns(series, scale: float = 100.0) -> np.ndarray:
    """Convert a positive level series into scaled log returns."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("series must contain at least two levels")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("series must contain positive, finite levels")
    result = np.diff(np.log(values)) * scale
    if len(result) == 0 or not np.isfinite(result).all():
        raise ValueError("series must contain at least two valid levels")
    return result


def forecast_volatility(returns, horizon: int = 5, p: int = 1, q: int = 1):
    """Backward-compatible GARCH volatility forecast from percentage returns."""
    model = GARCHVolatility(p=p, q=q).fit(np.asarray(returns, dtype=float) * 100)
    _, variance = model.forecast(horizon)
    return np.sqrt(variance) / 100
