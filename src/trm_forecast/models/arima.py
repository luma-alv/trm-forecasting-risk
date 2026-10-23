"""Reusable ARIMA fitting and train-only stationarity diagnostics."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ARIMAForecast:
    """Small wrapper around statsmodels' ARIMA implementation.

    For an integrated model, ``trend="t"`` represents a constant drift in
    the first differences. A literal constant is intentionally not used:
    statsmodels rejects it when ``d > 0``.
    """

    order: tuple[int, int, int] = (1, 1, 1)
    maxiter: int = 100
    fitted_model: object | None = None
    fit_status: str = "not_fitted"
    fit_warning: str | None = None
    trend: str | None = None

    def fit(self, y: pd.Series | np.ndarray) -> ARIMAForecast:
        values = np.asarray(y, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("y must be a non-empty one-dimensional finite series")
        if len(values) <= sum(self.order) + 1:
            self.fit_status = "insufficient_history"
            raise ValueError("insufficient history for the requested ARIMA order")
        try:
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise ImportError("install the 'models' extra to use ARIMA") from exc
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                self.fitted_model = ARIMA(values, order=self.order, trend=self.trend).fit(
                    method_kwargs={"maxiter": self.maxiter}
                )
            except Exception:
                self.fit_status = "fit_failed"
                raise
        convergence_warnings = [str(item.message) for item in caught]
        self.fit_warning = "; ".join(convergence_warnings) or None
        mle_retvals = getattr(self.fitted_model, "mle_retvals", {})
        converged = mle_retvals.get("converged", True) if isinstance(mle_retvals, dict) else True
        self.fit_status = "converged" if converged and not self.fit_warning else "converged_with_warning"
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.fitted_model is None:
            raise RuntimeError("fit must be called before predict")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        return np.asarray(self.fitted_model.forecast(steps=horizon), dtype=float)

    def predict_from_history(self, y: pd.Series | np.ndarray, horizon: int) -> np.ndarray:
        """Forecast from current history while reusing fitted parameters.

        This supports expanding-window evaluation with a refit schedule: the
        model parameters are held fixed between refits, but the state is
        updated with observations available at the current forecast origin.
        """
        if self.fitted_model is None:
            raise RuntimeError("fit must be called before predict_from_history")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        values = np.asarray(y, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("y must be a non-empty one-dimensional finite series")
        applied = self.fitted_model.apply(values, refit=False)
        return np.asarray(applied.forecast(steps=horizon), dtype=float)


def forecast_arima(y, horizon: int = 5, order=(1, 1, 1), trend: str | None = None) -> np.ndarray:
    """Backward-compatible one-shot ARIMA forecast."""
    return ARIMAForecast(tuple(order), trend=trend).fit(y).predict(horizon)


def stationarity_diagnostics(series: pd.Series) -> pd.DataFrame:
    """Run ADF and KPSS on levels and first differences."""
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError as exc:
        raise ImportError("install the 'models' extra to use stationarity diagnostics") from exc
    values = pd.to_numeric(series, errors="coerce").dropna()
    samples = {"level": values, "first_difference": values.diff().dropna()}
    rows: list[dict[str, object]] = []
    for name, sample in samples.items():
        row: dict[str, object] = {"series": name, "observations": len(sample)}
        try:
            adf_result = adfuller(sample, autolag="AIC")
            row.update({"adf_statistic": adf_result[0], "adf_pvalue": adf_result[1]})
        except (ValueError, np.linalg.LinAlgError) as exc:
            row.update({"adf_statistic": np.nan, "adf_pvalue": np.nan, "adf_error": str(exc)})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(sample, regression="c", nlags="auto")
            row.update({"kpss_statistic": kpss_result[0], "kpss_pvalue": kpss_result[1]})
        except (ValueError, np.linalg.LinAlgError) as exc:
            row.update({"kpss_statistic": np.nan, "kpss_pvalue": np.nan, "kpss_error": str(exc)})
        rows.append(row)
    return pd.DataFrame(rows)
