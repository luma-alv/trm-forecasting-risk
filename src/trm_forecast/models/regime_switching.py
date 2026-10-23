"""Two-state Markov-switching conditional-volatility model."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class MarkovSwitchingVolatility:
    """Two-state model with a common mean and regime-specific variances."""

    maxiter: int = 500
    fitted_model: object | None = None
    fit_status: str = "not_fitted"
    fit_warning: str | None = None

    def _validate(self, values: np.ndarray) -> None:
        if values.ndim != 1 or len(values) < 30 or not np.isfinite(values).all():
            raise ValueError("returns must contain at least 30 finite one-dimensional values")

    def _model(self, values: np.ndarray):
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        except ImportError as exc:
            raise ImportError("install the 'models' extra to use regime switching") from exc
        return MarkovRegression(
            values,
            k_regimes=2,
            trend="c",
            switching_trend=False,
            switching_variance=True,
        )

    def fit(self, returns: np.ndarray) -> MarkovSwitchingVolatility:
        values = np.asarray(returns, dtype=float)
        self._validate(values)
        model = self._model(values)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                self.fitted_model = model.fit(
                    disp=False,
                    maxiter=self.maxiter,
                    search_reps=0,
                )
            except Exception:
                self.fit_status = "fit_failed"
                raise
        converged = bool(getattr(self.fitted_model, "mle_retvals", {}).get("converged", False))
        self.fit_warning = "; ".join(str(item.message) for item in caught) or None
        self.fit_status = "converged" if converged else "converged_with_warning"
        self._parameters()
        return self

    def _parameters(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self.fitted_model is None:
            raise RuntimeError("fit must be called before accessing parameters")
        names = self.fitted_model.model.param_names
        params = np.asarray(self.fitted_model.params, dtype=float)
        variance_indices = [index for index, name in enumerate(names) if name.startswith("sigma2[")]
        if len(variance_indices) != 2:
            raise ValueError("expected two regime-specific variance parameters")
        variances = params[variance_indices]
        transition = np.asarray(self.fitted_model.regime_transition[:, :, 0], dtype=float)
        mean_indices = [index for index, name in enumerate(names) if name == "const"]
        if len(mean_indices) != 1:
            raise ValueError("expected one common constant mean parameter")
        mean = float(params[mean_indices[0]])
        if (
            transition.shape != (2, 2)
            or not np.isfinite(transition).all()
            or (transition < 0).any()
            or not np.allclose(transition.sum(axis=0), 1.0)
            or not np.isfinite(variances).all()
            or (variances <= 0).any()
        ):
            raise ValueError("estimated regime parameters are invalid")
        return transition, variances, mean

    def filtered_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """Return probabilities filtered using observations through this origin."""
        if self.fitted_model is None:
            raise RuntimeError("fit must be called before filtering")
        values = np.asarray(returns, dtype=float)
        self._validate(values)
        filtered = self._model(values).filter(self.fitted_model.params)
        probabilities = np.asarray(filtered.filtered_marginal_probabilities, dtype=float)
        if probabilities.ndim != 2:
            raise ValueError("filtered regime probabilities have an invalid shape")
        current = probabilities[:, -1] if probabilities.shape[0] == 2 else probabilities[-1, :]
        if not np.isfinite(current).all() or (current < 0).any() or not np.isclose(current.sum(), 1.0):
            raise ValueError("filtered regime probabilities are invalid")
        return current

    def forecast_from_history(
        self, returns: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecast returns and variance using filtered, not smoothed, probabilities."""
        if self.fitted_model is None:
            raise RuntimeError("fit must be called before forecasting")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        transition, variances, mean = self._parameters()
        state = self.filtered_probabilities(returns)
        means = np.full(horizon, mean, dtype=float)
        forecast_variances = np.empty(horizon, dtype=float)
        for step in range(horizon):
            state = transition @ state
            forecast_variances[step] = float(state @ variances)
        if not np.isfinite(forecast_variances).all() or (forecast_variances <= 0).any():
            raise ValueError("regime-switching variance forecast is invalid")
        return means, forecast_variances, state


def forecast_volatility(returns, horizon: int = 5):
    """Forecast conditional volatility from scaled log returns."""
    model = MarkovSwitchingVolatility().fit(np.asarray(returns, dtype=float))
    _, variance, _ = model.forecast_from_history(np.asarray(returns, dtype=float), horizon)
    return np.sqrt(variance)
