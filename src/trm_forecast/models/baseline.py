"""Simple forecasting baselines."""
import numpy as np
import pandas as pd


class NaiveForecast:
    """Last-observation and drift baselines with a common interface."""
    def __init__(self, method: str = "last"):
        if method not in {"last", "mean", "drift"}:
            raise ValueError("method must be last, mean, or drift")
        self.method = method
        self.history_: np.ndarray | None = None

    def fit(self, y):
        values = np.asarray(y, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("y must be a non-empty finite one-dimensional series")
        self.history_ = values
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.history_ is None:
            raise RuntimeError("fit must be called before predict")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if self.method == "last":
            return np.repeat(self.history_[-1], horizon)
        if self.method == "mean":
            return np.repeat(self.history_.mean(), horizon)
        drift = (self.history_[-1] - self.history_[0]) / max(len(self.history_) - 1, 1)
        return self.history_[-1] + drift * np.arange(1, horizon + 1)


def forecast_last(series: pd.Series, horizon: int = 5) -> pd.Series:
    """Convenience function returning a dated last-value forecast."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")
    model = NaiveForecast().fit(series.dropna().to_numpy())
    dates = pd.bdate_range(series.index[-1] + pd.offsets.BDay(), periods=horizon)
    return pd.Series(model.predict(horizon), index=dates, name="forecast")
