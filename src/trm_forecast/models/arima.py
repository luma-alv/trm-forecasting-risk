"""Optional ARIMA adapter."""
import numpy as np


def forecast_arima(y, horizon: int = 5, order=(1, 1, 1)):
    """Fit ARIMA and return a forecast; requires statsmodels."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError("install the 'models' extra to use ARIMA") from exc
    if horizon < 1:
        raise ValueError("horizon must be positive")
    fitted = ARIMA(np.asarray(y, dtype=float), order=order).fit()
    return np.asarray(fitted.forecast(steps=horizon))
