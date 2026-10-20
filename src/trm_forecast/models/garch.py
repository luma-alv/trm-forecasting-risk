"""Optional GARCH volatility adapter."""
import numpy as np


def forecast_volatility(returns, horizon: int = 5, p: int = 1, q: int = 1):
    """Forecast conditional volatility from percentage returns."""
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("install the 'models' extra to use GARCH") from exc
    fitted = arch_model(np.asarray(returns, dtype=float) * 100, vol="Garch", p=p, q=q, mean="Constant").fit(disp="off")
    variance = fitted.forecast(horizon=horizon).variance.iloc[-1].to_numpy()
    return np.sqrt(variance) / 100
