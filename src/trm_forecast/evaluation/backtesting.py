"""Expanding-window backtesting for one-step or multi-step forecasts."""
import numpy as np
import pandas as pd
from .metrics import mae, rmse, mape


def backtest(series, model_factory, initial_train_size: int = 60, horizon: int = 1, step: int = 1):
    """Run walk-forward evaluation and return predictions plus aggregate metrics."""
    values = series.to_numpy(dtype=float) if isinstance(series, pd.Series) else np.asarray(series, dtype=float)
    if initial_train_size < 1 or horizon < 1 or step < 1:
        raise ValueError("initial_train_size, horizon, and step must be positive")
    rows = []
    for origin in range(initial_train_size, len(values) - horizon + 1, step):
        model = model_factory()
        model.fit(values[:origin])
        predicted = np.asarray(model.predict(horizon), dtype=float)
        actual = values[origin:origin + horizon]
        for offset, (a, p) in enumerate(zip(actual, predicted), start=1):
            rows.append({"origin": origin, "horizon": offset, "actual": a, "predicted": p})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("series is too short for the requested backtest")
    summary = {"mae": mae(result.actual, result.predicted), "rmse": rmse(result.actual, result.predicted), "mape": mape(result.actual, result.predicted)}
    return result, summary
