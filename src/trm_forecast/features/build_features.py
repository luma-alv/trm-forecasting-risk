"""Leakage-aware feature engineering."""
import numpy as np
import pandas as pd


def build_features(series: pd.Series | pd.DataFrame, lags=(1, 2, 5, 10), rolling_windows=(5, 10, 20)) -> pd.DataFrame:
    """Build features known at the start of each day from a TRM series."""
    values = series.iloc[:, 0] if isinstance(series, pd.DataFrame) else series
    values = pd.to_numeric(values, errors="coerce").rename("trm").sort_index()
    result = pd.DataFrame(index=values.index)
    result["trm"] = values
    result["log_return"] = np.log(values).diff()
    for lag in lags:
        result[f"lag_{lag}"] = values.shift(lag)
    for window in rolling_windows:
        history = values.shift(1)
        result[f"rolling_mean_{window}"] = history.rolling(window).mean()
        result[f"rolling_std_{window}"] = history.rolling(window).std()
    result["day_of_week"] = result.index.dayofweek
    return result
