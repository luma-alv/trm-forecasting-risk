import numpy as np
import pandas as pd
from trm_forecast.evaluation.backtesting import backtest
from trm_forecast.models.baseline import NaiveForecast


def test_backtest_returns_metrics_and_predictions():
    series = pd.Series(np.arange(1, 31, dtype=float))
    predictions, metrics = backtest(series, NaiveForecast, initial_train_size=10)
    assert len(predictions) == 20
    assert metrics["mae"] == 1
