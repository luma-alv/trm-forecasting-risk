import numpy as np
import pandas as pd

from trm_forecast.features.build_features import build_features


def test_lags_and_rolling_statistics_use_only_past_observations():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    series = pd.Series(np.arange(100.0, 106.0), index=index)
    features = build_features(series, lags=(1,), rolling_windows=(3,))

    assert features.loc[index[3], "lag_1"] == 102
    assert features.loc[index[3], "rolling_mean_3"] == 101
    assert features.loc[index[3], "rolling_std_3"] == 1

    changed_future = series.copy()
    changed_future.loc[index[4]] = 10_000
    changed_features = build_features(changed_future, lags=(1,), rolling_windows=(3,))
    pd.testing.assert_frame_equal(
        features.loc[: index[3]], changed_features.loc[: index[3]], check_names=False
    )
