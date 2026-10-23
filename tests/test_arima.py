import numpy as np
import pandas as pd
import pytest

from trm_forecast.evaluation.arima import evaluate_arima
from trm_forecast.evaluation.persistence import evaluate_persistence
from trm_forecast.evaluation.protocol import PartitionSpec, TemporalEvaluationProtocol
from trm_forecast.models.arima import ARIMAForecast, stationarity_diagnostics


def _protocol_result():
    index = pd.date_range("2020-01-01", periods=24, freq="D")
    series = pd.Series(100 + np.arange(24, dtype=float), index=index)
    return TemporalEvaluationProtocol(
        as_of_date="2020-01-24",
        horizons=(1, 2),
        partitions=(PartitionSpec("train", "2020-01-01", "2020-01-12"),
                    PartitionSpec("validation", "2020-01-13", "2020-01-18"),
                    PartitionSpec("test", "2020-01-19", "2020-01-24")),
    ).split(series)


def test_arima_predictions_align_with_common_targets_and_no_future_leakage():
    result = _protocol_result()
    predictions, metrics = evaluate_arima(
        result, order=(0, 1, 0), horizons=(1, 2), min_history=5, partitions=("validation",)
    )
    persistence, _ = evaluate_persistence(result, horizons=(1, 2))
    expected = persistence.loc[persistence["partition"] == "validation", ["forecast_origin", "target_date", "horizon"]]
    actual = predictions[["forecast_origin", "target_date", "horizon"]]
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    assert set(metrics["horizon"]) == {1, 2}


def test_refit_frequency_updates_forecast_state_without_refitting_parameters():
    result = _protocol_result()
    arima_predictions, _ = evaluate_arima(
        result,
        order=(0, 1, 0),
        horizons=(1, 2),
        partitions=("validation",),
        min_history=5,
        refit_frequency=3,
    )
    persistence_predictions, _ = evaluate_persistence(result, horizons=(1, 2))
    persistence_predictions = persistence_predictions[
        persistence_predictions["partition"] == "validation"
    ]

    pd.testing.assert_series_equal(
        arima_predictions["prediction"].reset_index(drop=True),
        persistence_predictions["prediction"].reset_index(drop=True),
        check_names=False,
    )


def test_arima_order_and_insufficient_history_are_explicit():
    with pytest.raises(ValueError, match="insufficient history"):
        ARIMAForecast(order=(2, 1, 2)).fit(np.arange(5.0))


def test_arima_reproducibility_and_stationarity_diagnostics():
    values = pd.Series(100 + np.arange(30, dtype=float))
    first = stationarity_diagnostics(values)
    second = stationarity_diagnostics(values)
    pd.testing.assert_frame_equal(first, second)
    forecast_one = ARIMAForecast(order=(0, 1, 0)).fit(values).predict(2)
    forecast_two = ARIMAForecast(order=(0, 1, 0)).fit(values).predict(2)
    np.testing.assert_allclose(forecast_one, forecast_two)


def test_arima_drift_uses_linear_trend_for_integrated_model():
    values = pd.Series(100 + np.arange(30, dtype=float))
    model = ARIMAForecast(order=(0, 1, 0), trend="t").fit(values)
    forecast = model.predict(3)

    assert model.fitted_model.model.trend == "t"
    assert model.fitted_model.model.param_names[0] == "x1"
    assert np.all(np.diff(forecast) > 0)


def test_arima_drift_evaluation_preserves_target_alignment_and_identifier():
    result = _protocol_result()
    predictions, metrics = evaluate_arima(
        result,
        order=(0, 1, 0),
        horizons=(1, 2),
        partitions=("validation",),
        min_history=5,
        trend="t",
    )
    persistence, _ = evaluate_persistence(result, horizons=(1, 2))
    expected = persistence.loc[
        persistence["partition"] == "validation", ["forecast_origin", "target_date", "horizon"]
    ]
    actual = predictions[["forecast_origin", "target_date", "horizon"]]

    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    assert set(predictions["model"]) == {"arima(0, 1, 0)_drift"}
    assert set(metrics["horizon"]) == {1, 2}
