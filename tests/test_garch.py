import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

from trm_forecast.evaluation.garch import (
    block_bootstrap_coverage,
    evaluate_garch,
    evaluate_variance_benchmark,
    qlike,
)
from trm_forecast.evaluation.protocol import PartitionSpec, TemporalEvaluationProtocol
from trm_forecast.models.garch import GARCHVolatility, log_returns


def test_log_returns_are_scaled_and_validate_levels():
    levels = pd.Series([100.0, 101.0, 99.0])
    expected = np.diff(np.log(levels.to_numpy())) * 100
    np.testing.assert_allclose(log_returns(levels), expected)
    with pytest.raises(ValueError, match="positive"):
        log_returns([100.0, 0.0, 101.0])
    with pytest.raises(ValueError, match="at least two"):
        log_returns([100.0])


def test_garch_fit_and_forecast_are_valid_and_deterministic():
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.8, 120)
    first = GARCHVolatility().fit(returns)
    second = GARCHVolatility().fit(returns)
    first_mean, first_variance = first.forecast(3)
    second_mean, second_variance = second.forecast(3)
    np.testing.assert_allclose(first_mean, second_mean)
    np.testing.assert_allclose(first_variance, second_variance)
    assert np.isfinite(first_variance).all() and (first_variance > 0).all()
    with pytest.raises(RuntimeError, match="fit"):
        GARCHVolatility().forecast(1)


def test_garch_evaluation_uses_common_targets_and_origin_history_only():
    rng = np.random.default_rng(12)
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    levels = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    series = pd.Series(levels, index=dates)
    specs = (
        PartitionSpec("train", "2020-01-01", "2020-02-09"),
        PartitionSpec("validation", "2020-02-10", "2020-03-10"),
        PartitionSpec("test", "2020-03-11", None),
    )
    protocol = TemporalEvaluationProtocol("2020-03-30", (1, 2), specs)
    result = protocol.split(series)
    predictions, metrics = evaluate_garch(
        result, horizons=(1, 2), partitions=("validation",), min_history=21, refit_frequency=3
    )
    assert len(predictions) == 57
    assert predictions[["forecast_origin", "target_date", "horizon"]].duplicated().sum() == 0
    assert (pd.to_datetime(predictions["target_date"]) > pd.to_datetime(predictions["forecast_origin"])).all()
    assert (predictions["forecast_variance"] > 0).all()
    assert (predictions["lower_level"] < predictions["upper_level"]).all()
    assert metrics["failed_predictions"].sum() == 0
    first = predictions.iloc[0]
    expected_return = 100 * np.log(first["actual"] / first["origin_level"])
    assert first["realized_cumulative_return"] == pytest.approx(expected_return)

    benchmark_predictions, benchmark_metrics = evaluate_variance_benchmark(
        result, horizons=(1, 2), partitions=("validation",), scale=100
    )
    assert len(benchmark_predictions) == len(predictions)
    assert benchmark_metrics["qlike"].notna().all()

    altered = series.copy()
    altered.loc[pd.Timestamp("2020-02-11") :] *= 1.5
    altered_result = protocol.split(altered)
    altered_predictions, _ = evaluate_garch(
        altered_result, horizons=(1, 2), partitions=("validation",), min_history=21, refit_frequency=3
    )
    first_origin = predictions["forecast_origin"].min()
    first = predictions[predictions["forecast_origin"] == first_origin]
    altered_first = altered_predictions[altered_predictions["forecast_origin"] == first_origin]
    np.testing.assert_allclose(first["forecast_variance"], altered_first["forecast_variance"])


def test_qlike_rejects_invalid_variances():
    assert qlike([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.3465735903)
    with pytest.raises(ValueError, match="positive"):
        qlike([1.0], [0.0])


def test_block_bootstrap_coverage_is_deterministic_and_validates_input():
    covered = np.array([1, 1, 0, 1, 0, 1, 1, 0])
    first = block_bootstrap_coverage(covered, block_length=2, samples=100, seed=3)
    second = block_bootstrap_coverage(covered, block_length=2, samples=100, seed=3)
    assert first == second
    assert 0 <= first[0] <= first[1] <= 1
    with pytest.raises(ValueError, match="binary"):
        block_bootstrap_coverage([0, 2, 1])
