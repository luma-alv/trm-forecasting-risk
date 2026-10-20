import numpy as np
import pandas as pd
import pytest

from trm_forecast.evaluation.protocol import PartitionSpec, TemporalEvaluationProtocol
from trm_forecast.evaluation.regime_switching import evaluate_regime_switching
from trm_forecast.models.regime_switching import MarkovSwitchingVolatility


def _result(series):
    dates = series.index
    specs = (
        PartitionSpec("train", dates[0].date(), dates[39].date()),
        PartitionSpec("validation", dates[40].date(), dates[69].date()),
        PartitionSpec("test", dates[70].date(), None),
    )
    return TemporalEvaluationProtocol(dates[-1].date(), (1, 2), specs).split(series)


def test_regime_switching_forecast_is_positive_and_filtered():
    rng = np.random.default_rng(21)
    returns = np.r_[rng.normal(0, 0.4, 100), rng.normal(0, 2.0, 100)]
    model = MarkovSwitchingVolatility(maxiter=200).fit(returns)
    means, variances, final_probabilities = model.forecast_from_history(returns, 3)
    assert model.fit_status in {"converged", "converged_with_warning"}
    assert np.isfinite(means).all() and np.isfinite(variances).all()
    assert (variances > 0).all()
    assert np.isclose(final_probabilities.sum(), 1.0)
    assert (final_probabilities >= 0).all()


def test_regime_evaluation_aligns_targets_and_does_not_use_future_returns():
    rng = np.random.default_rng(22)
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    levels = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    series = pd.Series(levels, index=dates)
    result = _result(series)
    predictions, metrics = evaluate_regime_switching(
        result, horizons=(1, 2), partitions=("validation",), min_history=31, maxiter=200, refit_frequency=3
    )
    assert len(predictions) == 57
    assert predictions[["forecast_origin", "target_date", "horizon"]].duplicated().sum() == 0
    assert (pd.to_datetime(predictions["target_date"]) > pd.to_datetime(predictions["forecast_origin"])).all()
    assert np.isfinite(predictions["forecast_variance"]).all()
    assert (predictions["forecast_variance"] > 0).all()
    assert np.allclose(
        predictions["filtered_regime_0"] + predictions["filtered_regime_1"], 1.0
    )
    assert metrics["failed_predictions"].sum() == 0

    altered = series.copy()
    altered.loc[pd.Timestamp("2020-02-11") :] *= 1.5
    altered_predictions, _ = evaluate_regime_switching(
        _result(altered), horizons=(1, 2), partitions=("validation",), min_history=31,
        maxiter=200, refit_frequency=3
    )
    first_origin = predictions["forecast_origin"].min()
    first = predictions[predictions["forecast_origin"] == first_origin]
    altered_first = altered_predictions[altered_predictions["forecast_origin"] == first_origin]
    np.testing.assert_allclose(first["forecast_variance"], altered_first["forecast_variance"])


def test_regime_switching_rejects_short_or_unfitted_inputs():
    with pytest.raises(ValueError, match="at least 30"):
        MarkovSwitchingVolatility().fit(np.ones(29))
    with pytest.raises(RuntimeError, match="fit"):
        MarkovSwitchingVolatility().forecast_from_history(np.ones(30), 1)
