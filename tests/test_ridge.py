import numpy as np
import pandas as pd

from trm_forecast.business.evaluation import evaluate_business_value
from trm_forecast.evaluation.protocol import PartitionSpec, TemporalEvaluationProtocol
from trm_forecast.evaluation.ridge import evaluate_ridge
from trm_forecast.models.ridge import RidgeRegressor


def _protocol_result():
    index = pd.date_range("2020-01-01", periods=36, freq="D")
    series = pd.Series(100 + np.arange(36, dtype=float), index=index)
    return TemporalEvaluationProtocol(
        as_of_date="2020-02-05",
        horizons=(1, 2),
        partitions=(
            PartitionSpec("train", "2020-01-01", "2020-01-24"),
            PartitionSpec("validation", "2020-01-25", "2020-01-30"),
            PartitionSpec("test", "2020-01-31", "2020-02-05"),
        ),
    ).split(series)


def test_ridge_scaling_is_fit_on_training_rows_only():
    model = RidgeRegressor(alpha=1.0).fit(np.array([[0.0], [2.0]]), np.array([1.0, 3.0]))
    assert model.feature_mean_[0] == 1.0
    assert model.feature_scale_[0] == 1.0
    np.testing.assert_allclose(model.predict(np.array([[1.0]])), [2.0], atol=1e-8)


def test_ridge_evaluation_aligns_origins_targets_and_supports_business_value():
    result = _protocol_result()
    predictions, metrics = evaluate_ridge(
        result,
        alpha=1.0,
        horizons=(1, 2),
        partitions=("validation",),
        lags=(1, 2),
        rolling_windows=(3,),
    )

    assert len(predictions) == 9
    assert predictions["forecast_origin"].notna().all()
    assert predictions["target_date"].notna().all()
    assert set(predictions["horizon"]) == {1, 2}
    assert set(metrics["horizon"]) == {1, 2}
    assert predictions["status"].eq("converged").all()

    decisions, summary = evaluate_business_value(
        predictions, usd_amount=100, horizons=(1, 2), strict=False
    )
    assert len(decisions) == 8
    assert set(summary["operation"]) == {"importer", "exporter"}
