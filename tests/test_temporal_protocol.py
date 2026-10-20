import numpy as np
import pandas as pd
import pytest

from trm_forecast.evaluation.persistence import evaluate_persistence
from trm_forecast.evaluation.protocol import PartitionSpec, TemporalEvaluationProtocol


def synthetic_series(periods=30):
    dates = pd.date_range("2020-01-01", periods=periods, freq="D")
    return pd.Series(np.arange(periods, dtype=float), index=dates, name="trm")


def test_protocol_partitions_are_chronological_and_apply_as_of_cutoff():
    series = synthetic_series(30)
    protocol = TemporalEvaluationProtocol(
        as_of_date="2020-01-29",
        horizons=(1, 2),
        partitions=(
            PartitionSpec("train", "2020-01-01", "2020-01-10"),
            PartitionSpec("validation", "2020-01-11", "2020-01-20"),
            PartitionSpec("test", "2020-01-21", None),
        ),
    )

    result = protocol.split(series)

    assert result.excluded_after_cutoff == 1
    assert [summary.observation_count for summary in result.summaries] == [10, 10, 9]
    assert result.summaries[-1].latest_observation == "2020-01-29"
    assert result.series.index.max() == pd.Timestamp("2020-01-29")


def test_protocol_rejects_overlapping_boundaries_and_insufficient_history():
    overlapping = TemporalEvaluationProtocol(
        as_of_date="2020-01-30",
        horizons=(1,),
        partitions=(
            PartitionSpec("train", "2020-01-01", "2020-01-10"),
            PartitionSpec("test", "2020-01-10", None),
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        overlapping.split(synthetic_series())

    short = TemporalEvaluationProtocol(
        as_of_date="2020-01-03",
        horizons=(1, 2, 3),
        partitions=(PartitionSpec("test", "2020-01-01", None),),
    )
    with pytest.raises(ValueError, match="at least 4"):
        short.split(synthetic_series())

    future_partition = TemporalEvaluationProtocol(
        as_of_date="2020-01-10",
        horizons=(1,),
        partitions=(PartitionSpec("test", "2020-01-01", "2020-01-11"),),
    )
    with pytest.raises(ValueError, match="after as_of_date"):
        future_partition.split(synthetic_series())


def test_persistence_aligns_targets_by_observation_horizon_without_future_features():
    series = synthetic_series(12)
    protocol = TemporalEvaluationProtocol(
        as_of_date="2020-01-12",
        horizons=(1, 2),
        partitions=(PartitionSpec("test", "2020-01-01", None),),
    )
    result = protocol.split(series)
    predictions, metrics = evaluate_persistence(result, horizons=(1, 2))

    first = predictions.iloc[0]
    assert first["forecast_origin"] == "2020-01-01"
    assert first["target_date"] == "2020-01-02"
    assert first["horizon"] == 1
    assert first["prediction"] == 0
    assert first["actual"] == 1
    assert (pd.to_datetime(predictions["target_date"]) > pd.to_datetime(predictions["forecast_origin"])).all()
    origin_values = series.reindex(pd.to_datetime(predictions["forecast_origin"])).to_numpy()
    np.testing.assert_array_equal(predictions["prediction"].to_numpy(), origin_values)
    assert set(metrics["horizon"]) == {1, 2}
    assert metrics["mae"].notna().all() and metrics["rmse"].notna().all()


def test_persistence_evaluation_is_reproducible_and_rejects_bad_series():
    series = synthetic_series(12)
    protocol = TemporalEvaluationProtocol(
        as_of_date="2020-01-12",
        horizons=(1, 2),
        partitions=(PartitionSpec("test", "2020-01-01", None),),
    )
    result = protocol.split(series)
    first_predictions, first_metrics = evaluate_persistence(result)
    second_predictions, second_metrics = evaluate_persistence(result)

    pd.testing.assert_frame_equal(first_predictions, second_predictions)
    pd.testing.assert_frame_equal(first_metrics, second_metrics)

    with pytest.raises(ValueError, match="sorted"):
        protocol.split(series.sample(frac=1, random_state=1))
