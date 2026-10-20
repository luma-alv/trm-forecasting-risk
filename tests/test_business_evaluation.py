import pandas as pd
import pytest

from trm_forecast.business.evaluation import evaluate_business_value, validate_forecast_grid


def forecasts(predictions, actuals=None, model="demo"):
    actuals = actuals or {h: 4000 + h * 10 for h in range(1, 11)}
    return pd.DataFrame(
        {
            "partition": "validation",
            "forecast_origin": "2024-01-02",
            "target_date": [f"2024-01-{h + 2:02d}" for h in range(1, 11)],
            "horizon": list(range(1, 11)),
            "prediction": [predictions[h] for h in range(1, 11)],
            "actual": [actuals[h] for h in range(1, 11)],
            "model": model,
        }
    )


def test_importer_and_exporter_sign_conventions():
    frame = forecasts({h: 4100 - h * 10 for h in range(1, 11)})
    decisions, summary = evaluate_business_value(frame, usd_amount=100)

    importer = decisions.loc[decisions["operation"] == "importer"].iloc[0]
    exporter = decisions.loc[decisions["operation"] == "exporter"].iloc[0]
    assert importer.selected_horizon == 10
    assert importer.gross_cop_benefit == pytest.approx(-9000)
    assert exporter.selected_horizon == 1
    assert exporter.gross_cop_benefit == pytest.approx(0)
    assert len(summary) == 2


def test_horizon_one_decision_and_ties_choose_first_horizon():
    frame = forecasts({h: 4000 for h in range(1, 11)}, actuals={h: 4000 for h in range(1, 11)})
    decisions, _ = evaluate_business_value(frame, 50)
    assert set(decisions["selected_horizon"]) == {1}
    assert decisions["gross_cop_benefit"].eq(0).all()


def test_missing_horizon_is_explicit_and_can_be_excluded_non_strictly():
    frame = forecasts({h: 4000 for h in range(1, 11)}).query("horizon != 10")
    with pytest.raises(ValueError, match="missing_horizon"):
        evaluate_business_value(frame, 100)
    issues = validate_forecast_grid(frame, strict=False)
    assert "missing_horizon" in set(issues["issue"])
    decisions, summary = evaluate_business_value(frame, 100, strict=False)
    assert decisions.empty
    assert summary.empty


def test_duplicate_forecast_is_rejected_without_deduplication():
    frame = forecasts({h: 4000 for h in range(1, 11)})
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate_forecast"):
        evaluate_business_value(duplicated, 100)


@pytest.mark.parametrize("amount", [0, -1, float("nan"), float("inf")])
def test_invalid_amounts_are_rejected(amount):
    with pytest.raises(ValueError, match="usd_amount"):
        evaluate_business_value(forecasts({h: 4000 for h in range(1, 11)}), amount)


def test_aggregation_reports_mean_median_positive_rate_count_and_horizon():
    first = forecasts({h: 4000 for h in range(1, 11)}, model="a")
    second = forecasts({h: 4000 for h in range(1, 11)}, model="b")
    decisions, summary = evaluate_business_value(pd.concat([first, second]), 100)
    assert set(summary["model"]) == {"a", "b"}
    assert set(summary["operation"]) == {"importer", "exporter"}
    assert (summary["valid_decisions"] == 1).all()
    assert (summary["mean_selected_horizon"] == 1).all()
    assert (summary["positive_benefit_pct"] == 0).all()
    assert len(decisions) == 4
