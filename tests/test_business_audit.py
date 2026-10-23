import pandas as pd
import pytest

from trm_forecast.business.audit import (
    decision_distribution,
    matched_decision_comparison,
    positive_benefit_concentration,
    rescale_decisions,
    temporal_stability,
)


@pytest.fixture
def decisions():
    return pd.DataFrame(
        [
            {"partition": "test", "forecast_origin": "2024-01-02", "model": "ridge", "operation": "importer", "usd_amount": 100, "selected_horizon": 2, "gross_cop_benefit": 100},
            {"partition": "test", "forecast_origin": "2024-02-02", "model": "ridge", "operation": "importer", "usd_amount": 100, "selected_horizon": 1, "gross_cop_benefit": 0},
            {"partition": "test", "forecast_origin": "2024-03-02", "model": "ridge", "operation": "importer", "usd_amount": 100, "selected_horizon": 3, "gross_cop_benefit": -50},
            {"partition": "test", "forecast_origin": "2024-01-02", "model": "persistence", "operation": "importer", "usd_amount": 100, "selected_horizon": 1, "gross_cop_benefit": 0},
            {"partition": "test", "forecast_origin": "2024-02-02", "model": "persistence", "operation": "importer", "usd_amount": 100, "selected_horizon": 1, "gross_cop_benefit": 20},
            {"partition": "test", "forecast_origin": "2024-03-02", "model": "persistence", "operation": "importer", "usd_amount": 100, "selected_horizon": 1, "gross_cop_benefit": -10},
        ]
    )


def test_distribution_reports_percentiles_and_signs(decisions):
    result = decision_distribution(decisions)
    ridge = result.loc[result["model"] == "ridge"].iloc[0]
    assert ridge["mean_benefit"] == pytest.approx(50 / 3)
    assert ridge["positive_pct"] == pytest.approx(100 / 3)
    assert ridge["zero_pct"] == pytest.approx(100 / 3)
    assert ridge["negative_pct"] == pytest.approx(100 / 3)
    assert ridge["p05_benefit"] == pytest.approx(-45)


def test_concentration_uses_positive_decisions_only(decisions):
    result = positive_benefit_concentration(decisions, fractions=(0.5, 1.0))
    ridge = result.loc[result["model"] == "ridge"].iloc[0]
    assert ridge["positive_decisions"] == 1
    assert ridge["top_50_pct_positive_benefit_share"] == pytest.approx(100)


def test_matched_comparison_and_amount_sensitivity(decisions):
    matched, summary = matched_decision_comparison(decisions, "ridge", "persistence")
    assert len(matched) == 3
    assert summary.iloc[0]["candidate_better_pct"] == pytest.approx(100 / 3)
    scaled = rescale_decisions(decisions, 200)
    assert scaled.loc[scaled["model"] == "ridge", "gross_cop_benefit"].tolist() == [200, 0, -100]


def test_temporal_stability_uses_origin_periods(decisions):
    result = temporal_stability(decisions)
    assert set(result["origin_period"]) == {"2024Q1"}
    assert result["valid_decisions"].sum() == len(decisions)


def test_invalid_amount_is_rejected(decisions):
    with pytest.raises(ValueError, match="finite positive"):
        rescale_decisions(decisions, 0)
