import numpy as np
import pytest

from trm_forecast.business.exposure import exposure_value, scenario_pnl, value_at_risk


def test_exposure_and_scenario_pnl():
    assert exposure_value(100, 4000) == 400000
    assert scenario_pnl(100, 4000, 4100) == 10000


def test_historical_var_uses_the_upper_quantile_of_cop_losses():
    returns = np.array([-0.01, 0.02, 0.03])

    # Losses are [4000, -8000, -12000] COP. At 95%, numpy's linear
    # quantile is 2800 COP, consistent with the implementation's convention.
    assert value_at_risk(100, 4000, returns, 0.95) == pytest.approx(2800)


def test_historical_var_can_be_negative_at_low_confidence():
    # With this tiny sample, the 2/3 loss quantile is still a gain.
    assert value_at_risk(100, 4000, np.array([-0.01, 0.02, 0.03]), 2 / 3) == pytest.approx(-4000)
