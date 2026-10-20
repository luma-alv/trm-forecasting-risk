import numpy as np
from trm_forecast.business.exposure import exposure_value, scenario_pnl, value_at_risk


def test_exposure_and_scenario_pnl():
    assert exposure_value(100, 4000) == 400000
    assert scenario_pnl(100, 4000, 4100) == 10000


def test_historical_var():
    assert value_at_risk(100, 4000, np.array([-0.01, 0.02, 0.03]), 2 / 3) > 0
