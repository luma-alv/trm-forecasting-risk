import pandas as pd
import pytest
from trm_forecast.data.validation import DataValidationError, validate_trm


def test_validate_aliases_sorts_and_deduplicates():
    data = pd.DataFrame({"fecha": ["2024-01-02", "2024-01-01", "2024-01-01"], "valor": [4000, 3900, 3910]})
    result = validate_trm(data)
    assert list(result.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    assert result.loc["2024-01-01", "trm"] == 3910


def test_validate_rejects_non_positive_values():
    with pytest.raises(DataValidationError):
        validate_trm(pd.DataFrame({"date": ["2024-01-01"], "trm": [0]}))
