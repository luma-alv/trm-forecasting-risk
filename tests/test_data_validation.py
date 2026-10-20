import pandas as pd
import pytest

from trm_forecast.data.validation import DataValidationError, DuplicateDateError, validate_trm


def test_normalizes_valid_data_to_canonical_schema():
    raw = pd.DataFrame({"VIGENCIADESDE": ["2024-01-02", "2024-01-01"], "VALOR": [4000, "3900.5"]})
    result = validate_trm(raw, date_column="vigenciadesde", value_column="valor", source="test", retrieved_at="2024-02-01T12:00:00Z")
    assert list(result.columns) == ["date", "trm", "source", "retrieved_at"]
    assert result["date"].tolist() == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    assert result["trm"].tolist() == [3900.5, 4000.0]
    assert result["source"].eq("test").all()


@pytest.mark.parametrize("value", [None, "not-a-number", float("inf"), float("-inf"), 0, -1])
def test_rejects_invalid_trm_values(value):
    raw = pd.DataFrame({"date": ["2024-01-01"], "trm": [value]})
    with pytest.raises(DataValidationError):
        validate_trm(raw, source="test")


def test_rejects_invalid_dates():
    raw = pd.DataFrame({"date": ["not-a-date"], "trm": [4000]})
    with pytest.raises(DataValidationError, match="invalid dates"):
        validate_trm(raw, source="test")


def test_rejects_duplicate_dates_without_silent_deduplication():
    raw = pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "trm": [4000, 4001]})
    with pytest.raises(DuplicateDateError):
        validate_trm(raw, source="test")


def test_duplicate_policy_is_explicit_and_preserves_rows_when_enabled():
    raw = pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "trm": [4000, 4001]})
    result = validate_trm(raw, source="test", allow_duplicates=True)
    assert len(result) == 2
