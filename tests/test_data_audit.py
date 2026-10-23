import pandas as pd

from trm_forecast.data.audit import audit_trm


def test_audit_reports_count_and_date_range():
    report = audit_trm(pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "trm": [4000, 4010]}))

    assert report.observation_count == 2
    assert report.earliest_date == "2024-01-02"
    assert report.latest_date == "2024-01-03"
    assert report.unique_date_count == 2


def test_audit_detects_duplicates_and_unsorted_input():
    report = audit_trm(pd.DataFrame({"date": ["2024-01-03", "2024-01-03", "2024-01-02"], "trm": [3, 4, 2]}))

    assert report.duplicated_date_count == 1
    assert report.rows_in_duplicated_dates == 2
    assert report.multiple_observations_per_date is True
    assert report.is_sorted_chronologically is False


def test_audit_counts_missing_invalid_zero_and_negative_values():
    report = audit_trm(
        pd.DataFrame(
            {
                "date": ["2024-01-01", "bad", None, "2024-01-04", "2024-01-05", "2024-01-06"],
                "trm": ["not numeric", 4000, None, float("inf"), 0, -1],
            }
        )
    )

    assert report.missing_date_count == 1
    assert report.invalid_date_count == 1
    assert report.missing_value_count == 1
    assert report.non_numeric_value_count == 1
    assert report.non_finite_value_count == 1
    assert report.zero_value_count == 1
    assert report.negative_value_count == 1
    assert len(report.invalid_date_rows) == 2
    assert len(report.invalid_value_rows) == 5


def test_audit_calculates_calendar_and_generic_business_day_gaps():
    report = audit_trm(
        pd.DataFrame({"date": ["2024-01-03", "2024-01-04", "2024-01-08", "2024-01-10"], "trm": [1, 2, 3, 4]})
    )

    assert [gap["calendar_days"] for gap in report.calendar_day_gaps] == [1, 4, 2]
    assert [gap["business_days_monday_friday"] for gap in report.business_day_gaps] == [1, 2, 2]
    assert len(report.candidate_missing_periods) == 2


def test_empty_input_is_handled_and_results_are_reproducible():
    empty = pd.DataFrame({"date": pd.Series(dtype="object"), "trm": pd.Series(dtype="float64")})
    first = audit_trm(empty)
    second = audit_trm(empty)

    assert first.observation_count == 0
    assert first.earliest_date is None and first.latest_date is None
    assert first.to_dict() == second.to_dict()
