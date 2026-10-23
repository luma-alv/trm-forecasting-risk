"""Business-value evaluation of horizon-selection forecasts.

The evaluator deliberately separates forecast selection from realization: only
the predicted TRM values choose a horizon; actual TRM values are used afterward
to measure the outcome of that decision.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = tuple(range(1, 11))
OPERATIONS = ("importer", "exporter")


def _prediction_column(frame: pd.DataFrame) -> str:
    if "prediction" in frame.columns:
        return "prediction"
    if "predicted" in frame.columns:
        return "predicted"
    raise ValueError("forecast output must contain 'prediction' or 'predicted'")


def _validate_inputs(
    forecasts: pd.DataFrame,
    horizons: tuple[int, ...],
    usd_amount: float,
) -> tuple[pd.DataFrame, str, list[str]]:
    if not isinstance(forecasts, pd.DataFrame) or forecasts.empty:
        raise ValueError("forecasts must be a non-empty pandas DataFrame")
    if not horizons or any(not isinstance(h, (int, np.integer)) or h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    if not isinstance(usd_amount, (int, float, np.integer, np.floating)) or not np.isfinite(usd_amount):
        raise ValueError("usd_amount must be a finite positive number")
    if usd_amount <= 0:
        raise ValueError("usd_amount must be greater than zero")
    required = {"forecast_origin", "horizon", "actual", "model"}
    missing = sorted(required - set(forecasts.columns))
    if missing:
        raise ValueError(f"forecasts missing required columns: {missing}")
    prediction_column = _prediction_column(forecasts)
    frame = forecasts.copy()
    frame["_partition"] = frame.get("partition", "unspecified")
    numeric_columns = ["horizon", "actual", prediction_column]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["horizon"].isna().any() or (frame["horizon"] % 1 != 0).any():
        raise ValueError("horizon values must be integers")
    if not frame["model"].notna().all() or not frame["forecast_origin"].notna().all():
        raise ValueError("model and forecast_origin must not be missing")
    return frame, prediction_column, ["_partition", "forecast_origin", "model", "horizon"]


def validate_forecast_grid(
    forecasts: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Validate origin/model horizon groups and return explicit issue rows.

    A strict evaluation raises on the first issue. With ``strict=False``, issue
    rows are returned and the evaluator excludes those origin/model groups.
    No values are filled or deduplicated.
    """
    requested = tuple(int(h) for h in horizons)
    frame, prediction_column, keys = _validate_inputs(forecasts, requested, 1.0)
    issues: list[dict[str, object]] = []
    duplicate_mask = frame.duplicated(keys, keep=False)
    for key_values in frame.loc[duplicate_mask, keys].drop_duplicates().itertuples(index=False, name=None):
        issues.append({**dict(zip(keys, key_values)), "issue": "duplicate_forecast"})
    for key_values, group in frame.groupby(keys[:-1], sort=False, dropna=False):
        available = set(group["horizon"].astype(int))
        missing = sorted(set(requested) - available)
        invalid = group.loc[
            ~np.isfinite(group[prediction_column]) | ~np.isfinite(group["actual"])
        ]
        if missing:
            issues.append({**dict(zip(keys[:-1], key_values)), "issue": "missing_horizon", "missing_horizons": missing})
        if not invalid.empty:
            issues.append({**dict(zip(keys[:-1], key_values)), "issue": "invalid_forecast_or_actual"})
    result = pd.DataFrame(issues)
    if strict and not result.empty:
        issue = result.iloc[0].to_dict()
        raise ValueError(f"invalid forecast grid: {issue}")
    return result


def evaluate_business_value(
    forecasts: pd.DataFrame,
    usd_amount: float,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    operations: Iterable[str] = OPERATIONS,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate importer/exporter timing decisions for each valid origin/model.

    Returns ``(decisions, aggregation)``. ``strict=False`` excludes complete
    origin/model groups with duplicates, missing horizons, or non-finite values;
    use :func:`validate_forecast_grid` to inspect those exclusions.
    """
    requested = tuple(int(h) for h in horizons)
    operation_values = tuple(operation.lower() for operation in operations)
    if set(operation_values) != set(OPERATIONS):
        raise ValueError("operations must contain importer and exporter")
    frame, prediction_column, keys = _validate_inputs(forecasts, requested, usd_amount)
    issues = validate_forecast_grid(frame, requested, strict=strict)
    bad_groups = set()
    if not issues.empty:
        bad_groups = {
            tuple(row[key] for key in keys[:-1]) for _, row in issues.iterrows()
        }
    decisions: list[dict[str, object]] = []
    for key_values, group in frame.groupby(keys[:-1], sort=True, dropna=False):
        if tuple(key_values) in bad_groups:
            continue
        group = group.loc[group["horizon"].isin(requested)].sort_values("horizon")
        if len(group) != len(requested):
            continue
        selected_importer = group.loc[group[prediction_column].idxmin()]
        selected_exporter = group.loc[group[prediction_column].idxmax()]
        horizon_one = group.loc[group["horizon"] == 1].iloc[0]
        for operation, selected in (("importer", selected_importer), ("exporter", selected_exporter)):
            selected_actual = float(selected["actual"])
            actual_one = float(horizon_one["actual"])
            benefit = (
                usd_amount * (actual_one - selected_actual)
                if operation == "importer"
                else usd_amount * (selected_actual - actual_one)
            )
            row: dict[str, object] = {
                "partition": key_values[0],
                "forecast_origin": key_values[1],
                "model": key_values[2],
                "operation": operation,
                "usd_amount": float(usd_amount),
                "selected_horizon": int(selected["horizon"]),
                "predicted_trm_at_selection": float(selected[prediction_column]),
                "realized_trm_at_selection": selected_actual,
                "realized_trm_horizon_1": actual_one,
                "gross_cop_benefit": float(benefit),
            }
            if "target_date" in selected.index:
                row["selected_target_date"] = selected["target_date"]
            decisions.append(row)
    decisions_frame = pd.DataFrame(decisions)
    if decisions_frame.empty:
        return decisions_frame, pd.DataFrame(columns=_aggregation_columns())
    return decisions_frame, aggregate_business_value(decisions_frame)


def _aggregation_columns() -> list[str]:
    return [
        "partition", "model", "operation", "mean_gross_cop_benefit",
        "median_gross_cop_benefit", "positive_benefit_pct", "valid_decisions",
        "mean_selected_horizon",
    ]


def aggregate_business_value(decisions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate decision-level outcomes by partition, model, and operation."""
    required = {"partition", "model", "operation", "gross_cop_benefit", "selected_horizon"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"decisions missing required columns: {missing}")
    if decisions.empty:
        return pd.DataFrame(columns=_aggregation_columns())
    grouped = decisions.groupby(["partition", "model", "operation"], sort=True, dropna=False)
    result = grouped.agg(
        mean_gross_cop_benefit=("gross_cop_benefit", "mean"),
        median_gross_cop_benefit=("gross_cop_benefit", "median"),
        positive_benefit_pct=("gross_cop_benefit", lambda values: float((values > 0).mean() * 100)),
        valid_decisions=("gross_cop_benefit", "count"),
        mean_selected_horizon=("selected_horizon", "mean"),
    ).reset_index()
    return result[_aggregation_columns()]
