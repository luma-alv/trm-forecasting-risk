"""Robustness summaries for business-value decision outcomes."""
from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

_GROUP_COLUMNS = ["partition", "model", "operation"]


def _require_decisions(decisions: pd.DataFrame) -> None:
    required = set(_GROUP_COLUMNS) | {"gross_cop_benefit", "selected_horizon"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"decisions missing required columns: {missing}")


def decision_distribution(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return distributional and sign summaries for decision-level benefits."""
    _require_decisions(decisions)
    columns = _GROUP_COLUMNS + [
        "valid_decisions", "mean_benefit", "median_benefit", "std_benefit",
        "p05_benefit", "p25_benefit", "p75_benefit", "p95_benefit",
        "min_benefit", "max_benefit", "positive_pct", "zero_pct", "negative_pct",
        "total_benefit",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)

    def summarize(group: pd.DataFrame) -> pd.Series:
        benefits = pd.to_numeric(group["gross_cop_benefit"], errors="coerce").dropna()
        return pd.Series(
            {
                "valid_decisions": len(benefits),
                "mean_benefit": float(benefits.mean()),
                "median_benefit": float(benefits.median()),
                "std_benefit": float(benefits.std(ddof=1)) if len(benefits) > 1 else 0.0,
                "p05_benefit": float(benefits.quantile(0.05)),
                "p25_benefit": float(benefits.quantile(0.25)),
                "p75_benefit": float(benefits.quantile(0.75)),
                "p95_benefit": float(benefits.quantile(0.95)),
                "min_benefit": float(benefits.min()),
                "max_benefit": float(benefits.max()),
                "positive_pct": float((benefits > 0).mean() * 100),
                "zero_pct": float((benefits == 0).mean() * 100),
                "negative_pct": float((benefits < 0).mean() * 100),
                "total_benefit": float(benefits.sum()),
            }
        )

    return decisions.groupby(_GROUP_COLUMNS, sort=True, dropna=False).apply(
        summarize, include_groups=False
    ).reset_index()[columns]


def selected_horizon_distribution(decisions: pd.DataFrame) -> pd.DataFrame:
    """Count selected horizons and the share selected after horizon one."""
    _require_decisions(decisions)
    if decisions.empty:
        return pd.DataFrame(
            columns=_GROUP_COLUMNS + ["selected_horizon", "decisions", "pct_decisions"]
        )
    grouped = decisions.groupby(_GROUP_COLUMNS + ["selected_horizon"], sort=True, dropna=False)
    result = grouped.size().rename("decisions").reset_index()
    result["pct_decisions"] = result["decisions"] / result.groupby(_GROUP_COLUMNS)["decisions"].transform("sum") * 100
    return result


def positive_benefit_concentration(
    decisions: pd.DataFrame,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> pd.DataFrame:
    """Measure how much total positive benefit comes from the largest outcomes."""
    _require_decisions(decisions)
    if not fractions or any(fraction <= 0 or fraction > 1 for fraction in fractions):
        raise ValueError("fractions must be in the interval (0, 1]")
    columns = _GROUP_COLUMNS + ["positive_decisions", "total_positive_benefit"]
    columns += [f"top_{int(fraction * 100)}_pct_positive_benefit_share" for fraction in fractions]
    rows = []
    for keys, group in decisions.groupby(_GROUP_COLUMNS, sort=True, dropna=False):
        positive = pd.to_numeric(group["gross_cop_benefit"], errors="coerce")
        positive = positive[np.isfinite(positive) & (positive > 0)].sort_values(ascending=False)
        total = float(positive.sum())
        row = dict(zip(_GROUP_COLUMNS, keys))
        row["positive_decisions"] = len(positive)
        row["total_positive_benefit"] = total
        for fraction in fractions:
            count = max(1, ceil(len(positive) * fraction)) if len(positive) else 0
            share = float(positive.iloc[:count].sum() / total * 100) if total > 0 else np.nan
            row[f"top_{int(fraction * 100)}_pct_positive_benefit_share"] = share
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def temporal_stability(decisions: pd.DataFrame, frequency: str = "Q") -> pd.DataFrame:
    """Summarize benefit level and sign rates by non-overlapping origin periods."""
    _require_decisions(decisions)
    if "forecast_origin" not in decisions.columns:
        raise ValueError("decisions must contain forecast_origin")
    frame = decisions.copy()
    frame["origin_period"] = pd.to_datetime(frame["forecast_origin"]).dt.to_period(frequency).astype(str)
    grouped = frame.groupby(_GROUP_COLUMNS + ["origin_period"], sort=True, dropna=False)
    result = grouped.agg(
        valid_decisions=("gross_cop_benefit", "count"),
        mean_benefit=("gross_cop_benefit", "mean"),
        median_benefit=("gross_cop_benefit", "median"),
        positive_pct=("gross_cop_benefit", lambda values: float((values > 0).mean() * 100)),
        mean_selected_horizon=("selected_horizon", "mean"),
    ).reset_index()
    return result


def matched_decision_comparison(
    decisions: pd.DataFrame,
    candidate_model: str,
    reference_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare candidate and reference outcomes on identical origin/operation keys."""
    _require_decisions(decisions)
    keys = ["partition", "forecast_origin", "operation", "usd_amount"]
    missing = sorted(set(keys) - set(decisions.columns))
    if missing:
        raise ValueError(f"decisions missing matching columns: {missing}")
    subset = decisions.loc[decisions["model"].isin([candidate_model, reference_model])].copy()
    if subset.duplicated(keys + ["model"]).any():
        raise ValueError("each model must have at most one decision per matching key")
    candidate = subset.loc[subset["model"] == candidate_model, keys + ["gross_cop_benefit"]].rename(
        columns={"gross_cop_benefit": "candidate_benefit"}
    )
    reference = subset.loc[subset["model"] == reference_model, keys + ["gross_cop_benefit"]].rename(
        columns={"gross_cop_benefit": "reference_benefit"}
    )
    merged = candidate.merge(reference, on=keys, how="outer", indicator=True)
    matched = merged.loc[merged["_merge"] == "both"].drop(columns="_merge").copy()
    matched["benefit_difference"] = matched["candidate_benefit"] - matched["reference_benefit"]
    matched["candidate_better"] = matched["benefit_difference"] > 0
    matched["tie"] = matched["benefit_difference"] == 0
    matched["candidate_worse"] = matched["benefit_difference"] < 0

    rows = []
    for group_keys, group in merged.groupby(["partition", "operation"], sort=True, dropna=False):
        matched_group = group.loc[group["_merge"] == "both"]
        differences = matched_group["candidate_benefit"] - matched_group["reference_benefit"]
        row = dict(zip(["partition", "operation"], group_keys))
        row.update(
            {
                "candidate_model": candidate_model,
                "reference_model": reference_model,
                "candidate_decisions": int((group["_merge"] != "right_only").sum()),
                "reference_decisions": int((group["_merge"] != "left_only").sum()),
                "matched_decisions": len(matched_group),
                "candidate_only": int((group["_merge"] == "left_only").sum()),
                "reference_only": int((group["_merge"] == "right_only").sum()),
                "mean_benefit_difference": float(differences.mean()) if len(differences) else np.nan,
                "median_benefit_difference": float(differences.median()) if len(differences) else np.nan,
                "candidate_better_pct": float((differences > 0).mean() * 100) if len(differences) else np.nan,
                "ties_pct": float((differences == 0).mean() * 100) if len(differences) else np.nan,
                "candidate_worse_pct": float((differences < 0).mean() * 100) if len(differences) else np.nan,
            }
        )
        rows.append(row)
    return matched, pd.DataFrame(rows)


def rescale_decisions(decisions: pd.DataFrame, usd_amount: float) -> pd.DataFrame:
    """Rescale already-selected decisions to another USD notional."""
    _require_decisions(decisions)
    if not isinstance(usd_amount, (int, float, np.integer, np.floating)) or not np.isfinite(usd_amount) or usd_amount <= 0:
        raise ValueError("usd_amount must be a finite positive number")
    if decisions.empty:
        return decisions.copy()
    frame = decisions.copy()
    original_amount = pd.to_numeric(frame["usd_amount"], errors="coerce")
    if original_amount.isna().any() or (original_amount <= 0).any():
        raise ValueError("decisions must contain finite positive usd_amount values")
    frame["gross_cop_benefit"] = frame["gross_cop_benefit"] / original_amount * float(usd_amount)
    frame["usd_amount"] = float(usd_amount)
    return frame
