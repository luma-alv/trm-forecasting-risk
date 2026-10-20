"""Translate USD exposures and TRM scenarios into COP risk measures."""
import numpy as np


def exposure_value(amount_usd, trm):
    """COP value of a USD exposure at a given TRM."""
    if amount_usd < 0 or trm <= 0:
        raise ValueError("amount_usd must be non-negative and trm must be positive")
    return float(amount_usd * trm)


def scenario_pnl(amount_usd, current_trm, scenario_trm):
    """COP P&L for a USD receivable (positive when USD strengthens)."""
    if current_trm <= 0 or scenario_trm <= 0:
        raise ValueError("TRM values must be positive")
    return float(amount_usd * (scenario_trm - current_trm))


def value_at_risk(amount_usd, current_trm, returns, confidence_level=0.95):
    """Historical one-period VaR for a USD receivable, expressed as COP loss."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    losses = -amount_usd * current_trm * np.asarray(returns, dtype=float)
    return float(np.quantile(losses, confidence_level))
