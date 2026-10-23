"""Forecast error metrics."""
import numpy as np


def _arrays(actual, predicted):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape or a.size == 0:
        raise ValueError("actual and predicted must have the same non-empty shape")
    return a, p


def mae(actual, predicted):
    a, p = _arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted):
    a, p = _arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual, predicted):
    a, p = _arrays(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when actual values contain zero")
    return float(np.mean(np.abs((a - p) / a)) * 100)


def directional_accuracy(actual, predicted, previous=None):
    a, p = _arrays(actual, predicted)
    base = np.roll(a, 1) if previous is None else np.asarray(previous, dtype=float)
    if base.shape != a.shape:
        raise ValueError("previous must match actual shape")
    return float(np.mean(np.sign(a - base) == np.sign(p - base)))
