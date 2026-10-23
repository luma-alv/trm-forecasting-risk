"""Small dependency-light Ridge regression forecaster."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeRegressor:
    """Ridge regression with train-only feature standardization.

    The feature mean and scale are learned in ``fit`` and reused unchanged in
    ``predict``. The intercept is left unpenalized, as in standard Ridge
    implementations.
    """

    alpha: float = 1.0
    coefficients_: np.ndarray | None = None
    feature_mean_: np.ndarray | None = None
    feature_scale_: np.ndarray | None = None
    intercept_: float | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> RidgeRegressor:
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
            raise ValueError("features must be a non-empty 2D array aligned with target")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("features and target must be finite")
        self.feature_mean_ = x.mean(axis=0)
        self.feature_scale_ = x.std(axis=0)
        self.feature_scale_[self.feature_scale_ == 0] = 1.0
        scaled = (x - self.feature_mean_) / self.feature_scale_
        design = np.column_stack([np.ones(len(scaled)), scaled])
        penalty = np.diag(np.r_[0.0, np.full(scaled.shape[1], self.alpha)])
        normal = design.T @ design + penalty
        right_hand_side = design.T @ y
        try:
            solution = np.linalg.solve(normal, right_hand_side)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(normal, right_hand_side, rcond=None)[0]
        self.intercept_ = float(solution[0])
        self.coefficients_ = np.asarray(solution[1:], dtype=float)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("fit must be called before predict")
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != len(self.coefficients_):
            raise ValueError("features must have the same columns used during fit")
        if not np.isfinite(x).all():
            raise ValueError("features must be finite")
        scaled = (x - self.feature_mean_) / self.feature_scale_
        return self.intercept_ + scaled @ self.coefficients_
