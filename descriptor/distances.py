from __future__ import annotations

from typing import Tuple

import numpy as np


def covariance_inverse(X: np.ndarray, *, regularization: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    cov = np.cov(X, rowvar=False)
    if regularization:
        cov = cov + np.eye(cov.shape[0]) * regularization
    cov_inv = np.linalg.pinv(cov)
    return cov, cov_inv


def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> float:
    delta = x - mean
    dist_sq = float(delta.T @ cov_inv @ delta)
    return float(np.sqrt(max(0.0, dist_sq)))


def pairwise_mahalanobis(X: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    return np.array([mahalanobis_distance(row, mean, cov_inv) for row in X])
