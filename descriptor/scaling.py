from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class RobustScaler:
    median_: np.ndarray
    iqr_: np.ndarray
    feature_names: List[str]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def to_dict(self) -> dict:
        return {
            "median": self.median_.tolist(),
            "iqr": self.iqr_.tolist(),
            "feature_names": self.feature_names,
        }


def robust_scale(
    X: np.ndarray, feature_names: Iterable[str] | None = None
) -> Tuple[np.ndarray, RobustScaler]:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    median = np.median(X, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    iqr_safe = np.where(iqr == 0, 1.0, iqr)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    scaler = RobustScaler(median_=median, iqr_=iqr_safe, feature_names=list(feature_names))
    return scaler.transform(X), scaler
