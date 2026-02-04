from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


ObjectiveSpec = Dict[str, str]


def pareto_frontier(
    df: pd.DataFrame, objectives: ObjectiveSpec, *, return_mask: bool = False
) -> pd.DataFrame | np.ndarray:
    metrics = list(objectives.keys())
    values = df[metrics].to_numpy(dtype=float)
    directions = np.array([1.0 if objectives[m] == "max" else -1.0 for m in metrics])

    # Convert to maximization by flipping minimization objectives
    scores = values * directions

    n = scores.shape[0]
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[i]:
                continue
            if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                mask[i] = False
        
    return mask if return_mask else df.loc[mask].copy()


def normalize_objectives(
    df: pd.DataFrame, objectives: ObjectiveSpec
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    metrics = list(objectives.keys())
    norm_df = pd.DataFrame(index=df.index)
    bounds: Dict[str, Tuple[float, float]] = {}
    for metric in metrics:
        values = df[metric].astype(float)
        min_v = float(values.min())
        max_v = float(values.max())
        bounds[metric] = (min_v, max_v)
        if max_v == min_v:
            scaled = np.full_like(values, 0.5, dtype=float)
        else:
            scaled = (values - min_v) / (max_v - min_v)
        if objectives[metric] == "min":
            scaled = 1.0 - scaled
        norm_df[metric] = scaled
    return norm_df, bounds


def ideal_point_distance(
    df: pd.DataFrame,
    objectives: ObjectiveSpec,
    *,
    weights: Dict[str, float] | None = None,
) -> Tuple[pd.Series, np.ndarray, pd.DataFrame]:
    norm_df, _ = normalize_objectives(df, objectives)
    metrics = list(objectives.keys())
    if weights is None:
        weight_vec = np.ones(len(metrics), dtype=float)
    else:
        weight_vec = np.array([weights.get(metric, 1.0) for metric in metrics], dtype=float)
    ideal = np.ones(len(metrics), dtype=float)
    diff = (norm_df[metrics].to_numpy(dtype=float) - ideal) * weight_vec
    distances = np.linalg.norm(diff, axis=1)
    return pd.Series(distances, index=df.index, name="d_ideal"), ideal, norm_df
