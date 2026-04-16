from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AggregationConfig:
    method: str = "arithmetic"  # arithmetic | geometric | mahalanobis | topsis
    quality_weights: dict[str, float] | None = None
    cost_weights: dict[str, float] | None = None
    epsilon: float = 1e-6


def _resolve_weights(columns: Sequence[str], weights: dict[str, float] | None) -> dict[str, float]:
    if not columns:
        return {}
    if not weights:
        w = 1.0 / len(columns)
        return {c: w for c in columns}

    positive = {c: float(max(weights.get(c, 0.0), 0.0)) for c in columns}
    s = sum(positive.values())
    if s <= 0:
        w = 1.0 / len(columns)
        return {c: w for c in columns}
    return {c: v / s for c, v in positive.items()}


def _minmax_normalize(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for col in columns:
        values = pd.to_numeric(normalized[col], errors="coerce")
        min_v = values.min(skipna=True)
        max_v = values.max(skipna=True)
        if pd.isna(min_v) or pd.isna(max_v):
            normalized[f"{col}__norm"] = np.nan
        elif max_v - min_v <= 1e-12:
            normalized[f"{col}__norm"] = 0.0
        else:
            normalized[f"{col}__norm"] = (values - min_v) / (max_v - min_v)
    return normalized


def _weighted_arithmetic_row(row: pd.Series, columns: Sequence[str], weights: dict[str, float]) -> float:
    vals = []
    ws = []
    for col in columns:
        v = row.get(col)
        if pd.isna(v):
            continue
        vals.append(float(v))
        ws.append(weights[col])
    if not vals:
        return float("nan")
    ws_arr = np.asarray(ws, dtype=float)
    ws_arr = ws_arr / ws_arr.sum()
    return float(np.dot(np.asarray(vals, dtype=float), ws_arr))


def _weighted_geometric_row(
    row: pd.Series,
    columns: Sequence[str],
    weights: dict[str, float],
    *,
    epsilon: float,
) -> float:
    vals = []
    ws = []
    for col in columns:
        v = row.get(col)
        if pd.isna(v):
            continue
        vals.append(max(float(v), epsilon))
        ws.append(weights[col])
    if not vals:
        return float("nan")
    ws_arr = np.asarray(ws, dtype=float)
    ws_arr = ws_arr / ws_arr.sum()
    arr = np.asarray(vals, dtype=float)
    return float(np.exp(np.sum(ws_arr * np.log(arr))))


def _mahalanobis_distance_matrix(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.zeros(matrix.shape[0], dtype=float)
    cov = np.cov(matrix, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov_inv = np.linalg.pinv(cov)
    diff = matrix - target
    distances = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
    return distances


def _fill_with_col_mean(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce")
        if s.dropna().empty:
            out[col] = 0.0
        else:
            out[col] = s.fillna(float(s.mean()))
    return out


def compute_composites_for_group(
    group_df: pd.DataFrame,
    *,
    quality_cols: Sequence[str],
    cost_cols: Sequence[str],
    config: AggregationConfig,
) -> pd.DataFrame:
    """Compute Q/C/U for one benchmark group.

    Returns input frame with added columns: `Q`, `C`, `U`, `aggregation_method`.
    """
    frame = group_df.copy()
    frame["Q_distance_to_ideal"] = np.nan
    frame["C_distance_to_ideal"] = np.nan
    frame["Q_distance_to_anti_ideal"] = np.nan
    frame["C_distance_to_anti_ideal"] = np.nan

    q_cols = [c for c in quality_cols if c in frame.columns]
    c_cols = [c for c in cost_cols if c in frame.columns]

    # Quality values are expected in [0,1].
    for col in q_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").clip(0.0, 1.0)

    # Cost values are min-max normalized for arithmetic/geometric methods.
    if config.method in {"arithmetic", "geometric"}:
        frame = _minmax_normalize(frame, c_cols)
        cost_input_cols = [f"{c}__norm" for c in c_cols]
    else:
        cost_input_cols = list(c_cols)

    q_weights = _resolve_weights(q_cols, config.quality_weights)
    c_weights = _resolve_weights(cost_input_cols, config.cost_weights)

    if config.method == "arithmetic":
        frame["Q"] = frame.apply(
            _weighted_arithmetic_row,
            axis=1,
            columns=q_cols,
            weights=q_weights,
        )
        frame["C"] = frame.apply(
            _weighted_arithmetic_row,
            axis=1,
            columns=cost_input_cols,
            weights=c_weights,
        )

    elif config.method == "geometric":
        frame["Q"] = frame.apply(
            _weighted_geometric_row,
            axis=1,
            columns=q_cols,
            weights=q_weights,
            epsilon=config.epsilon,
        )
        frame["C"] = frame.apply(
            _weighted_geometric_row,
            axis=1,
            columns=cost_input_cols,
            weights=c_weights,
            epsilon=config.epsilon,
        )

    elif config.method in {"mahalanobis", "topsis"}:
        q_matrix = _fill_with_col_mean(frame[q_cols]) if q_cols else pd.DataFrame(index=frame.index)
        c_matrix = _fill_with_col_mean(frame[c_cols]) if c_cols else pd.DataFrame(index=frame.index)

        if q_cols:
            q_arr = q_matrix.to_numpy(dtype=float)
            q_ideal = np.ones(q_arr.shape[1], dtype=float)
            q_dist = _mahalanobis_distance_matrix(q_arr, q_ideal)
            q_max = float(np.max(q_dist)) if len(q_dist) else 0.0
            frame["Q_distance_to_ideal"] = q_dist
            frame["Q"] = 1.0 if q_max <= 1e-12 else 1.0 - (q_dist / q_max)
        else:
            frame["Q"] = np.nan

        if c_cols:
            c_arr = c_matrix.to_numpy(dtype=float)
            c_ideal = np.min(c_arr, axis=0)
            c_dist = _mahalanobis_distance_matrix(c_arr, c_ideal)
            c_max = float(np.max(c_dist)) if len(c_dist) else 0.0
            frame["C_distance_to_ideal"] = c_dist
            frame["C"] = 0.0 if c_max <= 1e-12 else (c_dist / c_max)
        else:
            frame["C"] = np.nan

        if config.method == "topsis":
            # TOPSIS-style proximity overlay (optional extension).
            if q_cols:
                q_arr = q_matrix.to_numpy(dtype=float)
                q_anti = np.min(q_arr, axis=0)
                q_ideal = np.max(q_arr, axis=0)
                dq_pos = _mahalanobis_distance_matrix(q_arr, q_ideal)
                dq_neg = _mahalanobis_distance_matrix(q_arr, q_anti)
                frame["Q_distance_to_anti_ideal"] = dq_neg
                q_topsis = np.divide(dq_neg, dq_pos + dq_neg, out=np.zeros_like(dq_neg), where=(dq_pos + dq_neg) > 0)
                frame["Q"] = 0.5 * frame["Q"].to_numpy(dtype=float) + 0.5 * q_topsis
            if c_cols:
                c_arr = c_matrix.to_numpy(dtype=float)
                c_ideal = np.min(c_arr, axis=0)
                c_anti = np.max(c_arr, axis=0)
                dc_pos = _mahalanobis_distance_matrix(c_arr, c_ideal)
                dc_neg = _mahalanobis_distance_matrix(c_arr, c_anti)
                frame["C_distance_to_anti_ideal"] = dc_neg
                c_topsis = np.divide(dc_pos, dc_pos + dc_neg, out=np.zeros_like(dc_pos), where=(dc_pos + dc_neg) > 0)
                frame["C"] = 0.5 * frame["C"].to_numpy(dtype=float) + 0.5 * c_topsis

    else:
        raise ValueError(f"Unsupported aggregation method: {config.method}")

    frame["Q"] = pd.to_numeric(frame["Q"], errors="coerce").clip(0.0, 1.0)
    frame["C"] = pd.to_numeric(frame["C"], errors="coerce").clip(0.0, 1.0)
    frame["U"] = frame["Q"] - frame["C"]
    frame["aggregation_method"] = config.method
    return frame


def compute_composites(
    summary_df: pd.DataFrame,
    *,
    quality_cols: Sequence[str],
    cost_cols: Sequence[str],
    config: AggregationConfig,
) -> pd.DataFrame:
    """Compute composites per benchmark to preserve fair normalization context."""
    parts: list[pd.DataFrame] = []
    for _, g in summary_df.groupby("benchmark", dropna=False):
        parts.append(
            compute_composites_for_group(
                g,
                quality_cols=quality_cols,
                cost_cols=cost_cols,
                config=config,
            )
        )
    if not parts:
        return summary_df.copy()
    return pd.concat(parts, ignore_index=True)
