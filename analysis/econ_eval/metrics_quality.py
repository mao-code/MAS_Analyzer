from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityMetricsConfig:
    pass_k_values: Sequence[int] = (1, 3, 5)


def pass_at_k(total_runs: int, success_count: int, k: int) -> float:
    """Compute pass@k with the standard estimator.

    Returns `nan` when `total_runs <= 0` or `k <= 0`.
    Returns `1.0` when `k > total_runs - success_count`.
    """
    if total_runs <= 0 or k <= 0:
        return math.nan
    if k > total_runs - success_count:
        return 1.0
    if k > total_runs:
        return math.nan

    try:
        numerator = math.comb(total_runs - success_count, k)
        denominator = math.comb(total_runs, k)
    except ValueError:
        return math.nan
    return 1.0 - (numerator / denominator)


def compute_quality_metrics(
    run_df: pd.DataFrame,
    *,
    config: QualityMetricsConfig | None = None,
) -> dict[str, float]:
    """Compute quality metrics for one system on one benchmark.

    Expected run columns: `success`, `completion`, `score`.
    """
    cfg = config or QualityMetricsConfig()
    successes = pd.to_numeric(run_df.get("success"), errors="coerce").fillna(0.0)
    completions = pd.to_numeric(run_df.get("completion"), errors="coerce").fillna(0.0)
    scores = pd.to_numeric(run_df.get("score"), errors="coerce")

    n = int(len(successes))
    c = int(successes.sum())

    success_rate = float(successes.mean()) if n else math.nan
    completion_rate = float(completions.mean()) if n else math.nan

    # Bernoulli variance normalized into [0, 1].
    if n >= 1:
        success_var = float(np.var(successes.to_numpy(dtype=float), ddof=0))
        stability = float(np.clip(1.0 - (success_var / 0.25), 0.0, 1.0))
    else:
        stability = math.nan

    eval_avg_score = float(scores.mean()) if not scores.dropna().empty else math.nan

    metrics: dict[str, float] = {
        "success_rate": success_rate,
        "stability": stability,
        "eval_avg_score": eval_avg_score,
        "completion_rate": completion_rate,
        "runs": float(n),
        "success_count": float(c),
    }
    for k in cfg.pass_k_values:
        metrics[f"pass_at_{k}"] = pass_at_k(n, c, int(k))
    return metrics
