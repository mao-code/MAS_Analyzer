from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostMetricsConfig:
    cps_fallback: str = "nan"  # nan | inf


def compute_cost_metrics(
    run_df: pd.DataFrame,
    *,
    success_rate: float,
    config: CostMetricsConfig | None = None,
) -> dict[str, float]:
    """Compute execution-cost and coordination diagnostics metrics.

    Expected run columns: `tokens_total`, `tool_calls_total`, `tool_error_count`,
        `communication_count`, `handoff_count`.

        Notes:
        - `communication_count` should represent agent-to-agent communication.
        - When available, `communication_count_total` and
            `communication_count_system_mediated` are also summarized as diagnostics.
    """
    cfg = config or CostMetricsConfig()

    tokens = pd.to_numeric(run_df.get("tokens_total"), errors="coerce")
    tool_calls = pd.to_numeric(run_df.get("tool_calls_total"), errors="coerce")
    tool_errors = pd.to_numeric(run_df.get("tool_error_count"), errors="coerce").fillna(0.0)
    communications = pd.to_numeric(run_df.get("communication_count"), errors="coerce")
    communications_total = pd.to_numeric(
        run_df.get("communication_count_total"), errors="coerce"
    )
    communications_system_mediated = pd.to_numeric(
        run_df.get("communication_count_system_mediated"), errors="coerce"
    )
    handoffs = pd.to_numeric(run_df.get("handoff_count"), errors="coerce")
    latencies_e2e = pd.to_numeric(run_df.get("latency_e2e"), errors="coerce")

    tokens_total = float(tokens.mean()) if not tokens.dropna().empty else math.nan

    if success_rate > 0 and np.isfinite(tokens_total):
        cost_per_success = tokens_total / success_rate
    elif cfg.cps_fallback == "inf":
        cost_per_success = math.inf
    else:
        cost_per_success = math.nan

    if len(tokens.dropna()) >= 2 and tokens_total > 0:
        tokens_cv = float(tokens.std(ddof=0) / tokens_total)
    else:
        tokens_cv = math.nan

    tool_calls_total = float(tool_calls.mean()) if not tool_calls.dropna().empty else math.nan

    tool_calls_sum = float(tool_calls.fillna(0.0).sum())
    tool_error_sum = float(tool_errors.sum())
    tool_error_rate = (tool_error_sum / tool_calls_sum) if tool_calls_sum > 0 else 0.0

    return {
        "tokens_total": tokens_total,
        "token_total": tokens_total,
        "latency_e2e": float(latencies_e2e.mean()) if not latencies_e2e.dropna().empty else math.nan,
        "cost_per_success": float(cost_per_success),
        "tokens_cv": tokens_cv,
        "tool_calls_total": tool_calls_total,
        # Coordination diagnostics
        "communication_count": float(communications.mean()) if not communications.dropna().empty else math.nan,
        "communication_count_total": (
            float(communications_total.mean()) if not communications_total.dropna().empty else math.nan
        ),
        "communication_count_system_mediated": (
            float(communications_system_mediated.mean())
            if not communications_system_mediated.dropna().empty
            else math.nan
        ),
        "handoff_count": float(handoffs.mean()) if not handoffs.dropna().empty else math.nan,
        "tool_error_rate": float(tool_error_rate),
    }
