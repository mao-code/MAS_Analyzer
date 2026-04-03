from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import TraceEvent
from .stages import compute_stage_metrics
from .utils import (
    compute_avg_branching,
    compute_communication_counts,
    compute_failure_mode_hist,
    compute_handoff_count,
    compute_loop_score,
    extract_tool_name,
    infer_completion,
    infer_success,
    is_tool_error,
)

QUALITY_METRICS = ("Q1_success_rate", "Q2_completion_rate")
COST_METRICS = ("C1_latency_p95", "C2_tokens_total", "C3_cost_total", "C4_tool_calls_total")
DIAGNOSTIC_METRICS = (
    "D1_tool_error_rate",
    "D2_communication_count",
    "D2_agent_to_agent_communication_count",
    "D2_system_mediated_communication_count",
    "D3_handoff_count",
)
RELIABILITY_METRICS = ("R1_success_var", "R2_latency_var", "R3_tokens_var")
PROCESS_METRICS = ("P1_steps_total", "P2_backtrack_rate", "P3_loop_score", "P4_verification_density")


@dataclass(frozen=True)
class RunOutcome:
    success: bool
    completion: bool
    score: float | None = None
    success_source: str = "trace"
    completion_source: str = "trace"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": bool(self.success),
            "completion": bool(self.completion),
            "success_source": self.success_source,
            "completion_source": self.completion_source,
        }
        if self.score is not None:
            payload["score"] = float(self.score)
        return payload


@dataclass
class ExtensionOptions:
    include_stage_metrics: bool = True
    avg_branching: bool = False
    unique_tools: bool = False
    failure_mode_hist: bool = False
    executability_score: bool = False


def resolve_run_outcome(
    events: Sequence[TraceEvent],
    *,
    evaluation: Any | None = None,
    final_answer: str | None = None,
    run_metadata: Mapping[str, object] | None = None,
) -> RunOutcome:
    success_source = "trace_inference"
    success = infer_success(events)
    score: float | None = None

    if evaluation is not None and getattr(evaluation, "success", None) is not None:
        success = bool(getattr(evaluation, "success"))
        success_source = "benchmark_evaluation"

    if evaluation is not None and getattr(evaluation, "score", None) is not None:
        score = float(getattr(evaluation, "score"))

    completion_source = "trace_inference"
    completion = infer_completion(events, final_answer=final_answer, run_metadata=run_metadata)
    if run_metadata:
        completion_source = "runtime_inference"
    if completion and final_answer and str(final_answer).strip():
        completion_source = "final_answer"

    return RunOutcome(
        success=bool(success),
        completion=bool(completion),
        score=score,
        success_source=success_source,
        completion_source=completion_source,
    )


def compute_run_metrics(
    events: Sequence[TraceEvent],
    *,
    outcome: RunOutcome | None = None,
    evaluation: Any | None = None,
    final_answer: str | None = None,
    run_metadata: Mapping[str, object] | None = None,
    extensions: ExtensionOptions | None = None,
) -> dict[str, Any]:
    if extensions is None:
        extensions = ExtensionOptions()
    if outcome is None:
        outcome = resolve_run_outcome(
            events,
            evaluation=evaluation,
            final_answer=final_answer,
            run_metadata=run_metadata,
        )

    steps_total = len(events)
    token_total = sum(event.token_in + event.token_out for event in events)
    latency_total = sum(event.latency_ms for event in events)
    cost_total = sum(event.cost_usd for event in events)

    tool_calls_total = sum(1 for event in events if event.event_type == "tool_call")
    tool_fail_total = sum(1 for event in events if is_tool_error(event))

    backtrack_count = sum(
        1 for event in events if event.event_type == "revise" or bool(event.payload.get("redo"))
    )
    backtrack_rate = backtrack_count / steps_total if steps_total else 0.0

    verify_count = sum(1 for event in events if event.event_type == "verify")
    verification_density = verify_count / steps_total if steps_total else 0.0

    loop_score = compute_loop_score(events)
    communication_counts = compute_communication_counts(events)
    handoff_count = compute_handoff_count(events)

    run_metrics: dict[str, Any] = {
        "success": bool(outcome.success),
        "completion": bool(outcome.completion),
        "success_source": outcome.success_source,
        "completion_source": outcome.completion_source,
        "latency_total": float(latency_total),
        "tokens_total": float(token_total),
        "cost_total": float(cost_total),
        "tool_calls_total": float(tool_calls_total),
        "tool_fail_total": float(tool_fail_total),
        "steps_total": float(steps_total),
        "backtrack_rate": float(backtrack_rate),
        "loop_score": float(loop_score),
        "verification_density": float(verification_density),
        "communication_count": float(communication_counts.total),
        "communication_count_agent_to_agent": float(communication_counts.agent_to_agent),
        "communication_count_system_mediated": float(communication_counts.system_mediated),
        "handoff_count": float(handoff_count),
    }
    if outcome.score is not None:
        run_metrics["score"] = float(outcome.score)

    if extensions.include_stage_metrics:
        run_metrics.update(compute_stage_metrics(events))

    if extensions.avg_branching:
        run_metrics["avg_branching"] = float(compute_avg_branching(events))

    if extensions.unique_tools:
        tools = {
            name
            for event in events
            if event.event_type in {"tool_call", "tool_result"}
            for name in [extract_tool_name(event)]
            if name
        }
        run_metrics["unique_tools"] = float(len(tools))

    if extensions.failure_mode_hist:
        run_metrics["failure_mode_hist"] = compute_failure_mode_hist(events)

    if extensions.executability_score:
        if tool_calls_total:
            run_metrics["executability_score"] = float(1.0 - (tool_fail_total / tool_calls_total))
        else:
            run_metrics["executability_score"] = 1.0

    return run_metrics


def _nanvar(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanvar(arr))


def compute_task_metrics(run_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not run_metrics:
        raise ValueError("At least one run metric entry is required")

    successes = np.array([1.0 if rm["success"] else 0.0 for rm in run_metrics])
    completions = np.array([1.0 if rm["completion"] else 0.0 for rm in run_metrics])

    latencies = np.array([float(rm["latency_total"]) for rm in run_metrics])
    tokens = np.array([float(rm["tokens_total"]) for rm in run_metrics])
    costs = np.array([float(rm["cost_total"]) for rm in run_metrics])
    tool_calls = np.array([float(rm["tool_calls_total"]) for rm in run_metrics])
    tool_fails = np.array([float(rm["tool_fail_total"]) for rm in run_metrics])

    steps = np.array([float(rm["steps_total"]) for rm in run_metrics])
    backtrack = np.array([float(rm["backtrack_rate"]) for rm in run_metrics])
    loop_scores = np.array([float(rm["loop_score"]) for rm in run_metrics])
    verify_density = np.array([float(rm["verification_density"]) for rm in run_metrics])
    communication_counts = np.array([float(rm.get("communication_count", 0.0)) for rm in run_metrics])
    communication_counts_agent = np.array(
        [float(rm.get("communication_count_agent_to_agent", 0.0)) for rm in run_metrics]
    )
    communication_counts_system = np.array(
        [float(rm.get("communication_count_system_mediated", 0.0)) for rm in run_metrics]
    )
    handoff_counts = np.array([float(rm.get("handoff_count", 0.0)) for rm in run_metrics])

    total_tool_calls = tool_calls.sum()
    total_tool_fails = tool_fails.sum()

    metrics: dict[str, Any] = {
        "Q1_success_rate": float(successes.mean()),
        "Q2_completion_rate": float(completions.mean()),
        "C1_latency_p95": float(np.percentile(latencies, 95)),
        "C2_tokens_total": float(tokens.mean()),
        "C3_cost_total": float(costs.mean()),
        "C4_tool_calls_total": float(tool_calls.mean()),
        "D1_tool_error_rate": float(total_tool_fails / total_tool_calls)
        if total_tool_calls
        else 0.0,
        "D2_communication_count": float(communication_counts.mean()),
        "D2_agent_to_agent_communication_count": float(communication_counts_agent.mean()),
        "D2_system_mediated_communication_count": float(communication_counts_system.mean()),
        "D3_handoff_count": float(handoff_counts.mean()),
        "R1_success_var": _nanvar(successes.tolist()),
        "R2_latency_var": _nanvar(latencies.tolist()),
        "R3_tokens_var": _nanvar(tokens.tolist()),
        "P1_steps_total": float(steps.mean()),
        "P2_backtrack_rate": float(backtrack.mean()),
        "P3_loop_score": float(loop_scores.mean()),
        "P4_verification_density": float(verify_density.mean()),
    }

    # Aggregate stage metrics if present
    stage_keys = [key for key in run_metrics[0].keys() if key.startswith("stage_")]
    for key in stage_keys:
        values = [float(rm.get(key, 0.0)) for rm in run_metrics]
        metrics[key] = float(np.mean(values))

    if "avg_branching" in run_metrics[0]:
        metrics["avg_branching"] = float(
            np.mean([rm.get("avg_branching", 0.0) for rm in run_metrics])
        )
    if "unique_tools" in run_metrics[0]:
        metrics["unique_tools"] = float(
            np.mean([rm.get("unique_tools", 0.0) for rm in run_metrics])
        )
    if "executability_score" in run_metrics[0]:
        metrics["executability_score"] = float(
            np.mean([rm.get("executability_score", 0.0) for rm in run_metrics])
        )

    return metrics


def aggregate_failure_modes(run_metrics: Sequence[dict[str, Any]]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for rm in run_metrics:
        data = rm.get("failure_mode_hist")
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            hist[key] = hist.get(key, 0) + int(value)
    return hist
