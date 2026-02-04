from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from .schema import TraceEvent
from .stages import compute_stage_metrics
from .utils import (
    compute_avg_branching,
    compute_failure_mode_hist,
    compute_loop_score,
    extract_tool_name,
    infer_completion,
    infer_success,
    is_tool_error,
)


class Evaluator(Protocol):
    def evaluate_success(self, events: Iterable[TraceEvent]) -> Optional[bool]:
        ...

    def evaluate_completion(self, events: Iterable[TraceEvent]) -> Optional[bool]:
        ...

    def evaluate_faithfulness(self, events: Iterable[TraceEvent]) -> Optional[float]:
        ...

    def evaluate_context_relevancy(self, events: Iterable[TraceEvent]) -> Optional[float]:
        ...


class NullEvaluator:
    def evaluate_success(self, events: Iterable[TraceEvent]) -> Optional[bool]:
        return None

    def evaluate_completion(self, events: Iterable[TraceEvent]) -> Optional[bool]:
        return None

    def evaluate_faithfulness(self, events: Iterable[TraceEvent]) -> Optional[float]:
        return None

    def evaluate_context_relevancy(self, events: Iterable[TraceEvent]) -> Optional[float]:
        return None


@dataclass
class ExtensionOptions:
    include_stage_metrics: bool = True
    avg_branching: bool = False
    unique_tools: bool = False
    failure_mode_hist: bool = False
    executability_score: bool = False


def compute_run_metrics(
    events: Sequence[TraceEvent],
    *,
    evaluator: Optional[Evaluator] = None,
    extensions: Optional[ExtensionOptions] = None,
) -> Dict[str, Any]:
    if evaluator is None:
        evaluator = NullEvaluator()
    if extensions is None:
        extensions = ExtensionOptions()

    success = evaluator.evaluate_success(events)
    if success is None:
        success = infer_success(events)

    completion = evaluator.evaluate_completion(events)
    if completion is None:
        completion = infer_completion(events)

    faithfulness = evaluator.evaluate_faithfulness(events)
    context_relevancy = evaluator.evaluate_context_relevancy(events)

    steps_total = len(events)
    token_total = sum(event.token_in + event.token_out for event in events)
    latency_total = sum(event.latency_ms for event in events)
    cost_total = sum(event.cost_usd for event in events)

    tool_calls_total = sum(1 for event in events if event.event_type == "tool_call")
    tool_fail_total = sum(1 for event in events if is_tool_error(event))

    backtrack_count = sum(
        1
        for event in events
        if event.event_type == "revise" or bool(event.payload.get("redo"))
    )
    backtrack_rate = backtrack_count / steps_total if steps_total else 0.0

    verify_count = sum(1 for event in events if event.event_type == "verify")
    verification_density = verify_count / steps_total if steps_total else 0.0

    loop_score = compute_loop_score(events)

    run_metrics: Dict[str, Any] = {
        "success": bool(success),
        "completion": bool(completion),
        "faithfulness": float(faithfulness) if faithfulness is not None else np.nan,
        "context_relevancy": float(context_relevancy)
        if context_relevancy is not None
        else np.nan,
        "latency_total": float(latency_total),
        "tokens_total": float(token_total),
        "cost_total": float(cost_total),
        "tool_calls_total": float(tool_calls_total),
        "tool_fail_total": float(tool_fail_total),
        "steps_total": float(steps_total),
        "backtrack_rate": float(backtrack_rate),
        "loop_score": float(loop_score),
        "verification_density": float(verification_density),
    }

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


def _nanmean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanvar(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanvar(arr))


def compute_task_metrics(run_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not run_metrics:
        raise ValueError("At least one run metric entry is required")

    successes = np.array([1.0 if rm["success"] else 0.0 for rm in run_metrics])
    completions = np.array([1.0 if rm["completion"] else 0.0 for rm in run_metrics])
    faithfulness = [float(rm.get("faithfulness", np.nan)) for rm in run_metrics]
    context_rel = [float(rm.get("context_relevancy", np.nan)) for rm in run_metrics]

    latencies = np.array([float(rm["latency_total"]) for rm in run_metrics])
    tokens = np.array([float(rm["tokens_total"]) for rm in run_metrics])
    costs = np.array([float(rm["cost_total"]) for rm in run_metrics])
    tool_calls = np.array([float(rm["tool_calls_total"]) for rm in run_metrics])
    tool_fails = np.array([float(rm["tool_fail_total"]) for rm in run_metrics])

    steps = np.array([float(rm["steps_total"]) for rm in run_metrics])
    backtrack = np.array([float(rm["backtrack_rate"]) for rm in run_metrics])
    loop_scores = np.array([float(rm["loop_score"]) for rm in run_metrics])
    verify_density = np.array([float(rm["verification_density"]) for rm in run_metrics])

    total_tool_calls = tool_calls.sum()
    total_tool_fails = tool_fails.sum()

    metrics: Dict[str, Any] = {
        "Q1_success_rate": float(successes.mean()),
        "Q2_completion_rate": float(completions.mean()),
        "Q3_faithfulness": _nanmean(faithfulness),
        "Q4_context_relevancy": _nanmean(context_rel),
        "C1_latency_p95": float(np.percentile(latencies, 95)),
        "C2_tokens_total": float(tokens.sum()),
        "C3_cost_total": float(costs.sum()),
        "C4_tool_calls_total": float(total_tool_calls),
        "C5_tool_error_rate": float(total_tool_fails / total_tool_calls) if total_tool_calls else 0.0,
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
        metrics["avg_branching"] = float(np.mean([rm.get("avg_branching", 0.0) for rm in run_metrics]))
    if "unique_tools" in run_metrics[0]:
        metrics["unique_tools"] = float(np.mean([rm.get("unique_tools", 0.0) for rm in run_metrics]))
    if "executability_score" in run_metrics[0]:
        metrics["executability_score"] = float(
            np.mean([rm.get("executability_score", 0.0) for rm in run_metrics])
        )

    return metrics


def aggregate_failure_modes(run_metrics: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for rm in run_metrics:
        data = rm.get("failure_mode_hist")
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            hist[key] = hist.get(key, 0) + int(value)
    return hist
