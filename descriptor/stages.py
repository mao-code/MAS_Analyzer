from __future__ import annotations

from typing import Dict, Iterable

from .schema import TraceEvent
from .utils import is_tool_error

STAGES = ("plan", "retrieve", "act", "verify", "revise", "finalize")

EVENT_STAGE_MAP = {
    "plan": "plan",
    "tool_call": "retrieve",
    "tool_result": "retrieve",
    "act": "act",
    "verify": "verify",
    "revise": "revise",
    "finalize": "finalize",
    "error": "finalize",
}


def stage_for_event(event: TraceEvent) -> str:
    return EVENT_STAGE_MAP.get(event.event_type, "act")


def compute_stage_metrics(events: Iterable[TraceEvent]) -> Dict[str, float]:
    counts = {stage: 0 for stage in STAGES}
    tokens = {stage: 0.0 for stage in STAGES}
    latency = {stage: 0.0 for stage in STAGES}
    tool_errors = {stage: 0.0 for stage in STAGES}
    verify_counts = {stage: 0 for stage in STAGES}

    for event in events:
        stage = stage_for_event(event)
        counts[stage] += 1
        tokens[stage] += event.token_in + event.token_out
        latency[stage] += event.latency_ms
        if is_tool_error(event):
            tool_errors[stage] += 1
        if event.event_type == "verify":
            verify_counts[stage] += 1

    metrics: Dict[str, float] = {}
    for stage in STAGES:
        count = counts[stage]
        metrics[f"stage_{stage}_events"] = float(count)
        metrics[f"stage_{stage}_tokens"] = float(tokens[stage])
        metrics[f"stage_{stage}_latency_ms"] = float(latency[stage])
        metrics[f"stage_{stage}_tool_errors"] = float(tool_errors[stage])
        metrics[f"stage_{stage}_verify_density"] = (
            float(verify_counts[stage]) / count if count else 0.0
        )
    return metrics
