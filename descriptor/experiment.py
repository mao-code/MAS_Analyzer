from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from .descriptor import (
    DescriptorResult,
    compute_descriptor_from_runs,
    write_descriptor_csv,
    write_descriptor_json,
)
from .io import write_trace_jsonl
from .metrics import ExtensionOptions
from .schema import TraceEvent


def write_run_trace(events: Iterable[TraceEvent], path: str | Path) -> None:
    """Write one run trace in the shared JSONL format."""

    write_trace_jsonl(events, path)


def analyze_task_runs(
    *,
    task_id: str,
    benchmark_name: str,
    run_traces: Sequence[Sequence[TraceEvent]],
    evaluations: Sequence[Any],
    output_dir: str | Path,
    extensions: ExtensionOptions | None = None,
) -> Dict[str, Any]:
    """Compute descriptor artifacts and emit analysis.json for one task."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptor_result: DescriptorResult = compute_descriptor_from_runs(
        run_traces,
        evaluator=None,
        extensions=extensions,
    )

    descriptor_json_path = output_dir / "descriptor.json"
    descriptor_csv_path = output_dir / "descriptor.csv"
    write_descriptor_json(descriptor_result, descriptor_json_path)
    write_descriptor_csv(descriptor_result, descriptor_csv_path)

    evaluation_summary = _summarize_evaluations(evaluations)
    stage_bottleneck = _infer_stage_bottleneck(descriptor_result.metrics)

    analysis = {
        "task_id": task_id,
        "benchmark": benchmark_name,
        "evaluation": evaluation_summary,
        "descriptor": descriptor_result.metrics,
        "descriptor_extensions": descriptor_result.extensions,
        "stage_bottleneck": stage_bottleneck,
        "per_run_descriptor": descriptor_result.per_run,
    }

    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return analysis


def _summarize_evaluations(evaluations: Sequence[Any]) -> Dict[str, Any]:
    scores = []
    successes = []
    items = []

    for entry in evaluations:
        score = float(getattr(entry, "score", 0.0))
        success = bool(getattr(entry, "success", False))
        task_id = str(getattr(entry, "task_id", ""))
        details = getattr(entry, "details", {})

        scores.append(score)
        successes.append(1.0 if success else 0.0)
        items.append(
            {
                "task_id": task_id,
                "score": score,
                "success": success,
                "details": details,
            }
        )

    if not scores:
        return {
            "count": 0,
            "avg_score": 0.0,
            "success_rate": 0.0,
            "runs": [],
        }

    return {
        "count": len(scores),
        "avg_score": float(sum(scores) / len(scores)),
        "success_rate": float(sum(successes) / len(successes)),
        "runs": items,
    }


def _infer_stage_bottleneck(metrics: Dict[str, Any]) -> Dict[str, Any]:
    stage_latency = {
        key: float(value)
        for key, value in metrics.items()
        if key.startswith("stage_") and key.endswith("_latency_ms")
    }
    stage_tokens = {
        key: float(value)
        for key, value in metrics.items()
        if key.startswith("stage_") and key.endswith("_tokens")
    }

    peak_latency_stage = max(stage_latency, key=stage_latency.get) if stage_latency else None
    peak_token_stage = max(stage_tokens, key=stage_tokens.get) if stage_tokens else None

    findings = []
    if float(metrics.get("P4_verification_density", 0.0)) < 0.05:
        findings.append("Verification is sparse; hallucination risk may be elevated")

    retrieve_errors = float(metrics.get("stage_retrieve_tool_errors", 0.0))
    if retrieve_errors > 0:
        findings.append("Retrieve stage has tool errors; retries may inflate latency")

    revise_tokens = float(metrics.get("stage_revise_tokens", 0.0))
    act_tokens = float(metrics.get("stage_act_tokens", 0.0))
    if revise_tokens > act_tokens and revise_tokens > 0:
        findings.append("Revise stage dominates token usage; execution loop may be unstable")

    return {
        "peak_latency_stage": peak_latency_stage,
        "peak_token_stage": peak_token_stage,
        "findings": findings,
    }
