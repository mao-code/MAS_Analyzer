"""Writing per-run and per-task artifacts into the experiment hierarchy."""

from __future__ import annotations

import time
import traceback
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import BenchmarkEvaluation
from descriptor.metrics import RunOutcome
from descriptor.schema import TraceEvent

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

from cli.common import _prompt_preview, _redact_secrets, _write_json
from cli.trajectory import (
    _fallback_interaction_logs,
    _render_trajectory_markdown,
    _trace_metrics_payload,
    _trajectory_payload,
)


def _write_eval(
    path: Path,
    evaluation: BenchmarkEvaluation,
    prediction: str,
    *,
    run_outcome: RunOutcome | None = None,
    metadata_summary: dict[str, Any] | None = None,
    metadata_path: Path | None = None,
) -> None:
    details = dict(evaluation.details)
    if "run_metadata" in details and metadata_summary is not None:
        details["run_metadata"] = dict(metadata_summary)
    if metadata_summary is not None:
        details["run_metadata_summary"] = dict(metadata_summary)
    if metadata_path is not None:
        details["run_metadata_path"] = str(metadata_path.resolve())

    payload = {
        "task_id": evaluation.task_id,
        "score": evaluation.score,
        "success": evaluation.success,
        "completion": bool(run_outcome.completion) if run_outcome is not None else None,
        "details": details,
        "prediction": prediction,
    }
    if run_outcome is not None:
        payload["outcome"] = run_outcome.to_dict()
    _write_json(path, payload)


def _write_raw_output(path: Path, *, final_answer: str, run_metadata: dict[str, Any]) -> None:
    payload = {
        "final_answer": final_answer,
        "run_metadata": _redact_secrets(run_metadata),
    }
    _write_json(path, payload)


def _classify_run_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}:{exc}".lower()
    if "fatal_tool_failure" in text or "tool_failure_circuit_breaker" in text:
        return "tool_failure"
    if "429" in text or "rate-limit" in text or "rate limit" in text:
        return "llm_rate_limited"
    if "timeout" in text or "exceeded configured timeout" in text:
        return "llm_timeout"
    if "jsondecodeerror" in text or "expecting value" in text:
        return "llm_invalid_response"
    if "connection error" in text or "connectionerror" in text:
        return "llm_connection_error"
    return "run_exception"


def _failed_run_result(
    *,
    task: Any,
    benchmark_name: str,
    system_info: dict[str, Any],
    run_index: int,
    seed: int,
    exc: Exception,
    started_s: float,
) -> tuple[str, list[TraceEvent], dict[str, Any], BenchmarkEvaluation, RunOutcome]:
    ended_s = time.time()
    error_type = type(exc).__name__
    error_message = str(exc)
    failure_category = _classify_run_exception(exc)
    final_answer = (
        "Run failed and was marked for rerun. "
        f"Failure category: {failure_category}. Error: {error_type}: {error_message}"
    )
    trace_events = [
        TraceEvent(
            timestamp_start=started_s,
            timestamp_end=ended_s,
            actor="system",
            event_type="error",
            payload={
                "status": "failed",
                "failure_category": failure_category,
                "error_type": error_type,
                "error_message": error_message,
                "needs_rerun": True,
            },
            token_in=0,
            token_out=0,
            latency_ms=max(0.0, (ended_s - started_s) * 1000.0),
            cost_usd=0.0,
            state_id=f"run_{run_index}_error",
        )
    ]
    run_metadata = {
        "task_id": str(getattr(task, "task_id", "")),
        "run_index": int(run_index),
        "seed": int(seed),
        "topology": str(system_info.get("topology", "")),
        "run_status": "failed",
        "fallback": True,
        "needs_rerun": True,
        "failure_category": failure_category,
        "fallback_reason": f"{error_type}:{error_message}",
        "error": {
            "type": error_type,
            "message": error_message,
            "category": failure_category,
            "traceback": traceback.format_exc(limit=12),
        },
        "turns_executed": 0,
        "messages_sent_total": 0,
        "messages_sent_by_agent": {},
        "tool_calls_total": 0,
        "tool_call_counts": {},
        "retrieved_docids": [],
        "vote_tally": {},
        "final_reason": f"run_failed:{failure_category}",
        "interaction_logs": _fallback_interaction_logs(
            task=task,
            system_info=system_info,
            final_answer=final_answer,
        ),
        "termination_history": [
            {
                "stage_name": "run_exception",
                "should_stop": True,
                "reason": failure_category,
                "reason_detail": f"{error_type}: {error_message}",
            }
        ],
    }
    evaluation = BenchmarkEvaluation(
        task_id=str(getattr(task, "task_id", "")),
        score=0.0,
        success=False,
        details={
            "eval_mode": "run_fallback",
            "prediction": final_answer,
            "run_failed": True,
            "fallback": True,
            "needs_rerun": True,
            "failure_category": failure_category,
            "error_type": error_type,
            "error_message": error_message,
            "benchmark": benchmark_name,
        },
    )
    run_outcome = RunOutcome(
        success=False,
        completion=False,
        score=0.0,
        success_source="run_fallback",
        completion_source="run_fallback",
    )
    return final_answer, trace_events, run_metadata, evaluation, run_outcome


def _task_manifest_payload(
    *,
    task: Any,
    benchmark_name: str,
    system_label: str,
    topology: str,
) -> dict[str, Any]:
    return {
        "task_id": str(task.task_id),
        "benchmark": benchmark_name,
        "system_label": system_label,
        "topology": topology,
        "prompt": getattr(task, "prompt", ""),
        "reference_answer": getattr(task, "reference_answer", ""),
        "metadata": dict(getattr(task, "metadata", {}) or {}),
    }


def _write_run_artifacts(
    *,
    task_dir: Path,
    benchmark_name: str,
    task: Any,
    run_index: int,
    final_answer: str,
    trace_events: Sequence[Any],
    evaluation: BenchmarkEvaluation,
    run_outcome: RunOutcome,
    run_metadata: dict[str, Any],
    system_info: dict[str, Any],
) -> dict[str, Any]:
    task_manifest_path = task_dir / "task.json"
    if not task_manifest_path.exists():
        _write_json(
            task_manifest_path,
            _task_manifest_payload(
                task=task,
                benchmark_name=benchmark_name,
                system_label=str(system_info.get("system_label", "")),
                topology=str(system_info.get("topology", "")),
            ),
        )

    answer_path = task_dir / f"run_{run_index}.answer.txt"
    metadata_path = task_dir / f"run_{run_index}.metadata.json"
    result_path = task_dir / f"run_{run_index}.result.json"
    trace_metrics_path = task_dir / f"run_{run_index}.trace_metrics.json"
    trajectory_json_path = task_dir / f"run_{run_index}.trajectory.json"
    trajectory_md_path = task_dir / f"run_{run_index}.trajectory.md"

    answer_path.write_text(final_answer, encoding="utf-8")
    _write_json(metadata_path, run_metadata)

    trace_metrics = _trace_metrics_payload(
        task=task,
        benchmark_name=benchmark_name,
        run_index=run_index,
        final_answer=final_answer,
        evaluation=evaluation,
        run_outcome=run_outcome,
        run_metadata=run_metadata,
        trace_events=trace_events,
    )
    _write_json(trace_metrics_path, trace_metrics)

    trajectory_payload = _trajectory_payload(
        task=task,
        benchmark_name=benchmark_name,
        system_info=system_info,
        run_index=run_index,
        final_answer=final_answer,
        run_metadata=run_metadata,
    )
    _write_json(trajectory_json_path, trajectory_payload)
    trajectory_md_path.write_text(
        _render_trajectory_markdown(trajectory_payload),
        encoding="utf-8",
    )

    result_payload = {
        "task_id": str(task.task_id),
        "benchmark": benchmark_name,
        "run_index": int(run_index),
        "system": system_info,
        "run_status": str(run_metadata.get("run_status", "completed") or "completed"),
        "fallback": bool(run_metadata.get("fallback", False)),
        "needs_rerun": bool(run_metadata.get("needs_rerun", False)),
        "failure_category": str(run_metadata.get("failure_category", "") or ""),
        "final_answer": final_answer,
        "evaluation": {
            "score": float(evaluation.score),
            "success": bool(evaluation.success),
            "details": {
                key: value
                for key, value in dict(evaluation.details).items()
                if key != "run_metadata"
            },
        },
        "outcome": run_outcome.to_dict(),
        "trace_metrics": trace_metrics["metrics"],
        "termination": trace_metrics["termination"],
        "run_summary": trace_metrics["runtime"],
        "artifacts": {
            "task_manifest_path": str(task_manifest_path.resolve()),
            "answer_path": str(answer_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
            "trace_metrics_path": str(trace_metrics_path.resolve()),
            "trajectory_json_path": str(trajectory_json_path.resolve()),
            "trajectory_md_path": str(trajectory_md_path.resolve()),
        },
    }
    _write_json(result_path, result_payload)

    return {
        "task_manifest_path": str(task_manifest_path.resolve()),
        "answer_path": str(answer_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "result_path": str(result_path.resolve()),
        "trace_metrics_path": str(trace_metrics_path.resolve()),
        "trajectory_json_path": str(trajectory_json_path.resolve()),
        "trajectory_md_path": str(trajectory_md_path.resolve()),
        "run_status": str(run_metadata.get("run_status", "completed") or "completed"),
        "fallback": bool(run_metadata.get("fallback", False)),
        "needs_rerun": bool(run_metadata.get("needs_rerun", False)),
        "failure_category": str(run_metadata.get("failure_category", "") or ""),
    }


def _summary_task_entry_from_payload(task_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    artifacts_payload = payload.get("artifacts", {})
    return {
        "task_id": str(payload.get("task_id", "")),
        "prompt_preview": payload.get("prompt_preview", ""),
        "reference_answer": payload.get("reference_answer", ""),
        "task_dir": str(task_dir.resolve()),
        "evaluation": dict(payload.get("evaluation", {})),
        "descriptor": dict(payload.get("descriptor", {})),
        "stage_bottleneck": payload.get("stage_bottleneck", {}),
        "run_status_summary": dict(payload.get("run_status_summary", {})),
        "run_failure_count": int(payload.get("run_failure_count", 0) or 0),
        "fallback_count": int(payload.get("fallback_count", 0) or 0),
        "needs_rerun": bool(payload.get("needs_rerun", False)),
        "artifacts": {
            "task_summary_path": str((task_dir / "task_summary.json").resolve()),
            "analysis_path": str(
                artifacts_payload.get("analysis_path", (task_dir / "analysis.json").resolve())
            ),
        },
    }


def _run_progress_summary(
    run_artifacts: Sequence[dict[str, Any]],
) -> tuple[dict[str, int], int, int]:
    run_status_summary: dict[str, int] = {}
    for artifact in run_artifacts:
        status = str(artifact.get("run_status", "completed") or "completed")
        run_status_summary[status] = run_status_summary.get(status, 0) + 1
    run_failure_count = sum(
        1
        for artifact in run_artifacts
        if str(artifact.get("run_status", "completed") or "completed") != "completed"
    )
    fallback_count = sum(1 for artifact in run_artifacts if bool(artifact.get("fallback", False)))
    return run_status_summary, run_failure_count, fallback_count


def _write_task_checkpoint(
    *,
    task_dir: Path,
    task: Any,
    benchmark_name: str,
    system_info: dict[str, Any],
    runs_per_task: int,
    run_artifacts: Sequence[dict[str, Any]],
    task_failure_details: Sequence[dict[str, Any]],
    checkpoint_reason: str,
) -> None:
    run_status_summary, run_failure_count, fallback_count = _run_progress_summary(run_artifacts)
    completed_runs = len(run_artifacts)
    needs_rerun = (
        completed_runs < runs_per_task
        or run_failure_count > 0
        or fallback_count > 0
        or bool(task_failure_details)
    )
    payload = {
        "task_id": str(task.task_id),
        "benchmark": benchmark_name,
        "system": system_info,
        "task_dir": str(task_dir.resolve()),
        "prompt_preview": _prompt_preview(task.prompt),
        "reference_answer": task.reference_answer,
        "checkpoint_complete": completed_runs >= runs_per_task,
        "checkpoint_reason": checkpoint_reason,
        "checkpoint_updated_at": datetime.now(UTC).isoformat(),
        "completed_runs": completed_runs,
        "runs_per_task": int(runs_per_task),
        "runs": list(run_artifacts),
        "run_status_summary": run_status_summary,
        "run_failure_count": run_failure_count,
        "fallback_count": fallback_count,
        "needs_rerun": needs_rerun,
        "rerun_details": {
            "run_failure_count": run_failure_count,
            "fallback_count": fallback_count,
            "run_status_summary": run_status_summary,
            "failure_details": list(task_failure_details),
        },
        "evaluation": {
            "count": completed_runs,
            "success_rate": None,
            "completion_rate": None,
            "avg_score": None,
            "checkpoint_note": "Final aggregate metrics are written after all runs finish.",
        },
        "descriptor": {
            "checkpoint_note": "Final descriptor metrics are written after all runs finish.",
        },
        "stage_bottleneck": {},
        "artifacts": {
            "task_summary_path": str((task_dir / "task_summary.json").resolve()),
            "analysis_path": str((task_dir / "analysis.json").resolve()),
            "descriptor_json_path": str((task_dir / "descriptor.json").resolve()),
            "descriptor_csv_path": str((task_dir / "descriptor.csv").resolve()),
        },
    }
    _write_json(task_dir / "task_summary.json", payload)


def _summary_row_from_analysis(
    *,
    benchmark_name: str,
    system_label: str,
    topology: str,
    agents: int,
    default_model: str,
    judge_model: str,
    task_id: str,
    task_dir: Path,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    evaluation = dict(analysis.get("evaluation", {}))
    descriptor = dict(analysis.get("descriptor", {}))
    row: dict[str, Any] = {
        "benchmark": benchmark_name,
        "system_label": system_label,
        "topology": topology,
        "agents": agents,
        "default_model": default_model,
        "judge_model": judge_model,
        "task_id": task_id,
        "runs": evaluation.get("count", 0),
        "accuracy": descriptor.get("accuracy", evaluation.get("accuracy", 0.0)),
        "eval_avg_score": evaluation.get("avg_score", 0.0),
        "eval_success_rate": evaluation.get("success_rate", 0.0),
        "eval_completion_rate": evaluation.get("completion_rate", 0.0),
        "latency_e2e": descriptor.get("latency_e2e"),
        "token_total": descriptor.get("token_total"),
        "task_dir": str(task_dir.resolve()),
    }
    row.update(descriptor)
    return row
