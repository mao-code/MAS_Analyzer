"""Detecting and reloading already-completed runs and tasks (resume support)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark import BenchmarkEvaluation
from cli.artifacts import _summary_row_from_analysis, _summary_task_entry_from_payload
from cli.common import _env_truthy, _log_progress
from descriptor.io import read_trace_jsonl
from descriptor.metrics import RunOutcome, resolve_run_outcome
from descriptor.schema import TraceEvent


def _llm_payload_needs_rerun(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if bool(payload.get("mock_used", False)):
        return True
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        if bool(metadata.get("empty_completion", False)):
            return True
        if metadata.get("failure_category"):
            return True
        if str(metadata.get("generation_status", "")).strip().lower() == "failed":
            return True
        if bool(metadata.get("tool_loop_timeout_recovered", False)):
            return True
        if metadata.get("tool_loop_recovery_mode"):
            return True
        if bool(metadata.get("tool_failure_circuit_breaker_triggered", False)):
            return True
        if metadata.get("fallback_reason"):
            return True
    return False


def _metadata_needs_rerun(run_metadata: dict[str, Any]) -> bool:
    if _env_truthy("MAS_FAIR_REPRODUCTION"):
        return False
    if str(run_metadata.get("run_status", "completed") or "completed") != "completed":
        return True
    if bool(run_metadata.get("fallback", False) or run_metadata.get("needs_rerun", False)):
        return True
    if run_metadata.get("failure_category") or run_metadata.get("fallback_reason"):
        return True
    if run_metadata.get("error"):
        return True

    for artifact in run_metadata.get("artifact_records", []):
        if isinstance(artifact, dict) and _llm_payload_needs_rerun(artifact.get("llm")):
            return True
    for log in run_metadata.get("interaction_logs", []):
        if isinstance(log, dict) and _llm_payload_needs_rerun(log.get("llm")):
            return True
    return False


def _task_payload_needs_rerun(task_dir: Path, payload: dict[str, Any]) -> bool:
    if payload.get("checkpoint_complete") is False:
        return True
    if _env_truthy("MAS_FAIR_REPRODUCTION"):
        return False
    if bool(payload.get("needs_rerun", False) or payload.get("fallback", False)):
        return True
    if int(payload.get("run_failure_count", 0) or 0) > 0:
        return True
    if int(payload.get("fallback_count", 0) or 0) > 0:
        return True

    runs = payload.get("runs", [])
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            if str(run.get("run_status", "completed") or "completed") != "completed":
                return True
            if bool(run.get("fallback", False) or run.get("needs_rerun", False)):
                return True
            metadata_path = run.get("metadata_path")
            if metadata_path:
                path = Path(str(metadata_path))
                if not path.exists():
                    path = task_dir / Path(str(metadata_path)).name
                if path.exists():
                    try:
                        metadata = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        return True
                    if isinstance(metadata, dict) and _metadata_needs_rerun(metadata):
                        return True

    for metadata_path in sorted(task_dir.glob("run_*.metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return True
        if isinstance(metadata, dict) and _metadata_needs_rerun(metadata):
            return True

    return False


def _run_artifact_paths(task_dir: Path, run_index: int) -> dict[str, Path]:
    return {
        "task_manifest_path": task_dir / "task.json",
        "answer_path": task_dir / f"run_{run_index}.answer.txt",
        "metadata_path": task_dir / f"run_{run_index}.metadata.json",
        "result_path": task_dir / f"run_{run_index}.result.json",
        "trace_path": task_dir / f"run_{run_index}.trace.jsonl",
        "eval_path": task_dir / f"run_{run_index}.eval.json",
        "trace_metrics_path": task_dir / f"run_{run_index}.trace_metrics.json",
        "trajectory_json_path": task_dir / f"run_{run_index}.trajectory.json",
        "trajectory_md_path": task_dir / f"run_{run_index}.trajectory.md",
    }


def _load_completed_run_resume(
    *,
    task_dir: Path,
    run_index: int,
) -> (
    tuple[str, list[TraceEvent], dict[str, Any], BenchmarkEvaluation, RunOutcome, dict[str, Any]]
    | None
):
    paths = _run_artifact_paths(task_dir, run_index)
    required = ("answer_path", "metadata_path", "trace_path", "eval_path")
    if any(not paths[key].exists() for key in required):
        return None

    try:
        metadata = json.loads(paths["metadata_path"].read_text(encoding="utf-8"))
        eval_payload = json.loads(paths["eval_path"].read_text(encoding="utf-8"))
        trace_events = read_trace_jsonl(paths["trace_path"], strict=False)
        final_answer = paths["answer_path"].read_text(encoding="utf-8")
    except Exception:
        return None

    if not isinstance(metadata, dict) or not isinstance(eval_payload, dict):
        return None
    if _metadata_needs_rerun(metadata):
        return None

    details = eval_payload.get("details", {})
    if not isinstance(details, dict):
        details = {}
    evaluation = BenchmarkEvaluation(
        task_id=str(eval_payload.get("task_id", "")),
        score=float(eval_payload.get("score", 0.0) or 0.0),
        success=bool(eval_payload.get("success", False)),
        details=details,
    )

    outcome_payload = eval_payload.get("outcome", {})
    if isinstance(outcome_payload, dict):
        score_value = outcome_payload.get("score")
        run_outcome = RunOutcome(
            success=bool(outcome_payload.get("success", evaluation.success)),
            completion=bool(
                outcome_payload.get("completion", eval_payload.get("completion", False))
            ),
            score=float(score_value) if score_value is not None else float(evaluation.score),
            success_source=str(outcome_payload.get("success_source", "resumed_eval")),
            completion_source=str(outcome_payload.get("completion_source", "resumed_eval")),
        )
    else:
        run_outcome = resolve_run_outcome(
            trace_events,
            evaluation=evaluation,
            final_answer=final_answer,
            run_metadata=metadata,
        )

    artifact_record = {
        "task_manifest_path": str(paths["task_manifest_path"].resolve()),
        "answer_path": str(paths["answer_path"].resolve()),
        "metadata_path": str(paths["metadata_path"].resolve()),
        "result_path": str(paths["result_path"].resolve()),
        "trace_metrics_path": str(paths["trace_metrics_path"].resolve()),
        "trajectory_json_path": str(paths["trajectory_json_path"].resolve()),
        "trajectory_md_path": str(paths["trajectory_md_path"].resolve()),
        "trace_path": str(paths["trace_path"].resolve()),
        "eval_path": str(paths["eval_path"].resolve()),
        "run_status": str(metadata.get("run_status", "completed") or "completed"),
        "fallback": bool(metadata.get("fallback", False)),
        "needs_rerun": bool(metadata.get("needs_rerun", False)),
        "failure_category": str(metadata.get("failure_category", "") or ""),
        "score": float(evaluation.score),
        "success": bool(evaluation.success),
        "completion": bool(run_outcome.completion),
    }
    return final_answer, trace_events, metadata, evaluation, run_outcome, artifact_record


def _load_completed_task_resume(
    *,
    task: Any,
    task_dir: Path,
    benchmark_name: str,
    system_info: dict[str, Any],
    runs_per_task: int,
    default_model: str,
    judge_model: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    task_summary_path = task_dir / "task_summary.json"
    if not task_summary_path.exists():
        return None

    try:
        payload = json.loads(task_summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log_progress(
            f"TASK_RESUME_INVALID task_id={task.task_id} path={task_summary_path} "
            f"error={type(exc).__name__}:{exc}"
        )
        return None

    if not isinstance(payload, dict):
        _log_progress(
            f"TASK_RESUME_INVALID task_id={task.task_id} path={task_summary_path} error=non_dict"
        )
        return None

    if str(payload.get("task_id", "")) != str(task.task_id):
        return None
    payload_benchmark = str(payload.get("benchmark", "")).strip()
    if payload_benchmark and payload_benchmark != str(benchmark_name):
        return None

    payload_system = payload.get("system", {})
    if not isinstance(payload_system, dict):
        return None
    for key in (
        "system_label",
        "topology",
        "agents",
        "max_turns",
        "discussion_rounds",
        "termination_consensus_mode",
        "final_vote_mode",
        "peer_artifact_max_chars",
        "communication_budget",
    ):
        if key in payload_system and payload_system.get(key) != system_info.get(key):
            return None

    runs = payload.get("runs", [])
    if not isinstance(runs, list) or len(runs) != runs_per_task:
        return None
    if _task_payload_needs_rerun(task_dir, payload):
        _log_progress(f"TASK_RESUME_RERUN_REQUIRED task_id={task.task_id} path={task_summary_path}")
        return None

    evaluation = payload.get("evaluation", {})
    descriptor = payload.get("descriptor", {})
    if not isinstance(evaluation, dict) or not isinstance(descriptor, dict):
        return None

    task_summary = _summary_task_entry_from_payload(task_dir, payload)
    row = _summary_row_from_analysis(
        benchmark_name=benchmark_name,
        system_label=str(system_info.get("system_label", "")),
        topology=str(system_info.get("topology", "")),
        agents=int(system_info.get("agents", 0) or 0),
        default_model=default_model,
        judge_model=judge_model,
        task_id=str(task.task_id),
        task_dir=task_dir,
        analysis={
            "evaluation": evaluation,
            "descriptor": descriptor,
        },
    )
    return task_summary, row
