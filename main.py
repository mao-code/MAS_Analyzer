from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import BenchmarkEvaluation, get_benchmark, list_benchmarks
from descriptor.experiment import analyze_task_runs, write_run_trace
from descriptor.io import read_trace_jsonl
from descriptor.metrics import RunOutcome, compute_run_metrics, resolve_run_outcome
from descriptor.schema import TraceEvent
from MAS import MASRunner, OpenRouterLLMClient, load_experiment_config
from MAS.langgraph_engine import ExperimentSpec, LangGraphMASEngine

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    from datetime import timezone

    UTC = timezone.utc


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _log_progress(message: str) -> None:
    print(f"[{_now_stamp()}] {message}", flush=True)


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OutputPaths:
    output_layout: str
    experiment_id: str
    experiment_root: Path
    benchmark_root: Path
    run_root: Path
    system_label: str


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    items = [item.strip() for item in str(raw).split(",")]
    values = [int(item) for item in items if item]
    return values or None


def _parse_str_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or None


def _benchmark_section_config(config: Any, benchmark_name: str) -> dict[str, Any]:
    if benchmark_name == "finance_agent":
        cfg = dict(config.finance_agent)
    elif benchmark_name == "browsecomp":
        cfg = dict(config.browsecomp)
    elif benchmark_name == "stabletoolbench":
        cfg = dict(config.stabletoolbench)
    elif benchmark_name == "plancraft":
        return dict(config.plancraft)
    elif benchmark_name == "workbench":
        return dict(config.workbench)
    elif benchmark_name == "scicode":
        return dict(config.scicode)
    elif benchmark_name == "agentbench":
        return dict(config.agentbench)
    elif benchmark_name == "webshop":
        return dict(config.webshop)
    else:
        return {}

    # Inject global openrouter config as fallback for LLM judge.
    # Benchmark-specific [browsecomp.openrouter] overrides take precedence.
    if "openrouter" not in cfg:
        cfg["openrouter"] = {}
    or_defaults = {
        "api_key": config.openrouter.api_key,
        "base_url": config.openrouter.base_url,
    }
    for key, value in or_defaults.items():
        if key not in cfg["openrouter"] and value:
            cfg["openrouter"][key] = value

    return cfg


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


def _write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sanitized_row = {
                key: ("" if isinstance(value, float) and math.isnan(value) else value)
                for key, value in row.items()
            }
            writer.writerow(sanitized_row)


def _default_system_label(config: Any) -> str:
    return str(config.mas.resolved_topology())


def _resolve_output_paths(
    *,
    args: argparse.Namespace,
    config: Any,
    benchmark_name: str,
    output_root: Path,
) -> OutputPaths:
    output_layout = str(args.output_layout)
    system_label = str(args.system_label or _default_system_label(config))

    if output_layout == "hierarchical":
        experiment_id = str(args.experiment_id or _now_stamp())
        experiment_root = output_root / experiment_id
        benchmark_root = experiment_root / benchmark_name
        run_root = benchmark_root / system_label
        return OutputPaths(
            output_layout=output_layout,
            experiment_id=experiment_id,
            experiment_root=experiment_root,
            benchmark_root=benchmark_root,
            run_root=run_root,
            system_label=system_label,
        )

    experiment_id = _now_stamp()
    experiment_root = output_root / experiment_id
    benchmark_root = experiment_root / benchmark_name
    return OutputPaths(
        output_layout=output_layout,
        experiment_id=experiment_id,
        experiment_root=experiment_root,
        benchmark_root=benchmark_root,
        run_root=experiment_root,
        system_label=system_label,
    )


def _apply_mas_overrides(config: Any, args: argparse.Namespace) -> None:
    mas_cfg = config.mas

    agents_per_level = _parse_int_list(args.agents_per_level)
    group_sizes = _parse_int_list(args.group_sizes)
    agent_types = _parse_str_list(args.agent_types)

    if args.topology is not None:
        mas_cfg.topology = str(args.topology)
    if args.agents is not None:
        mas_cfg.number_of_agents = int(args.agents)
        if agents_per_level is None:
            mas_cfg.agents_per_level = None
    if agents_per_level is not None:
        mas_cfg.agents_per_level = list(agents_per_level)
        mas_cfg.number_of_agents = int(sum(agents_per_level))
        mas_cfg.levels = len(agents_per_level)
    if group_sizes is not None:
        mas_cfg.group_sizes = list(group_sizes)
    if args.communication_budget is not None:
        mas_cfg.communication_count_internally = int(args.communication_budget)
    if args.mas_rounds is not None:
        mas_cfg.max_turns = max(1, int(args.mas_rounds))
        mas_cfg.turn_mode = "single_turn" if mas_cfg.max_turns <= 1 else "multi_turn"
    if args.discussion_rounds is not None:
        mas_cfg.discussion_rounds = max(1, int(args.discussion_rounds))
    if args.termination_consensus_mode is not None:
        mas_cfg.termination_consensus_mode = str(args.termination_consensus_mode)
    if getattr(args, "final_vote_mode", None) is not None:
        mas_cfg.final_vote_mode = str(args.final_vote_mode)
    if args.peer_artifact_max_chars is not None:
        mas_cfg.peer_artifact_max_chars = max(32, int(args.peer_artifact_max_chars))
    if args.default_model is not None:
        config.models["default"] = str(args.default_model)
    if args.judge_model is not None:
        config.models["judge"] = str(args.judge_model)
    if agent_types is not None:
        mas_cfg.agent_types = list(agent_types)
    if getattr(args, "no_dynamic_roles", False):
        mas_cfg.enable_dynamic_roles = False

    config.validate()


def _apply_benchmark_overrides(
    benchmark_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = dict(benchmark_cfg)
    benchmark_eval_judge_model = getattr(args, "benchmark_eval_judge_model", None)
    if benchmark_eval_judge_model is not None:
        cfg["judge_model"] = str(benchmark_eval_judge_model)
    return cfg


def _redact_secrets(data: Any, *, parent_key: str = "") -> Any:
    secret_markers = ("api_key", "token", "secret", "password")
    key_lower = parent_key.lower()

    if isinstance(data, dict):
        return {key: _redact_secrets(value, parent_key=str(key)) for key, value in data.items()}
    if isinstance(data, list):
        return [_redact_secrets(value, parent_key=parent_key) for value in data]
    if isinstance(data, tuple):
        return tuple(_redact_secrets(value, parent_key=parent_key) for value in data)
    if isinstance(data, str) and any(marker in key_lower for marker in secret_markers):
        return "***REDACTED***" if data else ""
    return data


def _mas_mode_label(config: Any) -> str:
    return "SAS" if config.mas.total_agents == 1 else "MAS"


def _runtime_tools(config: Any, benchmark_name: str, benchmark_cfg: dict[str, Any]) -> list[str]:
    tools: list[str] = []

    # Current MAS runtime only emits this synthetic coordination tool.
    if config.mas.communication_count_internally > 0 and config.mas.total_agents > 1:
        tools.append("inter_agent_send")

    if benchmark_name == "browsecomp":
        if bool(benchmark_cfg.get("enable_tools", True)):
            tools.append("search")
            if bool(benchmark_cfg.get("include_get_document", True)):
                tools.append("get_document")
        return tools
    if benchmark_name == "stabletoolbench":
        if bool(benchmark_cfg.get("enable_tools", True)):
            tools.append("stabletoolbench_virtual_api")
        return tools
    return tools


def _prompt_preview(prompt: Any, *, limit: int = 280) -> str:
    if isinstance(prompt, list):
        parts = []
        for item in prompt[:6]:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
            parts.append(f"{role}: {content}")
        text = "\n".join(parts)
    else:
        text = str(prompt)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _compact_run_metadata(run_metadata: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "task_id": run_metadata.get("task_id"),
        "run_index": run_metadata.get("run_index"),
        "seed": run_metadata.get("seed"),
        "topology": run_metadata.get("topology"),
        "run_status": run_metadata.get("run_status", "completed"),
        "fallback": bool(run_metadata.get("fallback", False)),
        "needs_rerun": bool(run_metadata.get("needs_rerun", False)),
        "turns_executed": run_metadata.get("turns_executed"),
        "messages_sent_total": run_metadata.get("messages_sent_total", 0),
        "messages_sent_by_agent": run_metadata.get("messages_sent_by_agent", {}),
        "tool_calls_total": run_metadata.get("tool_calls_total", 0),
        "tool_call_counts": run_metadata.get("tool_call_counts", {}),
        "retrieved_docids": run_metadata.get("retrieved_docids", []),
        "vote_tally": run_metadata.get("vote_tally", {}),
        "final_reason": run_metadata.get("final_reason", ""),
    }
    for key in (
        "reward",
        "num_steps",
        "terminated",
        "truncated",
        "function_calls",
        "error",
        "agentbench_status",
        "agentbench_result",
        "agentbench_turns",
        "steps_taken",
        "final_reward",
        "paper_score_100",
        "failure_category",
        "fallback_reason",
    ):
        if key in run_metadata:
            payload[key] = run_metadata[key]
    return payload


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


def _normalized_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _text_preview(text: Any, *, limit: int = 220) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _stage_metric_payload(
    run_metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metric_suffixes = (
        "events",
        "latency_ms",
        "tokens",
        "tool_errors",
        "verify_density",
    )
    core_metrics: dict[str, Any] = {}
    stage_metrics: dict[str, dict[str, Any]] = {}

    for key, value in run_metrics.items():
        if not key.startswith("stage_"):
            core_metrics[key] = value
            continue

        remainder = key[len("stage_") :]
        matched = False
        for suffix in metric_suffixes:
            marker = f"_{suffix}"
            if not remainder.endswith(marker):
                continue
            stage_name = remainder[: -len(marker)]
            stage_metrics.setdefault(stage_name, {})[suffix] = value
            matched = True
            break
        if not matched:
            core_metrics[key] = value

    return core_metrics, stage_metrics


def _trace_metrics_payload(
    *,
    task: Any,
    benchmark_name: str,
    run_index: int,
    final_answer: str,
    evaluation: BenchmarkEvaluation,
    run_outcome: RunOutcome,
    run_metadata: dict[str, Any],
    trace_events: Sequence[Any],
) -> dict[str, Any]:
    run_metrics = compute_run_metrics(
        trace_events,
        outcome=run_outcome,
        final_answer=final_answer,
        run_metadata=run_metadata,
    )
    core_metrics, stage_metrics = _stage_metric_payload(run_metrics)
    termination_history = list(run_metadata.get("termination_history", []))

    return {
        "task_id": str(task.task_id),
        "benchmark": benchmark_name,
        "run_index": int(run_index),
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
        "final_answer_preview": _text_preview(final_answer, limit=320),
        "metrics": core_metrics,
        "stages": stage_metrics,
        "runtime": _compact_run_metadata(run_metadata),
        "termination": termination_history[-1] if termination_history else None,
    }


def _fallback_interaction_logs(
    *,
    task: Any,
    system_info: dict[str, Any],
    final_answer: str,
) -> list[dict[str, Any]]:
    prompt = getattr(task, "prompt", "")
    if isinstance(prompt, list):
        prompt_messages = list(prompt)
    else:
        prompt_messages = [{"role": "user", "content": str(prompt)}]

    return [
        {
            "outer_step_index": 0,
            "dispatch_id": 0,
            "agent_id": "agent_0",
            "agent_role": system_info.get("mode", "agent"),
            "agent_type": "",
            "phase": "solve",
            "round_index": 0,
            "discussion_index": 0,
            "prompt_messages": prompt_messages,
            "visible_messages": [],
            "assistant_message": {"role": "assistant", "content": final_answer},
            "tool_calls": [],
        }
    ]


def _build_prompt_catalog(
    interaction_logs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, int]]:
    catalog: list[dict[str, Any]] = []
    prompt_id_by_key: dict[str, str] = {}
    prompt_order: dict[str, int] = {}

    for log in interaction_logs:
        for message in log.get("prompt_messages", []):
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            key = json.dumps(
                {"role": role, "content": content},
                sort_keys=True,
                ensure_ascii=False,
            )
            prompt_id = prompt_id_by_key.get(key)
            if prompt_id is None:
                prompt_id = f"p_{len(catalog) + 1}"
                prompt_id_by_key[key] = prompt_id
                prompt_order[prompt_id] = len(catalog)
                catalog.append(
                    {
                        "prompt_id": prompt_id,
                        "role": role,
                        "content": content,
                        "usage_count": 1,
                    }
                )
                continue

            catalog[prompt_order[prompt_id]]["usage_count"] = (
                int(catalog[prompt_order[prompt_id]].get("usage_count", 0)) + 1
            )

    return catalog, prompt_id_by_key, prompt_order


def _collect_message_catalog(
    *,
    run_metadata: dict[str, Any],
    interaction_logs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages_by_id: dict[str, dict[str, Any]] = {}

    def register(raw_message: dict[str, Any], *, outer_step_index: int) -> None:
        message_id = str(raw_message.get("message_id", ""))
        if not message_id:
            return
        recipients_value = raw_message.get("recipients", raw_message.get("to", []))
        if isinstance(recipients_value, str):
            recipients = [recipients_value]
        else:
            recipients = [str(item) for item in recipients_value]
        messages_by_id[message_id] = {
            "message_id": message_id,
            "outer_step_index": _normalized_int(
                raw_message.get("outer_step_index", outer_step_index)
            ),
            "dispatch_id": _normalized_int(raw_message.get("dispatch_id", 0)),
            "from": str(raw_message.get("sender", raw_message.get("from", "system"))),
            "to": recipients,
            "kind": str(raw_message.get("kind", "")),
            "phase": str(raw_message.get("phase", "")),
            "round_index": _normalized_int(
                raw_message.get("round", raw_message.get("round_index", 0))
            ),
            "discussion_index": _normalized_int(
                raw_message.get("discussion_index", raw_message.get("discussion", 0))
            ),
            "artifact_id": str(raw_message.get("artifact_id", ""))
            if raw_message.get("artifact_id")
            else "",
            "content": str(raw_message.get("content", "")),
        }

    for raw_message in run_metadata.get("relay_messages", []):
        if isinstance(raw_message, dict):
            register(
                raw_message,
                outer_step_index=_normalized_int(raw_message.get("outer_step_index", 0)),
            )

    for log in interaction_logs:
        outer_step_index = _normalized_int(log.get("outer_step_index", 0))
        for raw_message in log.get("visible_messages", []):
            if not isinstance(raw_message, dict):
                continue
            register(raw_message, outer_step_index=outer_step_index)

    return sorted(
        messages_by_id.values(),
        key=lambda item: (
            int(item.get("outer_step_index", 0)),
            int(item.get("dispatch_id", 0)),
            int(item.get("round_index", 0)),
            int(item.get("discussion_index", 0)),
            str(item.get("message_id", "")),
        ),
    )


def _trajectory_payload(
    *,
    task: Any,
    benchmark_name: str,
    system_info: dict[str, Any],
    run_index: int,
    final_answer: str,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    interaction_logs = list(run_metadata.get("interaction_logs", []))
    if not interaction_logs:
        interaction_logs = _fallback_interaction_logs(
            task=task,
            system_info=system_info,
            final_answer=final_answer,
        )

    raw_role_assignment = (
        dict(run_metadata.get("role_assignment", {}))
        if isinstance(run_metadata.get("role_assignment"), dict)
        else {}
    )
    role_assignment = {
        "enabled": bool(raw_role_assignment.get("enabled", False)),
        "benchmark_name": str(raw_role_assignment.get("benchmark_name", "")),
        "used_fallback": bool(raw_role_assignment.get("used_fallback", False)),
        "fallback_reason": str(raw_role_assignment.get("fallback_reason", "")),
        "prompt_messages": [
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
            for message in raw_role_assignment.get("prompt_messages", [])
            if isinstance(message, dict)
        ],
        "response": str(raw_role_assignment.get("response", "")),
        "llm": dict(raw_role_assignment.get("llm", {}))
        if isinstance(raw_role_assignment.get("llm"), dict)
        else {},
        "assignments": {
            str(agent_id): {
                "role_name": str(details.get("role_name", "")),
                "persona": str(details.get("persona", "")),
            }
            for agent_id, details in raw_role_assignment.get("assignments", {}).items()
            if isinstance(details, dict)
        }
        if isinstance(raw_role_assignment.get("assignments"), dict)
        else {},
    }

    prompt_catalog, prompt_id_by_key, prompt_order = _build_prompt_catalog(interaction_logs)
    message_catalog = _collect_message_catalog(
        run_metadata=run_metadata,
        interaction_logs=interaction_logs,
    )
    messages_by_group: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for message in message_catalog:
        group_key = (
            int(message.get("outer_step_index", 0)),
            int(message.get("dispatch_id", 0)),
        )
        messages_by_group.setdefault(group_key, []).append(message)

    termination_by_group: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for raw_item in run_metadata.get("termination_history", []):
        if not isinstance(raw_item, dict):
            continue
        entry = {
            "outer_step_index": _normalized_int(raw_item.get("outer_step_index", 0)),
            "dispatch_id": _normalized_int(raw_item.get("dispatch_id", 0)),
            "stage_name": str(raw_item.get("stage_name", "")),
            "next_step": str(raw_item.get("next_step", "")),
            "should_stop": bool(raw_item.get("should_stop", False)),
            "reason": str(raw_item.get("reason", "")),
            "reason_detail": str(raw_item.get("reason_detail", "")),
            "consensus_ratio": raw_item.get("consensus_ratio"),
            "average_confidence": raw_item.get("average_confidence"),
            "mean_delta": raw_item.get("mean_delta"),
            "progress_source": raw_item.get("progress_source"),
            "progress_status": raw_item.get("progress_status"),
            "expected_improvement": raw_item.get("expected_improvement"),
            "progress_explanation": raw_item.get("progress_explanation"),
            "valid_artifact_count": raw_item.get("valid_artifact_count"),
        }
        group_key = (entry["outer_step_index"], entry["dispatch_id"])
        termination_by_group.setdefault(group_key, []).append(entry)

    grouped_logs: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for log in interaction_logs:
        group_key = (
            _normalized_int(log.get("outer_step_index", 0)),
            _normalized_int(log.get("dispatch_id", 0)),
        )
        grouped_logs.setdefault(group_key, []).append(log)

    steps: list[dict[str, Any]] = []
    for step_index, group_key in enumerate(sorted(grouped_logs), start=1):
        logs = sorted(
            grouped_logs[group_key],
            key=lambda item: (
                str(item.get("agent_id", "")),
                str(item.get("phase", "")),
            ),
        )
        outer_step_index, dispatch_id = group_key
        phases = sorted({str(log.get("phase", "")) for log in logs if str(log.get("phase", ""))})
        round_index = _normalized_int(logs[0].get("round_index", 0))
        discussion_index = _normalized_int(logs[0].get("discussion_index", 0))

        agent_entries: list[dict[str, Any]] = []
        prompt_sets: list[set[str]] = []
        for log in logs:
            prompt_ids: list[str] = []
            for message in log.get("prompt_messages", []):
                role = str(message.get("role", "user"))
                content = str(message.get("content", ""))
                key = json.dumps(
                    {"role": role, "content": content},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                prompt_id = prompt_id_by_key[key]
                prompt_ids.append(prompt_id)

            prompt_sets.append(set(prompt_ids))
            tool_calls: list[dict[str, Any]] = []
            for call in log.get("tool_calls", []):
                entry = {
                    "tool_name": str(call.get("tool_name", "")),
                    "status": str(call.get("status", "")),
                    "arguments": call.get("arguments", {}),
                    "output_preview": str(call.get("output_preview", "")),
                }
                if call.get("error") is not None:
                    entry["error"] = call.get("error")
                tool_calls.append(entry)

            agent_entries.append(
                {
                    "agent_id": str(log.get("agent_id", "")),
                    "role": str(log.get("agent_role", "")),
                    "agent_type": str(log.get("agent_type", "")),
                    "prompt_ids": prompt_ids,
                    "inbox_message_ids": [
                        str(message.get("message_id", ""))
                        for message in log.get("visible_messages", [])
                        if str(message.get("message_id", ""))
                    ],
                    "tool_calls": tool_calls,
                    "response": str(log.get("assistant_message", {}).get("content", "")),
                    "llm": dict(log.get("llm", {})) if isinstance(log.get("llm"), dict) else {},
                }
            )

        shared_prompt_ids: set[str] = set()
        if prompt_sets:
            shared_prompt_ids = set.intersection(*prompt_sets)
        ordered_shared_prompt_ids = sorted(
            shared_prompt_ids,
            key=lambda prompt_id: prompt_order[prompt_id],
        )

        for entry in agent_entries:
            entry["prompt_ids"] = [
                prompt_id for prompt_id in entry["prompt_ids"] if prompt_id not in shared_prompt_ids
            ]

        steps.append(
            {
                "step_index": step_index,
                "outer_step_index": outer_step_index,
                "dispatch_id": dispatch_id,
                "phase": phases[0] if len(phases) == 1 else "",
                "phases": phases,
                "round_index": round_index,
                "discussion_index": discussion_index,
                "parallel": len(agent_entries) > 1,
                "shared_prompt_ids": ordered_shared_prompt_ids,
                "agents": agent_entries,
                "messages_sent": messages_by_group.get(group_key, []),
                "termination": termination_by_group.get(group_key, []),
            }
        )

    termination_history = list(run_metadata.get("termination_history", []))
    return {
        "task_id": str(task.task_id),
        "benchmark": benchmark_name,
        "run_index": int(run_index),
        "system": system_info,
        "tool_definitions": list(run_metadata.get("tool_definitions", [])),
        "role_assignment": role_assignment,
        "prompt_catalog": prompt_catalog,
        "steps": steps,
        "final": {
            "answer": final_answer,
            "answer_preview": _text_preview(final_answer, limit=320),
            "final_reason": str(run_metadata.get("final_reason", "")),
            "vote_tally": dict(run_metadata.get("vote_tally", {})),
            "last_termination": termination_history[-1] if termination_history else None,
        },
    }


def _append_markdown_fence(lines: list[str], content: Any, *, language: str = "text") -> None:
    text = str(content)
    fence = "```"
    while fence in text:
        fence += "`"
    lines.append(f"{fence}{language}")
    lines.append(text)
    lines.append(fence)


def _render_trajectory_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Trajectory: {payload.get('task_id', '')}",
        "",
        f"- Benchmark: {payload.get('benchmark', '')}",
        f"- System: {payload.get('system', {}).get('system_label', '')}",
        f"- Topology: {payload.get('system', {}).get('topology', '')}",
        f"- Run Index: {payload.get('run_index', 0)}",
        "",
        "## Final",
        "",
        f"- Final Reason: {payload.get('final', {}).get('final_reason', '') or '_None_'}",
        f"- Vote Tally: `{json.dumps(payload.get('final', {}).get('vote_tally', {}), sort_keys=True, ensure_ascii=False)}`",
        "",
        "### Final Answer",
        "",
        str(payload.get("final", {}).get("answer", "")),
        "",
        "## Tool Definitions",
        "",
    ]

    tool_definitions = list(payload.get("tool_definitions", []))
    if not tool_definitions:
        lines.append("_None_")
        lines.append("")
    else:
        for tool in tool_definitions:
            lines.append(f"### {tool.get('name', '')}")
            description = str(tool.get("description", "")).strip()
            if description:
                lines.append(description)
            parameters = tool.get("parameters", {})
            lines.append("")
            lines.append("```json")
            lines.append(
                json.dumps(parameters, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            )
            lines.append("```")
            lines.append("")

    lines.extend(["## Role Assignment", ""])
    role_assignment = payload.get("role_assignment", {})
    if not isinstance(role_assignment, dict):
        role_assignment = {}
    lines.append(f"- Enabled: {bool(role_assignment.get('enabled', False))}")
    benchmark_name = str(role_assignment.get("benchmark_name", "")).strip()
    if benchmark_name:
        lines.append(f"- Benchmark: {benchmark_name}")
    lines.append(f"- Used Fallback: {bool(role_assignment.get('used_fallback', False))}")
    lines.append(
        f"- Fallback Reason: {str(role_assignment.get('fallback_reason', '')).strip() or '_None_'}"
    )
    llm = role_assignment.get("llm", {})
    if isinstance(llm, dict) and llm:
        lines.append(
            f"- LLM: model={llm.get('model', '')} mock_used={bool(llm.get('mock_used', False))} token_in={llm.get('token_in', 0)} token_out={llm.get('token_out', 0)}"
        )
    lines.append("")
    lines.append("### Assigned Roles")
    lines.append("")
    assignments = role_assignment.get("assignments", {})
    if not isinstance(assignments, dict) or not assignments:
        lines.append("_None_")
        lines.append("")
    else:
        for agent_id, details in assignments.items():
            if not isinstance(details, dict):
                continue
            lines.append(f"- {agent_id}: {details.get('role_name', '') or '_None_'}")
            persona = str(details.get("persona", "")).strip()
            if persona:
                lines.append(f"  {persona}")
        lines.append("")

    lines.append("### Prompt")
    lines.append("")
    prompt_messages = role_assignment.get("prompt_messages", [])
    if not isinstance(prompt_messages, list) or not prompt_messages:
        lines.append("_None_")
        lines.append("")
    else:
        for index, message in enumerate(prompt_messages, start=1):
            if not isinstance(message, dict):
                continue
            lines.append(f"#### Prompt {index} [{str(message.get('role', '')).upper()}]")
            lines.append("")
            _append_markdown_fence(lines, message.get("content", ""))
            lines.append("")

    lines.append("### Response")
    lines.append("")
    response = str(role_assignment.get("response", ""))
    if response.strip():
        _append_markdown_fence(lines, response)
        lines.append("")
    else:
        lines.append("_None_")
        lines.append("")

    lines.extend(["## Prompt Catalog", ""])
    prompt_catalog = list(payload.get("prompt_catalog", []))
    if not prompt_catalog:
        lines.append("_None_")
        lines.append("")
    else:
        for prompt in prompt_catalog:
            lines.append(
                f"### {prompt.get('prompt_id', '')} [{str(prompt.get('role', '')).upper()}] x{prompt.get('usage_count', 0)}"
            )
            lines.append(str(prompt.get("content", "")))
            lines.append("")

    lines.extend(["## Communication Steps", ""])
    steps = list(payload.get("steps", []))
    if not steps:
        lines.append("_None_")
        lines.append("")
        return "\n".join(lines).strip() + "\n"

    for step in steps:
        step_header = (
            f"### Step {step.get('step_index', 0)}"
            f" · outer {step.get('outer_step_index', 0)}"
            f" · dispatch {step.get('dispatch_id', 0)}"
            f" · round {step.get('round_index', 0)}"
        )
        lines.append(step_header)
        lines.append("")
        phase = str(step.get("phase", "")).strip()
        if phase:
            lines.append(f"- Phase: {phase}")
        elif step.get("phases"):
            lines.append(f"- Phases: {', '.join(step.get('phases', []))}")
        lines.append(f"- Parallel: {bool(step.get('parallel', False))}")
        shared_prompt_ids = list(step.get("shared_prompt_ids", []))
        lines.append(
            f"- Shared Prompt IDs: {', '.join(shared_prompt_ids) if shared_prompt_ids else '_None_'}"
        )
        lines.append("")

        for agent in step.get("agents", []):
            lines.append(f"#### {agent.get('agent_id', '')} ({agent.get('role', '') or 'agent'})")
            lines.append(
                f"- Unique Prompt IDs: {', '.join(agent.get('prompt_ids', [])) if agent.get('prompt_ids') else '_None_'}"
            )
            lines.append(
                f"- Inbox Message IDs: {', '.join(agent.get('inbox_message_ids', [])) if agent.get('inbox_message_ids') else '_None_'}"
            )
            tool_calls = list(agent.get("tool_calls", []))
            if tool_calls:
                tool_summaries = [
                    f"{call.get('tool_name', '')} ({call.get('status', '')})" for call in tool_calls
                ]
                lines.append(f"- Tool Calls: {', '.join(tool_summaries)}")
            else:
                lines.append("- Tool Calls: _None_")
            lines.append("")
            _append_markdown_fence(lines, agent.get("response", ""))
            lines.append("")

        lines.append("#### Messages Sent")
        lines.append("")
        messages_sent = list(step.get("messages_sent", []))
        if not messages_sent:
            lines.append("_None_")
            lines.append("")
        else:
            for message in messages_sent:
                lines.append(
                    f"- {message.get('message_id', '')}: {message.get('from', '')} -> {', '.join(message.get('to', []))} [{message.get('kind', '')}]"
                )
                lines.append(f"  {message.get('content', '')}")
            lines.append("")

        lines.append("#### Termination")
        lines.append("")
        terminations = list(step.get("termination", []))
        if not terminations:
            lines.append("_None_")
            lines.append("")
        else:
            for termination in terminations:
                lines.append(
                    f"- {termination.get('stage_name', '')}: stop={termination.get('should_stop', False)} reason={termination.get('reason', '')}"
                )
                detail = str(termination.get("reason_detail", "")).strip()
                if detail:
                    lines.append(f"  {detail}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _matplotlib_positions(layout: Any) -> dict[str, tuple[float, float]]:
    topology = str(layout.topology)
    positions: dict[str, tuple[float, float]] = {}

    if topology == "sas":
        positions[layout.agent_ids[0]] = (0.5, 0.5)
        return positions

    if topology == "orchestrator_tree_structure":
        levels = []
        root = [layout.orchestrator_id] if layout.orchestrator_id else []
        if root:
            levels.append(root)
        if layout.managers:
            levels.append(list(layout.managers))
        if layout.leaves:
            levels.append(list(layout.leaves))
        for level_index, agents in enumerate(levels):
            y = 1.0 - (level_index / max(1, len(levels) - 1 or 1))
            for item_index, agent_id in enumerate(agents):
                x = (item_index + 1) / (len(agents) + 1)
                positions[agent_id] = (x, y)
        return positions

    if topology in {"orchestrator_no_discussion", "orchestrator_with_discussion"}:
        if layout.orchestrator_id:
            positions[layout.orchestrator_id] = (0.5, 0.9)
        for index, agent_id in enumerate(layout.specialists):
            positions[agent_id] = ((index + 1) / (len(layout.specialists) + 1), 0.2)
        return positions

    if topology == "group_chat_debate" and layout.groups:
        group_count = len(layout.groups)
        for group_index, group in enumerate(layout.groups):
            x_center = (group_index + 1) / (group_count + 1)
            for member_index, agent_id in enumerate(group):
                y = 0.8 - (member_index * 0.25)
                positions[agent_id] = (x_center, max(0.15, y))
        return positions

    total = max(1, len(layout.agent_ids))
    for index, agent_id in enumerate(layout.agent_ids):
        angle = (2.0 * math.pi * index) / total
        positions[agent_id] = (
            0.5 + 0.34 * math.cos(angle),
            0.5 + 0.34 * math.sin(angle),
        )
    return positions


def _write_matplotlib_graph_png(path: Path, layout: Any) -> None:
    import matplotlib.pyplot as plt

    positions = _matplotlib_positions(layout)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.axis("off")

    drawn: set[tuple[str, str]] = set()
    for source, targets in layout.adjacency.items():
        x1, y1 = positions[source]
        for target in targets:
            key = tuple(sorted((source, target)))
            if key in drawn or target not in positions:
                continue
            drawn.add(key)
            x2, y2 = positions[target]
            ax.plot([x1, x2], [y1, y2], color="#7c8695", linewidth=1.4, alpha=0.8, zorder=1)

    palette = {
        "orchestrator": "#ecb939",
        "root_orchestrator": "#ecb939",
        "manager": "#4a90e2",
        "leaf_worker": "#50c878",
        "specialist": "#50c878",
        "voter": "#f28c8c",
        "debater": "#b38bfa",
        "single_agent": "#ff9f43",
    }
    for agent_id, (x, y) in positions.items():
        role = str(layout.roles.get(agent_id, "agent"))
        color = palette.get(role, "#6cc4c4" if "representative" in role else "#8cbf88")
        ax.scatter([x], [y], s=1800, c=color, edgecolors="#243447", linewidths=1.4, zorder=2)
        ax.text(
            x,
            y,
            f"{agent_id}\n{role}",
            ha="center",
            va="center",
            fontsize=9,
            color="#111827",
            zorder=3,
        )

    ax.set_title(f"MAS Topology: {layout.topology}", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_workflow_matplotlib_graph_png(path: Path, workflow: Any) -> None:
    import matplotlib.pyplot as plt

    node_ids = ["START", *list(workflow.nodes.keys()), "END"]
    positions = {node_id: (index, 0.0) for index, node_id in enumerate(node_ids)}
    edges = LangGraphMASEngine._workflow_edges_from_documentation(workflow)

    fig, ax = plt.subplots(figsize=(max(10.0, len(node_ids) * 1.8), 3.8))
    ax.axis("off")

    for edge in edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": "#7c8695", "linewidth": 1.4},
            zorder=1,
        )

    for node_id, (x, y) in positions.items():
        if node_id in {"START", "END"}:
            color = "#d0d7de"
        elif "dispatch" in node_id:
            color = "#ecb939"
        elif "controller" in node_id or "checker" in node_id:
            color = "#f28c8c"
        elif "judge" in node_id or "voter" in node_id:
            color = "#b38bfa"
        elif node_id == "finalize":
            color = "#50c878"
        else:
            color = "#6cc4c4"
        ax.scatter([x], [y], s=2200, c=color, edgecolors="#243447", linewidths=1.2, zorder=2)
        ax.text(x, y, node_id, ha="center", va="center", fontsize=8.5, color="#111827", zorder=3)

    ax.set_title(f"Workflow: {workflow.topology}", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_system_graph_artifact(
    *,
    runner: MASRunner,
    config: Any,
    run_root: Path,
) -> dict[str, Any]:
    spec = ExperimentSpec(
        topology=config.mas.resolved_topology(),
        num_agents=config.mas.total_agents,
        rounds=max(1, int(config.mas.max_turns)),
        discussion_rounds=max(1, int(config.mas.discussion_rounds)),
        communication_budget_per_agent=int(config.mas.communication_count_internally),
        termination_consensus_mode=str(config.mas.termination_consensus_mode),
        peer_artifact_max_chars=int(config.mas.peer_artifact_max_chars),
        agents_per_level=(
            list(config.mas.agents_per_level) if config.mas.agents_per_level is not None else None
        ),
        group_sizes=(list(config.mas.group_sizes) if config.mas.group_sizes is not None else None),
    )

    graph_path = run_root / "mas_graph.png"
    mermaid_path = run_root / "mas_graph.mmd"
    metadata_path = run_root / "mas_graph.json"
    workflow_graph_path = run_root / "workflow_graph.png"
    workflow_mermaid_path = run_root / "workflow_graph.mmd"
    workflow_metadata_path = run_root / "workflow_graph.json"

    layout, visual_graph = runner.engine.build_topology_visual_graph(spec)
    mermaid_text = visual_graph.draw_mermaid()
    mermaid_path.write_text(mermaid_text, encoding="utf-8")

    render_backend = "langgraph_mermaid_api"
    render_error = ""
    try:
        png_bytes = visual_graph.draw_mermaid_png(
            output_file_path=str(graph_path),
            background_color="white",
            max_retries=0,
        )
        with contextlib.suppress(Exception):
            from IPython.display import Image as IPythonImage

            rendered = IPythonImage(data=png_bytes)
            if isinstance(getattr(rendered, "data", None), bytes | bytearray):
                png_bytes = bytes(rendered.data)
        graph_path.write_bytes(png_bytes)
    except Exception as exc:
        render_backend = "matplotlib_fallback"
        render_error = str(exc)
        _write_matplotlib_graph_png(graph_path, layout)

    workflow_definition, workflow_graph = runner.engine.build_workflow_visual_graph(spec)
    workflow_mermaid_text = workflow_graph.draw_mermaid()
    workflow_mermaid_path.write_text(workflow_mermaid_text, encoding="utf-8")

    workflow_render_backend = "langgraph_mermaid_api"
    workflow_render_error = ""
    try:
        workflow_png_bytes = workflow_graph.draw_mermaid_png(
            output_file_path=str(workflow_graph_path),
            background_color="white",
            max_retries=0,
        )
        with contextlib.suppress(Exception):
            from IPython.display import Image as IPythonImage

            rendered = IPythonImage(data=workflow_png_bytes)
            if isinstance(getattr(rendered, "data", None), bytes | bytearray):
                workflow_png_bytes = bytes(rendered.data)
        workflow_graph_path.write_bytes(workflow_png_bytes)
    except Exception as exc:
        workflow_render_backend = "matplotlib_fallback"
        workflow_render_error = str(exc)
        _write_workflow_matplotlib_graph_png(workflow_graph_path, workflow_definition)

    workflow_payload = {
        "topology": workflow_definition.topology,
        "render_backend": workflow_render_backend,
        "render_error": workflow_render_error,
        "png_path": str(workflow_graph_path.resolve()),
        "mermaid_path": str(workflow_mermaid_path.resolve()),
        "workflow": workflow_definition.to_payload(),
    }
    _write_json(workflow_metadata_path, workflow_payload)

    payload = {
        "topology": layout.topology,
        "render_backend": render_backend,
        "render_error": render_error,
        "png_path": str(graph_path.resolve()),
        "mermaid_path": str(mermaid_path.resolve()),
        "layout": layout.to_payload(),
        "workflow": workflow_payload,
    }
    _write_json(metadata_path, payload)
    return payload


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


def _experiment_settings_payload(
    *,
    args: argparse.Namespace,
    config: Any,
    benchmark_name: str,
    benchmark_cfg: dict[str, Any],
    task_limit: int | None,
    runs_per_task: int,
    seed: int,
    task_count: int,
    run_root: Path,
    output_paths: OutputPaths,
) -> dict[str, Any]:
    mas_cfg = config.mas
    benchmark_cfg_redacted = _redact_secrets(benchmark_cfg)

    return {
        "timestamp": output_paths.experiment_id,
        "experiment_id": output_paths.experiment_id,
        "output_layout": output_paths.output_layout,
        "run_root": str(run_root),
        "experiment_root": str(output_paths.experiment_root),
        "benchmark_root": str(output_paths.benchmark_root),
        "config_path": str(Path(args.config).resolve()),
        "benchmark": {
            "name": benchmark_name,
            "task_count": task_count,
            "task_limit": task_limit,
            "config": benchmark_cfg_redacted,
        },
        "runtime": {
            "runs_per_task": runs_per_task,
            "seed": seed,
            "output_dir": str(output_paths.experiment_root.parent),
        },
        "system": {
            "system_label": output_paths.system_label,
            "mode": _mas_mode_label(config),
            "mas": {
                "topology": mas_cfg.topology,
                "resolved_topology": mas_cfg.resolved_topology(),
                "levels": mas_cfg.levels,
                "number_of_agents": mas_cfg.total_agents,
                "agents_per_level": mas_cfg.resolved_agents_per_level(),
                "group_sizes": list(mas_cfg.group_sizes)
                if mas_cfg.group_sizes is not None
                else None,
                "agent_types": list(mas_cfg.agent_types),
                "turn_mode": mas_cfg.turn_mode,
                "max_turns": mas_cfg.max_turns,
                "discussion_rounds": mas_cfg.discussion_rounds,
                "termination_consensus_mode": mas_cfg.termination_consensus_mode,
                "final_vote_mode": mas_cfg.final_vote_mode,
                "peer_artifact_max_chars": mas_cfg.peer_artifact_max_chars,
                "communication_count_internally": mas_cfg.communication_count_internally,
                "intra_level_link_ratio": mas_cfg.intra_level_link_ratio,
                "full_linked": mas_cfg.full_linked,
                "topology_notes": (
                    "Intra-level edges are random unless full_linked=true. "
                    "Cross-level edges are full bipartite between adjacent levels."
                ),
            },
        },
        "models": dict(config.models),
        "openrouter": {
            "base_url": config.openrouter.base_url,
            "timeout_s": config.openrouter.timeout_s,
            "http_referer": config.openrouter.http_referer or "",
            "x_title": config.openrouter.x_title or "",
            "api_key_present": bool(config.openrouter.api_key),
        },
        "tools": {
            "agent_runtime_tools": _runtime_tools(config, benchmark_name, benchmark_cfg),
            "benchmark_eval_mode": str(benchmark_cfg.get("eval_mode", "")),
            "benchmark_judge_model": str(benchmark_cfg.get("judge_model", "")),
        },
        "raw_config_snapshot": _redact_secrets(asdict(config) if is_dataclass(config) else {}),
    }


def _write_experiment_settings(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


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


def _llm_payload_needs_rerun(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if bool(payload.get("mock_used", False)):
        return True
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
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
) -> tuple[str, list[TraceEvent], dict[str, Any], BenchmarkEvaluation, RunOutcome, dict[str, Any]] | None:
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
            completion=bool(outcome_payload.get("completion", eval_payload.get("completion", False))),
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
        _log_progress(
            f"TASK_RESUME_RERUN_REQUIRED task_id={task.task_id} path={task_summary_path}"
        )
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


def run_command(args: argparse.Namespace) -> int:
    # 1) Load runtime knobs (OpenRouter, MAS topology, model routing, benchmark settings).
    config = load_experiment_config(args.config)
    _apply_mas_overrides(config, args)

    benchmark_name = args.benchmark
    benchmark_cfg = _benchmark_section_config(config, benchmark_name)
    benchmark_cfg = _apply_benchmark_overrides(benchmark_cfg, args)
    # 2) Instantiate the benchmark adapter and MAS runtime.
    benchmark = get_benchmark(benchmark_name, config=benchmark_cfg)

    llm_client = OpenRouterLLMClient(config.openrouter, config.models)
    runner = MASRunner(config, llm_client)

    task_limit = args.task_limit if args.task_limit is not None else config.experiment.task_limit
    runs_per_task = (
        args.runs_per_task if args.runs_per_task is not None else config.experiment.runs_per_task
    )
    seed = args.seed if args.seed is not None else config.experiment.seed
    output_root = Path(args.output_dir or config.experiment.output_dir)
    output_paths = _resolve_output_paths(
        args=args,
        config=config,
        benchmark_name=benchmark_name,
        output_root=output_root,
    )
    output_paths.benchmark_root.mkdir(parents=True, exist_ok=True)
    output_paths.run_root.mkdir(parents=True, exist_ok=True)

    tasks = list(benchmark.load_tasks(task_limit=task_limit))
    if not tasks:
        raise RuntimeError(f"No tasks loaded for benchmark '{benchmark_name}'")
    _log_progress(
        "Loaded run configuration "
        f"benchmark={benchmark_name} system={args.system_label or config.mas.resolved_topology()} "
        f"tasks={len(tasks)} runs_per_task={runs_per_task} seed={seed} "
        f"topology={config.mas.resolved_topology()} agents={config.mas.total_agents} "
        f"rounds={int(config.mas.max_turns)} discussion_rounds={int(config.mas.discussion_rounds)} "
        f"communication_budget={int(config.mas.communication_count_internally)}"
    )

    experiment_settings = _experiment_settings_payload(
        args=args,
        config=config,
        benchmark_name=benchmark_name,
        benchmark_cfg=benchmark_cfg,
        task_limit=task_limit,
        runs_per_task=runs_per_task,
        seed=seed,
        task_count=len(tasks),
        run_root=output_paths.run_root,
        output_paths=output_paths,
    )
    settings_path = output_paths.run_root / "experiment_settings.json"
    _write_experiment_settings(settings_path, experiment_settings)

    graph_payload: dict[str, Any] | None = None
    if output_paths.output_layout == "hierarchical":
        graph_payload = _write_system_graph_artifact(
            runner=runner,
            config=config,
            run_root=output_paths.run_root,
        )

    system_info = {
        "system_label": output_paths.system_label,
        "mode": _mas_mode_label(config),
        "topology": config.mas.resolved_topology(),
        "agents": config.mas.total_agents,
        "agents_per_level": config.mas.resolved_agents_per_level(),
        "group_sizes": list(config.mas.group_sizes) if config.mas.group_sizes is not None else None,
        "agent_types": list(config.mas.agent_types),
        "max_turns": int(config.mas.max_turns),
        "discussion_rounds": int(config.mas.discussion_rounds),
        "termination_consensus_mode": str(config.mas.termination_consensus_mode),
        "final_vote_mode": str(config.mas.final_vote_mode),
        "peer_artifact_max_chars": int(config.mas.peer_artifact_max_chars),
        "communication_budget": int(config.mas.communication_count_internally),
    }

    summary_rows: list[dict[str, Any]] = []
    summary_json: dict[str, Any] = {
        "timestamp": output_paths.experiment_id,
        "experiment_id": output_paths.experiment_id,
        "output_layout": output_paths.output_layout,
        "benchmark": benchmark_name,
        "system_label": output_paths.system_label,
        "system": system_info,
        "config_path": str(Path(args.config).resolve()),
        "runs_per_task": runs_per_task,
        "task_count": len(tasks),
        "experiment_settings_path": str(settings_path.resolve()),
        "tasks": [],
    }
    if graph_payload is not None:
        summary_json["graph"] = graph_payload

    default_model = str(config.models.get("default", ""))
    judge_model = str(config.models.get("judge", config.models.get("default", "")))

    for task_idx, task in enumerate(tasks):
        task_dir = (
            output_paths.run_root / task.task_id
            if output_paths.output_layout == "hierarchical"
            else output_paths.benchmark_root / task.task_id
        )
        task_dir.mkdir(parents=True, exist_ok=True)

        resumed = _load_completed_task_resume(
            task=task,
            task_dir=task_dir,
            benchmark_name=benchmark_name,
            system_info=system_info,
            runs_per_task=runs_per_task,
            default_model=default_model,
            judge_model=judge_model,
        )
        if resumed is not None:
            task_summary, row = resumed
            summary_json["tasks"].append(task_summary)
            summary_rows.append(row)
            _log_progress(
                f"TASK_RESUME_SKIP index={task_idx + 1}/{len(tasks)} task_id={task.task_id} "
                f"path={task_dir / 'task_summary.json'}"
            )
            continue

        _log_progress(
            f"TASK_START index={task_idx + 1}/{len(tasks)} task_id={task.task_id}"
        )
        run_traces = []
        evaluations = []
        run_outcomes: list[RunOutcome] = []
        run_artifacts: list[dict[str, Any]] = []
        task_failure_details: list[dict[str, Any]] = []

        for run_index in range(runs_per_task):
            run_seed = seed + (task_idx * 1000) + run_index
            resumed_run = _load_completed_run_resume(task_dir=task_dir, run_index=run_index)
            if resumed_run is not None:
                (
                    final_answer,
                    trace_events,
                    run_metadata,
                    evaluation,
                    run_outcome,
                    artifact_record,
                ) = resumed_run
                run_traces.append(trace_events)
                evaluations.append(evaluation)
                run_outcomes.append(run_outcome)
                run_artifacts.append(artifact_record)
                _log_progress(
                    f"RUN_RESUME_SKIP task_id={task.task_id} run_index={run_index} "
                    f"path={task_dir / f'run_{run_index}.metadata.json'}"
                )
                continue

            run_started = datetime.now(UTC)
            run_started_s = time.time()
            _log_progress(
                f"RUN_START task_id={task.task_id} run_index={run_index} seed={run_seed}"
            )
            try:
                run = benchmark.run(
                    task=task,
                    runner=runner,
                    run_index=run_index,
                    seed=run_seed,
                )
            except Exception as exc:
                failure_category = _classify_run_exception(exc)
                _log_progress(
                    f"RUN_ERROR task_id={task.task_id} run_index={run_index} "
                    f"seed={run_seed} category={failure_category} "
                    f"error={type(exc).__name__}:{exc}"
                )
                (
                    final_answer,
                    trace_events,
                    run_metadata,
                    evaluation,
                    run_outcome,
                ) = _failed_run_result(
                    task=task,
                    benchmark_name=benchmark_name,
                    system_info=system_info,
                    run_index=run_index,
                    seed=run_seed,
                    exc=exc,
                    started_s=run_started_s,
                )
                task_failure_details.append(
                    {
                        "task_id": str(task.task_id),
                        "run_index": int(run_index),
                        "seed": int(run_seed),
                        "failure_category": failure_category,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            else:
                final_answer = run.final_answer
                trace_events = run.trace_events
                run_metadata = run.run_metadata
                evaluation = None
                run_outcome = None

            _log_progress(
                f"RUN_FINISH task_id={task.task_id} run_index={run_index} "
                f"elapsed_s={(datetime.now(UTC) - run_started).total_seconds():.2f} "
                f"trace_events={len(trace_events)} final_answer_chars={len(str(final_answer or ''))} "
                f"status={str(run_metadata.get('run_status', 'completed') or 'completed')}"
            )

            trace_path = task_dir / f"run_{run_index}.trace.jsonl"
            write_run_trace(trace_events, trace_path)
            run_traces.append(trace_events)
            _log_progress(
                f"TRACE_WRITTEN task_id={task.task_id} run_index={run_index} path={trace_path}"
            )

            raw_output_path = task_dir / f"run_{run_index}.raw.json"
            _write_raw_output(
                raw_output_path,
                final_answer=final_answer,
                run_metadata=run_metadata,
            )
            _log_progress(
                f"RAW_OUTPUT_WRITTEN task_id={task.task_id} run_index={run_index} path={raw_output_path}"
            )

            # 4) Let the benchmark score the model output.
            if evaluation is None:
                evaluation = benchmark.evaluate(
                    task,
                    final_answer,
                    run_metadata=run_metadata,
                )
            _log_progress(
                f"EVAL_FINISH task_id={task.task_id} run_index={run_index} "
                f"score={float(evaluation.score):.4f} success={bool(evaluation.success)}"
            )
            if run_outcome is None:
                run_outcome = resolve_run_outcome(
                    trace_events,
                    evaluation=evaluation,
                    final_answer=final_answer,
                    run_metadata=run_metadata,
                )
            if _metadata_needs_rerun(run_metadata):
                run_metadata["fallback"] = True
                run_metadata["needs_rerun"] = True
                run_metadata.setdefault("fallback_reason", "llm_or_tool_fallback")
            evaluations.append(evaluation)
            run_outcomes.append(run_outcome)

            artifact_paths = _write_run_artifacts(
                task_dir=task_dir,
                benchmark_name=benchmark_name,
                task=task,
                run_index=run_index,
                final_answer=final_answer,
                trace_events=trace_events,
                evaluation=evaluation,
                run_outcome=run_outcome,
                run_metadata=run_metadata,
                system_info=system_info,
            )
            eval_path = task_dir / f"run_{run_index}.eval.json"
            _write_eval(
                eval_path,
                evaluation,
                final_answer,
                run_outcome=run_outcome,
                metadata_summary=_compact_run_metadata(run_metadata),
                metadata_path=Path(artifact_paths["metadata_path"]),
            )
            run_artifacts.append(
                {
                    **artifact_paths,
                    "trace_path": str(trace_path.resolve()),
                    "eval_path": str(eval_path.resolve()),
                    "score": float(evaluation.score),
                    "success": bool(evaluation.success),
                    "completion": bool(run_outcome.completion),
                }
            )
            _log_progress(
                f"RUN_ARTIFACTS_WRITTEN task_id={task.task_id} run_index={run_index} "
                f"metadata_path={artifact_paths['metadata_path']} eval_path={eval_path}"
            )

        # 5) Convert trace+eval into descriptor artifacts and analysis outputs.
        _log_progress(f"TASK_ANALYZE_START task_id={task.task_id}")
        analysis = analyze_task_runs(
            task_id=task.task_id,
            benchmark_name=benchmark_name,
            run_traces=run_traces,
            evaluations=evaluations,
            run_outcomes=run_outcomes,
            output_dir=task_dir,
        )
        _log_progress(
            f"TASK_ANALYZE_FINISH task_id={task.task_id} "
            f"avg_score={float(analysis['evaluation'].get('avg_score', 0.0)):.4f} "
            f"success_rate={float(analysis['evaluation'].get('success_rate', 0.0)):.4f} "
            f"completion_rate={float(analysis['evaluation'].get('completion_rate', 0.0)):.4f}"
        )
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
        needs_rerun = run_failure_count > 0 or fallback_count > 0 or bool(task_failure_details)
        rerun_details = {
            "run_failure_count": run_failure_count,
            "fallback_count": fallback_count,
            "run_status_summary": run_status_summary,
            "failure_details": task_failure_details,
        }

        task_summary_payload = {
            "task_id": str(task.task_id),
            "benchmark": benchmark_name,
            "system": system_info,
            "task_dir": str(task_dir.resolve()),
            "prompt_preview": _prompt_preview(task.prompt),
            "reference_answer": task.reference_answer,
            "evaluation": analysis["evaluation"],
            "descriptor": analysis["descriptor"],
            "stage_bottleneck": analysis["stage_bottleneck"],
            "runs": run_artifacts,
            "run_status_summary": run_status_summary,
            "run_failure_count": run_failure_count,
            "fallback_count": fallback_count,
            "needs_rerun": needs_rerun,
            "rerun_details": rerun_details,
            "artifacts": {
                "analysis_path": str((task_dir / "analysis.json").resolve()),
                "descriptor_json_path": str((task_dir / "descriptor.json").resolve()),
                "descriptor_csv_path": str((task_dir / "descriptor.csv").resolve()),
            },
        }
        _write_json(task_dir / "task_summary.json", task_summary_payload)

        task_summary = {
            "task_id": task.task_id,
            "prompt_preview": _prompt_preview(task.prompt),
            "reference_answer": task.reference_answer,
            "task_dir": str(task_dir.resolve()),
            "evaluation": analysis["evaluation"],
            "descriptor": analysis["descriptor"],
            "stage_bottleneck": analysis["stage_bottleneck"],
            "run_status_summary": run_status_summary,
            "run_failure_count": run_failure_count,
            "fallback_count": fallback_count,
            "needs_rerun": needs_rerun,
            "artifacts": {
                "task_summary_path": str((task_dir / "task_summary.json").resolve()),
                "analysis_path": str((task_dir / "analysis.json").resolve()),
            },
        }
        summary_json["tasks"].append(task_summary)

        row = _summary_row_from_analysis(
            benchmark_name=benchmark_name,
            system_label=output_paths.system_label,
            topology=config.mas.resolved_topology(),
            agents=config.mas.total_agents,
            default_model=default_model,
            judge_model=judge_model,
            task_id=str(task.task_id),
            task_dir=task_dir,
            analysis=analysis,
        )
        row.update(
            {
                "run_failure_count": run_failure_count,
                "fallback_count": fallback_count,
                "needs_rerun": needs_rerun,
            }
        )
        summary_rows.append(row)
        _log_progress(
            f"TASK_FINISH index={task_idx + 1}/{len(tasks)} task_id={task.task_id} "
            f"task_dir={task_dir} failures={run_failure_count} fallbacks={fallback_count} "
            f"needs_rerun={needs_rerun}"
        )

    total_run_failures = sum(
        int(task.get("run_failure_count", 0) or 0)
        for task in summary_json["tasks"]
        if isinstance(task, dict)
    )
    total_fallbacks = sum(
        int(task.get("fallback_count", 0) or 0)
        for task in summary_json["tasks"]
        if isinstance(task, dict)
    )
    rerun_tasks = [
        str(task.get("task_id", ""))
        for task in summary_json["tasks"]
        if isinstance(task, dict) and bool(task.get("needs_rerun", False))
    ]
    summary_json["run_status_summary"] = {
        "run_failure_count": total_run_failures,
        "fallback_count": total_fallbacks,
        "needs_rerun_task_count": len(rerun_tasks),
        "needs_rerun_task_ids": rerun_tasks,
    }

    summary_json_path = output_paths.run_root / "summary.json"
    summary_csv_path = output_paths.run_root / "summary.csv"
    _write_json(summary_json_path, summary_json)
    _write_summary_csv(summary_csv_path, summary_rows)
    _log_progress(
        f"SUMMARY_WRITTEN summary_json={summary_json_path} summary_csv={summary_csv_path}"
    )
    _log_progress(
        "RUN_STATUS_SUMMARY "
        f"benchmark={benchmark_name} system={output_paths.system_label} "
        f"run_failures={total_run_failures} fallbacks={total_fallbacks} "
        f"needs_rerun_tasks={len(rerun_tasks)} "
        f"needs_rerun_task_ids={','.join(rerun_tasks) if rerun_tasks else 'none'}"
    )

    print(f"Run complete: {output_paths.run_root}")
    if rerun_tasks and _env_truthy("MAS_REQUIRE_LIVE_LLM") and not _env_truthy("MAS_DISABLE_LIVE_LLM"):
        return 2
    return 0


def list_benchmarks_command(_: argparse.Namespace) -> int:
    for name in list_benchmarks():
        print(name)
    return 0


def benchmark_info_command(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config) if args.config else None
    benchmark_cfg: dict[str, Any]
    if config is None:
        benchmark_cfg = {}
    else:
        benchmark_cfg = _benchmark_section_config(config, args.benchmark)

    benchmark = get_benchmark(args.benchmark, config=benchmark_cfg)
    info = benchmark.requirements()
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_experiment_command(args: argparse.Namespace) -> int:
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {experiment_root}")

    experiment_rows: list[dict[str, Any]] = []
    experiment_manifest: dict[str, Any] = {
        "experiment_root": str(experiment_root),
        "benchmarks": [],
    }

    for benchmark_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        benchmark_rows: list[dict[str, Any]] = []
        benchmark_manifest: dict[str, Any] = {
            "benchmark": benchmark_dir.name,
            "systems": [],
        }
        for system_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
            summary_json_path = system_dir / "summary.json"
            settings_path = system_dir / "experiment_settings.json"
            if not settings_path.exists():
                continue

            task_entries: list[dict[str, Any]] = []
            for task_summary_path in sorted(system_dir.glob("*/task_summary.json")):
                try:
                    payload = json.loads(task_summary_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                entry = _summary_task_entry_from_payload(task_summary_path.parent, payload)
                if _task_payload_needs_rerun(task_summary_path.parent, payload):
                    entry["needs_rerun"] = True
                    if int(entry.get("fallback_count", 0) or 0) == 0:
                        entry["fallback_count"] = 1
                task_entries.append(entry)

            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            if summary_json_path.exists():
                summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
                tasks = task_entries or list(summary.get("tasks", []))
            else:
                summary = {
                    "task_count": int(
                        settings.get("benchmark", {}).get("task_count", len(task_entries)) or 0
                    ),
                    "runs_per_task": int(
                        settings.get("runtime", {}).get("runs_per_task", 0) or 0
                    ),
                    "tasks": task_entries,
                }
                tasks = task_entries
            scoreable_tasks = [
                task
                for task in tasks
                if isinstance(task, dict)
                and not bool(task.get("needs_rerun", False))
                and int(task.get("evaluation", {}).get("valid_count", 1) or 0) > 0
            ]
            scores = [
                float(task.get("evaluation", {}).get("avg_score", 0.0))
                for task in scoreable_tasks
            ]
            success_rates = [
                float(task.get("evaluation", {}).get("success_rate", 0.0))
                for task in scoreable_tasks
            ]
            completion_rates = [
                float(task.get("evaluation", {}).get("completion_rate", 0.0))
                for task in scoreable_tasks
            ]
            run_failure_count = sum(
                int(task.get("run_failure_count", 0) or 0)
                for task in tasks
                if isinstance(task, dict)
            )
            fallback_count = sum(
                int(task.get("fallback_count", 0) or 0)
                for task in tasks
                if isinstance(task, dict)
            )
            needs_rerun_task_ids = [
                str(task.get("task_id", ""))
                for task in tasks
                if isinstance(task, dict) and bool(task.get("needs_rerun", False))
            ]

            row = {
                "benchmark": benchmark_dir.name,
                "system_label": system_dir.name,
                "topology": settings.get("system", {}).get("mas", {}).get("resolved_topology", ""),
                "agents": settings.get("system", {}).get("mas", {}).get("number_of_agents", 0),
                "default_model": settings.get("models", {}).get("default", ""),
                "judge_model": settings.get("models", {}).get(
                    "judge",
                    settings.get("models", {}).get("default", ""),
                ),
                "task_count": int(summary.get("task_count", 0)),
                "runs_per_task": int(summary.get("runs_per_task", 0)),
                "avg_task_score": _mean(scores),
                "avg_task_success_rate": _mean(success_rates),
                "avg_task_completion_rate": _mean(completion_rates),
                "completed_task_count": len(tasks),
                "scored_task_count": len(scoreable_tasks),
                "missing_task_count": max(int(summary.get("task_count", 0)) - len(tasks), 0),
                "run_failure_count": run_failure_count,
                "fallback_count": fallback_count,
                "needs_rerun_task_count": len(needs_rerun_task_ids),
                "needs_rerun_task_ids": ",".join(needs_rerun_task_ids),
                "system_root": str(system_dir.resolve()),
                "summary_json_path": str(summary_json_path.resolve()),
                "summary_csv_path": str((system_dir / "summary.csv").resolve()),
                "graph_png_path": str((system_dir / "mas_graph.png").resolve()),
            }
            benchmark_rows.append(row)
            experiment_rows.append(row)
            benchmark_manifest["systems"].append(row)

        if benchmark_rows:
            _write_json(benchmark_dir / "benchmark_summary.json", benchmark_manifest)
            _write_summary_csv(benchmark_dir / "benchmark_summary.csv", benchmark_rows)
            experiment_manifest["benchmarks"].append(benchmark_manifest)

    _write_json(experiment_root / "experiment_summary.json", experiment_manifest)
    _write_summary_csv(experiment_root / "experiment_summary.csv", experiment_rows)
    print(f"Experiment summary complete: {experiment_root}")
    total_missing = sum(int(row.get("missing_task_count", 0) or 0) for row in experiment_rows)
    total_run_failures = sum(int(row.get("run_failure_count", 0) or 0) for row in experiment_rows)
    total_fallbacks = sum(int(row.get("fallback_count", 0) or 0) for row in experiment_rows)
    total_rerun_tasks = sum(int(row.get("needs_rerun_task_count", 0) or 0) for row in experiment_rows)
    print(
        "Experiment run-status summary: "
        f"missing_tasks={total_missing} "
        f"run_failures={total_run_failures} "
        f"fallbacks={total_fallbacks} "
        f"needs_rerun_tasks={total_rerun_tasks}"
    )
    for row in experiment_rows:
        if (
            int(row.get("missing_task_count", 0) or 0) > 0
            or int(row.get("run_failure_count", 0) or 0) > 0
            or int(row.get("fallback_count", 0) or 0) > 0
            or int(row.get("needs_rerun_task_count", 0) or 0) > 0
        ):
            print(
                "  - "
                f"{row.get('benchmark')}::{row.get('system_label')} "
                f"completed={row.get('completed_task_count')}/{row.get('task_count')} "
                f"missing={row.get('missing_task_count')} "
                f"run_failures={row.get('run_failure_count')} "
                f"fallbacks={row.get('fallback_count')} "
                f"needs_rerun={row.get('needs_rerun_task_count')}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MAS experiments against benchmark adapters and descriptor analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("--config", required=True, help="Path to experiment TOML config")
    run_parser.add_argument(
        "--benchmark",
        required=True,
        choices=list_benchmarks(),
        help="Benchmark adapter to run",
    )
    run_parser.add_argument("--task-limit", type=int, default=None)
    run_parser.add_argument("--runs-per-task", type=int, default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument(
        "--output-layout",
        choices=["legacy", "hierarchical"],
        default="legacy",
        help="Output folder layout. 'hierarchical' writes experiment/benchmark/system/task.",
    )
    run_parser.add_argument("--experiment-id", default=None)
    run_parser.add_argument("--system-label", default=None)
    run_parser.add_argument("--topology", default=None)
    run_parser.add_argument("--agents", type=int, default=None)
    run_parser.add_argument("--mas-rounds", type=int, default=None)
    run_parser.add_argument("--discussion-rounds", type=int, default=None)
    run_parser.add_argument("--communication-budget", type=int, default=None)
    run_parser.add_argument(
        "--termination-consensus-mode",
        choices=["llm_judge", "lexical"],
        default=None,
    )
    run_parser.add_argument(
        "--final-vote-mode",
        choices=["llm_judge", "deterministic"],
        default=None,
    )
    run_parser.add_argument("--default-model", default=None)
    run_parser.add_argument("--judge-model", default=None)
    run_parser.add_argument(
        "--benchmark-eval-judge-model",
        default=None,
        help="Override benchmark-side evaluation judge_model without changing the MAS internal judge model.",
    )
    run_parser.add_argument("--peer-artifact-max-chars", type=int, default=None)
    run_parser.add_argument("--agents-per-level", default=None)
    run_parser.add_argument("--group-sizes", default=None)
    run_parser.add_argument("--agent-types", default=None)
    run_parser.add_argument(
        "--no-dynamic-roles",
        dest="no_dynamic_roles",
        action="store_true",
        default=False,
        help="Disable LLM-based dynamic role assignment and use only structural roles.",
    )
    run_parser.set_defaults(func=run_command)

    list_parser = subparsers.add_parser("list-benchmarks", help="List available benchmarks")
    list_parser.set_defaults(func=list_benchmarks_command)

    info_parser = subparsers.add_parser(
        "benchmark-info", help="Show benchmark requirements and setup notes"
    )
    info_parser.add_argument(
        "--benchmark",
        required=True,
        choices=list_benchmarks(),
    )
    info_parser.add_argument("--config", default=None)
    info_parser.set_defaults(func=benchmark_info_command)

    summarize_parser = subparsers.add_parser(
        "summarize-experiment",
        help="Aggregate hierarchical experiment outputs into benchmark and experiment summaries",
    )
    summarize_parser.add_argument("--experiment-root", required=True)
    summarize_parser.set_defaults(func=summarize_experiment_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
