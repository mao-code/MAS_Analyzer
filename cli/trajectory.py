"""Per-run trajectory payloads, prompt/message catalogues and markdown rendering."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from benchmark import BenchmarkEvaluation
from cli.common import _append_markdown_fence, _normalized_int, _text_preview
from descriptor.metrics import RunOutcome, compute_run_metrics


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
