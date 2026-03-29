from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark import BenchmarkEvaluation, get_benchmark, list_benchmarks
from descriptor.experiment import analyze_task_runs, write_run_trace
from MAS import MASRunner, OpenRouterLLMClient, load_experiment_config
from MAS.langgraph_engine import ExperimentSpec


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


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
        "details": details,
        "prediction": prediction,
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
            writer.writerow(row)


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
    if agent_types is not None:
        mas_cfg.agent_types = list(agent_types)

    config.validate()


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
    ):
        if key in run_metadata:
            payload[key] = run_metadata[key]
    return payload


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


def _trajectory_payload(
    *,
    task: Any,
    benchmark_name: str,
    system_info: dict[str, Any],
    run_index: int,
    final_answer: str,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    steps = list(run_metadata.get("interaction_logs", []))
    if not steps:
        prompt = getattr(task, "prompt", "")
        if isinstance(prompt, list):
            prompt_messages = list(prompt)
        else:
            prompt_messages = [{"role": "user", "content": str(prompt)}]
        steps = [
            {
                "dispatch_id": 0,
                "agent_id": "agent_0",
                "agent_role": system_info.get("mode", "agent"),
                "agent_type": "",
                "phase": "solve",
                "round_index": 0,
                "prompt_messages": prompt_messages,
                "visible_messages": [],
                "assistant_message": {"role": "assistant", "content": final_answer},
                "tool_calls": [],
                "llm": {},
            }
        ]

    return {
        "task_id": str(task.task_id),
        "benchmark": benchmark_name,
        "run_index": int(run_index),
        "system": system_info,
        "tool_definitions": list(run_metadata.get("tool_definitions", [])),
        "steps": steps,
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

    for index, step in enumerate(payload.get("steps", []), start=1):
        lines.append(
            f"## Step {index}: {step.get('agent_id', '')} ({step.get('phase', '')} / round {step.get('round_index', 0)})"
        )
        lines.append("")
        lines.append("### Prompt Messages")
        lines.append("")
        for message in step.get("prompt_messages", []):
            role = str(message.get("role", "user")).upper()
            lines.append(f"#### {role}")
            lines.append(str(message.get("content", "")))
            lines.append("")
        tool_calls = step.get("tool_calls", [])
        lines.append("### Tool Calls")
        lines.append("")
        if not tool_calls:
            lines.append("_None_")
            lines.append("")
        else:
            for call in tool_calls:
                lines.append(f"- `{call.get('tool_name', '')}` ({call.get('status', '')})")
                lines.append(
                    f"  args: `{json.dumps(call.get('arguments', {}), sort_keys=True, ensure_ascii=False, default=str)}`"
                )
                preview = str(call.get("output_preview", "")).strip()
                if preview:
                    lines.append(f"  output: {preview}")
            lines.append("")
        lines.append("### Assistant Message")
        lines.append("")
        lines.append(str(step.get("assistant_message", {}).get("content", "")))
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
        agents_per_level=(
            list(config.mas.agents_per_level) if config.mas.agents_per_level is not None else None
        ),
        group_sizes=(list(config.mas.group_sizes) if config.mas.group_sizes is not None else None),
    )

    graph_path = run_root / "mas_graph.png"
    mermaid_path = run_root / "mas_graph.mmd"
    metadata_path = run_root / "mas_graph.json"

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
            if isinstance(getattr(rendered, "data", None), (bytes, bytearray)):
                png_bytes = bytes(rendered.data)
        graph_path.write_bytes(png_bytes)
    except Exception as exc:
        render_backend = "matplotlib_fallback"
        render_error = str(exc)
        _write_matplotlib_graph_png(graph_path, layout)

    payload = {
        "topology": layout.topology,
        "render_backend": render_backend,
        "render_error": render_error,
        "png_path": str(graph_path.resolve()),
        "mermaid_path": str(mermaid_path.resolve()),
        "layout": layout.to_payload(),
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
    evaluation: BenchmarkEvaluation,
    run_metadata: dict[str, Any],
    system_info: dict[str, Any],
) -> dict[str, str]:
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
    trajectory_json_path = task_dir / f"run_{run_index}.trajectory.json"
    trajectory_md_path = task_dir / f"run_{run_index}.trajectory.md"

    answer_path.write_text(final_answer, encoding="utf-8")
    _write_json(metadata_path, run_metadata)

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
        "run_summary": _compact_run_metadata(run_metadata),
        "artifacts": {
            "task_manifest_path": str(task_manifest_path.resolve()),
            "answer_path": str(answer_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
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
        "trajectory_json_path": str(trajectory_json_path.resolve()),
        "trajectory_md_path": str(trajectory_md_path.resolve()),
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
                "group_sizes": list(mas_cfg.group_sizes) if mas_cfg.group_sizes is not None else None,
                "agent_types": list(mas_cfg.agent_types),
                "turn_mode": mas_cfg.turn_mode,
                "max_turns": mas_cfg.max_turns,
                "discussion_rounds": mas_cfg.discussion_rounds,
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


def run_command(args: argparse.Namespace) -> int:
    # 1) Load runtime knobs (OpenRouter, MAS topology, model routing, benchmark settings).
    config = load_experiment_config(args.config)
    _apply_mas_overrides(config, args)

    benchmark_name = args.benchmark
    benchmark_cfg = _benchmark_section_config(config, benchmark_name)
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

    for task_idx, task in enumerate(tasks):
        task_dir = (
            output_paths.run_root / task.task_id
            if output_paths.output_layout == "hierarchical"
            else output_paths.benchmark_root / task.task_id
        )
        task_dir.mkdir(parents=True, exist_ok=True)

        run_traces = []
        evaluations = []
        run_artifacts: list[dict[str, Any]] = []

        for run_index in range(runs_per_task):
            run_seed = seed + (task_idx * 1000) + run_index

            run = benchmark.run(
                task=task,
                runner=runner,
                run_index=run_index,
                seed=run_seed,
            )

            trace_path = task_dir / f"run_{run_index}.trace.jsonl"
            write_run_trace(run.trace_events, trace_path)
            run_traces.append(run.trace_events)

            # 4) Let the benchmark score the model output.
            evaluation = benchmark.evaluate(
                task,
                run.final_answer,
                run_metadata=run.run_metadata,
            )
            evaluations.append(evaluation)

            artifact_paths = _write_run_artifacts(
                task_dir=task_dir,
                benchmark_name=benchmark_name,
                task=task,
                run_index=run_index,
                final_answer=run.final_answer,
                evaluation=evaluation,
                run_metadata=run.run_metadata,
                system_info=system_info,
            )
            eval_path = task_dir / f"run_{run_index}.eval.json"
            _write_eval(
                eval_path,
                evaluation,
                run.final_answer,
                metadata_summary=_compact_run_metadata(run.run_metadata),
                metadata_path=Path(artifact_paths["metadata_path"]),
            )
            run_artifacts.append(
                {
                    **artifact_paths,
                    "trace_path": str(trace_path.resolve()),
                    "eval_path": str(eval_path.resolve()),
                    "score": float(evaluation.score),
                    "success": bool(evaluation.success),
                }
            )

        # 5) Convert trace+eval into descriptor artifacts and analysis outputs.
        analysis = analyze_task_runs(
            task_id=task.task_id,
            benchmark_name=benchmark_name,
            run_traces=run_traces,
            evaluations=evaluations,
            output_dir=task_dir,
        )

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
            "artifacts": {
                "task_summary_path": str((task_dir / "task_summary.json").resolve()),
                "analysis_path": str((task_dir / "analysis.json").resolve()),
            },
        }
        summary_json["tasks"].append(task_summary)

        row: dict[str, Any] = {
            "benchmark": benchmark_name,
            "system_label": output_paths.system_label,
            "topology": config.mas.resolved_topology(),
            "agents": config.mas.total_agents,
            "task_id": task.task_id,
            "runs": analysis["evaluation"].get("count", 0),
            "eval_avg_score": analysis["evaluation"].get("avg_score", 0.0),
            "eval_success_rate": analysis["evaluation"].get("success_rate", 0.0),
            "task_dir": str(task_dir.resolve()),
        }
        row.update(analysis["descriptor"])
        summary_rows.append(row)

    summary_json_path = output_paths.run_root / "summary.json"
    summary_csv_path = output_paths.run_root / "summary.csv"
    _write_json(summary_json_path, summary_json)
    _write_summary_csv(summary_csv_path, summary_rows)

    print(f"Run complete: {output_paths.run_root}")
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
            if not summary_json_path.exists() or not settings_path.exists():
                continue

            summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            tasks = list(summary.get("tasks", []))
            scores = [
                float(task.get("evaluation", {}).get("avg_score", 0.0))
                for task in tasks
                if isinstance(task, dict)
            ]
            success_rates = [
                float(task.get("evaluation", {}).get("success_rate", 0.0))
                for task in tasks
                if isinstance(task, dict)
            ]

            row = {
                "benchmark": benchmark_dir.name,
                "system_label": system_dir.name,
                "topology": settings.get("system", {}).get("mas", {}).get("resolved_topology", ""),
                "agents": settings.get("system", {}).get("mas", {}).get("number_of_agents", 0),
                "task_count": int(summary.get("task_count", 0)),
                "runs_per_task": int(summary.get("runs_per_task", 0)),
                "avg_task_score": _mean(scores),
                "avg_task_success_rate": _mean(success_rates),
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
    run_parser.add_argument("--agents-per-level", default=None)
    run_parser.add_argument("--group-sizes", default=None)
    run_parser.add_argument("--agent-types", default=None)
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
