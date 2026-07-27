"""Config resolution: benchmark sections, CLI overrides, output paths, run settings."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from cli.common import (
    OutputPaths,
    _now_stamp,
    _parse_int_list,
    _parse_str_list,
    _redact_secrets,
    _write_json,
)
from MAS.prompting_baselines import (
    BASELINE_DIRECT,
    normalize_prompting_baseline,
)


def _benchmark_section_config(config: Any, benchmark_name: str) -> dict[str, Any]:
    if benchmark_name == "finance_agent":
        cfg = dict(config.finance_agent)
    elif benchmark_name == "browsecomp":
        cfg = dict(config.browsecomp)
    elif benchmark_name == "stabletoolbench":
        cfg = dict(config.stabletoolbench)
    elif benchmark_name == "plancraft":
        return dict(config.plancraft)
    elif benchmark_name == "math500":
        return dict(config.math500)
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
    if getattr(args, "skill_update_batch_size", None) is not None:
        config.self_evolved.skill_update_batch_size = max(0, int(args.skill_update_batch_size))

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
    prompting_baseline = normalize_prompting_baseline(
        getattr(args, "prompting_baseline", BASELINE_DIRECT)
    )

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
            "prompting_baseline": {
                "name": prompting_baseline,
                "self_consistency_samples": int(getattr(args, "self_consistency_samples", 3) or 3),
                "self_refine_rounds": int(getattr(args, "self_refine_rounds", 3) or 3),
            },
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
