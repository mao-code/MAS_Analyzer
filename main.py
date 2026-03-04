from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from benchmark import BenchmarkEvaluation, get_benchmark, list_benchmarks
from descriptor.experiment import analyze_task_runs, write_run_trace
from MAS import MASRunner, OpenRouterLLMClient, load_experiment_config


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _benchmark_section_config(config: Any, benchmark_name: str) -> dict[str, Any]:
    if benchmark_name == "finance_agent":
        cfg = dict(config.finance_agent)
    elif benchmark_name == "browsecomp":
        cfg = dict(config.browsecomp)
    elif benchmark_name == "plancraft":
        return dict(config.plancraft)
    elif benchmark_name == "scicode":
        return dict(config.scicode)
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


def _write_eval(path: Path, evaluation: BenchmarkEvaluation, prediction: str) -> None:
    payload = {
        "task_id": evaluation.task_id,
        "score": evaluation.score,
        "success": evaluation.success,
        "details": evaluation.details,
        "prediction": prediction,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _redact_secrets(data: Any, *, parent_key: str = "") -> Any:
    secret_markers = ("api_key", "token", "secret", "password")
    key_lower = parent_key.lower()

    if isinstance(data, dict):
        return {
            key: _redact_secrets(value, parent_key=str(key))
            for key, value in data.items()
        }
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
) -> dict[str, Any]:
    mas_cfg = config.mas
    benchmark_cfg_redacted = _redact_secrets(benchmark_cfg)

    return {
        "timestamp": run_root.name,
        "run_root": str(run_root),
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
            "output_dir": str(run_root.parent),
        },
        "system": {
            "mode": _mas_mode_label(config),
            "mas": {
                "topology": mas_cfg.resolved_topology(),
                "levels": mas_cfg.levels,
                "number_of_agents": mas_cfg.total_agents,
                "agents_per_level": mas_cfg.resolved_agents_per_level(),
                "group_sizes": list(mas_cfg.group_sizes or []),
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_command(args: argparse.Namespace) -> int:
    # 1) Load runtime knobs (OpenRouter, MAS topology, model routing, benchmark settings).
    config = load_experiment_config(args.config)

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

    timestamp = _now_stamp()
    run_root = output_root / timestamp
    benchmark_root = run_root / benchmark_name
    benchmark_root.mkdir(parents=True, exist_ok=True)

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
        run_root=run_root,
    )
    _write_experiment_settings(run_root / "experiment_settings.json", experiment_settings)

    summary_rows: list[dict[str, Any]] = []
    summary_json: dict[str, Any] = {
        "timestamp": timestamp,
        "benchmark": benchmark_name,
        "config_path": str(Path(args.config).resolve()),
        "runs_per_task": runs_per_task,
        "task_count": len(tasks),
        "experiment_settings_path": str((run_root / "experiment_settings.json").resolve()),
        "tasks": [],
    }

    for task_idx, task in enumerate(tasks):
        task_dir = benchmark_root / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        run_traces = []
        evaluations = []

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

            eval_path = task_dir / f"run_{run_index}.eval.json"
            _write_eval(eval_path, evaluation, run.final_answer)

        # 5) Convert trace+eval into descriptor artifacts and analysis outputs.
        analysis = analyze_task_runs(
            task_id=task.task_id,
            benchmark_name=benchmark_name,
            run_traces=run_traces,
            evaluations=evaluations,
            output_dir=task_dir,
        )

        task_summary = {
            "task_id": task.task_id,
            "prompt": task.prompt,
            "reference_answer": task.reference_answer,
            "evaluation": analysis["evaluation"],
            "descriptor": analysis["descriptor"],
            "stage_bottleneck": analysis["stage_bottleneck"],
        }
        summary_json["tasks"].append(task_summary)

        row: dict[str, Any] = {
            "benchmark": benchmark_name,
            "task_id": task.task_id,
            "runs": analysis["evaluation"].get("count", 0),
            "eval_avg_score": analysis["evaluation"].get("avg_score", 0.0),
            "eval_success_rate": analysis["evaluation"].get("success_rate", 0.0),
        }
        row.update(analysis["descriptor"])
        summary_rows.append(row)

    summary_json_path = run_root / "summary.json"
    summary_csv_path = run_root / "summary.csv"
    summary_json_path.write_text(
        json.dumps(summary_json, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(summary_csv_path, summary_rows)

    print(f"Run complete: {run_root}")
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
