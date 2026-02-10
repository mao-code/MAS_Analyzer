from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from MAS import MASRunner, OpenRouterLLMClient, load_experiment_config
from benchmark import BenchmarkEvaluation, get_benchmark, list_benchmarks
from descriptor.experiment import analyze_task_runs, write_run_trace


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _benchmark_section_config(config: Any, benchmark_name: str) -> Dict[str, Any]:
    if benchmark_name == "finance_agent":
        return dict(config.finance_agent)
    if benchmark_name == "browsecomp":
        return dict(config.browsecomp)
    return {}


def _write_eval(path: Path, evaluation: BenchmarkEvaluation, prediction: str) -> None:
    payload = {
        "task_id": evaluation.task_id,
        "score": evaluation.score,
        "success": evaluation.success,
        "details": evaluation.details,
        "prediction": prediction,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_command(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)

    benchmark_name = args.benchmark
    benchmark_cfg = _benchmark_section_config(config, benchmark_name)
    benchmark = get_benchmark(benchmark_name, config=benchmark_cfg)

    llm_client = OpenRouterLLMClient(config.openrouter, config.models)
    runner = MASRunner(config, llm_client)

    task_limit = args.task_limit if args.task_limit is not None else config.experiment.task_limit
    runs_per_task = (
        args.runs_per_task
        if args.runs_per_task is not None
        else config.experiment.runs_per_task
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

    summary_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {
        "timestamp": timestamp,
        "benchmark": benchmark_name,
        "config_path": str(Path(args.config).resolve()),
        "runs_per_task": runs_per_task,
        "task_count": len(tasks),
        "tasks": [],
    }

    for task_idx, task in enumerate(tasks):
        task_dir = benchmark_root / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        run_traces = []
        evaluations = []

        for run_index in range(runs_per_task):
            run_seed = seed + (task_idx * 1000) + run_index
            run = runner.run_task(task=task, run_index=run_index, seed=run_seed)

            trace_path = task_dir / f"run_{run_index}.trace.jsonl"
            write_run_trace(run.trace_events, trace_path)
            run_traces.append(run.trace_events)

            evaluation = benchmark.evaluate(
                task,
                run.final_answer,
                run_metadata=run.run_metadata,
            )
            evaluations.append(evaluation)

            eval_path = task_dir / f"run_{run_index}.eval.json"
            _write_eval(eval_path, evaluation, run.final_answer)

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

        row: Dict[str, Any] = {
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
    benchmark_cfg: Dict[str, Any]
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
