from __future__ import annotations

import argparse
import json
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any

from benchmark import get_benchmark
from main import (
    _benchmark_section_config,
    _experiment_settings_payload,
    _write_experiment_settings,
    _write_summary_csv,
)
from MAS import load_experiment_config


def _load_analysis(task_dir: Path) -> dict[str, Any]:
    return json.loads((task_dir / "analysis.json").read_text(encoding="utf-8"))


def _task_dirs(benchmark_root: Path) -> list[Path]:
    return sorted(path for path in benchmark_root.iterdir() if path.is_dir())


def rehome_run(
    *,
    source_root: Path,
    benchmark_name: str,
    config_path: Path,
    destination_root: Path | None = None,
) -> Path:
    config = load_experiment_config(config_path)
    benchmark_cfg = _benchmark_section_config(config, benchmark_name)
    benchmark = get_benchmark(benchmark_name, config=benchmark_cfg)
    task_limit = config.experiment.task_limit
    tasks = list(benchmark.load_tasks(task_limit=task_limit))
    task_map = {task.task_id: task for task in tasks}

    source_benchmark_root = source_root / benchmark_name
    if not source_benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {source_benchmark_root}")

    if destination_root is None:
        destination_root = Path(config.experiment.output_dir) / source_root.name
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_benchmark_root = destination_root / benchmark_name
    destination_benchmark_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    summary_tasks: list[dict[str, Any]] = []

    for task_dir in _task_dirs(source_benchmark_root):
        destination_task_dir = destination_benchmark_root / task_dir.name
        shutil.copytree(task_dir, destination_task_dir, dirs_exist_ok=True)

        analysis = _load_analysis(task_dir)
        task = task_map.get(task_dir.name)

        task_summary = {
            "task_id": task_dir.name,
            "prompt": task.prompt if task is not None else "",
            "reference_answer": task.reference_answer if task is not None else "",
            "evaluation": analysis["evaluation"],
            "descriptor": analysis["descriptor"],
            "stage_bottleneck": analysis["stage_bottleneck"],
        }
        summary_tasks.append(task_summary)

        row: dict[str, Any] = {
            "benchmark": benchmark_name,
            "task_id": task_dir.name,
            "runs": analysis["evaluation"].get("count", 0),
            "eval_avg_score": analysis["evaluation"].get("avg_score", 0.0),
            "eval_success_rate": analysis["evaluation"].get("success_rate", 0.0),
        }
        row.update(analysis["descriptor"])
        summary_rows.append(row)

    args = Namespace(config=str(config_path))
    experiment_settings = _experiment_settings_payload(
        args=args,
        config=config,
        benchmark_name=benchmark_name,
        benchmark_cfg=benchmark_cfg,
        task_limit=task_limit,
        runs_per_task=config.experiment.runs_per_task,
        seed=config.experiment.seed,
        task_count=len(summary_tasks),
        run_root=destination_root,
    )
    _write_experiment_settings(destination_root / "experiment_settings.json", experiment_settings)

    summary_json = {
        "timestamp": destination_root.name,
        "benchmark": benchmark_name,
        "config_path": str(config_path.resolve()),
        "runs_per_task": config.experiment.runs_per_task,
        "task_count": len(summary_tasks),
        "experiment_settings_path": str((destination_root / "experiment_settings.json").resolve()),
        "tasks": summary_tasks,
    }
    (destination_root / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(destination_root / "summary.csv", summary_rows)
    return destination_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Split one benchmark run out of a mixed run root.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--destination-root", default=None)
    args = parser.parse_args()

    destination = rehome_run(
        source_root=Path(args.source_root).resolve(),
        benchmark_name=args.benchmark,
        config_path=Path(args.config).resolve(),
        destination_root=Path(args.destination_root).resolve() if args.destination_root else None,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
