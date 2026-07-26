from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path
from typing import Any

from benchmark import get_benchmark

from .run_existing_benchmarks import _inject_benchmark_runtime_config, _load_env_file

DEFAULT_BENCHMARKS = (
    "browsecomp",
    "math500",
    "plancraft",
    "stabletoolbench",
    "workbench",
)


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    result = run_preflight(args=args)
    print(_format_result(result))
    if not result["ok"]:
        raise SystemExit(1)


def run_preflight(*, args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    api_key_present = bool(str(os.getenv("OPENROUTER_API_KEY") or "").strip()) or bool(
        str(((data.get("openrouter") or {}).get("api_key")) or "").strip()
    )
    rows: dict[str, dict[str, Any]] = {}
    ok = api_key_present
    for benchmark_name in tuple(args.benchmark or DEFAULT_BENCHMARKS):
        section = dict(data.get(benchmark_name, {}) or {})
        _inject_benchmark_runtime_config(
            benchmark_cfg=section,
            benchmark_name=benchmark_name,
            args=args,
        )
        row: dict[str, Any] = {
            "ok": False,
            "task_count": 0,
            "validation_count": 0,
            "final_count": 0,
            "eval_mode": section.get("eval_mode"),
            "max_tool_iterations": section.get("max_tool_iterations"),
        }
        try:
            benchmark = get_benchmark(benchmark_name, config=section)
            tasks = list(benchmark.load_tasks(task_limit=args.task_limit))
            validation = tasks[: args.validation_task_limit]
            final = tasks[args.final_task_offset : args.final_task_offset + args.final_task_limit]
            row.update(
                {
                    "ok": len(tasks) >= args.final_task_offset + args.final_task_limit
                    and len(validation) == args.validation_task_limit
                    and len(final) == args.final_task_limit,
                    "task_count": len(tasks),
                    "validation_count": len(validation),
                    "final_count": len(final),
                    "first_final_task": final[0].task_id if final else None,
                    "requirements": benchmark.requirements(),
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        ok = ok and bool(row["ok"])
        rows[benchmark_name] = row
    return {
        "ok": ok,
        "config": str(config_path),
        "openrouter_api_key_present": api_key_present,
        "benchmarks": rows,
    }


def _format_result(result: dict[str, Any]) -> str:
    lines = [
        f"Config: {result['config']}",
        f"OpenRouter key present: {result['openrouter_api_key_present']}",
        "",
        "| Benchmark | OK | Tasks | Val | Final | Eval | Max Tools | Error |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for benchmark, row in result["benchmarks"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    benchmark,
                    str(row["ok"]),
                    str(row["task_count"]),
                    str(row["validation_count"]),
                    str(row["final_count"]),
                    str(row.get("eval_mode") or "--"),
                    str(row.get("max_tool_iterations") or "--"),
                    str(row.get("error") or "--"),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(f"Overall OK: {result['ok']}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight AgentSquare formal reproduction.")
    parser.add_argument("--config", default="config/reproduce_agentsquare.example.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--benchmark", action="append", default=None)
    parser.add_argument("--task-limit", type=int, default=40)
    parser.add_argument("--validation-task-limit", type=int, default=10)
    parser.add_argument("--final-task-offset", type=int, default=10)
    parser.add_argument("--final-task-limit", type=int, default=30)
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    return parser.parse_args()


if __name__ == "__main__":
    main()
