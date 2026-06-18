from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import get_benchmark, list_benchmarks
from MAS import OpenRouterLLMClient, load_experiment_config

from .executor import MASSCandidateExecutor
from .existing_benchmarks import ExistingBenchmarkMASSAdapter
from .framework import MASSFramework
from .models import MASSConfig, SearchSpace, StageResult

try:
    from datetime import UTC
except ImportError:  # pragma: no cover
    UTC = UTC


DEFAULT_EXCLUDED_BENCHMARKS = {"finance_agent"}
DEFAULT_ENABLED_BLOCKS = ("aggregate", "reflect", "debate", "execute")


def main() -> None:
    args = _parse_args()
    config = load_experiment_config(args.config)
    benchmarks = _resolve_benchmarks(args)
    output_root = Path(args.output_dir).expanduser().resolve()
    run_id = args.run_id or _now_stamp()
    experiment_root = output_root / run_id
    experiment_root.mkdir(parents=True, exist_ok=True)

    llm_client = OpenRouterLLMClient(config.openrouter, config.models)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "benchmarks": {},
        "excluded_benchmarks": sorted(DEFAULT_EXCLUDED_BENCHMARKS | set(args.exclude_benchmark)),
        "settings": {
            "task_limit": args.task_limit,
            "candidates_per_stage": args.candidates_per_stage,
            "max_validation_examples": args.max_validation_examples,
            "topology_temperature": args.topology_temperature,
            "max_agent_budget": args.max_agent_budget,
            "run_global_prompt_stage": not args.no_global_prompt_stage,
            "model_agent_type": args.model_agent_type,
        },
    }

    for benchmark_name in benchmarks:
        benchmark_dir = experiment_root / benchmark_name
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{_now_stamp()}] MASS_BENCH_START benchmark={benchmark_name}", flush=True)
        try:
            result_payload = _run_one_benchmark(
                benchmark_name=benchmark_name,
                args=args,
                config=config,
                llm_client=llm_client,
                output_dir=benchmark_dir,
            )
            summary["benchmarks"][benchmark_name] = result_payload
            print(
                f"[{_now_stamp()}] MASS_BENCH_DONE benchmark={benchmark_name} "
                f"best_score={result_payload.get('best_score')}",
                flush=True,
            )
        except Exception as exc:
            error_payload = {"error": f"{type(exc).__name__}: {exc}"}
            summary["benchmarks"][benchmark_name] = error_payload
            _write_json(benchmark_dir / "error.json", error_payload)
            print(
                f"[{_now_stamp()}] MASS_BENCH_ERROR benchmark={benchmark_name} "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )
            if not args.keep_going:
                raise

    _write_json(experiment_root / "summary.json", summary)
    print(f"[{_now_stamp()}] MASS_RUN_DONE output={experiment_root}", flush=True)


def _run_one_benchmark(
    *,
    benchmark_name: str,
    args: argparse.Namespace,
    config: Any,
    llm_client: OpenRouterLLMClient,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark_cfg = _benchmark_section_config(config, benchmark_name)
    benchmark = get_benchmark(benchmark_name, config=benchmark_cfg)
    tasks = list(benchmark.load_tasks(task_limit=args.task_limit))
    if not tasks:
        raise RuntimeError(f"No tasks loaded for benchmark '{benchmark_name}'")

    model_callback = _make_openrouter_callback(
        llm_client=llm_client,
        benchmark_name=benchmark_name,
        model_agent_type=args.model_agent_type,
        temperature=args.temperature,
    )
    adapter = ExistingBenchmarkMASSAdapter(
        benchmark=benchmark,
        tasks=tasks,
        executor=MASSCandidateExecutor(model_callback=model_callback),
        metadata={"benchmark_name": benchmark_name},
    )
    search_space = SearchSpace(
        enabled_blocks=tuple(args.enabled_block),
        max_agent_budget=int(args.max_agent_budget),
    )
    framework = MASSFramework(
        config=MASSConfig(
            task_name=benchmark_name,
            search_space=search_space,
            candidates_per_stage=int(args.candidates_per_stage),
            random_seed=int(args.seed),
            max_validation_examples=args.max_validation_examples,
            run_global_prompt_stage=not args.no_global_prompt_stage,
            keep_best_after_global_prompt_stage=args.keep_best_after_global_prompt_stage,
            topology_temperature=float(args.topology_temperature),
        ),
        benchmark=adapter,
    )
    results = framework.run()
    payload = _results_payload(results)
    payload["task_count"] = len(tasks)
    payload["tasks"] = [str(task.task_id) for task in tasks]
    payload["best_score"] = payload["final_stage"]["best_score"]
    _write_json(output_dir / "mass_results.json", payload)
    return payload


def _make_openrouter_callback(
    *,
    llm_client: OpenRouterLLMClient,
    benchmark_name: str,
    model_agent_type: str,
    temperature: float,
):
    def callback(role: str, prompt_text: str, example: Any, context: dict[str, Any]) -> str:
        task_id = f"{benchmark_name}:{example.example_id}:{role}"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a MASS reproduction agent block. Follow the assigned role, "
                    "use the provided context, and return only the block output."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Role: {role}\n"
                    f"Task:\n{example.prompt}\n\n"
                    f"Block prompt and context:\n{prompt_text}"
                ),
            },
        ]
        result = llm_client.generate(
            prompt=messages,
            agent_type=model_agent_type,
            task_id=task_id,
            run_index=0,
            agent_id=role,
            temperature=temperature,
        )
        context.setdefault("llm_calls", []).append(
            {
                "role": role,
                "model": result.model,
                "token_in": result.token_in,
                "token_out": result.token_out,
                "mock_used": result.mock_used,
            }
        )
        return result.text

    return callback


def _results_payload(results: dict[str, StageResult]) -> dict[str, Any]:
    final_key = list(results.keys())[-1]
    return {
        "final_stage_name": final_key,
        "final_stage": _stage_payload(results[final_key]),
        "stages": {key: _stage_payload(value) for key, value in results.items()},
    }


def _stage_payload(stage: StageResult) -> dict[str, Any]:
    return {
        "stage_name": stage.stage_name,
        "best_score": stage.best_score,
        "explored_candidates": stage.explored_candidates,
        "best_candidate": {
            "stage": stage.best_candidate.stage,
            "workflow": stage.best_candidate.workflow.to_payload(),
            "prompt_blocks": sorted(stage.best_candidate.prompts.keys()),
            "metadata": _jsonable(stage.best_candidate.metadata),
        },
        "metadata": _jsonable(stage.metadata),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_payload"):
        return _jsonable(value.to_payload())
    if hasattr(value, "__dict__") and value.__class__.__module__.startswith("reproduce."):
        return _jsonable(value.__dict__)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _benchmark_section_config(config: Any, benchmark_name: str) -> dict[str, Any]:
    cfg = dict(getattr(config, benchmark_name, {}) or {})
    if "openrouter" not in cfg:
        cfg["openrouter"] = {}
    if config.openrouter.api_key and "api_key" not in cfg["openrouter"]:
        cfg["openrouter"]["api_key"] = config.openrouter.api_key
    if config.openrouter.base_url and "base_url" not in cfg["openrouter"]:
        cfg["openrouter"]["base_url"] = config.openrouter.base_url
    return cfg


def _resolve_benchmarks(args: argparse.Namespace) -> list[str]:
    excluded = DEFAULT_EXCLUDED_BENCHMARKS | set(args.exclude_benchmark)
    if args.benchmark:
        requested = list(args.benchmark)
    else:
        requested = list_benchmarks()
    return [name for name in requested if name not in excluded]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone MASS reproduction framework on existing benchmarks."
    )
    parser.add_argument("--config", default="config/experiment.example.toml")
    parser.add_argument("--output-dir", default="outputs_mass_reproduce")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--exclude-benchmark", action="append", default=[])
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--candidates-per-stage", type=int, default=8)
    parser.add_argument("--max-agent-budget", type=int, default=12)
    parser.add_argument("--topology-temperature", type=float, default=1.0)
    parser.add_argument("--enabled-block", action="append", default=list(DEFAULT_ENABLED_BLOCKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model-agent-type", default="default")
    parser.add_argument("--no-global-prompt-stage", action="store_true")
    parser.add_argument("--keep-best-after-global-prompt-stage", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
