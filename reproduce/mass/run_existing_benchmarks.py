from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import get_benchmark, list_benchmarks

from .executor import MASSCandidateExecutor
from .existing_benchmarks import ExistingBenchmarkMASSAdapter
from .framework import MASSFramework
from .models import MASSConfig, SearchSpace, StageResult
from .paper_baselines import StandaloneOpenRouterClient, _load_env_file

try:
    from datetime import UTC
except ImportError:  # pragma: no cover
    UTC = UTC


DEFAULT_EXCLUDED_BENCHMARKS = {"finance_agent"}
DEFAULT_MODEL = "google/gemma-4-31b-it"
DEFAULT_VALIDATION_REPEATS = 3
DEFAULT_FINAL_EVALUATION_REPEATS = 3
DEFAULT_TOPOLOGY_CANDIDATES = 10
DEFAULT_TOPOLOGY_TEMPERATURE = 0.05
DEFAULT_MODEL_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    benchmarks = _resolve_benchmarks(args)
    output_root = Path(args.output_dir).expanduser().resolve()
    run_id = args.run_id or _now_stamp()
    experiment_root = output_root / run_id
    experiment_root.mkdir(parents=True, exist_ok=True)

    llm_client = StandaloneOpenRouterClient(
        model=args.model,
        base_url=args.openrouter_base_url,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
    )
    summary: dict[str, Any] = {
        "run_id": run_id,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "benchmarks": {},
        "excluded_benchmarks": sorted(DEFAULT_EXCLUDED_BENCHMARKS | set(args.exclude_benchmark)),
        "settings": {
            "task_limit": args.task_limit,
            "validation_task_limit": args.validation_task_limit,
            "final_task_limit": args.final_task_limit,
            "final_task_offset": args.final_task_offset,
            "candidates_per_stage": args.candidates_per_stage,
            "max_validation_examples": args.max_validation_examples,
            "validation_repeats": args.validation_repeats,
            "final_evaluation_repeats": args.final_evaluation_repeats,
            "topology_temperature": args.topology_temperature,
            "max_agent_budget": args.max_agent_budget,
            "run_global_prompt_stage": not args.no_global_prompt_stage,
            "model_agent_type": args.model_agent_type,
            "model": args.model,
            "max_tokens": args.max_tokens,
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
    llm_client: StandaloneOpenRouterClient,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark_cfg = _benchmark_section_config(args.config, benchmark_name)
    benchmark = get_benchmark(benchmark_name, config=benchmark_cfg)
    tasks = list(benchmark.load_tasks(task_limit=args.task_limit))
    if not tasks:
        raise RuntimeError(f"No tasks loaded for benchmark '{benchmark_name}'")
    validation_tasks, final_tasks, split_payload = _split_tasks_for_mass(args=args, tasks=tasks)

    model_callback = _make_openrouter_callback(
        llm_client=llm_client,
        benchmark_name=benchmark_name,
        model_agent_type=args.model_agent_type,
        temperature=args.temperature,
    )
    adapter = ExistingBenchmarkMASSAdapter(
        benchmark=benchmark,
        tasks=validation_tasks,
        executor=MASSCandidateExecutor(model_callback=model_callback),
        validation_repeats=args.validation_repeats,
        metadata={"benchmark_name": benchmark_name, "phase": "validation_search"},
    )
    search_space = _resolve_search_space(benchmark_name, args)
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
    final_evaluation = _evaluate_final_candidate(
        benchmark=benchmark,
        tasks=final_tasks,
        model_callback=model_callback,
        best_candidate=results[payload["final_stage_name"]].best_candidate,
        repeats=int(args.final_evaluation_repeats),
        benchmark_name=benchmark_name,
    )
    payload["task_count"] = len(tasks)
    payload["tasks"] = [str(task.task_id) for task in tasks]
    payload["validation_tasks"] = [str(task.task_id) for task in validation_tasks]
    payload["final_tasks"] = [str(task.task_id) for task in final_tasks]
    payload["task_split"] = split_payload
    payload["best_score"] = payload["final_stage"]["best_score"]
    payload["final_evaluation"] = final_evaluation
    _write_json(output_dir / "mass_results.json", payload)
    return payload


def _split_tasks_for_mass(
    *,
    args: argparse.Namespace,
    tasks: list[Any],
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    validation_limit = args.validation_task_limit
    final_limit = args.final_task_limit
    if validation_limit is None and final_limit is None and args.final_task_offset is None:
        task_ids = [str(task.task_id) for task in tasks]
        return (
            list(tasks),
            list(tasks),
            {
                "mode": "shared_loaded_tasks",
                "held_out": False,
                "validation_task_ids": task_ids,
                "final_task_ids": task_ids,
            },
        )

    validation_count = len(tasks) if validation_limit is None else max(1, int(validation_limit))
    validation_tasks = list(tasks[:validation_count])
    offset = args.final_task_offset
    if offset is None:
        offset = validation_count
    final_start = max(0, int(offset))
    if final_limit is None:
        final_tasks = list(tasks[final_start:])
    else:
        final_tasks = list(tasks[final_start : final_start + max(1, int(final_limit))])
    held_out = bool(final_tasks) and not {
        str(task.task_id) for task in validation_tasks
    }.intersection(str(task.task_id) for task in final_tasks)
    if not final_tasks:
        final_tasks = list(validation_tasks)
        held_out = False
    return (
        validation_tasks,
        final_tasks,
        {
            "mode": "deterministic_contiguous_split",
            "held_out": held_out,
            "validation_task_limit": validation_limit,
            "final_task_limit": final_limit,
            "final_task_offset": offset,
            "validation_task_ids": [str(task.task_id) for task in validation_tasks],
            "final_task_ids": [str(task.task_id) for task in final_tasks],
        },
    )


def _evaluate_final_candidate(
    *,
    benchmark: Any,
    tasks: list[Any],
    model_callback: Any,
    best_candidate: Any,
    repeats: int,
    benchmark_name: str,
) -> dict[str, Any]:
    adapter = ExistingBenchmarkMASSAdapter(
        benchmark=benchmark,
        tasks=tasks,
        executor=MASSCandidateExecutor(model_callback=model_callback),
        validation_repeats=max(1, repeats),
        metadata={"benchmark_name": benchmark_name, "phase": "final_evaluation"},
    )
    evaluation = adapter.evaluate_candidate(best_candidate, adapter.validation_examples())
    return {
        "score": float(evaluation.score),
        "repeats": max(1, repeats),
        "example_count": len(tasks),
        "details": _jsonable(evaluation.details),
    }


def _make_openrouter_callback(
    *,
    llm_client: StandaloneOpenRouterClient,
    benchmark_name: str,
    model_agent_type: str,
    temperature: float,
):
    del model_agent_type

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
            messages=messages,
            task_id=task_id,
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
    path = Path(config).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg = data.get(benchmark_name) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"[{benchmark_name}] config section must be a table when present.")
    return dict(cfg)


def _resolve_search_space(benchmark_name: str, args: argparse.Namespace) -> SearchSpace:
    if args.enabled_block:
        return SearchSpace(
            enabled_blocks=tuple(args.enabled_block),
            max_agent_budget=int(args.max_agent_budget),
        )
    family = _benchmark_family(benchmark_name)
    enabled_by_family = {
        "math_reasoning": ("aggregate", "reflect", "debate"),
        "discrete_reasoning": ("aggregate", "reflect", "debate"),
        "long_context": ("summarize", "aggregate", "reflect", "debate"),
        "coding": ("aggregate", "reflect", "debate", "execute"),
        "tool_or_web": ("aggregate", "reflect", "debate", "execute"),
        "general": ("aggregate", "reflect", "debate"),
    }
    return SearchSpace(
        enabled_blocks=enabled_by_family[family],
        max_agent_budget=int(args.max_agent_budget),
    )


def _benchmark_family(benchmark_name: str) -> str:
    normalized = benchmark_name.lower()
    if normalized in {"math", "math500", "gsm8k"}:
        return "math_reasoning"
    if normalized in {"drop"}:
        return "discrete_reasoning"
    if normalized in {"hotpotqa", "musique", "2wikimqa", "2wiki", "browsecomp"}:
        return "long_context"
    if normalized in {"mbpp", "humaneval", "livecodebench", "lcb", "scicode", "workbench"}:
        return "coding"
    if normalized in {"stabletoolbench", "webshop", "agentbench"}:
        return "tool_or_web"
    return "general"


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
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="outputs_mass_reproduce")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--exclude-benchmark", action="append", default=[])
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument(
        "--validation-task-limit",
        type=int,
        default=None,
        help="Use only the first N loaded tasks for MASS search/validation.",
    )
    parser.add_argument(
        "--final-task-limit",
        type=int,
        default=None,
        help="Use only N tasks for final evaluation after --final-task-offset.",
    )
    parser.add_argument(
        "--final-task-offset",
        type=int,
        default=None,
        help="Start final evaluation at this loaded-task offset; defaults after validation tasks.",
    )
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--candidates-per-stage", type=int, default=DEFAULT_TOPOLOGY_CANDIDATES)
    parser.add_argument("--max-agent-budget", type=int, default=12)
    parser.add_argument("--topology-temperature", type=float, default=DEFAULT_TOPOLOGY_TEMPERATURE)
    parser.add_argument("--validation-repeats", type=int, default=DEFAULT_VALIDATION_REPEATS)
    parser.add_argument(
        "--final-evaluation-repeats",
        type=int,
        default=DEFAULT_FINAL_EVALUATION_REPEATS,
        help="Paper-style repeated final evaluation runs for the optimized workflow.",
    )
    parser.add_argument("--enabled-block", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=DEFAULT_MODEL_TEMPERATURE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model-agent-type", default="default")
    parser.add_argument("--no-global-prompt-stage", action="store_true")
    parser.add_argument("--keep-best-after-global-prompt-stage", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
