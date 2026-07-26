from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmark import get_benchmark, list_benchmarks
from benchmark.base import BenchmarkEvaluation
from descriptor.metrics import compute_run_metrics, compute_task_metrics
from descriptor.schema import TraceEvent
from MAS import OpenRouterLLMClient, load_experiment_config

from .existing_benchmarks import ExistingBenchmarkMASSAdapter
from .framework import MASSFramework
from .interfaces import BenchmarkExample
from .models import (
    AgentPromptBundle,
    MASSCandidate,
    MASSConfig,
    SearchSpace,
    StageResult,
    WorkflowSpec,
)
from .optimizer import MIPROLikeConfig, MIPROLikePromptOptimizer
from .paper_baselines import _load_env_file
from .prompt_templates import family_prompt_templates
from .runtime_runner import MASSRuntimeRunner

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
DEFAULT_MAX_TOKENS = 0
DEFAULT_MAX_AGENT_BUDGET = 10


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    benchmarks = _resolve_benchmarks(args)
    output_root = Path(args.output_dir).expanduser().resolve()
    run_id = args.run_id or _now_stamp()
    experiment_root = output_root / run_id
    experiment_root.mkdir(parents=True, exist_ok=True)

    config = load_experiment_config(args.config)
    config.openrouter.base_url = args.openrouter_base_url
    config.openrouter.timeout_s = float(args.timeout_s)
    config.models["default"] = args.model
    if int(args.max_tokens) > 0:
        os.environ["OPENROUTER_MAX_TOKENS"] = str(int(args.max_tokens))
    else:
        os.environ.pop("OPENROUTER_MAX_TOKENS", None)
    llm_client = OpenRouterLLMClient(config.openrouter, config.models)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "benchmarks": {},
        "excluded_benchmarks": sorted(DEFAULT_EXCLUDED_BENCHMARKS | set(args.exclude_benchmark)),
        "settings": {
            "task_limit": args.task_limit,
            "validation_task_offset": args.validation_task_offset,
            "validation_task_limit": args.validation_task_limit,
            "final_task_limit": args.final_task_limit,
            "final_task_offset": args.final_task_offset,
            "candidates_per_stage": args.candidates_per_stage,
            "instruction_candidates": args.instruction_candidates,
            "prompt_search_rounds": args.prompt_search_rounds,
            "opt_val_limit": args.opt_val_limit,
            "paper_templates": not args.no_paper_templates,
            "llm_prompt_proposals": args.llm_prompt_proposals,
            "bootstrap_demos": not args.no_bootstrap_demos,
            "max_validation_examples": args.max_validation_examples,
            "validation_repeats": args.validation_repeats,
            "final_evaluation_repeats": args.final_evaluation_repeats,
            "topology_temperature": args.topology_temperature,
            "max_agent_budget": args.max_agent_budget,
            "run_global_prompt_stage": not args.no_global_prompt_stage,
            "model_agent_type": args.model_agent_type,
            "model": args.model,
            "temperature": args.temperature,
            "num_workers": args.num_workers,
            "resume": args.resume,
            "search_only": args.search_only,
            "skip_prompt_search": args.skip_prompt_search,
            "max_tokens": args.max_tokens if int(args.max_tokens) > 0 else None,
            "max_tokens_note": (
                "not set by runner; provider/model default applies"
                if int(args.max_tokens) <= 0
                else "sent as OpenRouter max_tokens"
            ),
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
    llm_client: OpenRouterLLMClient,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark_cfg = _benchmark_section_config(args.config, benchmark_name)
    search_benchmark_cfg = dict(benchmark_cfg)
    if args.max_tool_iterations is not None:
        benchmark_cfg["max_tool_iterations"] = max(1, int(args.max_tool_iterations))
        search_benchmark_cfg["max_tool_iterations"] = max(1, int(args.max_tool_iterations))
    if (
        benchmark_name.lower() == "plancraft"
        and args.plancraft_search_max_steps is not None
    ):
        search_benchmark_cfg["max_steps"] = max(1, int(args.plancraft_search_max_steps))
    benchmark = get_benchmark(benchmark_name, config=benchmark_cfg)
    search_benchmark = (
        get_benchmark(benchmark_name, config=search_benchmark_cfg)
        if search_benchmark_cfg != benchmark_cfg
        else benchmark
    )
    tasks = list(benchmark.load_tasks(task_limit=args.task_limit))
    if not tasks:
        raise RuntimeError(f"No tasks loaded for benchmark '{benchmark_name}'")
    validation_tasks, final_tasks, split_payload = _split_tasks_for_mass(args=args, tasks=tasks)

    adapter = ExistingBenchmarkMASSAdapter(
        benchmark=search_benchmark,
        tasks=validation_tasks,
        validation_repeats=args.validation_repeats,
        metadata={"benchmark_name": benchmark_name, "phase": "validation_search"},
        runtime_llm_client=llm_client,
        model_agent_type=args.model_agent_type,
        temperature=args.temperature,
        seed=args.seed,
    )
    search_space = _resolve_search_space(benchmark_name, args)
    prompt_templates = (
        {}
        if args.no_paper_templates
        else family_prompt_templates(_benchmark_family(benchmark_name), benchmark_name=benchmark_name)
    )
    prompt_checkpoint = _prompt_search_checkpoint_path(output_dir)
    payload: dict[str, Any] | None = None
    best_candidate: MASSCandidate | None = None
    prompt_source = Path(args.prompt_search_source).expanduser() if args.prompt_search_source else None
    if prompt_source is not None:
        if prompt_source.is_dir():
            prompt_source = _prompt_search_checkpoint_path(prompt_source)
        if not prompt_source.exists():
            raise FileNotFoundError(f"--prompt-search-source not found: {prompt_source}")
        try:
            payload = json.loads(prompt_source.read_text(encoding="utf-8"))
            payload = _select_final_stage_payload(
                payload,
                keep_best_after_global_prompt_stage=bool(
                    args.keep_best_after_global_prompt_stage
                ),
            )
            best_candidate = _candidate_from_payload(payload["final_stage"]["best_candidate"])
            payload["prompt_search_source"] = str(prompt_source.resolve())
            _write_json(prompt_checkpoint, payload)
            print(
                f"[{_now_stamp()}] MASS_PROMPT_SEARCH_SOURCE "
                f"benchmark={benchmark_name} source={prompt_source} checkpoint={prompt_checkpoint}",
                flush=True,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid --prompt-search-source {prompt_source}: {type(exc).__name__}:{exc}"
            ) from exc
    elif args.resume and prompt_checkpoint.exists():
        try:
            payload = json.loads(prompt_checkpoint.read_text(encoding="utf-8"))
            payload = _select_final_stage_payload(
                payload,
                keep_best_after_global_prompt_stage=bool(
                    args.keep_best_after_global_prompt_stage
                ),
            )
            best_candidate = _candidate_from_payload(payload["final_stage"]["best_candidate"])
            print(
                f"[{_now_stamp()}] MASS_PROMPT_SEARCH_RESUME "
                f"benchmark={benchmark_name} checkpoint={prompt_checkpoint}",
                flush=True,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"[{_now_stamp()}] MASS_PROMPT_SEARCH_RESUME_SKIP "
                f"benchmark={benchmark_name} reason={type(exc).__name__}:{exc}",
                flush=True,
            )
            payload = None
            best_candidate = None

    if (payload is None or best_candidate is None) and args.skip_prompt_search:
        best_candidate = _default_candidate_for_smoke(
            benchmark_name=benchmark_name,
            prompt_templates=prompt_templates,
        )
        payload = _candidate_only_payload(best_candidate)
        _write_json(prompt_checkpoint, payload)
        print(
            f"[{_now_stamp()}] MASS_PROMPT_SEARCH_SKIPPED "
            f"benchmark={benchmark_name} checkpoint={prompt_checkpoint}",
            flush=True,
        )

    if payload is None or best_candidate is None:
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
                prompt_templates=prompt_templates,
            ),
            benchmark=adapter,
            prompt_optimizer=_make_prompt_optimizer(
                args=args,
                llm_client=llm_client,
                benchmark_name=benchmark_name,
                checkpoint_dir=output_dir / "checkpoints" / "prompt_search_steps",
            ),
        )
        results = framework.run()
        payload = _results_payload(results)
        payload = _select_final_stage_payload(
            payload,
            keep_best_after_global_prompt_stage=bool(args.keep_best_after_global_prompt_stage),
        )
        best_candidate = _candidate_from_payload(payload["final_stage"]["best_candidate"])
        _write_json(prompt_checkpoint, payload)
        print(
            f"[{_now_stamp()}] MASS_PROMPT_SEARCH_CHECKPOINT "
            f"benchmark={benchmark_name} checkpoint={prompt_checkpoint}",
            flush=True,
        )
    if args.search_only:
        final_evaluation = {
            "search_only": True,
            "task_count": 0,
            "repeat_count": 0,
            "scores": [],
            "mean_score": None,
            "executions": [],
        }
        print(
            f"[{_now_stamp()}] MASS_FINAL_EVAL_SKIPPED benchmark={benchmark_name} "
            "reason=search_only",
            flush=True,
        )
    else:
        final_evaluation = _evaluate_final_candidate(
            benchmark=benchmark,
            tasks=final_tasks,
            llm_client=llm_client,
            best_candidate=best_candidate,
            repeats=int(args.final_evaluation_repeats),
            benchmark_name=benchmark_name,
            args=args,
            checkpoint_dir=output_dir / "checkpoints" / "final_evaluation",
        )
    payload["task_count"] = len(tasks)
    payload["tasks"] = [str(task.task_id) for task in tasks]
    payload["validation_tasks"] = [str(task.task_id) for task in validation_tasks]
    payload["final_tasks"] = [str(task.task_id) for task in final_tasks]
    payload["task_split"] = split_payload
    payload["best_score"] = payload["final_stage"]["best_score"]
    payload["final_evaluation"] = final_evaluation
    _write_json(output_dir / "mass_results.json", payload)
    _write_analysis_input(
        experiment_root=output_dir.parent,
        benchmark_name=benchmark_name,
        final_evaluation=final_evaluation,
    )
    return payload


def _split_tasks_for_mass(
    *,
    args: argparse.Namespace,
    tasks: list[Any],
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    tasks_by_id = {str(task.task_id): task for task in tasks}
    validation_ids = _parse_task_id_list(args.validation_task_ids)
    final_ids = _parse_task_id_list(args.final_task_ids)
    if validation_ids is not None or final_ids is not None:
        if validation_ids is None:
            validation_ids = []
        if final_ids is None:
            final_ids = []
        missing = [
            task_id
            for task_id in [*validation_ids, *final_ids]
            if task_id not in tasks_by_id
        ]
        if missing:
            raise ValueError(f"Requested task ids are not loaded: {missing}")
        validation_tasks = [tasks_by_id[task_id] for task_id in validation_ids]
        final_tasks = [tasks_by_id[task_id] for task_id in final_ids]
        validation_required = not (args.prompt_search_source or args.skip_prompt_search)
        if validation_required and not validation_tasks:
            raise ValueError("--validation-task-ids must select at least one task")
        if not final_tasks:
            raise ValueError("--final-task-ids must select at least one task")
        return (
            validation_tasks,
            final_tasks,
            {
                "mode": "explicit_ids",
                "validation_task_ids": validation_ids,
                "final_task_ids": final_ids,
            },
        )

    validation_limit = args.validation_task_limit
    validation_offset = max(0, int(args.validation_task_offset or 0))
    final_limit = args.final_task_limit
    if (
        validation_limit is None
        and validation_offset == 0
        and final_limit is None
        and args.final_task_offset is None
    ):
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
    validation_tasks = list(tasks[validation_offset : validation_offset + validation_count])
    offset = args.final_task_offset
    if offset is None:
        offset = validation_offset + validation_count
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
            "validation_task_offset": validation_offset,
            "validation_task_limit": validation_limit,
            "final_task_limit": final_limit,
            "final_task_offset": offset,
            "validation_task_ids": [str(task.task_id) for task in validation_tasks],
            "final_task_ids": [str(task.task_id) for task in final_tasks],
        },
    )


def _parse_task_id_list(value: Any) -> list[str] | None:
    if value in (None, "", False):
        return None
    if isinstance(value, str):
        path = Path(value).expanduser()
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                payload = json.loads(text)
                if isinstance(payload, list):
                    return [str(item).strip() for item in payload if str(item).strip()]
                raise ValueError(f"Task id JSON file must contain a list: {path}")
            value = text
        return [item.strip() for item in re.split(r"[,\n]", value) if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("task ids must be a comma-separated string, file path, or list")


def _evaluate_final_candidate(
    *,
    benchmark: Any,
    tasks: list[Any],
    llm_client: OpenRouterLLMClient,
    best_candidate: Any,
    repeats: int,
    benchmark_name: str,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    repeat_count = max(1, int(repeats))
    jobs = [
        (task_index, repeat_index, task)
        for repeat_index in range(repeat_count)
        for task_index, task in enumerate(tasks)
    ]
    completed: list[dict[str, Any]] = []
    pending: list[tuple[int, int, Any]] = []
    resumed_count = 0
    for task_index, repeat_index, task in jobs:
        checkpoint_path = _final_checkpoint_path(
            checkpoint_dir=checkpoint_dir,
            task_id=str(task.task_id),
            repeat_index=repeat_index,
        )
        if args.resume and checkpoint_path.exists():
            try:
                completed.append(json.loads(checkpoint_path.read_text(encoding="utf-8")))
                resumed_count += 1
                continue
            except json.JSONDecodeError:
                pass
        pending.append((task_index, repeat_index, task))

    print(
        f"[{_now_stamp()}] MASS_FINAL_EVAL_START benchmark={benchmark_name} "
        f"total={len(jobs)} resumed={resumed_count} pending={len(pending)} "
        f"workers={max(1, int(args.num_workers))}",
        flush=True,
    )
    if max(1, int(args.num_workers)) == 1:
        for task_index, repeat_index, task in pending:
            completed.append(
                _run_final_eval_job(
                    benchmark=benchmark,
                    task=task,
                    task_index=task_index,
                    repeat_index=repeat_index,
                    best_candidate=best_candidate,
                    llm_client=llm_client,
                    benchmark_name=benchmark_name,
                    args=args,
                    checkpoint_dir=checkpoint_dir,
                )
            )
    elif pending:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(args.num_workers))
        ) as executor:
            future_to_job = {
                executor.submit(
                    _run_final_eval_job,
                    benchmark=benchmark,
                    task=task,
                    task_index=task_index,
                    repeat_index=repeat_index,
                    best_candidate=best_candidate,
                    llm_client=llm_client,
                    benchmark_name=benchmark_name,
                    args=args,
                    checkpoint_dir=checkpoint_dir,
                ): (task_index, repeat_index, task)
                for task_index, repeat_index, task in pending
            }
            for future in concurrent.futures.as_completed(future_to_job):
                task_index, repeat_index, task = future_to_job[future]
                try:
                    completed.append(future.result())
                except Exception as exc:
                    error_payload = {
                        "example_id": str(task.task_id),
                        "validation_repeat": repeat_index,
                        "score": 0.0,
                        "success": False,
                        "details": {"error": f"{type(exc).__name__}: {exc}"},
                        "final_answer": "",
                        "turn_count": 0,
                        "trace_events": [],
                        "run_metadata": {
                            "mass_reproduce": True,
                            "benchmark_name": benchmark_name,
                            "phase": "final_evaluation",
                            "task_id": str(task.task_id),
                            "run_index": repeat_index,
                            "seed": int(args.seed) + 100000 + repeat_index * 1000 + task_index,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        "evaluation": {"error": f"{type(exc).__name__}: {exc}"},
                        "checkpoint_status": "error",
                    }
                    _write_json(
                        _final_checkpoint_path(
                            checkpoint_dir=checkpoint_dir,
                            task_id=str(task.task_id),
                            repeat_index=repeat_index,
                        ),
                        error_payload,
                    )
                    completed.append(error_payload)
                    if not args.keep_going:
                        raise

    completed = sorted(
        completed,
        key=lambda item: (str(item.get("example_id")), int(item.get("validation_repeat", 0) or 0)),
    )
    scores = [float(item.get("score", 0.0) or 0.0) for item in completed]
    score = sum(scores) / len(scores) if scores else 0.0
    benchmark_evaluations = [
        {
            "example_id": str(item.get("example_id")),
            "validation_repeat": int(item.get("validation_repeat", 0) or 0),
            "score": float(item.get("score", 0.0) or 0.0),
            "success": bool(item.get("success", False)),
            "details": item.get("details") or {},
        }
        for item in completed
    ]
    executions = [
        {
            "example_id": str(item.get("example_id")),
            "final_answer": str(item.get("final_answer") or ""),
            "turn_count": int(item.get("turn_count", 0) or 0),
            "trace_events": list(item.get("trace_events") or []),
            "run_metadata": dict(item.get("run_metadata") or {}),
            "evaluation": item.get("evaluation") or item.get("details") or {},
            "validation_repeat": int(item.get("validation_repeat", 0) or 0),
        }
        for item in completed
    ]
    details = {
        "scores": scores,
        "benchmark_scores": list(scores),
        "benchmark_evaluations": benchmark_evaluations,
        "executions": executions,
        "validation_repeats": repeat_count,
        "candidate": {
            "stage": best_candidate.stage,
            "workflow": best_candidate.workflow.to_payload(),
            "prompt_blocks": sorted(best_candidate.prompts.keys()),
        },
        "adapter_metadata": {"benchmark_name": benchmark_name, "phase": "final_evaluation"},
        "execution_path": "benchmark.run",
        "checkpoint_dir": str(checkpoint_dir),
        "resumed_count": resumed_count,
        "completed_count": len(completed),
    }
    return {
        "score": float(score),
        "repeats": repeat_count,
        "example_count": len(tasks),
        "benchmark_scores": list(details.get("benchmark_scores") or details.get("scores") or []),
        "benchmark_evaluations": list(details.get("benchmark_evaluations") or []),
        "executions": list(details.get("executions") or []),
        "details": details,
    }


def _run_final_eval_job(
    *,
    benchmark: Any,
    task: Any,
    task_index: int,
    repeat_index: int,
    best_candidate: Any,
    llm_client: OpenRouterLLMClient,
    benchmark_name: str,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    seed = int(args.seed) + 100000 + repeat_index * 1000 + task_index
    runner = MASSRuntimeRunner(
        candidate=best_candidate,
        llm_client=llm_client,
        model_agent_type=args.model_agent_type,
        temperature=args.temperature,
    )
    print(
        f"[{_now_stamp()}] MASS_FINAL_EVAL_JOB_START benchmark={benchmark_name} "
        f"task_id={task.task_id} repeat={repeat_index}",
        flush=True,
    )
    run_result = benchmark.run(
        task=task,
        runner=runner,
        run_index=repeat_index,
        seed=seed,
    )
    evaluation: BenchmarkEvaluation = benchmark.evaluate(
        task,
        run_result.final_answer,
        run_metadata=dict(run_result.run_metadata),
    )
    payload = {
        "example_id": str(task.task_id),
        "validation_repeat": repeat_index,
        "score": float(evaluation.score),
        "success": bool(evaluation.success),
        "details": evaluation.details,
        "final_answer": run_result.final_answer,
        "turn_count": int(
            dict(run_result.run_metadata).get("execution", {}).get(
                "turn_count", len(run_result.trace_events)
            )
        ),
        "trace_events": [event.to_dict() for event in run_result.trace_events],
        "run_metadata": dict(run_result.run_metadata),
        "evaluation": evaluation.details,
        "checkpoint_status": "completed",
    }
    _write_json(
        _final_checkpoint_path(
            checkpoint_dir=checkpoint_dir,
            task_id=str(task.task_id),
            repeat_index=repeat_index,
        ),
        payload,
    )
    print(
        f"[{_now_stamp()}] MASS_FINAL_EVAL_JOB_DONE benchmark={benchmark_name} "
        f"task_id={task.task_id} repeat={repeat_index} score={float(evaluation.score):.4f}",
        flush=True,
    )
    return payload


def _final_checkpoint_path(*, checkpoint_dir: Path, task_id: str, repeat_index: int) -> Path:
    return checkpoint_dir / f"{_safe_path_part(task_id)}__run_{int(repeat_index)}.json"


def _prompt_search_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoints" / "prompt_search.json"


def _default_candidate_for_smoke(
    *,
    benchmark_name: str,
    prompt_templates: dict[str, AgentPromptBundle],
) -> MASSCandidate:
    prompts = dict(prompt_templates)
    if "predictor" not in prompts:
        prompts["predictor"] = AgentPromptBundle(
            system_instruction=f"Solve the {benchmark_name} task and return the final answer.",
            input_fields=("Task",),
            output_fields=("Answer",),
            output_contract="Return the actual final answer.",
            metadata={"source": "skip_prompt_search_default"},
        )
    workflow = WorkflowSpec(
        summarize_rounds=0,
        aggregate_width=1,
        reflect_rounds=0,
        debate_rounds=0,
        execute_enabled=False,
        order=("aggregate",),
    )
    return MASSCandidate(
        workflow=workflow,
        prompts={"predictor": prompts["predictor"]},
        stage="skip_prompt_search_default",
        metadata={
            "benchmark_name": benchmark_name,
            "skip_prompt_search": True,
            "note": "Smoke/debug path; official reproduction should run prompt search.",
        },
    )


def _candidate_only_payload(candidate: MASSCandidate) -> dict[str, Any]:
    stage = StageResult(
        stage_name=candidate.stage,
        best_candidate=candidate,
        best_score=0.0,
        explored_candidates=1,
        metadata={"skip_prompt_search": True},
    )
    payload = _results_payload({"skip_prompt_search_default": stage})
    payload["skip_prompt_search"] = True
    return payload


def _results_payload(results: dict[str, StageResult]) -> dict[str, Any]:
    final_key = list(results.keys())[-1]
    return {
        "final_stage_name": final_key,
        "final_stage": _stage_payload(results[final_key]),
        "stages": {key: _stage_payload(value) for key, value in results.items()},
    }


def _select_final_stage_payload(
    payload: dict[str, Any],
    *,
    keep_best_after_global_prompt_stage: bool,
) -> dict[str, Any]:
    if not keep_best_after_global_prompt_stage:
        return payload
    stages = payload.get("stages")
    if not isinstance(stages, dict) or not stages:
        return payload

    def score_item(item: tuple[str, Any]) -> tuple[float, str]:
        key, stage_payload = item
        if not isinstance(stage_payload, dict):
            return (float("-inf"), key)
        try:
            score = float(stage_payload.get("best_score", float("-inf")))
        except (TypeError, ValueError):
            score = float("-inf")
        return (score, key)

    best_key, best_stage = max(stages.items(), key=score_item)
    if not isinstance(best_stage, dict):
        return payload
    selected = dict(payload)
    selected["final_stage_name"] = str(best_key)
    selected["final_stage"] = best_stage
    selected["best_score"] = best_stage.get("best_score")
    return selected


def _stage_payload(stage: StageResult) -> dict[str, Any]:
    return {
        "stage_name": stage.stage_name,
        "best_score": stage.best_score,
        "explored_candidates": stage.explored_candidates,
        "best_candidate": {
            "stage": stage.best_candidate.stage,
            "workflow": stage.best_candidate.workflow.to_payload(),
            "prompt_blocks": sorted(stage.best_candidate.prompts.keys()),
            "prompts": {
                name: {
                    "system_instruction": bundle.system_instruction,
                    "input_fields": list(bundle.input_fields),
                    "output_fields": list(bundle.output_fields),
                    "output_contract": bundle.output_contract,
                    "exemplar": bundle.exemplar,
                    "metadata": _jsonable(bundle.metadata),
                }
                for name, bundle in sorted(stage.best_candidate.prompts.items())
            },
            "metadata": _jsonable(stage.best_candidate.metadata),
        },
        "metadata": _jsonable(stage.metadata),
    }


def _candidate_from_payload(payload: dict[str, Any]) -> MASSCandidate:
    workflow_payload = dict(payload["workflow"])
    workflow = WorkflowSpec(
        summarize_rounds=int(workflow_payload.get("summarize_rounds", 0) or 0),
        aggregate_width=int(workflow_payload.get("aggregate_width", 1) or 1),
        reflect_rounds=int(workflow_payload.get("reflect_rounds", 0) or 0),
        debate_rounds=int(workflow_payload.get("debate_rounds", 0) or 0),
        execute_enabled=bool(workflow_payload.get("execute_enabled", False)),
        order=tuple(str(item) for item in workflow_payload.get("order", ())),
    )
    prompt_payload = payload.get("prompts")
    if not isinstance(prompt_payload, dict) or not prompt_payload:
        raise ValueError("prompt_search checkpoint does not contain candidate prompts")
    prompts: dict[str, AgentPromptBundle] = {}
    for name, item in prompt_payload.items():
        if not isinstance(item, dict):
            raise ValueError(f"prompt payload for {name!r} is not an object")
        prompts[str(name)] = AgentPromptBundle(
            system_instruction=str(item.get("system_instruction") or ""),
            input_fields=tuple(str(field) for field in item.get("input_fields") or ()),
            output_fields=tuple(str(field) for field in item.get("output_fields") or ()),
            output_contract=str(item.get("output_contract") or ""),
            exemplar=str(item.get("exemplar") or ""),
            metadata=dict(item.get("metadata") or {}),
        )
    return MASSCandidate(
        workflow=workflow,
        prompts=prompts,
        stage=str(payload.get("stage") or "resumed_prompt_search"),
        metadata=dict(payload.get("metadata") or {}),
    )


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


def _write_analysis_input(
    *,
    experiment_root: Path,
    benchmark_name: str,
    final_evaluation: dict[str, Any],
) -> None:
    system_dir = experiment_root / "analysis_input" / benchmark_name / "mass"
    details = dict(final_evaluation.get("details") or {})
    executions = list(final_evaluation.get("executions") or details.get("executions") or [])
    evaluations_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    benchmark_evaluations = (
        final_evaluation.get("benchmark_evaluations") or details.get("benchmark_evaluations") or []
    )
    for item in benchmark_evaluations:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("example_id")), int(item.get("validation_repeat", 0) or 0))
        evaluations_by_key[key] = item

    task_runs: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for execution in executions:
        if not isinstance(execution, dict):
            continue
        task_id = str(execution.get("example_id", "unknown"))
        run_index = int(execution.get("validation_repeat", 0) or 0)
        evaluation_payload = evaluations_by_key.get((task_id, run_index))
        if evaluation_payload is None:
            evaluation_details = dict(execution.get("evaluation") or {})
            score = float(evaluation_details.get("score", 0.0) or 0.0)
            success = bool(evaluation_details.get("success", score > 0.0))
        else:
            score = float(evaluation_payload.get("score", 0.0) or 0.0)
            success = bool(evaluation_payload.get("success", score > 0.0))
            evaluation_details = dict(evaluation_payload.get("details") or {})
        trace_events = [
            TraceEvent.from_dict(event, strict=False)
            for event in execution.get("trace_events", [])
            if isinstance(event, dict)
        ]
        run_metadata = dict(execution.get("run_metadata") or {})
        evaluation = SimpleNamespace(score=score, success=success, details=evaluation_details)
        metrics = compute_run_metrics(
            trace_events,
            evaluation=evaluation,
            final_answer=str(execution.get("final_answer") or ""),
            run_metadata=run_metadata,
        )
        payload = {
            "task_id": task_id,
            "run_index": run_index,
            "prediction": str(execution.get("final_answer") or ""),
            "evaluation": {
                "score": score,
                "success": success,
                "details": evaluation_details,
            },
            "runtime": run_metadata,
            "metrics": metrics,
            "trace": [event.to_dict() for event in trace_events],
        }
        task_dir = system_dir / _safe_path_part(task_id)
        _write_json(task_dir / f"run_{run_index}.trace_metrics.json", payload)
        task_runs.setdefault(task_id, []).append(payload)

    for task_id, runs in sorted(task_runs.items()):
        task_metrics = compute_task_metrics([dict(run["metrics"]) for run in runs])
        summary_rows.append(
            {
                "task_id": task_id,
                "prediction": runs[-1].get("prediction", ""),
                "score": float(task_metrics.get("eval_avg_score", 0.0) or 0.0),
                "success": bool(float(task_metrics.get("success_rate", 0.0) or 0.0) > 0.0),
                "success_rate": float(task_metrics.get("success_rate", 0.0) or 0.0),
                **task_metrics,
                "benchmark": benchmark_name,
                "system_label": "mass",
                "topology": "mass",
            }
        )
    _write_summary_csv(system_dir / "summary.csv", summary_rows)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_path_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe or "unknown"


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


def _make_prompt_optimizer(
    *,
    args: argparse.Namespace,
    llm_client: OpenRouterLLMClient,
    benchmark_name: str = "",
    checkpoint_dir: Path | None = None,
) -> MIPROLikePromptOptimizer:
    proposer = None
    if args.llm_prompt_proposals:
        proposer = _make_llm_instruction_proposer(
            llm_client=llm_client,
            agent_type=args.model_agent_type,
            temperature=args.temperature,
        )
    return MIPROLikePromptOptimizer(
        MIPROLikeConfig(
            instruction_candidates=int(args.instruction_candidates),
            rounds_per_agent=int(args.prompt_search_rounds),
            instruction_proposer=proposer,
            validation_limit=args.opt_val_limit,
            checkpoint_dir=checkpoint_dir,
            bootstrap_demos=not args.no_bootstrap_demos,
            benchmark_name=benchmark_name,
        )
    )


def _make_llm_instruction_proposer(
    *,
    llm_client: OpenRouterLLMClient,
    agent_type: str,
    temperature: float,
) -> Any:
    def propose(
        block_name: str,
        seed_prompt: AgentPromptBundle,
        examples: list[BenchmarkExample],
        workflow: Any,
        scope: str,
        count: int,
    ) -> list[str]:
        example_lines = []
        for example in list(examples)[:3]:
            example_lines.append(
                f"- id={example.example_id}; task={_short_text(example.prompt, limit=220)}; "
                f"reference={_short_text(example.reference_answer, limit=120)}"
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You propose concise system instructions for MASS prompt search. "
                    "Return only a numbered list. Each item must be one instruction. "
                    "Only rewrite the <Instruction> text. Do not change, mention, rename, "
                    "or remove the fixed Input/Output fields."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Block: {block_name}\n"
                    f"Scope: {scope}\n"
                    f"Seed instruction: {seed_prompt.system_instruction}\n"
                    f"Fixed input fields: {list(seed_prompt.input_fields)}\n"
                    f"Fixed output fields: {list(seed_prompt.output_fields)}\n"
                    f"Output contract: {seed_prompt.output_contract or 'benchmark default'}\n"
                    f"Workflow: {workflow.to_payload()}\n"
                    f"Validation examples:\n" + "\n".join(example_lines) + "\n\n"
                    f"Propose {count} alternative instructions optimized for benchmark validation."
                ),
            },
        ]
        print(
            f"[{_now_stamp()}] MASS_PROMPT_PROPOSAL_START block={block_name} "
            f"scope={scope} count={count}",
            flush=True,
        )
        result = llm_client.generate(
            prompt=messages,
            agent_type=agent_type,
            task_id=f"mass_prompt_proposal:{block_name}",
            run_index=0,
            agent_id=f"prompt_optimizer_{block_name}",
            tools=[],
            max_tool_iterations=1,
            temperature=temperature,
        )
        proposals = _parse_instruction_proposals(result.text, limit=count)
        print(
            f"[{_now_stamp()}] MASS_PROMPT_PROPOSAL_DONE block={block_name} "
            f"scope={scope} proposals={len(proposals)} token_in={result.token_in} "
            f"token_out={result.token_out}",
            flush=True,
        )
        return proposals

    return propose


def _parse_instruction_proposals(text: str, *, limit: int) -> list[str]:
    proposals: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.lstrip("-* ")
        if "." in line:
            prefix, rest = line.split(".", 1)
            if prefix.strip().isdigit():
                line = rest.strip()
        if line and line not in proposals:
            proposals.append(line)
        if len(proposals) >= limit:
            break
    return proposals


def _short_text(value: Any, *, limit: int) -> str:
    text = str(value).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _benchmark_family(benchmark_name: str) -> str:
    normalized = benchmark_name.lower()
    if normalized in {"math", "math500", "gsm8k"}:
        return "math_reasoning"
    if normalized in {"drop"}:
        return "discrete_reasoning"
    if normalized in {"hotpotqa", "musique", "2wikimqa", "2wiki", "browsecomp"}:
        return "long_context"
    if normalized in {"mbpp", "humaneval", "livecodebench", "lcb", "scicode"}:
        return "coding"
    if normalized in {"stabletoolbench", "webshop", "agentbench", "workbench"}:
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
        "--validation-task-offset",
        type=int,
        default=0,
        help="Start MASS search/validation at this loaded-task offset.",
    )
    parser.add_argument(
        "--validation-task-limit",
        type=int,
        default=None,
        help="Use N loaded tasks for MASS search/validation after --validation-task-offset.",
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
    parser.add_argument(
        "--validation-task-ids",
        default=None,
        help=(
            "Comma/newline-separated task ids or a file path selecting MASS search/validation "
            "tasks. Overrides validation offset/limit when provided."
        ),
    )
    parser.add_argument(
        "--final-task-ids",
        default=None,
        help=(
            "Comma/newline-separated task ids or a file path selecting final-evaluation tasks. "
            "Overrides final offset/limit when provided."
        ),
    )
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--candidates-per-stage", type=int, default=DEFAULT_TOPOLOGY_CANDIDATES)
    parser.add_argument("--instruction-candidates", type=int, default=10)
    parser.add_argument("--prompt-search-rounds", type=int, default=10)
    parser.add_argument(
        "--opt-val-limit",
        type=int,
        default=None,
        help=(
            "Cap how many validation examples each prompt candidate is scored on during "
            "MIPRO-style search (None = full validation set, the paper default). Lower this "
            "to cut prompt-optimization cost when debugging."
        ),
    )
    parser.add_argument(
        "--llm-prompt-proposals",
        action="store_true",
        help="Use the configured OpenRouter model to propose prompt candidates before validation scoring.",
    )
    parser.add_argument(
        "--no-bootstrap-demos",
        action="store_true",
        help="Skip bootstrapping few-shot demos from validation predictions during prompt search.",
    )
    parser.add_argument("--max-agent-budget", type=int, default=DEFAULT_MAX_AGENT_BUDGET)
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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="OpenRouter max_tokens. Use 0 to omit max_tokens and let the provider/model default apply.",
    )
    parser.add_argument("--model-agent-type", default="default")
    parser.add_argument(
        "--max-tool-iterations",
        type=int,
        default=None,
        help="Override benchmark max_tool_iterations for tool-enabled MASS runs.",
    )
    parser.add_argument(
        "--plancraft-search-max-steps",
        type=int,
        default=None,
        help=(
            "Override only PlanCraft prompt/topology-search max_steps. Final evaluation "
            "still uses the benchmark config max_steps."
        ),
    )
    parser.add_argument(
        "--no-paper-templates",
        action="store_true",
        help="Use the framework's generic role prompts instead of the App. D per-family templates.",
    )
    parser.add_argument("--no-global-prompt-stage", action="store_true")
    parser.add_argument("--keep-best-after-global-prompt-stage", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers for final evaluation only. Prompt/topology search remains serial.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed final-evaluation checkpoints under the same run id.",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Run prompt/topology/workflow search and write mass_results.json without final evaluation.",
    )
    parser.add_argument(
        "--skip-prompt-search",
        action="store_true",
        help=(
            "Debug/smoke path: skip MASS prompt/topology search and directly final-evaluate "
            "the default predictor-only paper template. Do not use for official reproduction."
        ),
    )
    parser.add_argument(
        "--prompt-search-source",
        default=None,
        help=(
            "Reuse a completed prompt_search.json, or a run/benchmark directory containing "
            "checkpoints/prompt_search.json, and run only final evaluation for the current split."
        ),
    )
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
