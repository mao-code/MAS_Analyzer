from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import get_benchmark
from descriptor.metrics import compute_run_metrics
from descriptor.schema import TraceEvent
from MAS import OpenRouterLLMClient, load_experiment_config

from .models import AgentSquareConfig, AgentSquareModule
from .modules import DEFAULT_MODULE_POOLS, spec_from_names
from .runtime_runner import AgentSquareRuntimeRunner

try:
    from datetime import UTC
except ImportError:  # pragma: no cover
    UTC = None  # type: ignore[assignment]


DEFAULT_MODEL = "google/gemma-4-31b-it"


def main() -> None:
    args = _parse_args()
    _load_env_file(args.env_file)
    if int(args.max_tokens) > 0:
        os.environ["OPENROUTER_MAX_TOKENS"] = str(int(args.max_tokens))
    else:
        os.environ.pop("OPENROUTER_MAX_TOKENS", None)

    config = load_experiment_config(args.config)
    config.openrouter.base_url = args.openrouter_base_url
    config.openrouter.timeout_s = float(args.timeout_s)
    config.models["default"] = args.model
    llm_client = OpenRouterLLMClient(config.openrouter, config.models)

    run_id = args.run_id or _now_stamp()
    output_root = Path(args.output_dir).expanduser().resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "method": "agentsquare",
        "run_id": run_id,
        "settings": {
            "model": args.model,
            "temperature": args.temperature,
            "task_limit": args.task_limit,
            "validation_task_limit": args.validation_task_limit,
            "final_task_limit": args.final_task_limit,
            "final_task_offset": args.final_task_offset,
            "runs_per_task": args.runs_per_task,
            "workers": args.workers,
            "planning": args.planning,
            "reasoning": args.reasoning,
            "tooluse": args.tooluse,
            "memory": args.memory,
            "search": args.search,
            "max_search_candidates": args.max_search_candidates,
            "search_iterations": args.search_iterations,
            "module_evolution_mode": args.module_evolution_mode,
            "predictor_mode": args.predictor_mode,
            "predictor_top_k": args.predictor_top_k,
            "validation_repeats": args.validation_repeats,
            "max_tokens": args.max_tokens if int(args.max_tokens) > 0 else None,
        },
        "benchmarks": {},
    }

    for benchmark_name in args.benchmark:
        print(f"[{_now_stamp()}] AGENTSQUARE_BENCH_START benchmark={benchmark_name}", flush=True)
        try:
            payload = _run_one_benchmark(
                benchmark_name=benchmark_name,
                args=args,
                llm_client=llm_client,
                output_dir=output_root / benchmark_name,
            )
            summary["benchmarks"][benchmark_name] = payload
            print(
                f"[{_now_stamp()}] AGENTSQUARE_BENCH_DONE benchmark={benchmark_name} "
                f"score={payload.get('score')}",
                flush=True,
            )
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            summary["benchmarks"][benchmark_name] = payload
            _write_json(output_root / benchmark_name / "error.json", payload)
            print(
                f"[{_now_stamp()}] AGENTSQUARE_BENCH_ERROR benchmark={benchmark_name} "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )
            if not args.keep_going:
                raise

    _write_json(output_root / "summary.json", summary)
    print(f"[{_now_stamp()}] AGENTSQUARE_RUN_DONE output={output_root}", flush=True)


def _run_one_benchmark(
    *,
    benchmark_name: str,
    args: argparse.Namespace,
    llm_client: OpenRouterLLMClient,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_cfg = _benchmark_section_config(args.config, benchmark_name)
    _inject_benchmark_runtime_config(
        benchmark_cfg=benchmark_cfg,
        benchmark_name=benchmark_name,
        args=args,
    )
    if args.max_tool_iterations is not None:
        benchmark_cfg["max_tool_iterations"] = max(1, int(args.max_tool_iterations))
    benchmark = get_benchmark(benchmark_name, config=benchmark_cfg)
    tasks = list(benchmark.load_tasks(task_limit=args.task_limit))
    validation_tasks = _select_validation_tasks(args=args, tasks=tasks)
    final_tasks = _select_final_tasks(args=args, tasks=tasks)
    if not final_tasks:
        raise RuntimeError(f"No final tasks selected for benchmark '{benchmark_name}'")

    initial_spec = spec_from_names(
        planning=args.planning,
        reasoning=args.reasoning,
        tooluse=args.tooluse,
        memory=args.memory,
    )
    spec = initial_spec
    generated_modules: dict[str, dict[str, AgentSquareModule]] = {}
    search_payload: dict[str, Any] | None = None
    search_source: Path | None = None
    if args.search_source:
        search_source = _resolve_search_source(args.search_source)
        search_payload = json.loads(search_source.read_text(encoding="utf-8"))
        generated_modules = _modules_from_archive_payload(search_payload.get("module_archive", {}))
        spec = spec_from_names(**search_payload["best_spec_names"], extra_modules=generated_modules)
    elif args.search:
        if not validation_tasks:
            raise RuntimeError("--search requires --validation-task-limit > 0")
        search_payload = _run_search(
            benchmark=benchmark,
            benchmark_name=benchmark_name,
            args=args,
            llm_client=llm_client,
            validation_tasks=validation_tasks,
            output_dir=output_dir / "search",
        )
        generated_modules = _modules_from_archive_payload(search_payload.get("module_archive", {}))
        spec = spec_from_names(**search_payload["best_spec_names"], extra_modules=generated_modules)

    runner = AgentSquareRuntimeRunner(
        spec=spec,
        llm_client=llm_client,
        config=AgentSquareConfig(
            model_agent_type=args.model_agent_type,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens) if int(args.max_tokens) > 0 else None,
        ),
    )
    _write_json(output_dir / "agentsquare_spec.json", spec.to_payload())
    _write_json(
        output_dir / "split.json",
        {
            "task_limit": args.task_limit,
            "validation_task_limit": args.validation_task_limit,
            "validation_task_ids": [task.task_id for task in validation_tasks],
            "final_task_offset": args.final_task_offset,
            "final_task_limit": args.final_task_limit,
            "final_task_ids": [task.task_id for task in final_tasks],
        },
    )

    jobs = [(task, run_index) for task in final_tasks for run_index in range(args.runs_per_task)]
    if int(args.workers) <= 1:
        run_payloads = [
            _execute_one(
                benchmark=benchmark,
                runner=runner,
                task=task,
                run_index=run_index,
                seed=args.seed + run_index,
                benchmark_name=benchmark_name,
                output_dir=output_dir / "final",
                max_tool_iterations=args.max_tool_iterations,
                resume=args.resume,
            )
            for task, run_index in jobs
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = [
                pool.submit(
                    _execute_one,
                    benchmark=benchmark,
                    runner=runner,
                    task=task,
                    run_index=run_index,
                    seed=args.seed + run_index,
                    benchmark_name=benchmark_name,
                    output_dir=output_dir / "final",
                    max_tool_iterations=args.max_tool_iterations,
                    resume=args.resume,
                )
                for task, run_index in jobs
            ]
            run_payloads = [future.result() for future in concurrent.futures.as_completed(futures)]

    score = sum(float(item["score"]) for item in run_payloads) / len(run_payloads)
    by_run: dict[int, list[float]] = {}
    for item in run_payloads:
        by_run.setdefault(int(item["run_index"]), []).append(float(item["score"]))
    per_run_scores = {
        str(run_index): sum(values) / len(values)
        for run_index, values in sorted(by_run.items())
        if values
    }
    payload = {
        "benchmark": benchmark_name,
        "score": score,
        "success_rate": score,
        "task_count": len(final_tasks),
        "run_count": len(run_payloads),
        "per_run_scores": per_run_scores,
        "runs": run_payloads,
    }
    if search_payload is not None:
        payload["search"] = search_payload
    if search_source is not None:
        payload["transfer_source"] = str(search_source)
    _write_json(output_dir / "results.json", payload)
    return payload


def _resolve_search_source(value: str) -> Path:
    source = Path(value).expanduser().resolve()
    if source.is_dir():
        for relative in ("search_results.json", "search/search_results.json"):
            candidate = source / relative
            if candidate.is_file():
                return candidate
    if source.is_file():
        return source
    raise FileNotFoundError(f"AgentSquare search source not found: {source}")


def _run_search(
    *,
    benchmark: Any,
    benchmark_name: str,
    args: argparse.Namespace,
    llm_client: OpenRouterLLMClient,
    validation_tasks: list[Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    search_results_path = output_dir / "search_results.json"
    if args.resume and search_results_path.exists():
        return json.loads(search_results_path.read_text(encoding="utf-8"))
    seed_spec = {
        "planning": args.planning,
        "reasoning": args.reasoning,
        "tooluse": args.tooluse,
        "memory": args.memory,
    }
    tested_cases: list[dict[str, Any]] = []
    measured_scores: dict[tuple[tuple[str, str], ...], float] = {}
    current_agent = dict(seed_spec)
    generated_modules: dict[str, dict[str, AgentSquareModule]] = {}
    best_case: dict[str, Any] | None = None
    iterations: list[dict[str, Any]] = []

    for iteration in range(max(1, int(args.search_iterations))):
        generated_payload = _generate_module_proposals(
            current_agent=current_agent,
            generated_modules=generated_modules,
            llm_client=llm_client,
            args=args,
            benchmark_name=benchmark_name,
            iteration=iteration,
            tested_cases=tested_cases,
        )
        for module_type, module in generated_payload["modules"].items():
            generated_modules.setdefault(module_type, {})[module.name] = module
        generated_specs = _generated_module_specs(current_agent, generated_payload["modules"])
        evolved = _evolution_specs(current_agent, generated_modules=generated_modules)
        recombined = _candidate_specs(args, generated_modules=generated_modules)
        candidate_pool = _unique_specs([current_agent, *generated_specs, *evolved, *recombined])
        candidate_pool = candidate_pool[: max(1, int(args.max_search_candidates))]
        predictor_payload = _predict_candidate_scores(
            candidates=candidate_pool,
            current_agent=current_agent,
            measured_scores=measured_scores,
            generated_modules=generated_modules,
            llm_client=llm_client,
            args=args,
            benchmark_name=benchmark_name,
            iteration=iteration,
            tested_cases=tested_cases,
        )
        predictions = list(predictor_payload["ranked_candidates"])
        ranked = sorted(predictions, key=lambda item: item["predicted_score"], reverse=True)
        selected = ranked[: max(1, int(args.predictor_top_k))]
        iteration_cases: list[dict[str, Any]] = []
        _write_json(
            output_dir / f"iteration_{iteration:03d}" / "predictor_rankings.json",
            {
                "iteration": iteration,
                "current_agent": current_agent,
                "candidate_count": len(candidate_pool),
                "generated_modules": generated_payload["artifact"],
                "ranked_candidates": ranked,
                "selected_for_validation": selected,
                "predictor": predictor_payload["predictor"],
            },
        )
        for selected_index, prediction in enumerate(selected):
            spec_names = dict(prediction["spec_names"])
            key = _spec_key(spec_names)
            if key in measured_scores:
                score = measured_scores[key]
                case = {
                    "iteration": iteration,
                    "candidate_index": len(tested_cases),
                    "selected_index": selected_index,
                    "spec_names": dict(spec_names),
                    "score": score,
                    "predicted_score": prediction["predicted_score"],
                    "prediction_reason": prediction["reason"],
                    "cached": True,
                    "task_count": len(validation_tasks),
                    "run_count": 0,
                }
                tested_cases.append(case)
                iteration_cases.append(case)
                continue
            case = _evaluate_search_candidate(
                benchmark=benchmark,
                benchmark_name=benchmark_name,
                args=args,
                llm_client=llm_client,
                validation_tasks=validation_tasks,
                output_dir=output_dir
                / f"iteration_{iteration:03d}"
                / f"candidate_{selected_index:03d}",
                spec_names=spec_names,
                generated_modules=generated_modules,
                candidate_index=len(tested_cases),
                selected_index=selected_index,
                iteration=iteration,
                predicted_score=float(prediction["predicted_score"]),
                prediction_reason=str(prediction["reason"]),
            )
            score = float(case["score"])
            measured_scores[key] = score
            tested_cases.append(case)
            iteration_cases.append(case)
            if best_case is None or score > float(best_case["score"]):
                best_case = case
                current_agent = dict(spec_names)
                _write_json(
                    output_dir / "best_spec.json",
                    spec_from_names(**current_agent, extra_modules=generated_modules).to_payload(),
                )
                _write_json(output_dir / "best_case.json", best_case)
        if iteration_cases:
            best_iteration_case = max(iteration_cases, key=lambda item: float(item["score"]))
            if best_case is None or float(best_iteration_case["score"]) >= float(
                best_case["score"]
            ):
                current_agent = dict(best_iteration_case["spec_names"])
        iterations.append(
            {
                "iteration": iteration,
                "current_agent_after_iteration": dict(current_agent),
                "evolution_candidate_count": len(evolved),
                "generated_evolution_candidate_count": len(generated_specs),
                "recombination_candidate_count": len(recombined),
                "candidate_pool_count": len(candidate_pool),
                "tested_cases": iteration_cases,
            }
        )
        _write_json(
            output_dir / f"iteration_{iteration:03d}" / "iteration_result.json",
            iterations[-1],
        )

    if best_case is None:
        raise RuntimeError("AgentSquare search did not evaluate any candidates")
    payload = {
        "search_type": "agentsquare_iterative_module_search",
        "note": (
            "This implements AgentSquare's control flow over this repo's standardized "
            "module pool: module evolution as one-slot module mutations, recombination "
            "over module slots, LLM or fallback predictor ranking, validation testing, "
            "and best-agent updates. LLM-generated modules are accepted as validated "
            "prompt modules, not executed as arbitrary Python code."
        ),
        "predictor_mode": args.predictor_mode,
        "iteration_count": max(1, int(args.search_iterations)),
        "candidate_count": len(tested_cases),
        "tested_cases": tested_cases,
        "iterations": iterations,
        "best_spec_names": best_case["spec_names"],
        "best_score": best_case["score"],
        "module_archive": _module_archive_payload(generated_modules),
    }
    _write_json(search_results_path, payload)
    return payload


def _evaluate_search_candidate(
    *,
    benchmark: Any,
    benchmark_name: str,
    args: argparse.Namespace,
    llm_client: OpenRouterLLMClient,
    validation_tasks: list[Any],
    output_dir: Path,
    spec_names: dict[str, str],
    generated_modules: dict[str, dict[str, AgentSquareModule]],
    candidate_index: int,
    selected_index: int,
    iteration: int,
    predicted_score: float,
    prediction_reason: str,
) -> dict[str, Any]:
    spec = spec_from_names(**spec_names, extra_modules=generated_modules)
    runner = AgentSquareRuntimeRunner(
        spec=spec,
        llm_client=llm_client,
        config=AgentSquareConfig(
            model_agent_type=args.model_agent_type,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens) if int(args.max_tokens) > 0 else None,
        ),
    )
    _write_json(output_dir / "spec.json", spec.to_payload())
    jobs = [
        (task, run_index)
        for task in validation_tasks
        for run_index in range(max(1, int(args.validation_repeats)))
    ]
    run_payloads = [
        _execute_one(
            benchmark=benchmark,
            runner=runner,
            task=task,
            run_index=run_index,
            seed=args.seed + run_index,
            benchmark_name=benchmark_name,
            output_dir=output_dir,
            max_tool_iterations=args.max_tool_iterations,
            resume=args.resume,
        )
        for task, run_index in jobs
    ]
    score = sum(float(item["score"]) for item in run_payloads) / len(run_payloads)
    case = {
        "iteration": iteration,
        "candidate_index": candidate_index,
        "selected_index": selected_index,
        "spec_names": dict(spec_names),
        "predicted_score": predicted_score,
        "prediction_reason": prediction_reason,
        "score": score,
        "task_count": len(validation_tasks),
        "run_count": len(run_payloads),
        "cached": False,
        "result_path": str((output_dir / "runs").resolve()),
    }
    _write_json(output_dir / "result.json", case)
    print(
        f"[{_now_stamp()}] AGENTSQUARE_SEARCH_CANDIDATE "
        f"benchmark={benchmark_name} iteration={iteration} candidate={candidate_index} "
        f"predicted={predicted_score:.4f} score={score:.4f} spec={spec_names}",
        flush=True,
    )
    return case


def _evolution_specs_with_pool(
    current_agent: dict[str, str],
    pools: dict[str, tuple[AgentSquareModule, ...]],
) -> list[dict[str, str]]:
    evolved: list[dict[str, str]] = []
    for module_type, pool in pools.items():
        for module in pool:
            candidate = dict(current_agent)
            candidate[module_type] = module.name
            if candidate != current_agent:
                evolved.append(candidate)
    return _unique_specs(evolved)


def _evolution_specs(
    current_agent: dict[str, str],
    *,
    generated_modules: dict[str, dict[str, AgentSquareModule]] | None = None,
) -> list[dict[str, str]]:
    """AgentSquare-style module evolution over base and generated module archives."""

    pools = _module_pools(generated_modules or {})
    return _evolution_specs_with_pool(current_agent, pools)


def _module_pools(
    generated_modules: dict[str, dict[str, AgentSquareModule]],
) -> dict[str, tuple[AgentSquareModule, ...]]:
    pools: dict[str, tuple[AgentSquareModule, ...]] = {}
    for module_type, base_pool in DEFAULT_MODULE_POOLS.items():
        extras = tuple(generated_modules.get(module_type, {}).values())
        pools[module_type] = tuple(base_pool) + extras
    return pools


def _generated_module_specs(
    current_agent: dict[str, str],
    generated: dict[str, AgentSquareModule],
) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for module_type, module in generated.items():
        candidate = dict(current_agent)
        candidate[module_type] = module.name
        specs.append(candidate)
    return specs


def _generate_module_proposals(
    *,
    current_agent: dict[str, str],
    generated_modules: dict[str, dict[str, AgentSquareModule]],
    llm_client: OpenRouterLLMClient,
    args: argparse.Namespace,
    benchmark_name: str,
    iteration: int,
    tested_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    if getattr(args, "module_evolution_mode", "llm") == "off":
        return {
            "modules": {},
            "artifact": {
                "mode": "off",
                "fallback_used": False,
                "reason": "module_evolution_mode_off",
            },
        }
    try:
        prompt = _build_module_proposal_prompt(
            current_agent=current_agent,
            generated_modules=generated_modules,
            benchmark_name=benchmark_name,
            iteration=iteration,
            tested_cases=tested_cases,
        )
        result = llm_client.generate(
            prompt=prompt,
            agent_type=args.model_agent_type,
            task_id=f"agentsquare_module_evolution:{benchmark_name}:iteration_{iteration}",
            run_index=iteration,
            agent_id="agentsquare_module_evolution",
            tools=[],
            max_tool_iterations=1,
            temperature=float(args.module_evolution_temperature),
            max_tokens=int(args.module_evolution_max_tokens),
        )
        if result.mock_used:
            raise RuntimeError("module_evolution_mock_used")
        modules = _parse_module_proposals(
            result.text,
            benchmark_name=benchmark_name,
            iteration=iteration,
        )
        return {
            "modules": modules,
            "artifact": {
                "mode": "llm",
                "fallback_used": False,
                "module_count": len(modules),
                "modules": {k: v.to_payload() for k, v in modules.items()},
                "raw_text": result.text,
                "model": result.model,
                "token_in": result.token_in,
                "token_out": result.token_out,
                "metadata": dict(result.metadata),
            },
        }
    except Exception as exc:
        return {
            "modules": {},
            "artifact": {
                "mode": "none",
                "requested_mode": getattr(args, "module_evolution_mode", "llm"),
                "fallback_used": True,
                "reason": f"{type(exc).__name__}: {exc}",
            },
        }


def _build_module_proposal_prompt(
    *,
    current_agent: dict[str, str],
    generated_modules: dict[str, dict[str, AgentSquareModule]],
    benchmark_name: str,
    iteration: int,
    tested_cases: list[dict[str, Any]],
) -> list[dict[str, str]]:
    archive = {
        module_type: [
            module.to_payload()
            for module in (
                *DEFAULT_MODULE_POOLS[module_type],
                *generated_modules.get(module_type, {}).values(),
            )
            if module.name.lower() != "none"
        ]
        for module_type in DEFAULT_MODULE_POOLS
    }
    recent = [
        {
            "spec_names": case.get("spec_names"),
            "score": case.get("score"),
            "iteration": case.get("iteration"),
        }
        for case in tested_cases[-12:]
    ]
    payload = {
        "task": "Generate new AgentSquare modules for module evolution.",
        "benchmark": benchmark_name,
        "iteration": iteration,
        "current_agent": current_agent,
        "recent_validation_history": recent,
        "module_archive": archive,
        "constraints": [
            "Return strict JSON only.",
            "Generate prompt-level module definitions, not executable code.",
            "Produce at most one module for each of planning, reasoning, tooluse, memory.",
            "Names must be short identifiers and unique enough for this iteration.",
            "Each prompt must describe how that module should transform task/context into output.",
        ],
        "output_schema": {
            "modules": [
                {
                    "module_type": "planning|reasoning|tooluse|memory",
                    "name": "short name",
                    "thought": "one sentence rationale",
                    "prompt": "module instruction prompt",
                    "code": "optional pseudo-code or original code text for audit only",
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are AgentSquare's module evolution operator. Create new "
                "agent module variants from the archive and feedback. Return JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]


def _parse_module_proposals(
    text: str,
    *,
    benchmark_name: str,
    iteration: int,
) -> dict[str, AgentSquareModule]:
    payload = _extract_json_object(text)
    raw_modules = payload.get("modules")
    if not isinstance(raw_modules, list):
        raise ValueError("missing modules list")
    modules: dict[str, AgentSquareModule] = {}
    for index, item in enumerate(raw_modules):
        if not isinstance(item, dict):
            continue
        module_type = str(item.get("module_type") or item.get("module type") or "").lower()
        if module_type not in DEFAULT_MODULE_POOLS:
            continue
        prompt = str(item.get("prompt") or item.get("instruction") or "").strip()
        thought = str(item.get("thought") or item.get("description") or "").strip()
        if not prompt or not thought:
            continue
        raw_name = str(item.get("name") or f"Generated{module_type.title()}{index}").strip()
        safe_name = _safe_module_name(raw_name, module_type, benchmark_name, iteration, index)
        modules[module_type] = AgentSquareModule(
            name=safe_name,
            module_type=module_type,
            thought=thought[:500],
            prompt=prompt[:4000],
            metadata={
                "generated_by": "agentsquare_module_evolution",
                "benchmark": benchmark_name,
                "iteration": iteration,
                "raw_name": raw_name,
                "code_audit_only": str(item.get("code") or "")[:4000],
            },
        )
    if not modules:
        raise ValueError("no valid module proposals")
    return modules


def _safe_module_name(
    raw_name: str,
    module_type: str,
    benchmark_name: str,
    iteration: int,
    index: int,
) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw_name.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        cleaned = f"Generated_{module_type}_{index}"
    return f"GEN_{benchmark_name}_{iteration}_{module_type}_{cleaned}"[:96]


def _module_archive_payload(
    generated_modules: dict[str, dict[str, AgentSquareModule]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        module_type: [module.to_payload() for module in modules.values()]
        for module_type, modules in generated_modules.items()
    }


def _modules_from_archive_payload(
    payload: Any,
) -> dict[str, dict[str, AgentSquareModule]]:
    output: dict[str, dict[str, AgentSquareModule]] = {}
    if not isinstance(payload, dict):
        return output
    for module_type, modules in payload.items():
        if module_type not in DEFAULT_MODULE_POOLS or not isinstance(modules, list):
            continue
        for item in modules:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            prompt = str(item.get("prompt") or "")
            thought = str(item.get("thought") or "")
            if not name or not prompt or not thought:
                continue
            output.setdefault(module_type, {})[name] = AgentSquareModule(
                name=name,
                module_type=module_type,
                thought=thought,
                prompt=prompt,
                metadata=dict(item.get("metadata") or {}),
            )
    return output


def _predict_candidate_scores(
    *,
    candidates: list[dict[str, str]],
    current_agent: dict[str, str],
    measured_scores: dict[tuple[tuple[str, str], ...], float],
    generated_modules: dict[str, dict[str, AgentSquareModule]] | None = None,
    llm_client: OpenRouterLLMClient | None = None,
    args: argparse.Namespace | None = None,
    benchmark_name: str = "",
    iteration: int = 0,
    tested_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rank candidate module tuples with an AgentSquare-style predictor."""

    heuristic_ranked = _heuristic_candidate_scores(
        candidates=candidates,
        current_agent=current_agent,
        measured_scores=measured_scores,
    )
    if args is None or getattr(args, "predictor_mode", "heuristic") == "heuristic":
        return {
            "ranked_candidates": heuristic_ranked,
            "predictor": {
                "mode": "heuristic",
                "fallback_used": False,
                "reason": "predictor_mode_heuristic",
            },
        }
    try:
        llm_ranked, llm_info = _llm_predict_candidate_scores(
            candidates=candidates,
            current_agent=current_agent,
            heuristic_ranked=heuristic_ranked,
            measured_scores=measured_scores,
            generated_modules=generated_modules or {},
            llm_client=llm_client,
            args=args,
            benchmark_name=benchmark_name,
            iteration=iteration,
            tested_cases=tested_cases or [],
        )
        return {
            "ranked_candidates": llm_ranked,
            "predictor": llm_info,
        }
    except Exception as exc:
        return {
            "ranked_candidates": heuristic_ranked,
            "predictor": {
                "mode": "heuristic",
                "requested_mode": "llm",
                "fallback_used": True,
                "reason": f"{type(exc).__name__}: {exc}",
            },
        }


def _heuristic_candidate_scores(
    *,
    candidates: list[dict[str, str]],
    current_agent: dict[str, str],
    measured_scores: dict[tuple[tuple[str, str], ...], float],
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for spec in candidates:
        key = _spec_key(spec)
        if key in measured_scores:
            score = measured_scores[key] + 0.1
            reason = "previous_validation_score"
        else:
            diff_count = sum(1 for slot, value in spec.items() if current_agent.get(slot) != value)
            score = 0.5
            if diff_count == 1:
                score += 0.08
            if spec.get("reasoning") in {"COT", "COT-SC"}:
                score += 0.04
            if spec.get("tooluse") == "IO":
                score += 0.03
            if spec.get("planning") in {"IO", "DEPS"}:
                score += 0.02
            if spec.get("memory") != "None":
                score += 0.01
            reason = f"heuristic_unmeasured_diff_{diff_count}"
        ranked.append(
            {
                "spec_names": dict(spec),
                "predicted_score": round(float(score), 6),
                "reason": reason,
                "predictor_source": "heuristic",
            }
        )
    return ranked


def _llm_predict_candidate_scores(
    *,
    candidates: list[dict[str, str]],
    current_agent: dict[str, str],
    heuristic_ranked: list[dict[str, Any]],
    measured_scores: dict[tuple[tuple[str, str], ...], float],
    generated_modules: dict[str, dict[str, AgentSquareModule]],
    llm_client: OpenRouterLLMClient | None,
    args: argparse.Namespace,
    benchmark_name: str,
    iteration: int,
    tested_cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if llm_client is None:
        raise RuntimeError("llm_client_missing")
    prompt = _build_predictor_prompt(
        candidates=candidates,
        current_agent=current_agent,
        measured_scores=measured_scores,
        generated_modules=generated_modules,
        benchmark_name=benchmark_name,
        iteration=iteration,
        tested_cases=tested_cases,
    )
    result = llm_client.generate(
        prompt=prompt,
        agent_type=args.model_agent_type,
        task_id=f"agentsquare_predictor:{benchmark_name}:iteration_{iteration}",
        run_index=iteration,
        agent_id="agentsquare_predictor",
        tools=[],
        max_tool_iterations=1,
        temperature=0.0,
        max_tokens=int(args.predictor_max_tokens),
    )
    if result.mock_used:
        raise RuntimeError("predictor_mock_used")
    parsed = _parse_predictor_response(result.text, candidates=candidates)
    if not parsed:
        raise RuntimeError("predictor_returned_no_valid_candidates")
    heuristic_by_key = {_spec_key(item["spec_names"]): item for item in heuristic_ranked}
    parsed_by_key = {_spec_key(item["spec_names"]): item for item in parsed}
    ranked: list[dict[str, Any]] = []
    for item in parsed:
        key = _spec_key(item["spec_names"])
        merged = dict(item)
        merged["predictor_source"] = "llm"
        if key in heuristic_by_key:
            merged["heuristic_score"] = heuristic_by_key[key]["predicted_score"]
        ranked.append(merged)
    for candidate in heuristic_ranked:
        key = _spec_key(candidate["spec_names"])
        if key not in parsed_by_key:
            missing = dict(candidate)
            missing["reason"] = f"heuristic_append_after_llm: {missing.get('reason', '')}"
            ranked.append(missing)
    info = {
        "mode": "llm",
        "fallback_used": False,
        "model": result.model,
        "mock_used": result.mock_used,
        "token_in": result.token_in,
        "token_out": result.token_out,
        "raw_text": result.text,
        "metadata": dict(result.metadata),
    }
    return ranked, info


def _build_predictor_prompt(
    *,
    candidates: list[dict[str, str]],
    current_agent: dict[str, str],
    measured_scores: dict[tuple[tuple[str, str], ...], float],
    generated_modules: dict[str, dict[str, AgentSquareModule]],
    benchmark_name: str,
    iteration: int,
    tested_cases: list[dict[str, Any]],
) -> list[dict[str, str]]:
    history = [
        {
            "spec_names": case.get("spec_names"),
            "score": case.get("score"),
            "iteration": case.get("iteration"),
        }
        for case in tested_cases[-12:]
    ]
    candidates_payload = []
    for index, spec in enumerate(candidates):
        candidates_payload.append(
            {
                "index": index,
                "spec_names": spec,
                "measured_score": measured_scores.get(_spec_key(spec)),
                "module_thoughts": _module_thoughts(spec, generated_modules=generated_modules),
            }
        )
    return [
        {
            "role": "system",
            "content": (
                "You are the AgentSquare performance predictor. Rank candidate "
                "agent module combinations before validation. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Predict validation success for AgentSquare module candidates.",
                    "benchmark": benchmark_name,
                    "iteration": iteration,
                    "current_agent": current_agent,
                    "recent_validation_history": history,
                    "candidates": candidates_payload,
                    "output_schema": {
                        "ranked_candidates": [
                            {
                                "index": 0,
                                "predicted_score": "float in [0, 1]",
                                "reason": "short reason",
                            }
                        ]
                    },
                    "instructions": (
                        "Use measured_score when present. Otherwise infer from module fit. "
                        "Include every candidate index once if possible."
                    ),
                },
                ensure_ascii=True,
            ),
        },
    ]


def _module_thoughts(
    spec: dict[str, str],
    *,
    generated_modules: dict[str, dict[str, AgentSquareModule]] | None = None,
) -> dict[str, str]:
    thoughts: dict[str, str] = {}
    for module_type, module_name in spec.items():
        generated = (generated_modules or {}).get(module_type, {})
        if module_name in generated:
            module = generated[module_name]
            thoughts[module_type] = f"{module.thought}\nPrompt: {module.prompt}"
            continue
        for module in DEFAULT_MODULE_POOLS[module_type]:
            if module.name == module_name:
                thoughts[module_type] = module.thought
                break
    return thoughts


def _parse_predictor_response(
    text: str, *, candidates: list[dict[str, str]]
) -> list[dict[str, Any]]:
    payload = _extract_json_object(text)
    ranked_items = payload.get("ranked_candidates")
    if not isinstance(ranked_items, list):
        raise ValueError("missing ranked_candidates list")
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in ranked_items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["index"])
        except Exception:
            continue
        if index < 0 or index >= len(candidates) or index in seen:
            continue
        seen.add(index)
        try:
            score = float(item.get("predicted_score", 0.0))
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))
        output.append(
            {
                "spec_names": dict(candidates[index]),
                "predicted_score": round(score, 6),
                "reason": str(item.get("reason") or "llm_predictor"),
            }
        )
    return output


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("empty predictor response")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("predictor response is not an object")
    return payload


def _unique_specs(specs: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    output: list[dict[str, str]] = []
    for spec in specs:
        key = _spec_key(spec)
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(spec))
    return output


def _spec_key(spec: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in spec.items()))


def _candidate_specs(
    args: argparse.Namespace,
    *,
    generated_modules: dict[str, dict[str, AgentSquareModule]] | None = None,
) -> list[dict[str, str]]:
    seed_spec = {
        "planning": args.planning,
        "reasoning": args.reasoning,
        "tooluse": args.tooluse,
        "memory": args.memory,
    }
    candidates = [seed_spec]
    pools = _module_pools(generated_modules or {})
    for planning in pools["planning"]:
        for reasoning in pools["reasoning"]:
            for tooluse in pools["tooluse"]:
                for memory in pools["memory"]:
                    spec = {
                        "planning": planning.name,
                        "reasoning": reasoning.name,
                        "tooluse": tooluse.name,
                        "memory": memory.name,
                    }
                    if spec not in candidates:
                        candidates.append(spec)
    max_candidates = max(1, int(args.max_search_candidates))
    return candidates[:max_candidates]


def _execute_one(
    *,
    benchmark: Any,
    runner: AgentSquareRuntimeRunner,
    task: Any,
    run_index: int,
    seed: int,
    benchmark_name: str,
    output_dir: Path,
    max_tool_iterations: int | None,
    resume: bool,
) -> dict[str, Any]:
    task_dir = output_dir / "runs" / str(task.task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    run_path = task_dir / f"run_{run_index}.json"
    if resume and run_path.exists():
        return json.loads(run_path.read_text(encoding="utf-8"))

    result = benchmark.run(task=task, runner=runner, run_index=run_index, seed=seed)
    evaluation = benchmark.evaluate(
        task,
        result.final_answer,
        run_metadata=result.run_metadata,
    )
    metrics = compute_run_metrics(
        result.trace_events,
        evaluation=evaluation,
        final_answer=result.final_answer,
        run_metadata=result.run_metadata,
    )
    payload = {
        "task_id": str(task.task_id),
        "run_index": int(run_index),
        "seed": int(seed),
        "prediction": result.final_answer,
        "score": float(evaluation.score),
        "success": bool(evaluation.success),
        "metrics": metrics,
        "run_metadata": result.run_metadata,
        "evaluation_details": evaluation.details,
        "trace": [_trace_to_dict(event) for event in result.trace_events],
    }
    _write_json(run_path, payload)
    print(
        f"[{_now_stamp()}] AGENTSQUARE_RUN_SAVED benchmark={benchmark_name} "
        f"task_id={task.task_id} run_index={run_index} score={evaluation.score}",
        flush=True,
    )
    return payload


def _select_final_tasks(*, args: argparse.Namespace, tasks: list[Any]) -> list[Any]:
    if args.final_task_limit is None:
        return tasks
    start = int(args.final_task_offset)
    end = start + int(args.final_task_limit)
    return tasks[start:end]


def _select_validation_tasks(*, args: argparse.Namespace, tasks: list[Any]) -> list[Any]:
    limit = int(args.validation_task_limit or 0)
    if limit <= 0:
        return []
    return tasks[:limit]


def _benchmark_section_config(config_path: str, benchmark_name: str) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.exists():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    section = data.get(benchmark_name, {})
    return dict(section) if isinstance(section, dict) else {}


def _inject_benchmark_runtime_config(
    *,
    benchmark_cfg: dict[str, Any],
    benchmark_name: str,
    args: argparse.Namespace,
) -> None:
    if benchmark_name == "browsecomp":
        openrouter_cfg = dict(benchmark_cfg.get("openrouter") or {})
        openrouter_cfg.setdefault("base_url", args.openrouter_base_url)
        api_key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
        if api_key:
            openrouter_cfg.setdefault("api_key", api_key)
        benchmark_cfg["openrouter"] = openrouter_cfg
    if benchmark_name == "stabletoolbench":
        benchmark_cfg.setdefault("judge_api_base", args.openrouter_base_url)


def _trace_to_dict(event: TraceEvent) -> dict[str, Any]:
    return {
        "timestamp_start": event.timestamp_start,
        "timestamp_end": event.timestamp_end,
        "actor": event.actor,
        "event_type": event.event_type,
        "payload": event.payload,
        "token_in": event.token_in,
        "token_out": event.token_out,
        "latency_ms": event.latency_ms,
        "cost_usd": event.cost_usd,
        "state_id": event.state_id,
        "extra": event.extra,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _now_stamp() -> str:
    if UTC is not None:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AgentSquare-style baselines on repo benchmarks."
    )
    parser.add_argument("--config", default="config/reproduce_agentsquare.example.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="outputs_agentsquare_reproduce")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--task-limit", type=int, default=1)
    parser.add_argument("--validation-task-limit", type=int, default=0)
    parser.add_argument("--final-task-limit", type=int, default=None)
    parser.add_argument("--final-task-offset", type=int, default=0)
    parser.add_argument("--runs-per-task", type=int, default=1)
    parser.add_argument("--search", action="store_true")
    parser.add_argument(
        "--search-source",
        default=None,
        help="Load an existing search_results.json and skip search.",
    )
    parser.add_argument("--max-search-candidates", type=int, default=8)
    parser.add_argument("--search-iterations", type=int, default=2)
    parser.add_argument("--module-evolution-mode", choices=("llm", "off"), default="llm")
    parser.add_argument("--module-evolution-temperature", type=float, default=0.8)
    parser.add_argument("--module-evolution-max-tokens", type=int, default=2400)
    parser.add_argument("--predictor-mode", choices=("llm", "heuristic"), default="llm")
    parser.add_argument("--predictor-top-k", type=int, default=2)
    parser.add_argument("--predictor-max-tokens", type=int, default=1200)
    parser.add_argument("--validation-repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-agent-type", default="default")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tool-iterations", type=int, default=None)
    parser.add_argument("--planning", default="None")
    parser.add_argument("--reasoning", default="IO")
    parser.add_argument("--tooluse", default="None")
    parser.add_argument("--memory", default="None")
    args = parser.parse_args()
    if args.search and args.search_source:
        parser.error("--search and --search-source are mutually exclusive")
    return args


if __name__ == "__main__":
    main()
