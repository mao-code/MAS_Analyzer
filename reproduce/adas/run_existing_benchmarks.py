from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import statistics
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import get_benchmark
from descriptor.metrics import compute_run_metrics
from descriptor.schema import TraceEvent
from MAS import OpenRouterLLMClient, load_experiment_config

from .models import ADASConfig, ADASSolution
from .prompts import BASELINE_SOLUTIONS, build_reflexion_prompts, build_search_prompt
from .runtime_runner import ADASRuntimeRunner

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

    summary: dict[str, Any] = {
        "method": "adas_meta_agent_search",
        "upstream": {
            "repo": "https://github.com/ShengranHu/ADAS",
            "commit": "2702bee8fefda42255efc5be9f60e3bd3db96ae4",
            "license": "Apache-2.0",
            "adaptation": (
                "Meta Agent Search archive/reflexion/debug loop adapted to MAS_Analyzer "
                "benchmarks, OpenRouter, benchmark-owned tools/judges, and trace artifacts."
            ),
        },
        "run_id": run_id,
        "settings": vars(args),
        "benchmarks": {},
    }

    for benchmark_name in args.benchmark:
        print(f"[{_now_stamp()}] ADAS_BENCH_START benchmark={benchmark_name}", flush=True)
        try:
            payload = _run_one_benchmark(
                benchmark_name=benchmark_name,
                args=args,
                llm_client=llm_client,
                output_dir=output_root / benchmark_name,
            )
            summary["benchmarks"][benchmark_name] = payload
            print(
                f"[{_now_stamp()}] ADAS_BENCH_DONE benchmark={benchmark_name} "
                f"score={payload.get('score')}",
                flush=True,
            )
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            summary["benchmarks"][benchmark_name] = payload
            _write_json(output_root / benchmark_name / "error.json", payload)
            print(
                f"[{_now_stamp()}] ADAS_BENCH_ERROR benchmark={benchmark_name} "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )
            if not args.keep_going:
                raise

    _write_json(output_root / "summary.json", summary)
    print(f"[{_now_stamp()}] ADAS_RUN_DONE output={output_root}", flush=True)


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

    if args.search:
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
        solution = ADASSolution.from_payload(search_payload["best_solution"])
    else:
        solution = ADASSolution.from_payload(BASELINE_SOLUTIONS[0].to_payload())
        search_payload = None

    _write_json(output_dir / "adas_solution.json", solution.to_payload())
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

    runner = ADASRuntimeRunner(
        solution=solution,
        llm_client=llm_client,
        config=ADASConfig(
            model_agent_type=args.model_agent_type,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens) if int(args.max_tokens) > 0 else None,
            max_tool_iterations=max(1, int(args.max_tool_iterations or 8)),
        ),
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
    payload = _result_payload(
        benchmark_name=benchmark_name,
        final_tasks=final_tasks,
        run_payloads=run_payloads,
        search_payload=search_payload,
    )
    _write_json(output_dir / "results.json", payload)
    return payload


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
    archive_path = output_dir / "archive.json"
    if args.resume and search_results_path.exists():
        return json.loads(search_results_path.read_text(encoding="utf-8"))
    if args.resume and archive_path.exists():
        archive = [
            ADASSolution.from_payload(item)
            for item in json.loads(archive_path.read_text(encoding="utf-8"))
        ]
    else:
        archive = [ADASSolution.from_payload(item.to_payload()) for item in BASELINE_SOLUTIONS]

    evaluated: list[dict[str, Any]] = []
    for idx, solution in enumerate(archive):
        if solution.fitness is not None:
            continue
        print(
            f"[{_now_stamp()}] ADAS_SEARCH_INITIAL benchmark={benchmark_name} "
            f"index={idx} name={solution.name}",
            flush=True,
        )
        case = _evaluate_solution(
            benchmark=benchmark,
            benchmark_name=benchmark_name,
            args=args,
            llm_client=llm_client,
            validation_tasks=validation_tasks,
            solution=solution,
            output_dir=output_dir / "initial" / f"solution_{idx:03d}",
        )
        solution.fitness = float(case["score"])
        solution.validation_scores = [float(x) for x in case.get("scores", [])]
        solution.metadata["validation_result"] = case
        evaluated.append(case)
        _write_json(archive_path, [item.to_payload() for item in archive])

    for generation in range(max(0, int(args.search_generations))):
        candidate_path = output_dir / f"generation_{generation:03d}" / "candidate.json"
        result_path = output_dir / f"generation_{generation:03d}" / "result.json"
        if args.resume and result_path.exists():
            candidate = ADASSolution.from_payload(json.loads(candidate_path.read_text(encoding="utf-8")))
            case = json.loads(result_path.read_text(encoding="utf-8"))
            if all(candidate.name != item.name or candidate.code != item.code for item in archive):
                candidate.fitness = float(case["score"])
                candidate.validation_scores = [float(x) for x in case.get("scores", [])]
                candidate.metadata["validation_result"] = case
                archive.append(candidate)
            continue
        print(
            f"[{_now_stamp()}] ADAS_SEARCH_GENERATION benchmark={benchmark_name} "
            f"generation={generation}",
            flush=True,
        )
        candidate = _propose_solution(
            archive=archive,
            llm_client=llm_client,
            args=args,
            benchmark_name=benchmark_name,
            generation=generation,
            benchmark_description=_benchmark_description(benchmark_name),
            output_dir=output_dir / f"generation_{generation:03d}",
        )
        candidate.generation = generation
        _write_json(candidate_path, candidate.to_payload())
        case: dict[str, Any] | None = None
        for debug_attempt in range(max(1, int(args.debug_max))):
            try:
                case = _evaluate_solution(
                    benchmark=benchmark,
                    benchmark_name=benchmark_name,
                    args=args,
                    llm_client=llm_client,
                    validation_tasks=validation_tasks,
                    solution=candidate,
                    output_dir=output_dir
                    / f"generation_{generation:03d}"
                    / f"debug_{debug_attempt:03d}",
                )
                if float(case["score"]) <= 0.0 and args.reject_all_zero:
                    raise RuntimeError("All 0 validation accuracy")
                break
            except Exception as exc:
                _write_json(
                    output_dir / f"generation_{generation:03d}" / f"debug_{debug_attempt:03d}" / "error.json",
                    {"error": f"{type(exc).__name__}: {exc}", "candidate": candidate.to_payload()},
                )
                candidate = _debug_solution(
                    archive=archive,
                    candidate=candidate,
                    error=f"{type(exc).__name__}: {exc}",
                    llm_client=llm_client,
                    args=args,
                    benchmark_name=benchmark_name,
                    generation=generation,
                    debug_attempt=debug_attempt,
                    output_dir=output_dir / f"generation_{generation:03d}",
                )
                _write_json(candidate_path, candidate.to_payload())
        if case is None:
            continue
        candidate.fitness = float(case["score"])
        candidate.validation_scores = [float(x) for x in case.get("scores", [])]
        candidate.metadata["validation_result"] = case
        archive.append(candidate)
        evaluated.append(case)
        _write_json(result_path, case)
        _write_json(archive_path, [item.to_payload() for item in archive])

    scored = [solution for solution in archive if solution.fitness is not None]
    if not scored:
        raise RuntimeError("ADAS search did not evaluate any candidate")
    best = max(scored, key=lambda item: float(item.fitness or 0.0))
    payload = {
        "search_type": "adas_meta_agent_search",
        "note": (
            "Adapted from ShengranHu/ADAS Meta Agent Search: initialized archive, "
            "meta-agent code proposal, two reflexion prompts, validation evaluation, "
            "debug refinement, archive update, and best-agent selection."
        ),
        "archive": [solution.to_payload() for solution in archive],
        "best_solution": best.to_payload(),
        "best_score": best.fitness,
        "evaluated_cases": evaluated,
    }
    _write_json(search_results_path, payload)
    return payload


def _propose_solution(
    *,
    archive: list[ADASSolution],
    llm_client: OpenRouterLLMClient,
    args: argparse.Namespace,
    benchmark_name: str,
    generation: int,
    benchmark_description: str,
    output_dir: Path,
) -> ADASSolution:
    system_prompt, user_prompt = build_search_prompt(
        archive=archive,
        benchmark_name=benchmark_name,
        benchmark_description=benchmark_description,
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    payload = _json_llm_with_retry(
        llm_client=llm_client,
        messages=messages,
        args=args,
        task_id=f"adas_propose:{benchmark_name}:generation_{generation}",
        run_index=generation,
        agent_id="adas_meta_agent",
        temperature=float(args.meta_temperature),
        max_tokens=int(args.meta_max_tokens),
        output_dir=output_dir / "meta_propose",
        validator=lambda payload: _validate_solution_code(
            _solution_from_llm_payload(payload, generation=generation)
        ),
    )
    prompt1, prompt2 = build_reflexion_prompts(archive[-1] if archive else None)
    messages.append({"role": "assistant", "content": json.dumps(payload)})
    messages.append({"role": "user", "content": prompt1})
    payload = _json_llm_with_retry(
        llm_client=llm_client,
        messages=messages,
        args=args,
        task_id=f"adas_reflexion1:{benchmark_name}:generation_{generation}",
        run_index=generation,
        agent_id="adas_meta_agent_reflexion_1",
        temperature=float(args.meta_temperature),
        max_tokens=int(args.meta_max_tokens),
        output_dir=output_dir / "meta_reflexion_1",
        validator=lambda payload: _validate_solution_code(
            _solution_from_llm_payload(payload, generation=generation)
        ),
    )
    messages.append({"role": "assistant", "content": json.dumps(payload)})
    messages.append({"role": "user", "content": prompt2})
    payload = _json_llm_with_retry(
        llm_client=llm_client,
        messages=messages,
        args=args,
        task_id=f"adas_reflexion2:{benchmark_name}:generation_{generation}",
        run_index=generation,
        agent_id="adas_meta_agent_reflexion_2",
        temperature=float(args.meta_temperature),
        max_tokens=int(args.meta_max_tokens),
        output_dir=output_dir / "meta_reflexion_2",
        validator=lambda payload: _validate_solution_code(
            _solution_from_llm_payload(payload, generation=generation)
        ),
    )
    solution = _solution_from_llm_payload(payload, generation=generation)
    _validate_solution_code(solution)
    return solution


def _debug_solution(
    *,
    archive: list[ADASSolution],
    candidate: ADASSolution,
    error: str,
    llm_client: OpenRouterLLMClient,
    args: argparse.Namespace,
    benchmark_name: str,
    generation: int,
    debug_attempt: int,
    output_dir: Path,
) -> ADASSolution:
    messages = [
        {
            "role": "system",
            "content": "You are debugging generated ADAS agent code. Return JSON.",
        },
        {
            "role": "user",
            "content": (
                "Archive summary:\n"
                + json.dumps([s.to_payload() for s in archive[-5:]], indent=2)
                + "\n\nCurrent candidate:\n"
                + json.dumps(candidate.to_payload(), indent=2)
                + f"\n\nError during validation:\n{error}\n\n"
                "Carefully debug the implementation while preserving the same high-level thought. "
                "Reply exactly as JSON with keys `thought`, `debug_thought`, `name`, and `code`."
            ),
        },
    ]
    payload = _json_llm_with_retry(
        llm_client=llm_client,
        messages=messages,
        args=args,
        task_id=f"adas_debug:{benchmark_name}:generation_{generation}:attempt_{debug_attempt}",
        run_index=generation,
        agent_id="adas_meta_agent_debug",
        temperature=float(args.meta_temperature),
        max_tokens=int(args.meta_max_tokens),
        output_dir=output_dir / f"debug_{debug_attempt:03d}" / "meta_debug",
        validator=lambda payload: _validate_solution_code(
            _solution_from_llm_payload(payload, generation=generation)
        ),
    )
    solution = _solution_from_llm_payload(payload, generation=generation)
    _validate_solution_code(solution)
    solution.metadata["debug_error"] = error
    solution.metadata["debug_attempt"] = debug_attempt
    if payload.get("debug_thought"):
        solution.metadata["debug_thought"] = str(payload.get("debug_thought"))
    return solution


def _json_llm_with_retry(
    *,
    llm_client: OpenRouterLLMClient,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    task_id: str,
    run_index: int,
    agent_id: str,
    temperature: float,
    max_tokens: int,
    output_dir: Path,
    validator: Any | None = None,
) -> dict[str, Any]:
    attempts = max(1, int(getattr(args, "meta_retry_attempts", 1) or 1))
    working_messages = [dict(message) for message in messages]
    last_error = ""
    for attempt in range(attempts):
        try:
            payload = _json_llm(
                llm_client=llm_client,
                messages=working_messages,
                args=args,
                task_id=f"{task_id}:attempt_{attempt}",
                run_index=run_index,
                agent_id=f"{agent_id}_attempt_{attempt}",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if validator is not None:
                validator(payload)
            _write_json(output_dir / f"attempt_{attempt:02d}.json", {"payload": payload})
            return payload
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _write_json(
                output_dir / f"attempt_{attempt:02d}_error.json",
                {"attempt": attempt, "error": last_error},
            )
            if attempt >= attempts - 1:
                break
            working_messages = [
                *working_messages,
                {
                    "role": "user",
                    "content": (
                        "Your previous response could not be used.\n"
                        f"Error: {last_error}\n\n"
                        "Retry now. Return exactly one valid JSON object with the required keys. "
                        "The `code` field must contain a complete `def forward(self, taskInfo):` "
                        "function and no markdown fences."
                    ),
                },
            ]
    raise RuntimeError(f"ADAS meta-agent failed after {attempts} attempts: {last_error}")


def _json_llm(
    *,
    llm_client: OpenRouterLLMClient,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    task_id: str,
    run_index: int,
    agent_id: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    result = llm_client.generate(
        prompt=messages,
        agent_type=args.model_agent_type,
        task_id=task_id,
        run_index=run_index,
        agent_id=agent_id,
        tools=[],
        max_tool_iterations=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _parse_json_object(result.text)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(str(text or "").strip())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    import re

    match = re.search(r"\{.*\}", str(text or ""), flags=re.DOTALL)
    if match:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    raise ValueError("LLM response did not contain a JSON object")


def _solution_from_llm_payload(payload: dict[str, Any], *, generation: int | str) -> ADASSolution:
    code = str(payload.get("code") or "")
    if "def forward" not in code:
        raise ValueError("Generated solution missing `def forward`")
    return ADASSolution(
        name=str(payload.get("name") or "Unnamed ADAS Agent"),
        thought=str(payload.get("thought") or payload.get("reflection") or ""),
        code=code,
        generation=generation,
        metadata={k: v for k, v in payload.items() if k not in {"name", "thought", "code"}},
    )


def _validate_solution_code(solution: ADASSolution) -> None:
    code = solution.code.strip()
    if not code:
        raise ValueError("Generated solution code is empty")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Generated solution has invalid Python syntax: {exc}") from exc

    forward_defs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    if not forward_defs:
        raise ValueError("Generated solution must define a top-level `forward` function")
    if len(forward_defs) > 1:
        raise ValueError("Generated solution must define exactly one top-level `forward` function")

    forward = forward_defs[0]
    if len(forward.args.args) < 2:
        raise ValueError("Generated `forward` must accept runtime/self and task_info arguments")

    forbidden_imports = {
        "asyncio",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "threading",
    }
    forbidden_calls = {
        "__import__",
        "compile",
        "eval",
        "exec",
        "globals",
        "locals",
        "open",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name.split(".", 1)[0] for alias in node.names}
            blocked = sorted(names & forbidden_imports)
            if blocked:
                raise ValueError(f"Generated solution imports forbidden module(s): {blocked}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module in forbidden_imports:
                raise ValueError(f"Generated solution imports forbidden module: {module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_calls:
                raise ValueError(f"Generated solution calls forbidden function: {node.func.id}")

    compile(tree, "<adas_generated_solution>", "exec")


def _evaluate_solution(
    *,
    benchmark: Any,
    benchmark_name: str,
    args: argparse.Namespace,
    llm_client: OpenRouterLLMClient,
    validation_tasks: list[Any],
    solution: ADASSolution,
    output_dir: Path,
) -> dict[str, Any]:
    runner = ADASRuntimeRunner(
        solution=solution,
        llm_client=llm_client,
        config=ADASConfig(
            model_agent_type=args.model_agent_type,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens) if int(args.max_tokens) > 0 else None,
            max_tool_iterations=max(1, int(args.max_tool_iterations or 8)),
        ),
    )
    _write_json(output_dir / "solution.json", solution.to_payload())
    jobs = [
        (task, run_index)
        for task in validation_tasks
        for run_index in range(max(1, int(args.validation_repeats)))
    ]
    if int(args.workers) <= 1:
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
                    output_dir=output_dir,
                    max_tool_iterations=args.max_tool_iterations,
                    resume=args.resume,
                )
                for task, run_index in jobs
            ]
            run_payloads = [future.result() for future in concurrent.futures.as_completed(futures)]
    scores = [float(item["score"]) for item in run_payloads]
    case = {
        "solution_name": solution.name,
        "generation": solution.generation,
        "score": sum(scores) / len(scores) if scores else 0.0,
        "scores": scores,
        "task_count": len(validation_tasks),
        "run_count": len(run_payloads),
        "result_path": str((output_dir / "runs").resolve()),
    }
    _write_json(output_dir / "result.json", case)
    print(
        f"[{_now_stamp()}] ADAS_SEARCH_CANDIDATE benchmark={benchmark_name} "
        f"name={solution.name!r} generation={solution.generation} score={case['score']:.4f}",
        flush=True,
    )
    return case


def _execute_one(
    *,
    benchmark: Any,
    runner: ADASRuntimeRunner,
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
    evaluation = benchmark.evaluate(task, result.final_answer, run_metadata=result.run_metadata)
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
        f"[{_now_stamp()}] ADAS_RUN_SAVED benchmark={benchmark_name} "
        f"task_id={task.task_id} run_index={run_index} score={evaluation.score}",
        flush=True,
    )
    return payload


def _result_payload(
    *,
    benchmark_name: str,
    final_tasks: list[Any],
    run_payloads: list[dict[str, Any]],
    search_payload: dict[str, Any] | None,
) -> dict[str, Any]:
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


def _benchmark_description(benchmark_name: str) -> str:
    return {
        "browsecomp": "Search-and-answer benchmark requiring evidence gathering with search tools.",
        "stabletoolbench": "Tool/API-use benchmark requiring valid tool calls and response grounding.",
        "plancraft": "Planning benchmark requiring valid next actions in a constrained environment.",
        "workbench": "Workflow/tool side-effect benchmark requiring correct use of provided tools.",
        "math500": "Mathematical reasoning benchmark requiring exact final answers.",
    }.get(benchmark_name, "General agentic task-solving benchmark.")


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
    parser = argparse.ArgumentParser(description="Run ADAS Meta Agent Search on repo benchmarks.")
    parser.add_argument("--config", default="config/reproduce_agentsquare.example.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="outputs_adas_reproduce")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--task-limit", type=int, default=1)
    parser.add_argument("--validation-task-limit", type=int, default=0)
    parser.add_argument("--final-task-limit", type=int, default=None)
    parser.add_argument("--final-task-offset", type=int, default=0)
    parser.add_argument("--runs-per-task", type=int, default=1)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--search-generations", type=int, default=3)
    parser.add_argument("--debug-max", type=int, default=3)
    parser.add_argument("--reject-all-zero", action="store_true")
    parser.add_argument("--validation-repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-agent-type", default="default")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--meta-temperature", type=float, default=0.8)
    parser.add_argument("--meta-max-tokens", type=int, default=4096)
    parser.add_argument("--meta-retry-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tool-iterations", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
