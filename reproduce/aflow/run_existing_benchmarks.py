from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import get_benchmark, list_benchmarks
from MAS import OpenRouterLLMClient, load_experiment_config

try:
    from datetime import UTC
except ImportError:  # pragma: no cover
    UTC = UTC


DEFAULT_EXCLUDED_BENCHMARKS = {"finance_agent"}


@dataclass(frozen=True)
class WorkflowCandidate:
    name: str
    operators: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "operators": list(self.operators)}


DEFAULT_WORKFLOWS = (
    WorkflowCandidate("generate", ("Generate",)),
    WorkflowCandidate("review", ("Generate", "Review")),
    WorkflowCandidate("test_review", ("Generate", "Test", "Review")),
    WorkflowCandidate("ensemble", ("Generate", "Generate", "Generate", "Ensemble")),
)


def main() -> None:
    args = _parse_args()
    config = load_experiment_config(args.config)
    llm_client = OpenRouterLLMClient(config.openrouter, config.models)
    benchmarks = _resolve_benchmarks(args)
    run_id = args.run_id or _now_stamp()
    output_root = Path(args.output_dir).expanduser().resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "method": "aflow_style",
        "benchmarks": {},
        "excluded_benchmarks": sorted(DEFAULT_EXCLUDED_BENCHMARKS | set(args.exclude_benchmark)),
        "settings": vars(args),
    }

    for benchmark_name in benchmarks:
        print(f"[{_now_stamp()}] AFLOW_BENCH_START benchmark={benchmark_name}", flush=True)
        bench_dir = output_root / benchmark_name
        bench_dir.mkdir(parents=True, exist_ok=True)
        try:
            payload = _run_benchmark(
                benchmark_name=benchmark_name,
                config=config,
                llm_client=llm_client,
                args=args,
                output_dir=bench_dir,
            )
            summary["benchmarks"][benchmark_name] = payload
            print(
                f"[{_now_stamp()}] AFLOW_BENCH_DONE benchmark={benchmark_name} "
                f"best_score={payload['best_score']}",
                flush=True,
            )
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            summary["benchmarks"][benchmark_name] = payload
            _write_json(bench_dir / "error.json", payload)
            print(
                f"[{_now_stamp()}] AFLOW_BENCH_ERROR benchmark={benchmark_name} "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )
            if not args.keep_going:
                raise

    _write_json(output_root / "summary.json", summary)
    print(f"[{_now_stamp()}] AFLOW_RUN_DONE output={output_root}", flush=True)


def _run_benchmark(
    *,
    benchmark_name: str,
    config: Any,
    llm_client: OpenRouterLLMClient,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark = get_benchmark(
        benchmark_name, config=_benchmark_section_config(config, benchmark_name)
    )
    tasks = list(benchmark.load_tasks(task_limit=args.task_limit))
    if not tasks:
        raise RuntimeError(f"No tasks loaded for benchmark '{benchmark_name}'")

    workflows = list(DEFAULT_WORKFLOWS[: max(1, args.sample)])
    evaluated: list[dict[str, Any]] = []
    for round_index in range(max(1, args.max_rounds)):
        for workflow in workflows:
            scores = []
            task_payloads = []
            for task in tasks[: max(1, args.validation_rounds)]:
                prediction, trace = _execute_workflow(
                    llm_client=llm_client,
                    workflow=workflow,
                    task_prompt=task.prompt,
                    task_id=f"{benchmark_name}:{task.task_id}:{workflow.name}",
                    model_agent_type=args.model_agent_type,
                    temperature=args.temperature,
                )
                evaluation = benchmark.evaluate(
                    task,
                    prediction,
                    run_metadata={
                        "aflow_reproduce": True,
                        "workflow": workflow.to_payload(),
                        "trace": trace,
                    },
                )
                scores.append(float(evaluation.score))
                task_payloads.append(
                    {
                        "task_id": str(task.task_id),
                        "prediction": prediction,
                        "score": float(evaluation.score),
                        "success": bool(evaluation.success),
                        "trace": trace,
                    }
                )
            avg_score = sum(scores) / len(scores) if scores else 0.0
            evaluated.append(
                {
                    "round": round_index + 1,
                    "workflow": workflow.to_payload(),
                    "score": avg_score,
                    "tasks": task_payloads,
                }
            )

    best = max(evaluated, key=lambda item: float(item["score"]))
    payload = {
        "benchmark": benchmark_name,
        "task_count": len(tasks),
        "best_score": float(best["score"]),
        "best_workflow": best["workflow"],
        "evaluated": evaluated,
    }
    _write_json(output_dir / "aflow_results.json", payload)
    return payload


def _execute_workflow(
    *,
    llm_client: OpenRouterLLMClient,
    workflow: WorkflowCandidate,
    task_prompt: Any,
    task_id: str,
    model_agent_type: str,
    temperature: float,
) -> tuple[str, list[dict[str, Any]]]:
    drafts: list[str] = []
    feedback = ""
    trace: list[dict[str, Any]] = []
    for idx, operator in enumerate(workflow.operators):
        prompt = _operator_prompt(operator, task_prompt, drafts=drafts, feedback=feedback)
        result = llm_client.generate(
            prompt=[
                {
                    "role": "system",
                    "content": "You are executing an AFlow-style workflow operator. Return only the operator output.",
                },
                {"role": "user", "content": prompt},
            ],
            agent_type=model_agent_type,
            task_id=task_id,
            run_index=0,
            agent_id=f"{operator.lower()}_{idx}",
            temperature=temperature,
        )
        trace.append(
            {
                "operator": operator,
                "text": result.text,
                "model": result.model,
                "token_in": result.token_in,
                "token_out": result.token_out,
                "mock_used": result.mock_used,
            }
        )
        if operator == "Generate":
            drafts.append(result.text)
        elif operator == "Review":
            drafts = [result.text]
        elif operator == "Test":
            feedback = result.text
        elif operator == "Ensemble":
            drafts = [result.text]
    return (drafts[-1] if drafts else trace[-1]["text"]), trace


def _operator_prompt(operator: str, task_prompt: Any, *, drafts: list[str], feedback: str) -> str:
    if operator == "Generate":
        return f"Solve the task.\n\nTask:\n{task_prompt}"
    if operator == "Review":
        return (
            "Review and revise the current draft. Return the final answer.\n\n"
            f"Task:\n{task_prompt}\n\nDrafts:\n{_join(drafts)}\n\nFeedback:\n{feedback}"
        )
    if operator == "Test":
        return (
            "Check the draft for correctness and produce concise feedback.\n\n"
            f"Task:\n{task_prompt}\n\nDrafts:\n{_join(drafts)}"
        )
    if operator == "Ensemble":
        return (
            "Select or synthesize the best final answer from these candidates.\n\n"
            f"Task:\n{task_prompt}\n\nCandidates:\n{_join(drafts)}"
        )
    return f"{operator}\n\nTask:\n{task_prompt}\n\nDrafts:\n{_join(drafts)}"


def _join(items: list[str]) -> str:
    return "\n\n".join(f"[{idx}] {item}" for idx, item in enumerate(items)) or "(none)"


def _benchmark_section_config(config: Any, benchmark_name: str) -> dict[str, Any]:
    cfg = dict(getattr(config, benchmark_name, {}) or {})
    cfg.setdefault("openrouter", {})
    if config.openrouter.api_key and "api_key" not in cfg["openrouter"]:
        cfg["openrouter"]["api_key"] = config.openrouter.api_key
    if config.openrouter.base_url and "base_url" not in cfg["openrouter"]:
        cfg["openrouter"]["base_url"] = config.openrouter.base_url
    return cfg


def _resolve_benchmarks(args: argparse.Namespace) -> list[str]:
    excluded = DEFAULT_EXCLUDED_BENCHMARKS | set(args.exclude_benchmark)
    requested = args.benchmark or list_benchmarks()
    return [name for name in requested if name not in excluded]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AFlow-style workflows on existing benchmarks."
    )
    parser.add_argument("--config", default="config/experiment.example.toml")
    parser.add_argument("--output-dir", default="outputs_aflow_reproduce")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--exclude-benchmark", action="append", default=[])
    parser.add_argument("--task-limit", type=int, default=1)
    parser.add_argument("--sample", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--validation-rounds", type=int, default=1)
    parser.add_argument("--model-agent-type", default="default")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
