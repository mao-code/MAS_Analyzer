from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from MAS import OpenRouterLLMClient, load_experiment_config
from reproduce.aflow.official.optimizer import OfficialAFlowBenchmarkOptimizer
from reproduce.aflow.run_existing_benchmarks import (
    ROOT,
    _benchmark_section_config,
    _configure_live_mode,
    _prepare_benchmark_environment,
    _validate_live_config,
)

try:
    from datetime import UTC
except ImportError:  # pragma: no cover
    UTC = None  # type: ignore[assignment]


def main() -> None:
    args = _parse_args()
    load_dotenv(ROOT / ".env")
    if args.env_file:
        load_dotenv(args.env_file, override=True)
    _configure_live_mode(args)
    config = load_experiment_config(args.config)
    _validate_live_config(config, args)

    source = Path(args.workflow_dir).expanduser().resolve()
    if not (source / "graph.py").exists() or not (source / "prompt.py").exists():
        raise FileNotFoundError(f"workflow_dir must contain graph.py and prompt.py: {source}")

    run_id = args.run_id or f"aflow_fixed_workflow_{_now_stamp()}"
    output_dir = Path(args.output_dir).expanduser().resolve() / run_id / args.benchmark
    round_dir = output_dir / "workflows" / f"round_{args.round_number}"
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "__init__.py").write_text("", encoding="utf-8")
    (round_dir / "graph.py").write_text((source / "graph.py").read_text(encoding="utf-8"), encoding="utf-8")
    (round_dir / "prompt.py").write_text((source / "prompt.py").read_text(encoding="utf-8"), encoding="utf-8")

    benchmark_config = _benchmark_section_config(config, args.benchmark)
    if args.max_tool_iterations is not None:
        benchmark_config["max_tool_iterations"] = max(1, int(args.max_tool_iterations))
    _prepare_benchmark_environment(args.benchmark, benchmark_config)
    optimizer = OfficialAFlowBenchmarkOptimizer(
        benchmark_name=args.benchmark,
        benchmark_config=benchmark_config,
        llm_client=OpenRouterLLMClient(config.openrouter, config.models),
        output_dir=output_dir,
        task_limit=max(1, int(args.task_limit)),
        validation_rounds=1,
        test_task_limit=max(1, int(args.test_task_limit)),
        test_offset=max(0, int(args.test_offset)),
        runs_per_task=max(1, int(args.runs_per_task)),
        workers=max(1, int(args.workers)),
        retries=max(0, int(args.retries)),
        max_rounds=1,
        sample=1,
        seed=int(args.seed),
        model_agent_type=str(args.model_agent_type),
        temperature=float(args.temperature),
        allow_mock=bool(args.allow_mock),
    )
    print(
        f"[{_now_stamp()}] AFLOW_FIXED_TEST_START benchmark={args.benchmark} "
        f"workflow={source} round={args.round_number}",
        flush=True,
    )
    test_payload = optimizer._evaluate_round(args.round_number, initial=False, split="test")
    payload: dict[str, Any] = {
        "method": "aflow_fixed_best_workflow",
        "benchmark": args.benchmark,
        "workflow_dir": str(source),
        "round": args.round_number,
        "test_score": test_payload["score"],
        "test": test_payload,
        "settings": vars(args),
        "time": _now_stamp(),
    }
    (output_dir / "aflow_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    summary = {
        "run_id": run_id,
        "benchmarks": {args.benchmark: payload},
        "settings": vars(args),
    }
    (output_dir.parent / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[{_now_stamp()}] AFLOW_FIXED_TEST_DONE benchmark={args.benchmark} "
        f"score={test_payload['score']} output={output_dir.parent}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fixed AFlow best_workflow on test only.")
    parser.add_argument("--config", default="config/reproduce_gemma/baseline_gemma_30x3.toml")
    parser.add_argument("--benchmark", default="browsecomp")
    parser.add_argument("--workflow-dir", required=True)
    parser.add_argument("--output-dir", default="outputs_aflow_reproduce")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--round-number", type=int, default=1)
    parser.add_argument("--task-limit", type=int, default=40)
    parser.add_argument("--test-task-limit", type=int, default=30)
    parser.add_argument("--test-offset", type=int, default=10)
    parser.add_argument("--runs-per-task", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-tool-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-agent-type", default="default")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--allow-mock", action="store_true")
    return parser.parse_args()


def _now_stamp() -> str:
    if UTC is None:
        return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    main()
