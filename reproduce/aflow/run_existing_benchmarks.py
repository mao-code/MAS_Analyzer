from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmark import list_benchmarks
from MAS import OpenRouterLLMClient, load_experiment_config
from reproduce.aflow.official.optimizer import OfficialAFlowBenchmarkOptimizer

try:
    from datetime import UTC
except ImportError:  # pragma: no cover
    UTC = UTC


DEFAULT_EXCLUDED_BENCHMARKS = {"finance_agent"}
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    args = _parse_args()
    _load_reproduce_env(args.env_file)
    _configure_live_mode(args)
    config = load_experiment_config(args.config)
    _validate_live_config(config, args)
    llm_client = OpenRouterLLMClient(config.openrouter, config.models)
    benchmarks = _resolve_benchmarks(args)
    run_id = args.run_id or _now_stamp()
    output_root = Path(args.output_dir).expanduser().resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "method": "aflow_official_adapter",
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
    benchmark_config = _benchmark_section_config(config, benchmark_name)
    _prepare_benchmark_environment(benchmark_name, benchmark_config)
    optimizer = OfficialAFlowBenchmarkOptimizer(
        benchmark_name=benchmark_name,
        benchmark_config=benchmark_config,
        llm_client=llm_client,
        output_dir=output_dir,
        task_limit=max(1, int(args.task_limit)),
        validation_rounds=max(1, int(args.validation_rounds)),
        test_task_limit=max(1, int(args.test_task_limit or args.task_limit)),
        test_offset=max(
            0, int(args.test_offset if args.test_offset is not None else args.validation_rounds)
        ),
        runs_per_task=max(1, int(args.runs_per_task)),
        retries=max(0, int(args.retries)),
        max_rounds=max(1, int(args.max_rounds)),
        sample=max(1, int(args.sample)),
        seed=int(args.seed),
        model_agent_type=str(args.model_agent_type),
        temperature=float(args.temperature),
        allow_mock=bool(args.allow_mock),
    )
    return optimizer.optimize()


def _benchmark_section_config(config: Any, benchmark_name: str) -> dict[str, Any]:
    cfg = dict(getattr(config, benchmark_name, {}) or {})
    cfg.setdefault("openrouter", {})
    if config.openrouter.api_key and "api_key" not in cfg["openrouter"]:
        cfg["openrouter"]["api_key"] = config.openrouter.api_key
    if config.openrouter.base_url and "base_url" not in cfg["openrouter"]:
        cfg["openrouter"]["base_url"] = config.openrouter.base_url
    return cfg


def _prepare_benchmark_environment(benchmark_name: str, benchmark_config: dict[str, Any]) -> None:
    if benchmark_name != "stabletoolbench":
        return
    from scripts.full_experiment import prepare_stabletoolbench

    prepare_stabletoolbench(dict(benchmark_config))


def _load_reproduce_env(env_file: str | None) -> None:
    candidates: list[Path] = []
    if env_file:
        candidates.append(Path(env_file).expanduser())
    candidates.extend(
        [
            ROOT / ".env",
            ROOT.parent / "MAS_Analyzer" / ".env",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            print(f"[{_now_stamp()}] AFLOW_ENV_LOADED path={candidate}", flush=True)
            return
    if env_file:
        raise FileNotFoundError(f"Requested env file not found: {env_file}")


def _configure_live_mode(args: argparse.Namespace) -> None:
    if args.allow_mock:
        return
    if _env_flag("MAS_DISABLE_LIVE_LLM"):
        raise RuntimeError(
            "Live OpenRouter reproduce is required, but MAS_DISABLE_LIVE_LLM is set. "
            "Unset it or pass --allow-mock for plumbing-only smoke tests."
        )
    os.environ.setdefault("MAS_REQUIRE_LIVE_LLM", "1")


def _validate_live_config(config: Any, args: argparse.Namespace) -> None:
    if args.allow_mock:
        return
    if not getattr(config.openrouter, "api_key", None):
        raise RuntimeError(
            "Live OpenRouter reproduce is required, but OPENROUTER_API_KEY is missing. "
            "Export it, put it in .env, or pass --env-file."
        )


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_benchmarks(args: argparse.Namespace) -> list[str]:
    excluded = set(args.exclude_benchmark)
    if args.benchmark:
        requested = args.benchmark
    else:
        excluded |= DEFAULT_EXCLUDED_BENCHMARKS
        requested = list_benchmarks()
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
    parser.add_argument(
        "--test-task-limit",
        type=int,
        default=None,
        help="Number of held-out tasks to run with the best workflow. Defaults to --task-limit.",
    )
    parser.add_argument(
        "--test-offset",
        type=int,
        default=None,
        help="Offset for held-out test tasks. Defaults to --validation-rounds to avoid overlap.",
    )
    parser.add_argument("--sample", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--validation-rounds", type=int, default=1)
    parser.add_argument("--model-agent-type", default="default")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--runs-per-task", type=int, default=1)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Per task/run retry count after the first attempt.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-file", default=None)
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow deterministic mock fallback. Use only for plumbing smoke tests.",
    )
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
