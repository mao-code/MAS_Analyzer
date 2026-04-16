from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import get_benchmark
from benchmark.base import BenchmarkTask
from MAS.config import load_experiment_config
from MAS.llm import OpenRouterLLMClient
from MAS.runner import MASRunner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run StableToolBench tasks with SAS (single-agent) and export a solvability table."
        )
    )
    parser.add_argument(
        "--config",
        default="config/benchmarks/stabletoolbench_10.toml",
        help="Path to TOML config.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Limit number of tasks loaded from benchmark.",
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="Comma-separated task_ids to run (optional).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Directory for outputs. Default: artifacts/stabletoolbench_sas_solvability/<timestamp>"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--default-model",
        default="",
        help="Override models.default from config.",
    )
    parser.add_argument(
        "--skip-missing-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip task execution when any required API cache file is missing.",
    )
    return parser.parse_args()


def _extract_event_value(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _extract_tool_error_count(trace_events: list[Any]) -> int:
    errors = 0
    for event in trace_events:
        event_type = str(_extract_event_value(event, "event_type", "") or "")
        if event_type != "tool_call_result":
            continue
        payload = _extract_event_value(event, "payload", {}) or {}
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status", "") or "").lower()
        output_text = str(payload.get("output", "") or "").lower()
        if status in {"error", "failed"} or '"error"' in output_text:
            errors += 1
    return errors


def _task_query(task: BenchmarkTask) -> str:
    raw = str(task.metadata.get("query", "") or "")
    if raw:
        return raw
    prompt = task.prompt or []
    if prompt and isinstance(prompt[0], dict):
        return str(prompt[0].get("content", "") or "")
    return ""


def _std(value: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_]", "_", str(value or ""))
    out = re.sub(r"_+", "_", out).strip("_").lower()
    if out and out[0].isdigit():
        out = f"get_{out}"
    return out


def _candidate_paths(cache_root: Path, category: str, tool: str, api: str) -> list[Path]:
    category_dir = cache_root / str(category or "").strip()
    tool_dir = category_dir / f"{_std(tool)}_for_{_std(category)}"
    api_filename = f"{_std(api)}.json"
    api_filename_raw = f"{str(api or '').strip()}.json"
    return [
        tool_dir / api_filename,
        tool_dir / api_filename_raw,
        category_dir / api_filename,
        category_dir / api_filename_raw,
        cache_root / api_filename,
        cache_root / api_filename_raw,
    ]


def _missing_cache_apis(benchmark: Any, task: BenchmarkTask) -> list[str]:
    cache_root = Path(
        getattr(benchmark, "tool_cache_dir", "benchmark/stabletoolbench/tool_response_cache")
    )
    api_list = list(getattr(benchmark, "_task_api_lists", {}).get(str(task.task_id), []))
    missing: list[str] = []
    for item in api_list:
        category = str(item.get("category_name", "") or "")
        tool = str(item.get("tool_name", "") or "")
        api = str(item.get("api_name", "") or "")
        paths = _candidate_paths(cache_root, category, tool, api)
        if not any(path.exists() for path in paths):
            missing.append(f"{category}/{tool}/{api}")
    return missing


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _selected_tasks(tasks: list[BenchmarkTask], task_ids_raw: str) -> list[BenchmarkTask]:
    if not task_ids_raw.strip():
        return tasks
    wanted = {part.strip() for part in task_ids_raw.split(",") if part.strip()}
    return [task for task in tasks if str(task.task_id) in wanted]


def main() -> int:
    args = _parse_args()
    cfg = load_experiment_config(args.config)

    # Force SAS runtime shape.
    cfg.mas.topology = "sas"
    cfg.mas.levels = 1
    cfg.mas.number_of_agents = 1
    cfg.mas.agents_per_level = [1]
    cfg.mas.group_sizes = None
    cfg.mas.agent_types = ["general"]
    cfg.mas.communication_count_internally = 0
    cfg.mas.discussion_rounds = 1
    cfg.mas.max_turns = 1

    if args.default_model.strip():
        cfg.models["default"] = args.default_model.strip()

    cfg.validate()

    benchmark_cfg = asdict(cfg).get("stabletoolbench", {})
    benchmark = get_benchmark("stabletoolbench", config=benchmark_cfg)

    llm_client = OpenRouterLLMClient(cfg.openrouter, cfg.models)
    runner = MASRunner(cfg, llm_client)

    tasks = list(benchmark.load_tasks(task_limit=args.task_limit))
    tasks = _selected_tasks(tasks, args.task_ids)
    if not tasks:
        raise RuntimeError("No tasks selected.")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir.strip()
        else Path("artifacts/stabletoolbench_sas_solvability").resolve() / _now_stamp()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        run_seed = int(args.seed) + idx
        started = datetime.now(UTC)
        runtime_error = ""
        final_answer = ""
        score = 0.0
        success = False
        reason = ""
        tool_errors = 0
        trace_events: list[Any] = []
        missing_cache = _missing_cache_apis(benchmark, task)
        cache_ready = len(missing_cache) == 0

        if args.skip_missing_cache and not cache_ready:
            runtime_error = "SKIPPED_MISSING_CACHE"
            reason = "Missing required API cache files."
        else:
            try:
                run = benchmark.run(task=task, runner=runner, run_index=0, seed=run_seed)
                trace_events = list(run.trace_events or [])
                tool_errors = _extract_tool_error_count(trace_events)
                final_answer = str(run.final_answer or "")
                ev = benchmark.evaluate(task, final_answer, run_metadata=run.run_metadata)
                score = float(ev.score)
                success = bool(ev.success)
                details = ev.details or {}
                reason = str(details.get("reason", "") or "")
            except Exception as exc:  # noqa: BLE001
                runtime_error = f"{type(exc).__name__}: {exc}"

        elapsed_s = (datetime.now(UTC) - started).total_seconds()
        row = {
            "task_id": str(task.task_id),
            "query_id": str(task.metadata.get("query_id", task.task_id)),
            "cache_ready": cache_ready,
            "missing_cache_count": len(missing_cache),
            "missing_cache_apis": " | ".join(missing_cache),
            "solvable_by_sas": bool(success and not runtime_error),
            "runtime_ok": not bool(runtime_error),
            "score": score,
            "success": success,
            "elapsed_s": round(elapsed_s, 3),
            "trace_events": len(trace_events),
            "tool_error_count": tool_errors,
            "runtime_error": runtime_error,
            "reason": reason,
            "query_preview": _task_query(task)[:200].replace("\n", " "),
            "final_answer_preview": final_answer[:200].replace("\n", " "),
        }
        rows.append(row)
        print(
            f"[{idx + 1}/{len(tasks)}] task_id={row['task_id']} "
            f"solvable={row['solvable_by_sas']} score={row['score']:.3f}"
        )

    csv_path = output_dir / "stabletoolbench_sas_solvability.csv"
    json_path = output_dir / "stabletoolbench_sas_solvability.json"
    summary_path = output_dir / "summary.json"

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(rows)
    runtime_ok = sum(1 for row in rows if row["runtime_ok"])
    solvable = sum(1 for row in rows if row["solvable_by_sas"])
    avg_score = (sum(float(row["score"]) for row in rows) / total) if total else 0.0
    summary = {
        "total_tasks": total,
        "runtime_ok_tasks": runtime_ok,
        "solvable_by_sas_tasks": solvable,
        "runtime_failure_tasks": total - runtime_ok,
        "avg_score": avg_score,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
