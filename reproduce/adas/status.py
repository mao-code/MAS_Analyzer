from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_BENCHMARKS = (
    "browsecomp",
    "stabletoolbench",
    "plancraft",
    "workbench",
    "math500",
)


def main() -> None:
    args = _parse_args()
    payload = collect_status(
        run_root=Path(args.run_root).expanduser().resolve(),
        benchmarks=tuple(args.benchmark or DEFAULT_BENCHMARKS),
    )
    print(_format_status(payload))
    if args.json:
        print(json.dumps(payload, indent=2, default=str))


def collect_status(*, run_root: Path, benchmarks: tuple[str, ...]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        bench_root = run_root / benchmark
        split = _read_json(bench_root / "split.json")
        results = _read_json(bench_root / "results.json")
        search_results = _read_json(bench_root / "search" / "search_results.json")
        error = _read_json(bench_root / "error.json")
        active_error = None if isinstance(results, dict) else error
        final_run_files = sorted((bench_root / "final" / "runs").glob("*/run_*.json"))
        expected_runs = None
        if isinstance(split, dict):
            expected_runs = len(split.get("final_task_ids") or []) * _runs_per_task(results)
        rows[benchmark] = {
            "exists": bench_root.exists(),
            "search_done": isinstance(search_results, dict),
            "results_done": isinstance(results, dict),
            "error": active_error,
            "expected_runs": expected_runs,
            "completed_runs": len(final_run_files),
            "score": results.get("score") if isinstance(results, dict) else None,
            "best_solution": (
                (search_results.get("best_solution") or {}).get("name")
                if isinstance(search_results, dict)
                else None
            ),
        }
    top_summary = _read_json(run_root / "summary.json")
    table_summary = _read_json(run_root / "adas_summary.json")
    return {
        "run_root": str(run_root),
        "exists": run_root.exists(),
        "top_summary_done": isinstance(top_summary, dict),
        "table_summary_done": isinstance(table_summary, dict),
        "benchmarks": rows,
    }


def _runs_per_task(results: Any) -> int:
    if not isinstance(results, dict):
        return 0
    task_count = int(results.get("task_count") or 0)
    run_count = int(results.get("run_count") or 0)
    if task_count <= 0:
        return 0
    return max(1, run_count // task_count)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def _format_status(payload: dict[str, Any]) -> str:
    lines = [
        f"Run root: {payload['run_root']}",
        f"Exists: {payload['exists']}",
        f"Top summary: {payload['top_summary_done']} | Table summary: {payload['table_summary_done']}",
        "",
        "| Benchmark | Search | Results | Runs | Score | Best Solution | Error |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for benchmark, row in payload["benchmarks"].items():
        expected = row.get("expected_runs")
        completed = row.get("completed_runs")
        runs = f"{completed}/{expected}" if expected else str(completed)
        error = "--"
        if row.get("error"):
            error_payload = row["error"]
            error = str(
                error_payload.get("error") if isinstance(error_payload, dict) else error_payload
            )
        score = "--" if row.get("score") is None else f"{float(row['score']) * 100.0:.1f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    benchmark,
                    str(bool(row.get("search_done"))),
                    str(bool(row.get("results_done"))),
                    runs,
                    score,
                    str(row.get("best_solution") or "--"),
                    error,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show ADAS run progress from artifacts.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
