from __future__ import annotations

import argparse
import json
import statistics
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
    run_root = Path(args.run_root).expanduser().resolve()
    benchmarks = tuple(args.benchmark or DEFAULT_BENCHMARKS)
    payload = summarize_run(run_root=run_root, benchmarks=benchmarks)
    if args.output:
        _write_json(Path(args.output).expanduser(), payload)
    print(_format_markdown(payload))
    print()
    print(_format_latex_row(payload, system_name=args.system_name))


def summarize_run(*, run_root: Path, benchmarks: tuple[str, ...]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        result_path = run_root / benchmark / "results.json"
        if not result_path.exists():
            rows[benchmark] = {
                "available": False,
                "path": str(result_path),
                "mean": None,
                "std": None,
                "per_run_scores": {},
            }
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        per_run_scores = _per_run_scores(result)
        values = list(per_run_scores.values())
        mean = sum(values) / len(values) if values else float(result.get("score", 0.0))
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        rows[benchmark] = {
            "available": True,
            "path": str(result_path),
            "mean": mean,
            "std": std,
            "mean_pct": mean * 100.0,
            "std_pct": std * 100.0,
            "task_count": result.get("task_count"),
            "run_count": result.get("run_count"),
            "per_run_scores": per_run_scores,
            "best_solution": ((result.get("search") or {}).get("best_solution") or {}).get("name"),
        }
    return {
        "method": "ADAS",
        "run_root": str(run_root),
        "benchmarks": rows,
        "average": _average_summary(rows=rows, benchmarks=benchmarks),
    }


def _per_run_scores(result: dict[str, Any]) -> dict[str, float]:
    raw = result.get("per_run_scores")
    if isinstance(raw, dict) and raw:
        return {
            str(key): float(value)
            for key, value in sorted(raw.items(), key=lambda item: str(item[0]))
        }
    by_run: dict[str, list[float]] = {}
    for item in result.get("runs", []):
        if not isinstance(item, dict):
            continue
        run_index = str(item.get("run_index", "0"))
        by_run.setdefault(run_index, []).append(float(item.get("score", 0.0)))
    return {
        run_index: sum(values) / len(values)
        for run_index, values in sorted(by_run.items())
        if values
    }


def _average_summary(
    *, rows: dict[str, dict[str, Any]], benchmarks: tuple[str, ...]
) -> dict[str, Any]:
    available = [rows[name] for name in benchmarks if rows.get(name, {}).get("available")]
    if not available:
        return {"available": False, "mean": None, "std": None}
    means = [float(row["mean"]) for row in available]
    run_ids = sorted(
        {run_id for row in available for run_id in row.get("per_run_scores", {}).keys()}
    )
    per_run_average: dict[str, float] = {}
    for run_id in run_ids:
        values = [
            float(row["per_run_scores"][run_id])
            for row in available
            if run_id in row.get("per_run_scores", {})
        ]
        if values:
            per_run_average[run_id] = sum(values) / len(values)
    mean = sum(means) / len(means)
    std = statistics.pstdev(list(per_run_average.values())) if len(per_run_average) > 1 else 0.0
    return {
        "available": True,
        "mean": mean,
        "std": std,
        "mean_pct": mean * 100.0,
        "std_pct": std * 100.0,
        "per_run_scores": per_run_average,
        "benchmark_count": len(available),
    }


def _format_markdown(payload: dict[str, Any]) -> str:
    lines = ["| Benchmark | Mean ± Std | Runs | Best Solution |", "|---|---:|---:|---|"]
    for benchmark, row in payload["benchmarks"].items():
        if not row["available"]:
            lines.append(f"| {benchmark} | -- | -- | -- |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    benchmark,
                    _fmt_pm(row["mean_pct"], row["std_pct"]),
                    str(row.get("run_count") or "--"),
                    str(row.get("best_solution") or "--"),
                ]
            )
            + " |"
        )
    average = payload["average"]
    if average["available"]:
        lines.append(f"| Average | {_fmt_pm(average['mean_pct'], average['std_pct'])} | -- | -- |")
    return "\n".join(lines)


def _format_latex_row(payload: dict[str, Any], *, system_name: str) -> str:
    cells = [system_name]
    for row in payload["benchmarks"].values():
        if not row["available"]:
            cells.append("--")
        else:
            cells.append(f"${row['mean_pct']:.1f}_{{\\pm {row['std_pct']:.1f}}}$")
    average = payload["average"]
    cells.append(
        f"${average['mean_pct']:.1f}_{{\\pm {average['std_pct']:.1f}}}$"
        if average["available"]
        else "--"
    )
    return " & ".join(cells) + r" \\"


def _fmt_pm(mean_pct: float, std_pct: float) -> str:
    return f"{mean_pct:.1f} ± {std_pct:.1f}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ADAS reproduction results.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--output", default="")
    parser.add_argument("--system-name", default="ADAS")
    return parser.parse_args()


if __name__ == "__main__":
    main()
