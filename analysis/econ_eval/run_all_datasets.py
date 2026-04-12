from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analysis.econ_eval.run_pipeline import PipelineConfig, run_economic_pipeline


def _has_data(benchmark_dir: Path) -> bool:
    if not benchmark_dir.is_dir():
        return False
    has_summary = any(path.name == "summary.csv" for path in benchmark_dir.rglob("summary.csv"))
    has_traces = any(benchmark_dir.rglob("run_*.trace_metrics.json"))
    return has_summary and has_traces


def discover_benchmarks_with_data(experiment_root: Path) -> list[str]:
    benchmarks: list[str] = []
    for candidate in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        if _has_data(candidate):
            benchmarks.append(candidate.name)
    return benchmarks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run economic SAS-vs-MAS pipeline for every dataset with available artifacts"
    )
    parser.add_argument("--experiment-root", required=True, help="Experiment folder path")
    parser.add_argument(
        "--output-base",
        default=None,
        help="Base folder for outputs. Defaults to <experiment-root>/Plot",
    )
    parser.add_argument(
        "--primary-method",
        default="arithmetic",
        choices=["arithmetic", "geometric", "mahalanobis", "topsis"],
    )
    parser.add_argument(
        "--pass-k",
        nargs="*",
        type=int,
        default=[1, 3, 5, 8],
        help="pass@k values, e.g. --pass-k 1 3 5 8",
    )
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_base = (
        Path(args.output_base).expanduser().resolve()
        if args.output_base
        else (experiment_root / "Plot").resolve()
    )
    output_base.mkdir(parents=True, exist_ok=True)

    benchmarks = discover_benchmarks_with_data(experiment_root)
    if not benchmarks:
        raise SystemExit(f"No datasets with artifacts found under {experiment_root}")

    cfg = PipelineConfig(
        pass_k_values=tuple(args.pass_k),
        primary_method=args.primary_method,
        methods=("arithmetic", "geometric", "mahalanobis", "topsis"),
    )

    outputs: dict[str, str] = {}

    # 1) One combined run over all datasets with data.
    all_out = output_base / "all"
    run_economic_pipeline(
        experiment_root=experiment_root,
        output_dir=all_out,
        benchmarks=benchmarks,
        config=cfg,
    )
    outputs["all"] = str(all_out)

    # 2) Per-dataset outputs (plots/tables for each benchmark).
    for benchmark in benchmarks:
        out_dir = output_base / benchmark
        run_economic_pipeline(
            experiment_root=experiment_root,
            output_dir=out_dir,
            benchmarks=[benchmark],
            config=cfg,
        )
        outputs[benchmark] = str(out_dir)

    payload = {
        "experiment_root": str(experiment_root),
        "benchmarks_with_data": benchmarks,
        "output_dirs": outputs,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
