from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .aggregation import AggregationConfig, compute_composites
from .benchmark_metadata import attach_benchmark_metadata
from .metrics_cost import CostMetricsConfig, compute_cost_metrics
from .metrics_quality import QualityMetricsConfig, compute_quality_metrics
from .plotting import (
    plot_benchmark_grouped,
    plot_gain_cost_plane,
    plot_mahalanobis_diagnostics,
    plot_pareto_frontier,
    plot_sensitivity,
    plot_utility_comparison,
)
from .utility_analysis import compare_sas_vs_mas


@dataclass(frozen=True)
class PipelineConfig:
    pass_k_values: Sequence[int] = (1, 3, 5)
    methods: Sequence[str] = ("arithmetic", "geometric", "mahalanobis", "topsis")
    quality_weights: dict[str, float] | None = None
    cost_weights: dict[str, float] | None = None
    primary_method: str = "arithmetic"
    cps_fallback: str = "nan"


def _plot_path(output_dir: Path, category: str, filename: str) -> Path:
    return output_dir / category / filename


def _to_float_01(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if v >= 1.0 else (0.0 if v <= 0.0 else v)


def _discover_task_dirs(system_dir: Path) -> list[Path]:
    return sorted(path for path in system_dir.iterdir() if path.is_dir())


def _extract_run_index(path: Path) -> int | None:
    match = re.search(r"run_(\d+)\.trace_metrics\.json$", path.name)
    if not match:
        return None
    return int(match.group(1))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_per_run_table(experiment_root: Path, benchmarks: Sequence[str] | None = None) -> pd.DataFrame:
    """Build a run-level processed table from trace/result artifacts."""
    wanted = {b.lower() for b in (benchmarks or [])}
    rows: list[dict[str, Any]] = []

    for benchmark_dir in sorted(p for p in experiment_root.iterdir() if p.is_dir()):
        benchmark = benchmark_dir.name
        if wanted and benchmark.lower() not in wanted:
            continue

        for system_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
            if not (system_dir / "summary.csv").exists():
                continue
            system_name = system_dir.name
            system_type = "SAS" if system_name.lower() == "sas" else "MAS"

            for task_dir in _discover_task_dirs(system_dir):
                task_id = task_dir.name
                for trace_path in sorted(task_dir.glob("run_*.trace_metrics.json")):
                    run_index = _extract_run_index(trace_path)
                    if run_index is None:
                        continue
                    result_path = task_dir / f"run_{run_index}.result.json"
                    eval_path = task_dir / f"run_{run_index}.eval.json"

                    trace_payload = _load_json(trace_path)
                    metrics = trace_payload.get("metrics", {})

                    success = _to_float_01(metrics.get("success"))
                    completion = _to_float_01(metrics.get("completion"))
                    score = metrics.get("score")
                    if score is None and eval_path.exists():
                        score = _load_json(eval_path).get("score")

                    if result_path.exists():
                        result_payload = _load_json(result_path)
                        system_mode = (
                            str(result_payload.get("system", {}).get("mode", "")).upper().strip()
                        )
                        if system_mode in {"SAS", "MAS"}:
                            system_type = system_mode

                    tool_calls_total = float(metrics.get("tool_calls_total", 0.0) or 0.0)
                    tool_error_count = float(metrics.get("tool_fail_total", 0.0) or 0.0)
                    tool_error_rate = (
                        tool_error_count / tool_calls_total if tool_calls_total > 0 else 0.0
                    )

                    communication_total = metrics.get("communication_count")
                    communication_agent_to_agent = metrics.get("communication_count_agent_to_agent")
                    communication_system_mediated = metrics.get(
                        "communication_count_system_mediated"
                    )

                    # Chapter-2 alignment:
                    # `communication_count` is defined as agent-to-agent communication.
                    communication_count = (
                        communication_agent_to_agent
                        if communication_agent_to_agent is not None
                        else communication_total
                    )

                    rows.append(
                        {
                            "benchmark": benchmark,
                            "system_name": system_name,
                            "system_type": system_type,
                            "task_id": task_id,
                            "run_id": f"{task_id}_{run_index}",
                            "run_index": run_index,
                            "success": success,
                            "completion": completion,
                            "score": score,
                            "tokens_total": metrics.get("tokens_total"),
                            "tool_calls_total": tool_calls_total,
                            "communication_count": communication_count,
                            "communication_count_total": communication_total,
                            "communication_count_agent_to_agent": communication_agent_to_agent,
                            "communication_count_system_mediated": communication_system_mediated,
                            "handoff_count": metrics.get("handoff_count"),
                            "tool_error_count": tool_error_count,
                            "tool_error_rate": tool_error_rate,
                            "latency_total": metrics.get("latency_total"),
                            "steps_total": metrics.get("steps_total"),
                        }
                    )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No run-level artifacts found under {experiment_root}")
    numeric_cols = [
        "success",
        "completion",
        "score",
        "tokens_total",
        "tool_calls_total",
        "communication_count",
        "communication_count_total",
        "communication_count_agent_to_agent",
        "communication_count_system_mediated",
        "handoff_count",
        "tool_error_count",
        "tool_error_rate",
        "latency_total",
        "steps_total",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def build_system_summary(
    run_df: pd.DataFrame,
    *,
    pass_k_values: Sequence[int],
    cps_fallback: str,
) -> pd.DataFrame:
    """Aggregate run-level rows to per-system per-benchmark summary."""
    rows: list[dict[str, Any]] = []
    for (benchmark, system_name, system_type), group in run_df.groupby(
        ["benchmark", "system_name", "system_type"], dropna=False
    ):
        quality = compute_quality_metrics(
            group,
            config=QualityMetricsConfig(pass_k_values=tuple(pass_k_values)),
        )
        cost = compute_cost_metrics(
            group,
            success_rate=float(quality["success_rate"]),
            config=CostMetricsConfig(cps_fallback=cps_fallback),
        )
        row = {
            "benchmark": benchmark,
            "system_name": system_name,
            "system_type": system_type,
            **quality,
            **cost,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    return attach_benchmark_metadata(summary)


def run_economic_pipeline(
    *,
    experiment_root: Path,
    output_dir: Path,
    benchmarks: Sequence[str] | None = None,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PipelineConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    run_df = build_per_run_table(experiment_root, benchmarks)
    run_csv = output_dir / "per_run_processed.csv"
    run_df.to_csv(run_csv, index=False)

    summary_base = build_system_summary(
        run_df,
        pass_k_values=cfg.pass_k_values,
        cps_fallback=cfg.cps_fallback,
    )

    quality_cols = [
        "success_rate",
        *[f"pass_at_{k}" for k in cfg.pass_k_values],
        "stability",
        "eval_avg_score",
    ]
    cost_cols = ["tokens_total", "cost_per_success", "tokens_cv", "tool_calls_total"]

    all_method_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []

    for method in cfg.methods:
        scored = compute_composites(
            summary_base,
            quality_cols=quality_cols,
            cost_cols=cost_cols,
            config=AggregationConfig(
                method=method,
                quality_weights=cfg.quality_weights,
                cost_weights=cfg.cost_weights,
            ),
        )
        all_method_frames.append(scored)
        comparison_frames.append(compare_sas_vs_mas(scored))

    sensitivity_df = pd.concat(all_method_frames, ignore_index=True)
    sensitivity_csv = output_dir / "sensitivity_system_summary.csv"
    sensitivity_df.to_csv(sensitivity_csv, index=False)

    sensitivity_comp_df = pd.concat(comparison_frames, ignore_index=True)
    sensitivity_comp_csv = output_dir / "sensitivity_sas_vs_mas_comparison.csv"
    sensitivity_comp_df.to_csv(sensitivity_comp_csv, index=False)

    primary_summary = sensitivity_df[sensitivity_df["aggregation_method"] == cfg.primary_method].copy()
    primary_comp = sensitivity_comp_df[
        sensitivity_comp_df["aggregation_method"] == cfg.primary_method
    ].copy()

    system_csv = output_dir / "per_system_benchmark_summary.csv"
    primary_summary.to_csv(system_csv, index=False)
    comp_csv = output_dir / "sas_vs_mas_comparison.csv"
    primary_comp.to_csv(comp_csv, index=False)

    plot_paths = [
        plot_utility_comparison(primary_comp, output_dir / "RQ1"),
        plot_gain_cost_plane(primary_comp, output_dir / "RQ2"),
        plot_benchmark_grouped(primary_summary, output_dir / "RQ1"),
        plot_sensitivity(sensitivity_df, output_dir / "THEORY"),
        plot_pareto_frontier(primary_summary, output_dir / "THEORY"),
    ]

    # Method-specific plots to make aggregation effects explicit.
    for method in cfg.methods:
        method_summary = sensitivity_df[sensitivity_df["aggregation_method"] == method].copy()
        method_comp = sensitivity_comp_df[
            sensitivity_comp_df["aggregation_method"] == method
        ].copy()
        plot_paths.extend(
            [
                plot_utility_comparison(method_comp, output_dir / "RQ1", file_suffix=method),
                plot_gain_cost_plane(method_comp, output_dir / "RQ2", file_suffix=method),
                plot_pareto_frontier(method_summary, output_dir / "THEORY", file_suffix=method),
                plot_mahalanobis_diagnostics(method_summary, output_dir / "THEORY", file_suffix=method),
            ]
        )
    plot_paths = [p for p in plot_paths if p]

    ranking_stability = (
        sensitivity_df.assign(
            rank=sensitivity_df.groupby(["benchmark", "aggregation_method"])["U"].rank(
                ascending=False,
                method="dense",
            )
        )
        .groupby(["benchmark", "system_name"], as_index=False)["rank"]
        .std()
        .rename(columns={"rank": "utility_rank_std"})
    )
    ranking_stability_csv = output_dir / "ranking_stability.csv"
    ranking_stability.to_csv(ranking_stability_csv, index=False)

    payload = {
        "experiment_root": str(experiment_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "benchmarks": sorted(run_df["benchmark"].unique().tolist()),
        "methods": list(cfg.methods),
        "artifacts": {
            "per_run_processed_csv": str(run_csv.resolve()),
            "per_system_benchmark_summary_csv": str(system_csv.resolve()),
            "sas_vs_mas_comparison_csv": str(comp_csv.resolve()),
            "sensitivity_system_summary_csv": str(sensitivity_csv.resolve()),
            "sensitivity_sas_vs_mas_comparison_csv": str(sensitivity_comp_csv.resolve()),
            "ranking_stability_csv": str(ranking_stability_csv.resolve()),
            "plots": plot_paths,
        },
    }
    payload_path = output_dir / "analysis_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Economic SAS vs MAS evaluation pipeline")
    parser.add_argument("--experiment-root", required=True, help="Experiment folder path")
    parser.add_argument("--output-dir", required=True, help="Output folder path")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Optional benchmark filter list, e.g. browsecomp plancraft",
    )
    parser.add_argument(
        "--primary-method",
        default="arithmetic",
        choices=["arithmetic", "geometric", "mahalanobis", "topsis"],
    )
    args = parser.parse_args()

    payload = run_economic_pipeline(
        experiment_root=Path(args.experiment_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        benchmarks=args.benchmarks,
        config=PipelineConfig(primary_method=args.primary_method),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
