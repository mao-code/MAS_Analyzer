from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_METRICS = [
    ("eval_avg_score", "Task Score", "task_score"),
    ("C1_latency_p95", "Latency (ms)", "latency_ms"),
    ("C2_tokens_total", "Tokens", "tokens_total"),
    ("D2_communication_count", "Communication Count", "communication_count"),
]


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _system_order(frame: pd.DataFrame) -> list[str]:
    summary = (
        frame.groupby("system_label", as_index=False)
        .agg(
            mean_score=("eval_avg_score", "mean"),
            mean_success=("eval_success_rate", "mean"),
            mean_tokens=("C2_tokens_total", "mean"),
        )
        .sort_values(
            by=["mean_score", "mean_success", "mean_tokens", "system_label"],
            ascending=[False, False, True, True],
        )
    )
    ordered = summary["system_label"].tolist()
    if "sas" in ordered:
        ordered = ["sas"] + [item for item in ordered if item != "sas"]
    return ordered


def load_task_rows(experiment_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for benchmark_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        for system_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
            summary_csv = system_dir / "summary.csv"
            if not summary_csv.exists():
                continue
            frame = pd.read_csv(summary_csv)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["benchmark"] = frame.get("benchmark", benchmark_dir.name)
            frame["system_label"] = frame.get("system_label", system_dir.name)
            frame["topology"] = frame.get("topology", system_dir.name)
            frame["system_root"] = str(system_dir.resolve())
            frames.append(frame)

    if not frames:
        raise ValueError(f"No summary.csv files found under {experiment_root}")
    return pd.concat(frames, ignore_index=True)


def aggregate_system_metrics(task_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        task_df.groupby(["benchmark", "system_label", "topology"], as_index=False)
        .agg(
            task_count=("task_id", "nunique"),
            avg_score=("eval_avg_score", "mean"),
            median_score=("eval_avg_score", "median"),
            avg_success_rate=("eval_success_rate", "mean"),
            avg_latency_ms=("C1_latency_p95", "mean"),
            median_latency_ms=("C1_latency_p95", "median"),
            avg_tokens=("C2_tokens_total", "mean"),
            median_tokens=("C2_tokens_total", "median"),
            avg_cost_usd=("C3_cost_total", "mean"),
            avg_tool_calls=("C4_tool_calls_total", "mean"),
            avg_communication=("D2_communication_count", "mean"),
            avg_handoffs=("D3_handoff_count", "mean"),
            avg_steps=("P1_steps_total", "mean"),
            avg_loop_score=("P3_loop_score", "mean"),
            avg_verification_density=("P4_verification_density", "mean"),
        )
        .sort_values(
            by=["benchmark", "avg_score", "avg_success_rate", "avg_tokens", "system_label"],
            ascending=[True, False, False, True, True],
        )
    )
    return grouped


def compute_vs_sas(task_df: pd.DataFrame) -> pd.DataFrame:
    baseline = task_df[task_df["system_label"] == "sas"][
        [
            "benchmark",
            "task_id",
            "eval_avg_score",
            "eval_success_rate",
            "C1_latency_p95",
            "C2_tokens_total",
            "D2_communication_count",
            "C4_tool_calls_total",
        ]
    ].rename(
        columns={
            "eval_avg_score": "sas_score",
            "eval_success_rate": "sas_success_rate",
            "C1_latency_p95": "sas_latency_ms",
            "C2_tokens_total": "sas_tokens_total",
            "D2_communication_count": "sas_communication_count",
            "C4_tool_calls_total": "sas_tool_calls_total",
        }
    )

    merged = task_df.merge(baseline, on=["benchmark", "task_id"], how="left")
    if merged["sas_score"].isna().all():
        return pd.DataFrame()

    merged = merged[merged["system_label"] != "sas"].copy()
    merged["score_delta_vs_sas"] = merged["eval_avg_score"] - merged["sas_score"]
    merged["success_delta_vs_sas"] = merged["eval_success_rate"] - merged["sas_success_rate"]
    merged["latency_delta_ms_vs_sas"] = merged["C1_latency_p95"] - merged["sas_latency_ms"]
    merged["tokens_delta_vs_sas"] = merged["C2_tokens_total"] - merged["sas_tokens_total"]
    merged["communication_delta_vs_sas"] = (
        merged["D2_communication_count"] - merged["sas_communication_count"]
    )
    merged["tool_calls_delta_vs_sas"] = (
        merged["C4_tool_calls_total"] - merged["sas_tool_calls_total"]
    )
    merged["score_win_vs_sas"] = (merged["score_delta_vs_sas"] > 0).astype(int)
    merged["score_tie_vs_sas"] = (merged["score_delta_vs_sas"] == 0).astype(int)
    merged["score_loss_vs_sas"] = (merged["score_delta_vs_sas"] < 0).astype(int)
    return merged


def aggregate_vs_sas(vs_sas_df: pd.DataFrame) -> pd.DataFrame:
    if vs_sas_df.empty:
        return pd.DataFrame()
    return (
        vs_sas_df.groupby(["benchmark", "system_label"], as_index=False)
        .agg(
            task_count=("task_id", "nunique"),
            mean_score_delta_vs_sas=("score_delta_vs_sas", "mean"),
            mean_success_delta_vs_sas=("success_delta_vs_sas", "mean"),
            mean_latency_delta_ms_vs_sas=("latency_delta_ms_vs_sas", "mean"),
            mean_tokens_delta_vs_sas=("tokens_delta_vs_sas", "mean"),
            mean_communication_delta_vs_sas=("communication_delta_vs_sas", "mean"),
            mean_tool_calls_delta_vs_sas=("tool_calls_delta_vs_sas", "mean"),
            score_wins_vs_sas=("score_win_vs_sas", "sum"),
            score_ties_vs_sas=("score_tie_vs_sas", "sum"),
            score_losses_vs_sas=("score_loss_vs_sas", "sum"),
        )
        .sort_values(
            by=["benchmark", "mean_score_delta_vs_sas", "mean_tokens_delta_vs_sas", "system_label"],
            ascending=[True, False, True, True],
        )
    )


def _save_boxplot(frame: pd.DataFrame, *, benchmark: str, metric: str, label: str, slug: str, out_dir: Path) -> str | None:
    metric_values = frame[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if metric_values.empty:
        return None

    systems = _system_order(frame)
    series = [frame.loc[frame["system_label"] == system, metric].dropna().to_numpy() for system in systems]
    if not any(len(values) for values in series):
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(systems) * 1.1), 5))
    boxplot = ax.boxplot(series, labels=systems, patch_artist=True)
    for patch, system in zip(boxplot.get("boxes", []), systems):
        if system == "sas":
            patch.set_facecolor("#d6eaf8")
        else:
            patch.set_facecolor("#e8f6ef")
    ax.set_title(f"{benchmark}: {label} by Topology")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_{slug}_boxplot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.resolve())


def _save_heatmap(frame: pd.DataFrame, *, benchmark: str, out_dir: Path) -> str | None:
    pivot = frame.pivot_table(
        index="task_id",
        columns="system_label",
        values="eval_avg_score",
        aggfunc="mean",
    )
    if pivot.empty:
        return None
    ordered_cols = [col for col in _system_order(frame) if col in pivot.columns]
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), max(4, len(pivot.index) * 0.8)))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"{benchmark}: Task Score Heatmap")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Task Score")
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_task_score_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.resolve())


def _save_efficiency_scatter(frame: pd.DataFrame, *, benchmark: str, out_dir: Path) -> str | None:
    summary = (
        frame.groupby("system_label", as_index=False)
        .agg(
            avg_score=("eval_avg_score", "mean"),
            avg_tokens=("C2_tokens_total", "mean"),
            avg_latency_ms=("C1_latency_p95", "mean"),
        )
    )
    if summary.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in summary.iterrows():
        color = "#1f77b4" if row["system_label"] == "sas" else "#2ca02c"
        ax.scatter(row["avg_tokens"], row["avg_score"], s=100, color=color)
        ax.annotate(str(row["system_label"]), (row["avg_tokens"], row["avg_score"]), xytext=(4, 4), textcoords="offset points")
    ax.set_title(f"{benchmark}: Accuracy vs Token Cost")
    ax.set_xlabel("Average Tokens")
    ax.set_ylabel("Average Task Score")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_accuracy_vs_tokens.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.resolve())


def _save_overall_score_bars(system_df: pd.DataFrame, out_dir: Path) -> str | None:
    if system_df.empty:
        return None

    benchmarks = sorted(system_df["benchmark"].unique())
    systems = sorted(system_df["system_label"].unique())
    x = np.arange(len(systems))
    width = 0.8 / max(1, len(benchmarks))

    fig, ax = plt.subplots(figsize=(max(9, len(systems) * 1.2), 5))
    for idx, benchmark in enumerate(benchmarks):
        subset = system_df[system_df["benchmark"] == benchmark].set_index("system_label")
        values = [float(subset.loc[system, "avg_score"]) if system in subset.index else math.nan for system in systems]
        ax.bar(x + (idx - (len(benchmarks) - 1) / 2) * width, values, width=width, label=benchmark)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=25, ha="right")
    ax.set_ylabel("Average Task Score")
    ax.set_title("Average Task Score by Topology and Benchmark")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "overall_avg_task_score.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.resolve())


def write_report(
    *,
    experiment_root: Path,
    out_dir: Path,
    task_df: pd.DataFrame,
    system_df: pd.DataFrame,
    vs_sas_df: pd.DataFrame,
    plots: dict[str, list[str] | str],
) -> Path:
    lines = [
        f"# Experiment Analysis: {experiment_root.name}",
        "",
        f"- Experiment Root: `{experiment_root}`",
        f"- Benchmarks: {task_df['benchmark'].nunique()}",
        f"- Topologies: {task_df['system_label'].nunique()}",
        f"- Task rows: {len(task_df)}",
        f"- Unique tasks: {task_df[['benchmark', 'task_id']].drop_duplicates().shape[0]}",
        "",
        "## Headline Findings",
        "",
    ]

    def render_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_None_"
        return "```text\n" + frame.to_string(index=False, justify="left") + "\n```"

    for benchmark in sorted(task_df["benchmark"].unique()):
        benchmark_systems = system_df[system_df["benchmark"] == benchmark].copy()
        best = benchmark_systems.sort_values(
            by=["avg_score", "avg_success_rate", "avg_tokens", "system_label"],
            ascending=[False, False, True, True],
        ).iloc[0]
        lines.append(
            f"- `{benchmark}`: best mean score is `{best['system_label']}` "
            f"with score `{best['avg_score']:.3f}`, success `{best['avg_success_rate']:.3f}`, "
            f"and mean tokens `{best['avg_tokens']:.1f}`."
        )
        if "sas" in benchmark_systems["system_label"].values:
            sas_row = benchmark_systems[benchmark_systems["system_label"] == "sas"].iloc[0]
            lines.append(
                f"- `{benchmark}` SAS baseline: score `{sas_row['avg_score']:.3f}`, "
                f"success `{sas_row['avg_success_rate']:.3f}`, mean tokens `{sas_row['avg_tokens']:.1f}`."
            )
        if not vs_sas_df.empty:
            gains = vs_sas_df[vs_sas_df["benchmark"] == benchmark]
            if not gains.empty:
                leader = gains.sort_values(
                    by=["mean_score_delta_vs_sas", "mean_tokens_delta_vs_sas", "system_label"],
                    ascending=[False, True, True],
                ).iloc[0]
                lines.append(
                    f"- `{benchmark}` strongest score lift vs SAS: `{leader['system_label']}` "
                    f"with mean score delta `{leader['mean_score_delta_vs_sas']:+.3f}` "
                    f"over `{int(leader['task_count'])}` tasks."
                )
        lines.append("")

    lines.extend(["## Topology Table", ""])
    lines.append(render_table(system_df.round(3)))
    lines.append("")

    if not vs_sas_df.empty:
        lines.extend(["## Topology Delta vs SAS", ""])
        lines.append(render_table(vs_sas_df.round(3)))
        lines.append("")

    lines.extend(["## Plot Inventory", ""])
    for key, value in plots.items():
        if isinstance(value, list):
            for item in value:
                lines.append(f"- `{key}`: `{item}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `C3_cost_total` is omitted from most figures when it is identically zero across the experiment.")
    lines.append("- This experiment uses only two tasks per benchmark and one run per task, so boxplots are descriptive rather than inferential.")
    lines.append("- `workbench` appears to complete runs structurally but still scores zero on task evaluation across all tested topologies.")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return report_path


def analyze_experiment(experiment_root: Path, output_dir: Path) -> dict[str, Any]:
    task_df = load_task_rows(experiment_root)
    system_df = aggregate_system_metrics(task_df)
    vs_sas_task_df = compute_vs_sas(task_df)
    vs_sas_system_df = aggregate_vs_sas(vs_sas_task_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    task_csv = output_dir / "task_level_metrics.csv"
    system_csv = output_dir / "system_level_metrics.csv"
    vs_sas_task_csv = output_dir / "task_level_vs_sas.csv"
    vs_sas_system_csv = output_dir / "system_level_vs_sas.csv"
    task_df.to_csv(task_csv, index=False)
    system_df.to_csv(system_csv, index=False)
    if not vs_sas_task_df.empty:
        vs_sas_task_df.to_csv(vs_sas_task_csv, index=False)
    if not vs_sas_system_df.empty:
        vs_sas_system_df.to_csv(vs_sas_system_csv, index=False)

    plots: dict[str, list[str] | str] = {}
    overall_plot = _save_overall_score_bars(system_df, output_dir)
    if overall_plot:
        plots["overall_avg_task_score"] = overall_plot

    for benchmark in sorted(task_df["benchmark"].unique()):
        benchmark_frame = task_df[task_df["benchmark"] == benchmark].copy()
        benchmark_plots: list[str] = []
        for metric, label, slug in PLOT_METRICS:
            path = _save_boxplot(
                benchmark_frame,
                benchmark=benchmark,
                metric=metric,
                label=label,
                slug=slug,
                out_dir=output_dir,
            )
            if path:
                benchmark_plots.append(path)
        heatmap_path = _save_heatmap(benchmark_frame, benchmark=benchmark, out_dir=output_dir)
        if heatmap_path:
            benchmark_plots.append(heatmap_path)
        scatter_path = _save_efficiency_scatter(benchmark_frame, benchmark=benchmark, out_dir=output_dir)
        if scatter_path:
            benchmark_plots.append(scatter_path)
        plots[benchmark] = benchmark_plots

    report_path = write_report(
        experiment_root=experiment_root,
        out_dir=output_dir,
        task_df=task_df,
        system_df=system_df,
        vs_sas_df=vs_sas_system_df,
        plots=plots,
    )

    analysis_payload = {
        "experiment_root": str(experiment_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "task_rows": len(task_df),
        "benchmark_count": int(task_df["benchmark"].nunique()),
        "topology_count": int(task_df["system_label"].nunique()),
        "artifacts": {
            "task_level_metrics_csv": str(task_csv.resolve()),
            "system_level_metrics_csv": str(system_csv.resolve()),
            "report_md": str(report_path.resolve()),
            "plots": plots,
        },
        "headline": {
            benchmark: (
                system_df[system_df["benchmark"] == benchmark]
                .sort_values(
                    by=["avg_score", "avg_success_rate", "avg_tokens", "system_label"],
                    ascending=[False, False, True, True],
                )
                .iloc[0][["system_label", "avg_score", "avg_success_rate", "avg_tokens"]]
                .to_dict()
            )
            for benchmark in sorted(task_df["benchmark"].unique())
        },
    }
    if not vs_sas_task_df.empty:
        analysis_payload["artifacts"]["task_level_vs_sas_csv"] = str(vs_sas_task_csv.resolve())
    if not vs_sas_system_df.empty:
        analysis_payload["artifacts"]["system_level_vs_sas_csv"] = str(vs_sas_system_csv.resolve())

    analysis_json = output_dir / "analysis.json"
    analysis_json.write_text(json.dumps(analysis_payload, indent=2, sort_keys=True), encoding="utf-8")
    return analysis_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a hierarchical experiment root and generate comparison plots.")
    parser.add_argument("--experiment-root", required=True, help="Path to artifacts/full_experiment/<experiment-id>")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for analysis artifacts. Defaults to <experiment-root>/analysis",
    )
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else experiment_root / "analysis"
    )
    analyze_experiment(experiment_root, output_dir)
    print(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
