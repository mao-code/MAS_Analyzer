from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.analyze_experiment import (  # noqa: E402
    _axis_system_label,
    _color_for_system,
    _format_compact_number,
    _marker_for_topology,
    aggregate_system_metrics,
    aggregate_vs_sas,
    compute_vs_sas,
    load_task_rows,
)


PLOT_DPI = 300
DEFAULT_EXPERIMENTS = {
    "GPT-OSS-120B": Path(
        "20260427T134706Z__openai_gpt_oss_120b/20260427T134706Z__openai_gpt_oss_120b"
    ),
    "Gemma-4-31B-IT": Path(
        "20260427T134706Z__google_gemma_4_31b_it_nitro/"
        "20260427T134706Z__google_gemma_4_31b_it_nitro"
    ),
    "Qwen3-32B": Path(
        "20260426_qwen3_32b_paper_run__qwen_qwen3_32b_nitro/"
        "20260426_qwen3_32b_paper_run__qwen_qwen3_32b_nitro"
    ),
}
MODEL_COLORS = {
    "GPT-OSS-120B": "#1f77b4",
    "Gemma-4-31B-IT": "#2ca02c",
    "Qwen3-32B": "#d62728",
}
MODEL_ORDER = list(DEFAULT_EXPERIMENTS)
BENCHMARK_ORDER = ["browsecomp", "finance_agent", "plancraft", "stabletoolbench", "workbench"]


matplotlib.rcParams.update(
    {
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8.5,
        "legend.frameon": False,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.65,
        "lines.linewidth": 1.45,
        "patch.linewidth": 0.6,
    }
)


def _ordered(values: pd.Series, preferred: list[str]) -> list[str]:
    present = values.dropna().astype(str).unique().tolist()
    ordered = [item for item in preferred if item in present]
    return ordered + sorted(item for item in present if item not in ordered)


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        ax.tick_params(axis="both", which="both", direction="out", length=3.0, width=0.7)
    fig.patch.set_facecolor("white")
    fig.savefig(path, dpi=PLOT_DPI)
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return str(path.resolve())


def _load_all(experiments: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_frames: list[pd.DataFrame] = []
    system_frames: list[pd.DataFrame] = []
    vs_sas_frames: list[pd.DataFrame] = []

    for model_name, root in experiments.items():
        task_df = load_task_rows(root)
        task_df.insert(0, "model", model_name)
        task_df.insert(1, "experiment_root", str(root.resolve()))
        system_df = aggregate_system_metrics(task_df)
        vs_sas_df = aggregate_vs_sas(compute_vs_sas(task_df))
        system_df.insert(0, "model", model_name)
        if not vs_sas_df.empty:
            vs_sas_df.insert(0, "model", model_name)
        task_frames.append(task_df)
        system_frames.append(system_df)
        vs_sas_frames.append(vs_sas_df)

    task_all = pd.concat(task_frames, ignore_index=True)
    system_all = pd.concat(system_frames, ignore_index=True)
    vs_sas_all = pd.concat([frame for frame in vs_sas_frames if not frame.empty], ignore_index=True)
    system_all["model"] = pd.Categorical(system_all["model"], MODEL_ORDER, ordered=True)
    task_all["model"] = pd.Categorical(task_all["model"], MODEL_ORDER, ordered=True)
    if not vs_sas_all.empty:
        vs_sas_all["model"] = pd.Categorical(vs_sas_all["model"], MODEL_ORDER, ordered=True)
    return task_all, system_all, vs_sas_all


def _plot_success_heatmap(system_df: pd.DataFrame, out_dir: Path) -> str:
    frame = system_df.copy()
    benchmarks = _ordered(frame["benchmark"], BENCHMARK_ORDER)
    systems = _ordered(frame["system_label"], ["sas"])
    rows = [(benchmark, system) for benchmark in benchmarks for system in systems]
    models = [model for model in MODEL_ORDER if model in frame["model"].astype(str).unique()]
    matrix = np.full((len(rows), len(models)), np.nan)
    lookup = frame.set_index(["benchmark", "system_label", "model"])["avg_success_rate"]
    for row_idx, (benchmark, system) in enumerate(rows):
        for col_idx, model in enumerate(models):
            key = (benchmark, system, model)
            if key in lookup.index:
                matrix[row_idx, col_idx] = float(lookup.loc[key])

    fig_height = max(7.5, len(rows) * 0.31)
    fig, ax = plt.subplots(figsize=(6.3, fig_height))
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    labels = [f"{benchmark} | {_axis_system_label(system).replace(chr(10), ' ')}" for benchmark, system in rows]
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Success rate by benchmark, topology, and model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Benchmark | topology")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=6.8, color="white" if value > 0.55 else "#222222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Average success rate")
    fig.tight_layout()
    return _save(fig, out_dir / "model_success_heatmap.png")


def _plot_quality_cost_frontier(system_df: pd.DataFrame, out_dir: Path) -> str:
    frame = system_df.dropna(subset=["avg_success_rate", "avg_tokens_total"]).copy()
    benchmarks = _ordered(frame["benchmark"], BENCHMARK_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), sharex=False, sharey=True)
    axes_flat = axes.ravel()
    for idx, benchmark in enumerate(benchmarks):
        ax = axes_flat[idx]
        subset = frame[frame["benchmark"] == benchmark]
        for _, row in subset.iterrows():
            model = str(row["model"])
            system = str(row["system_label"])
            ax.scatter(
                float(row["avg_tokens_total"]),
                float(row["avg_success_rate"]),
                s=54,
                color=MODEL_COLORS.get(model, "#777777"),
                marker=_marker_for_topology(system),
                edgecolors="#111111",
                linewidths=0.55,
                alpha=0.88,
            )
        ax.set_title(benchmark)
        ax.set_xlabel("Mean tokens")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_compact_number))
        ax.grid(True, axis="both", alpha=0.35)
        ax.set_ylim(-0.03, 1.03)
    for ax in axes_flat[len(benchmarks) :]:
        ax.axis("off")
    axes_flat[0].set_ylabel("Average success rate")
    axes_flat[3].set_ylabel("Average success rate")

    model_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=color, markeredgecolor="#111111", markersize=6.5, label=model)
        for model, color in MODEL_COLORS.items()
        if model in frame["model"].astype(str).unique()
    ]
    topology_handles = [
        Line2D([0], [0], marker=_marker_for_topology(system), linestyle="None", color="none", markerfacecolor="#888888", markeredgecolor="#111111", markersize=6.5, label=_axis_system_label(system).replace("\n", " "))
        for system in _ordered(frame["system_label"], ["sas"])
    ]
    fig.legend(handles=model_handles, title="Model", loc="center left", bbox_to_anchor=(0.86, 0.62))
    fig.legend(handles=topology_handles, title="Topology", loc="center left", bbox_to_anchor=(0.86, 0.25))
    fig.suptitle("Quality-cost frontier across model runs", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 0.96))
    return _save(fig, out_dir / "model_quality_cost_frontier.png")


def _plot_best_delta_vs_sas(vs_sas_df: pd.DataFrame, out_dir: Path) -> str | None:
    if vs_sas_df.empty:
        return None
    frame = vs_sas_df.dropna(subset=["mean_success_rate_delta_vs_sas"]).copy()
    if frame.empty:
        return None
    idx = frame.groupby(["model", "benchmark"], observed=True)["mean_success_rate_delta_vs_sas"].idxmax()
    best = frame.loc[idx].copy()
    benchmarks = _ordered(best["benchmark"], BENCHMARK_ORDER)
    models = [model for model in MODEL_ORDER if model in best["model"].astype(str).unique()]
    x = np.arange(len(benchmarks))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for model_idx, model in enumerate(models):
        values = []
        labels = []
        for benchmark in benchmarks:
            subset = best[(best["model"].astype(str) == model) & (best["benchmark"] == benchmark)]
            if subset.empty:
                values.append(np.nan)
                labels.append("")
            else:
                row = subset.iloc[0]
                values.append(float(row["mean_success_rate_delta_vs_sas"]))
                labels.append(_axis_system_label(str(row["system_label"])).replace("\n", " "))
        positions = x + (model_idx - (len(models) - 1) / 2) * width
        bars = ax.bar(positions, values, width=width, color=MODEL_COLORS.get(model, "#777777"), label=model, alpha=0.86)
        for bar, value, label in zip(bars, values, labels, strict=False):
            if not np.isfinite(value):
                continue
            y = value + (0.012 if value >= 0 else -0.018)
            ax.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va="bottom" if value >= 0 else "top", fontsize=6.7, rotation=90)
    ax.axhline(0.0, color="#333333", linewidth=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=20, ha="right")
    ax.set_ylabel("Best MAS success-rate delta vs SAS")
    ax.set_title("Best topology lift over SAS by model and benchmark")
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir / "model_best_delta_vs_sas.png")


def _plot_model_topology_summary(system_df: pd.DataFrame, out_dir: Path) -> str:
    frame = (
        system_df.groupby(["model", "system_label"], observed=True, as_index=False)
        .agg(
            mean_success=("avg_success_rate", "mean"),
            mean_tokens=("avg_tokens_total", "mean"),
            mean_stability=("avg_stability", "mean"),
        )
        .sort_values(["system_label", "model"])
    )
    systems = _ordered(frame["system_label"], ["sas"])
    models = [model for model in MODEL_ORDER if model in frame["model"].astype(str).unique()]
    x = np.arange(len(systems))
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    for model_idx, model in enumerate(models):
        values = []
        for system in systems:
            subset = frame[(frame["model"].astype(str) == model) & (frame["system_label"] == system)]
            values.append(float(subset.iloc[0]["mean_success"]) if not subset.empty else np.nan)
        ax.bar(
            x + (model_idx - (len(models) - 1) / 2) * width,
            values,
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, "#777777"),
            alpha=0.86,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_axis_system_label(system) for system in systems], rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean success rate across benchmarks")
    ax.set_title("Topology-level performance by model")
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    return _save(fig, out_dir / "model_topology_success_summary.png")


def _ci95(values: pd.Series) -> tuple[float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan, np.nan, np.nan
    mean = float(numeric.mean())
    if len(numeric) < 2:
        return mean, 0.0, 0.0
    half_width = float(1.96 * numeric.std(ddof=1) / np.sqrt(len(numeric)))
    return mean, half_width, half_width


def _plot_main_result_by_benchmark(task_df: pd.DataFrame, out_dir: Path) -> str:
    frame = (
        task_df.groupby(["benchmark", "system_label", "model"], observed=True)
        .agg(
            mean_success=("success_rate", "mean"),
            n_tasks=("task_id", "nunique"),
            success_std=("success_rate", "std"),
        )
        .reset_index()
    )
    frame["success_sem95"] = 1.96 * frame["success_std"].fillna(0.0) / np.sqrt(
        frame["n_tasks"].clip(lower=1)
    )
    benchmarks = _ordered(frame["benchmark"], BENCHMARK_ORDER)
    systems = _ordered(frame["system_label"], ["sas"])
    models = [model for model in MODEL_ORDER if model in frame["model"].astype(str).unique()]

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), sharey=True)
    axes_flat = axes.ravel()
    x = np.arange(len(systems))
    offsets = np.linspace(-0.24, 0.24, len(models))
    for idx, benchmark in enumerate(benchmarks):
        ax = axes_flat[idx]
        subset = frame[frame["benchmark"] == benchmark]
        for model, offset in zip(models, offsets, strict=False):
            model_subset = subset[subset["model"].astype(str) == model].set_index("system_label")
            y_values = [model_subset.loc[system, "mean_success"] if system in model_subset.index else np.nan for system in systems]
            errors = [model_subset.loc[system, "success_sem95"] if system in model_subset.index else np.nan for system in systems]
            ax.errorbar(
                x + offset,
                y_values,
                yerr=errors,
                fmt="o",
                markersize=5.8,
                capsize=2.5,
                color=MODEL_COLORS.get(model, "#777777"),
                markeredgecolor="#111111",
                markeredgewidth=0.45,
                linewidth=1.0,
                label=model,
            )
        ax.set_title(benchmark)
        ax.set_xticks(x)
        ax.set_xticklabels([_axis_system_label(system) for system in systems], rotation=0)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, axis="y", alpha=0.4)
    for ax in axes_flat[len(benchmarks) :]:
        ax.axis("off")
    axes_flat[0].set_ylabel("Mean task success")
    axes_flat[3].set_ylabel("Mean task success")
    handles = [
        Line2D([0], [0], marker="o", color=color, markeredgecolor="#111111", linestyle="None", markersize=6, label=model)
        for model, color in MODEL_COLORS.items()
        if model in models
    ]
    fig.legend(handles=handles, title="Model", loc="lower right", bbox_to_anchor=(0.965, 0.12))
    fig.suptitle("Main result: topology performance is model- and benchmark-dependent", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.96))
    return _save(fig, out_dir / "main_result_by_benchmark.png")


def _pareto_mask(points: pd.DataFrame) -> pd.Series:
    mask = []
    for idx, row in points.iterrows():
        dominated = (
            (points["avg_tokens_total"] <= row["avg_tokens_total"])
            & (points["avg_success_rate"] >= row["avg_success_rate"])
            & (
                (points["avg_tokens_total"] < row["avg_tokens_total"])
                | (points["avg_success_rate"] > row["avg_success_rate"])
            )
        ).any()
        mask.append(not dominated)
    return pd.Series(mask, index=points.index)


def _plot_paper_quality_cost_pareto(system_df: pd.DataFrame, out_dir: Path) -> str:
    frame = system_df.dropna(subset=["avg_success_rate", "avg_tokens_total"]).copy()
    benchmarks = _ordered(frame["benchmark"], BENCHMARK_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), sharey=True)
    axes_flat = axes.ravel()
    for idx, benchmark in enumerate(benchmarks):
        ax = axes_flat[idx]
        subset = frame[frame["benchmark"] == benchmark].copy()
        pareto = subset[_pareto_mask(subset)].sort_values("avg_tokens_total")
        for _, row in subset.iterrows():
            model = str(row["model"])
            system = str(row["system_label"])
            ax.scatter(
                float(row["avg_tokens_total"]),
                float(row["avg_success_rate"]),
                s=54,
                color=MODEL_COLORS.get(model, "#777777"),
                marker=_marker_for_topology(system),
                edgecolors="#111111",
                linewidths=0.5,
                alpha=0.45 if row.name not in pareto.index else 0.95,
                zorder=2,
            )
        if len(pareto) >= 2:
            ax.plot(
                pareto["avg_tokens_total"],
                pareto["avg_success_rate"],
                color="#111111",
                linewidth=1.05,
                alpha=0.65,
                zorder=1,
            )
        for _, row in pareto.iterrows():
            label = f"{str(row['model']).split('-')[0]} / {_axis_system_label(str(row['system_label'])).replace(chr(10), ' ')}"
            ax.annotate(
                label,
                (float(row["avg_tokens_total"]), float(row["avg_success_rate"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6.4,
                color="#222222",
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
            )
        ax.set_title(benchmark)
        ax.set_xlabel("Mean tokens")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_compact_number))
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, axis="both", alpha=0.35)
    for ax in axes_flat[len(benchmarks) :]:
        ax.axis("off")
    axes_flat[0].set_ylabel("Mean task success")
    axes_flat[3].set_ylabel("Mean task success")
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=color, markeredgecolor="#111111", markersize=6.3, label=model)
        for model, color in MODEL_COLORS.items()
        if model in frame["model"].astype(str).unique()
    ]
    topology_handles = [
        Line2D([0], [0], marker=_marker_for_topology(system), linestyle="None", color="none", markerfacecolor="#888888", markeredgecolor="#111111", markersize=6.3, label=_axis_system_label(system).replace("\n", " "))
        for system in _ordered(frame["system_label"], ["sas"])
    ]
    fig.legend(handles=model_handles, title="Model", loc="center left", bbox_to_anchor=(0.86, 0.62))
    fig.legend(handles=topology_handles, title="Topology", loc="center left", bbox_to_anchor=(0.86, 0.25))
    fig.suptitle("Quality-cost Pareto frontier: collaboration gains are not only token scaling", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 0.96))
    return _save(fig, out_dir / "quality_cost_pareto_by_benchmark.png")


def _plot_delta_vs_sas_quadrant(vs_sas_df: pd.DataFrame, out_dir: Path) -> str | None:
    if vs_sas_df.empty:
        return None
    frame = vs_sas_df.dropna(
        subset=["mean_success_rate_delta_vs_sas", "mean_tokens_total_delta_vs_sas"]
    ).copy()
    if frame.empty:
        return None
    benchmarks = _ordered(frame["benchmark"], BENCHMARK_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    for idx, benchmark in enumerate(benchmarks):
        ax = axes_flat[idx]
        subset = frame[frame["benchmark"] == benchmark]
        for _, row in subset.iterrows():
            model = str(row["model"])
            system = str(row["system_label"])
            x = float(row["mean_tokens_total_delta_vs_sas"])
            y = float(row["mean_success_rate_delta_vs_sas"])
            ax.scatter(
                x,
                y,
                s=58,
                color=MODEL_COLORS.get(model, "#777777"),
                marker=_marker_for_topology(system),
                edgecolors="#111111",
                linewidths=0.55,
                alpha=0.9,
            )
        leaders = subset.sort_values(
            ["mean_success_rate_delta_vs_sas", "mean_tokens_total_delta_vs_sas"],
            ascending=[False, True],
        ).head(3)
        for _, row in leaders.iterrows():
            label = f"{str(row['model']).split('-')[0]} / {_axis_system_label(str(row['system_label'])).replace(chr(10), ' ')}"
            ax.annotate(
                label,
                (float(row["mean_tokens_total_delta_vs_sas"]), float(row["mean_success_rate_delta_vs_sas"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6.6,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
            )
        ax.axhline(0.0, color="#333333", linewidth=0.85)
        ax.axvline(0.0, color="#333333", linewidth=0.85)
        ax.set_title(benchmark)
        ax.set_xlabel("Mean token delta vs SAS")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_compact_number))
        ax.grid(True, axis="both", alpha=0.35)
    for ax in axes_flat[len(benchmarks) :]:
        ax.axis("off")
    axes_flat[0].set_ylabel("Mean success delta vs SAS")
    axes_flat[3].set_ylabel("Mean success delta vs SAS")
    fig.text(
        0.08,
        0.935,
        "Upper-left is best: higher success with fewer tokens than SAS.",
        fontsize=8.5,
        color="#444444",
    )
    handles = [
        Line2D([0], [0], marker="o", color=color, markeredgecolor="#111111", linestyle="None", markersize=6, label=model)
        for model, color in MODEL_COLORS.items()
        if model in frame["model"].astype(str).unique()
    ]
    fig.legend(handles=handles, title="Model", loc="lower right", bbox_to_anchor=(0.965, 0.11))
    fig.suptitle("Gain-cost plane: MAS lift relative to SAS", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.94))
    return _save(fig, out_dir / "delta_vs_sas_quadrant.png")


def _plot_ranking_consistency_heatmap(system_df: pd.DataFrame, out_dir: Path) -> str:
    frame = system_df.dropna(subset=["avg_success_rate", "avg_tokens_total"]).copy()
    frame["rank"] = frame.groupby(["model", "benchmark"], observed=True).apply(
        lambda group: group.sort_values(
            ["avg_success_rate", "avg_tokens_total", "system_label"],
            ascending=[False, True, True],
        ).assign(rank=np.arange(1, len(group) + 1))["rank"]
    ).reset_index(level=[0, 1], drop=True)
    systems = _ordered(frame["system_label"], ["sas"])
    row_keys = [
        (model, benchmark)
        for model in MODEL_ORDER
        for benchmark in BENCHMARK_ORDER
        if ((frame["model"].astype(str) == model) & (frame["benchmark"] == benchmark)).any()
    ]
    matrix = np.full((len(row_keys), len(systems)), np.nan)
    lookup = frame.set_index(["model", "benchmark", "system_label"])["rank"]
    for row_idx, (model, benchmark) in enumerate(row_keys):
        for col_idx, system in enumerate(systems):
            key = (model, benchmark, system)
            if key in lookup.index:
                matrix[row_idx, col_idx] = float(lookup.loc[key])
    fig, ax = plt.subplots(figsize=(8.4, max(5.8, len(row_keys) * 0.36)))
    cmap = matplotlib.colormaps["magma_r"].copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(matrix, aspect="auto", vmin=1, vmax=len(systems), cmap=cmap)
    ax.set_xticks(np.arange(len(systems)))
    ax.set_xticklabels([_axis_system_label(system).replace("\n", " ") for system in systems], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_yticklabels([f"{model} | {benchmark}" for model, benchmark in row_keys])
    ax.set_title("Topology ranking consistency across models and benchmarks")
    ax.set_xlabel("Topology")
    ax.set_ylabel("Model | benchmark")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{int(value)}", ha="center", va="center", fontsize=7.0, color="white" if value > len(systems) / 2 else "#111111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Rank by success, then token cost")
    fig.tight_layout()
    return _save(fig, out_dir / "ranking_consistency_heatmap.png")


def _plot_stability_ci_by_topology(system_df: pd.DataFrame, out_dir: Path) -> str:
    frame = system_df.dropna(subset=["avg_stability"]).copy()
    summary = (
        frame.groupby(["model", "system_label"], observed=True)
        .agg(mean_stability=("avg_stability", "mean"), std_stability=("avg_stability", "std"), n=("benchmark", "nunique"))
        .reset_index()
    )
    summary["sem95"] = 1.96 * summary["std_stability"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))
    systems = _ordered(summary["system_label"], ["sas"])
    models = [model for model in MODEL_ORDER if model in summary["model"].astype(str).unique()]
    x = np.arange(len(systems))
    offsets = np.linspace(-0.24, 0.24, len(models))
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for model, offset in zip(models, offsets, strict=False):
        subset = summary[summary["model"].astype(str) == model].set_index("system_label")
        values = [subset.loc[system, "mean_stability"] if system in subset.index else np.nan for system in systems]
        errors = [subset.loc[system, "sem95"] if system in subset.index else np.nan for system in systems]
        ax.errorbar(
            x + offset,
            values,
            yerr=errors,
            fmt="o",
            capsize=2.5,
            markersize=5.8,
            color=MODEL_COLORS.get(model, "#777777"),
            markeredgecolor="#111111",
            markeredgewidth=0.45,
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_axis_system_label(system) for system in systems])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Mean stability across benchmarks")
    ax.set_title("Appendix: stability by topology and model")
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    return _save(fig, out_dir / "stability_ci_by_topology.png")


def build_model_comparison(experiments: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    resolved = {name: path.expanduser().resolve() for name, path in experiments.items()}
    task_df, system_df, vs_sas_df = _load_all(resolved)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_csv = out_dir / "combined_task_level_metrics.csv"
    system_csv = out_dir / "combined_system_level_metrics.csv"
    vs_sas_csv = out_dir / "combined_system_level_vs_sas.csv"
    task_df.to_csv(task_csv, index=False)
    system_df.to_csv(system_csv, index=False)
    if not vs_sas_df.empty:
        vs_sas_df.to_csv(vs_sas_csv, index=False)

    plots = [
        _plot_main_result_by_benchmark(task_df, out_dir),
        _plot_paper_quality_cost_pareto(system_df, out_dir),
        _plot_ranking_consistency_heatmap(system_df, out_dir),
        _plot_stability_ci_by_topology(system_df, out_dir),
    ]
    delta_plot = _plot_delta_vs_sas_quadrant(vs_sas_df, out_dir)
    if delta_plot:
        plots.append(delta_plot)

    payload = {
        "experiments": {name: str(path) for name, path in resolved.items()},
        "task_rows": int(len(task_df)),
        "system_rows": int(len(system_df)),
        "benchmarks": sorted(task_df["benchmark"].dropna().astype(str).unique().tolist()),
        "models": MODEL_ORDER,
        "artifacts": {
            "task_level_metrics_csv": str(task_csv.resolve()),
            "system_level_metrics_csv": str(system_csv.resolve()),
            "plots": plots,
        },
    }
    if not vs_sas_df.empty:
        payload["artifacts"]["system_level_vs_sas_csv"] = str(vs_sas_csv.resolve())
    (out_dir / "model_comparison_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready plots comparing the OpenAI, Gemma, and Qwen experiment runs."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/overall",
        help="Directory for combined comparison plots and CSVs.",
    )
    args = parser.parse_args()
    payload = build_model_comparison(DEFAULT_EXPERIMENTS, Path(args.output_dir).resolve())
    print(payload["artifacts"]["plots"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
