from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.resolve())


def _suffix(name: str) -> str:
    return f"_{name}" if name else ""


def plot_utility_comparison(comp_df: pd.DataFrame, out_dir: Path, *, file_suffix: str = "") -> str | None:
    if comp_df.empty:
        return None
    frame = comp_df.copy().sort_values(["benchmark", "delta_U"], ascending=[True, False])
    labels = [f"{b}:{s}" for b, s in zip(frame["benchmark"], frame["system_name"], strict=False)]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 5))
    ax.bar(x - width / 2, frame["U_sas"], width=width, label="SAS utility")
    ax.bar(x + width / 2, frame["U_mas"], width=width, label="MAS utility")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Utility (U = Q - C)")
    ax.set_title("SAS vs MAS utility comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return _save(fig, out_dir / f"utility_sas_vs_mas{_suffix(file_suffix)}.png")


def plot_gain_cost_plane(comp_df: pd.DataFrame, out_dir: Path, *, file_suffix: str = "") -> str | None:
    if comp_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    for benchmark, group in comp_df.groupby("benchmark"):
        ax.scatter(group["G"], group["K"], label=benchmark, s=80)
    lim = float(max(np.nanmax(np.abs(comp_df["G"])), np.nanmax(np.abs(comp_df["K"])), 1e-6))
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.2, label="G = K")
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlim(-lim * 1.05, lim * 1.05)
    ax.set_ylim(-lim * 1.05, lim * 1.05)
    ax.set_xlabel("Collaboration gain G")
    ax.set_ylabel("Coordination cost K")
    ax.set_title("Gain-cost scatter with decision boundary")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, out_dir / f"gain_cost_plane{_suffix(file_suffix)}.png")


def plot_benchmark_grouped(summary_df: pd.DataFrame, out_dir: Path, *, file_suffix: str = "") -> str | None:
    if summary_df.empty:
        return None
    frame = summary_df.copy()
    plot_cols = ["Q", "C", "U", "success_rate", "completion_rate"]
    melted = frame.melt(
        id_vars=["benchmark", "system_name", "aggregation_method"],
        value_vars=plot_cols,
        var_name="metric",
        value_name="value",
    )

    fig, axes = plt.subplots(1, len(plot_cols), figsize=(max(13, len(frame) * 0.5), 4), sharey=False)
    if len(plot_cols) == 1:
        axes = [axes]

    for i, metric in enumerate(plot_cols):
        ax = axes[i]
        subset = melted[melted["metric"] == metric]
        if subset.empty:
            ax.axis("off")
            continue
        pivot = subset.pivot_table(index="benchmark", columns="system_name", values="value", aggfunc="mean")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        if i > 0 and ax.get_legend() is not None:
            ax.get_legend().remove()
    if len(axes) > 0 and axes[0].get_legend() is not None:
        axes[0].legend(fontsize=8)
    fig.suptitle("Benchmark-grouped metrics")
    return _save(fig, out_dir / f"benchmark_grouped_metrics{_suffix(file_suffix)}.png")


def plot_sensitivity(sensitivity_df: pd.DataFrame, out_dir: Path, *, file_suffix: str = "") -> str | None:
    if sensitivity_df.empty:
        return None
    rank_df = sensitivity_df.copy()
    rank_df["rank"] = rank_df.groupby(["benchmark", "aggregation_method"])["U"].rank(
        ascending=False, method="dense"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for system_name, group in rank_df.groupby("system_name"):
        mean_rank = group.groupby("aggregation_method")["rank"].mean().sort_index()
        ax.plot(mean_rank.index.tolist(), mean_rank.values.tolist(), marker="o", label=system_name)
    ax.set_ylabel("Mean utility rank (lower is better)")
    ax.set_title("Sensitivity across aggregation methods")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out_dir / f"sensitivity_ranking{_suffix(file_suffix)}.png")


def plot_pareto_frontier(summary_df: pd.DataFrame, out_dir: Path, *, file_suffix: str = "") -> str | None:
    if summary_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    for benchmark, group in summary_df.groupby("benchmark"):
        ax.scatter(group["C"], group["Q"], label=benchmark, s=70)
    ax.set_xlabel("Cost composite C (lower is better)")
    ax.set_ylabel("Quality composite Q (higher is better)")
    ax.set_title("Quality-cost Pareto view")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_dir / f"quality_cost_pareto{_suffix(file_suffix)}.png")


def plot_mahalanobis_diagnostics(summary_df: pd.DataFrame, out_dir: Path, *, file_suffix: str = "") -> str | None:
    """Plot Mahalanobis distances to ideal points for quality and cost."""
    frame = summary_df.copy()
    required = ["Q_distance_to_ideal", "C_distance_to_ideal", "system_name"]
    if frame.empty or any(col not in frame.columns for col in required):
        return None
    frame = frame.dropna(subset=["Q_distance_to_ideal", "C_distance_to_ideal"])
    if frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 6))
    for benchmark, group in frame.groupby("benchmark"):
        ax.scatter(
            group["C_distance_to_ideal"],
            group["Q_distance_to_ideal"],
            s=70,
            label=benchmark,
        )
    ax.set_xlabel("Mahalanobis distance to cost ideal (lower better)")
    ax.set_ylabel("Mahalanobis distance to quality ideal (lower better)")
    ax.set_title("Mahalanobis ideal-point distance diagnostics")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_dir / f"mahalanobis_distance_diagnostics{_suffix(file_suffix)}.png")
