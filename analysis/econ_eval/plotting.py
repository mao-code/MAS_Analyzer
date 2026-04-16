from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.resolve())


def _suffix(name: str) -> str:
    return f"_{name}" if name else ""


def _marker_for_system(name: str) -> str:
    key = str(name).lower()
    mapping = {
        "sas": "X",
        "orchestrator_with_discussion": "o",
        "orchestrator_no_discussion": "s",
        "orchestrator_tree_structure": "^",
        "group_chat_debate": "D",
        "fully_linked_debate": "P",
        "only_voting": "v",
    }
    return mapping.get(key, "o")


def _benchmark_color_map(benchmarks: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.cm.get_cmap("tab10")
    return {benchmark: cmap(i % 10) for i, benchmark in enumerate(sorted(benchmarks))}


def _topology_handles(topologies: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=_marker_for_system(topology),
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="#222222",
            markersize=7,
            linestyle="None",
            label=topology,
        )
        for topology in sorted(set(topologies))
    ]


def _benchmark_handles(color_map: dict[str, tuple[float, float, float, float]]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="#222222",
            markersize=7,
            linestyle="None",
            label=benchmark,
        )
        for benchmark, color in color_map.items()
    ]


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
    ax.text(
        0.01,
        0.98,
        "Each bar pair = one benchmark × system configuration",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        alpha=0.8,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return _save(fig, out_dir / f"utility_sas_vs_mas{_suffix(file_suffix)}.png")


def plot_gain_cost_plane(comp_df: pd.DataFrame, out_dir: Path, *, file_suffix: str = "") -> str | None:
    if comp_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    color_map = _benchmark_color_map(comp_df["benchmark"].dropna().astype(str).unique().tolist())
    topologies: list[str] = []
    for benchmark, group in comp_df.groupby("benchmark"):
        for _, row in group.iterrows():
            marker = _marker_for_system(row.get("system_name", ""))
            topologies.append(str(row.get("system_name", "")))
            ax.scatter(
                float(row["G"]),
                float(row["K"]),
                s=65,
                marker=marker,
                alpha=0.85,
                color=color_map.get(str(benchmark), "#1f77b4"),
            )
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
    topo_legend = ax.legend(
        handles=_topology_handles(topologies),
        title="Topology",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(topo_legend)
    ax.legend(
        handles=_benchmark_handles(color_map),
        title="Benchmark",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.56),
    )
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
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        if i > 0 and ax.get_legend() is not None:
            ax.get_legend().remove()
    if len(axes) > 0 and axes[0].get_legend() is not None:
        axes[0].legend(fontsize=8)
    fig.suptitle("Benchmark-grouped metrics (each bar = benchmark-average for one system)")
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
    ax.text(
        0.01,
        0.98,
        "Each line = one system; rank is averaged over benchmarks",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        alpha=0.8,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out_dir / f"sensitivity_ranking{_suffix(file_suffix)}.png")


def plot_pareto_frontier(summary_df: pd.DataFrame, out_dir: Path, *, file_suffix: str = "") -> str | None:
    if summary_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    color_map = _benchmark_color_map(summary_df["benchmark"].dropna().astype(str).unique().tolist())
    topologies: list[str] = []
    for benchmark, group in summary_df.groupby("benchmark"):
        for _, row in group.iterrows():
            marker = _marker_for_system(row.get("system_name", ""))
            topologies.append(str(row.get("system_name", "")))
            ax.scatter(
                float(row["C"]),
                float(row["Q"]),
                s=60,
                marker=marker,
                alpha=0.85,
                color=color_map.get(str(benchmark), "#1f77b4"),
            )
    ax.set_xlabel("Cost composite C (lower is better)")
    ax.set_ylabel("Quality composite Q (higher is better)")
    ax.set_title("Quality-cost Pareto view")
    ax.grid(alpha=0.3)
    topo_legend = ax.legend(
        handles=_topology_handles(topologies),
        title="Topology",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(topo_legend)
    ax.legend(
        handles=_benchmark_handles(color_map),
        title="Benchmark",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.56),
    )
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

    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    color_map = _benchmark_color_map(frame["benchmark"].dropna().astype(str).unique().tolist())
    topologies: list[str] = []
    for benchmark, group in frame.groupby("benchmark"):
        for _, row in group.iterrows():
            marker = _marker_for_system(row.get("system_name", ""))
            topologies.append(str(row.get("system_name", "")))
            ax.scatter(
                float(row["C_distance_to_ideal"]),
                float(row["Q_distance_to_ideal"]),
                s=60,
                marker=marker,
                alpha=0.85,
                color=color_map.get(str(benchmark), "#1f77b4"),
            )
    ax.set_xlabel("Mahalanobis distance to cost ideal (lower better)")
    ax.set_ylabel("Mahalanobis distance to quality ideal (lower better)")
    ax.set_title("Mahalanobis ideal-point distance diagnostics")
    ax.grid(alpha=0.3)
    topo_legend = ax.legend(
        handles=_topology_handles(topologies),
        title="Topology",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(topo_legend)
    ax.legend(
        handles=_benchmark_handles(color_map),
        title="Benchmark",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.56),
    )
    return _save(fig, out_dir / f"mahalanobis_distance_diagnostics{_suffix(file_suffix)}.png")
