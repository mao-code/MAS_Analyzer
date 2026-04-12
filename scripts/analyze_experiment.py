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


PASS_AT_K_COLUMNS = ("pass_at_1", "pass_at_3", "pass_at_5", "pass_at_8")
DELTA_HEATMAP_COLUMNS = (
    ("mean_success_rate_delta_vs_sas", "success_rate"),
    ("mean_stability_delta_vs_sas", "stability"),
    ("mean_tokens_total_delta_vs_sas", "tokens_total"),
    ("mean_cost_per_success_delta_vs_sas", "cost_per_success"),
)


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _plot_dir(output_dir: Path, benchmark: str, category: str) -> Path:
    return output_dir / benchmark / category


def _save_fig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.resolve())


def _ordered_systems(
    frame: pd.DataFrame,
    *,
    success_col: str,
    stability_col: str,
    tokens_col: str,
) -> list[str]:
    summary = (
        frame.groupby("system_label", as_index=False)
        .agg(
            order_success=(success_col, "mean"),
            order_stability=(stability_col, "mean"),
            order_tokens=(tokens_col, "mean"),
        )
        .sort_values(
            by=["order_success", "order_stability", "order_tokens", "system_label"],
            ascending=[False, False, True, True],
        )
    )
    ordered = summary["system_label"].tolist()
    if "sas" in ordered:
        ordered = ["sas"] + [item for item in ordered if item != "sas"]
    return ordered


def _maybe_add_alias(
    frame: pd.DataFrame,
    target: str,
    candidates: list[str],
) -> None:
    if target in frame.columns:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        return
    for candidate in candidates:
        if candidate in frame.columns:
            frame[target] = pd.to_numeric(frame[candidate], errors="coerce")
            return


def normalize_task_metrics(task_df: pd.DataFrame) -> pd.DataFrame:
    frame = task_df.copy()

    alias_map = {
        "eval_avg_score": ["eval_avg_score"],
        "success_rate": ["success_rate", "Q1_success_rate", "eval_success_rate"],
        "tokens_total": ["tokens_total", "C2_tokens_total"],
        "tool_calls_total": ["tool_calls_total", "C4_tool_calls_total"],
        "tool_error_rate": ["tool_error_rate", "D1_tool_error_rate"],
        "communication_count": ["communication_count", "D2_communication_count"],
        "handoff_count": ["handoff_count", "D3_handoff_count"],
        "agent_to_agent_communication_count": [
            "agent_to_agent_communication_count",
            "D2_agent_to_agent_communication_count",
        ],
        "system_mediated_communication_count": [
            "system_mediated_communication_count",
            "D2_system_mediated_communication_count",
        ],
    }
    for target, candidates in alias_map.items():
        _maybe_add_alias(frame, target, candidates)

    if "stability" in frame.columns:
        frame["stability"] = pd.to_numeric(frame["stability"], errors="coerce")
    elif "R1_success_var" in frame.columns:
        success_var = pd.to_numeric(frame["R1_success_var"], errors="coerce")
        run_counts = (
            pd.to_numeric(frame["runs"], errors="coerce")
            if "runs" in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        frame["stability"] = np.where(
            run_counts >= 2,
            np.clip(1.0 - (success_var / 0.25), 0.0, 1.0),
            np.nan,
        )

    if "tokens_cv" in frame.columns:
        frame["tokens_cv"] = pd.to_numeric(frame["tokens_cv"], errors="coerce")
    elif "R3_tokens_var" in frame.columns and "tokens_total" in frame.columns:
        token_var = pd.to_numeric(frame["R3_tokens_var"], errors="coerce")
        token_mean = pd.to_numeric(frame["tokens_total"], errors="coerce")
        run_counts = (
            pd.to_numeric(frame["runs"], errors="coerce")
            if "runs" in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        frame["tokens_cv"] = np.where(
            (run_counts >= 2) & (token_mean > 0),
            np.sqrt(token_var) / token_mean,
            np.nan,
        )

    if "cost_per_success" in frame.columns:
        frame["cost_per_success"] = pd.to_numeric(frame["cost_per_success"], errors="coerce")
    elif "tokens_total" in frame.columns and "success_rate" in frame.columns:
        tokens_total = pd.to_numeric(frame["tokens_total"], errors="coerce")
        success_rate = pd.to_numeric(frame["success_rate"], errors="coerce")
        frame["cost_per_success"] = np.where(
            success_rate > 0,
            tokens_total / success_rate,
            np.nan,
        )

    if "pass_at_1" not in frame.columns and "success_rate" in frame.columns:
        frame["pass_at_1"] = pd.to_numeric(frame["success_rate"], errors="coerce")
    for column in PASS_AT_K_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in [
        "eval_avg_score",
        "success_rate",
        "tokens_total",
        "tool_calls_total",
        "tool_error_rate",
        "communication_count",
        "handoff_count",
        "agent_to_agent_communication_count",
        "system_mediated_communication_count",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


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
    return normalize_task_metrics(pd.concat(frames, ignore_index=True))


def aggregate_system_metrics(task_df: pd.DataFrame) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {
        "task_count": ("task_id", "nunique"),
        "avg_eval_score": ("eval_avg_score", "mean"),
        "median_eval_score": ("eval_avg_score", "median"),
        "avg_success_rate": ("success_rate", "mean"),
        "avg_stability": ("stability", "mean"),
        "avg_tokens_total": ("tokens_total", "mean"),
        "median_tokens_total": ("tokens_total", "median"),
        "avg_cost_per_success": ("cost_per_success", "mean"),
        "avg_tokens_cv": ("tokens_cv", "mean"),
        "avg_tool_calls_total": ("tool_calls_total", "mean"),
        "avg_tool_error_rate": ("tool_error_rate", "mean"),
        "avg_communication_count": ("communication_count", "mean"),
        "avg_handoff_count": ("handoff_count", "mean"),
    }
    if "agent_to_agent_communication_count" in task_df.columns:
        agg_spec["avg_agent_to_agent_communication_count"] = (
            "agent_to_agent_communication_count",
            "mean",
        )
    if "system_mediated_communication_count" in task_df.columns:
        agg_spec["avg_system_mediated_communication_count"] = (
            "system_mediated_communication_count",
            "mean",
        )
    for column in PASS_AT_K_COLUMNS:
        if column in task_df.columns:
            agg_spec[f"avg_{column}"] = (column, "mean")

    return (
        task_df.groupby(["benchmark", "system_label", "topology"], as_index=False)
        .agg(**agg_spec)
        .sort_values(
            by=["benchmark", "avg_success_rate", "avg_stability", "avg_tokens_total", "system_label"],
            ascending=[True, False, False, True, True],
        )
    )


def compute_vs_sas(task_df: pd.DataFrame) -> pd.DataFrame:
    baseline = task_df[task_df["system_label"] == "sas"][
        [
            "benchmark",
            "task_id",
            "eval_avg_score",
            "success_rate",
            "stability",
            "tokens_total",
            "cost_per_success",
        ]
    ].rename(
        columns={
            "eval_avg_score": "sas_eval_avg_score",
            "success_rate": "sas_success_rate",
            "stability": "sas_stability",
            "tokens_total": "sas_tokens_total",
            "cost_per_success": "sas_cost_per_success",
        }
    )

    merged = task_df.merge(baseline, on=["benchmark", "task_id"], how="left")
    if merged["sas_success_rate"].isna().all():
        return pd.DataFrame()

    merged = merged[merged["system_label"] != "sas"].copy()
    merged["eval_score_delta_vs_sas"] = merged["eval_avg_score"] - merged["sas_eval_avg_score"]
    merged["success_rate_delta_vs_sas"] = merged["success_rate"] - merged["sas_success_rate"]
    merged["stability_delta_vs_sas"] = merged["stability"] - merged["sas_stability"]
    merged["tokens_total_delta_vs_sas"] = merged["tokens_total"] - merged["sas_tokens_total"]
    merged["cost_per_success_delta_vs_sas"] = (
        merged["cost_per_success"] - merged["sas_cost_per_success"]
    )
    return merged


def aggregate_vs_sas(vs_sas_df: pd.DataFrame) -> pd.DataFrame:
    if vs_sas_df.empty:
        return pd.DataFrame()
    return (
        vs_sas_df.groupby(["benchmark", "system_label"], as_index=False)
        .agg(
            task_count=("task_id", "nunique"),
            mean_eval_score_delta_vs_sas=("eval_score_delta_vs_sas", "mean"),
            mean_success_rate_delta_vs_sas=("success_rate_delta_vs_sas", "mean"),
            mean_stability_delta_vs_sas=("stability_delta_vs_sas", "mean"),
            mean_tokens_total_delta_vs_sas=("tokens_total_delta_vs_sas", "mean"),
            mean_cost_per_success_delta_vs_sas=("cost_per_success_delta_vs_sas", "mean"),
        )
        .sort_values(
            by=[
                "benchmark",
                "mean_success_rate_delta_vs_sas",
                "mean_stability_delta_vs_sas",
                "system_label",
            ],
            ascending=[True, False, False, True],
        )
    )


def _color_for_system(system_label: str) -> str:
    return "#1f77b4" if system_label == "sas" else "#2ca02c"


def _save_pass_at_k_chart(frame: pd.DataFrame, *, benchmark: str, out_dir: Path) -> str | None:
    pass_columns = [
        (f"avg_{column}", int(column.rsplit("_", 1)[1]))
        for column in PASS_AT_K_COLUMNS
        if f"avg_{column}" in frame.columns and not frame[f"avg_{column}"].dropna().empty
    ]
    if not pass_columns:
        return None

    systems = _ordered_systems(
        frame,
        success_col="avg_success_rate",
        stability_col="avg_stability",
        tokens_col="avg_tokens_total",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for system_label in systems:
        system_rows = frame[frame["system_label"] == system_label]
        if system_rows.empty:
            continue
        row = system_rows.iloc[0]
        x_values: list[int] = []
        y_values: list[float] = []
        for column, k in pass_columns:
            value = row[column]
            if pd.isna(value):
                continue
            x_values.append(k)
            y_values.append(float(value))
        if not x_values:
            continue
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            color=_color_for_system(system_label),
            label=system_label,
        )
    if not ax.lines:
        plt.close(fig)
        return None

    ax.set_title(f"{benchmark}: pass@k Reliability")
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([k for _, k in pass_columns])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_pass_at_k.png"
    return _save_fig(fig, path)


def _save_success_vs_tokens_frontier(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = frame.dropna(subset=["avg_tokens_total", "avg_success_rate"]).copy()
    if subset.empty:
        return None

    systems = _ordered_systems(
        subset,
        success_col="avg_success_rate",
        stability_col="avg_stability",
        tokens_col="avg_tokens_total",
    )
    subset["system_label"] = pd.Categorical(
        subset["system_label"], categories=systems, ordered=True
    )
    subset = subset.sort_values("system_label")

    fig, ax = plt.subplots(figsize=(8, 6))
    stability_values = subset["avg_stability"].to_numpy(dtype=float)
    finite_stability = stability_values[np.isfinite(stability_values)]
    color_norm = None
    if finite_stability.size:
        color_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        color_map = plt.cm.viridis
    else:
        color_map = None

    for _, row in subset.iterrows():
        system_label = str(row["system_label"])
        color = _color_for_system(system_label)
        if color_map is not None and not pd.isna(row["avg_stability"]):
            color = color_map(color_norm(float(row["avg_stability"])))
        ax.scatter(
            float(row["avg_tokens_total"]),
            float(row["avg_success_rate"]),
            s=220 if system_label == "sas" else 120,
            marker="*" if system_label == "sas" else "o",
            color=color,
            edgecolors="#111111",
            linewidths=0.8,
        )
        ax.annotate(
            system_label,
            (float(row["avg_tokens_total"]), float(row["avg_success_rate"])),
            xytext=(5, 5),
            textcoords="offset points",
        )

    if color_map is not None and color_norm is not None:
        mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
        colorbar = fig.colorbar(mappable, ax=ax)
        colorbar.set_label("stability")

    ax.set_title(f"{benchmark}: success_rate vs tokens_total")
    ax.set_xlabel("tokens_total")
    ax.set_ylabel("success_rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_success_vs_tokens_frontier.png"
    return _save_fig(fig, path)


def _format_delta(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if abs(value) >= 100:
        return f"{value:+.0f}"
    if abs(value) >= 10:
        return f"{value:+.1f}"
    return f"{value:+.2f}"


def _save_vs_sas_delta_heatmap(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = frame[frame["benchmark"] == benchmark].copy()
    if subset.empty:
        return None

    metric_pairs = [
        (column, label)
        for column, label in DELTA_HEATMAP_COLUMNS
        if column in subset.columns and not subset[column].dropna().empty
    ]
    if not metric_pairs:
        return None

    systems = subset["system_label"].tolist()
    matrix = np.array(
        [
            [
                float(subset.loc[subset["system_label"] == system_label, column].iloc[0])
                for column, _ in metric_pairs
            ]
            for system_label in systems
        ],
        dtype=float,
    )
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return None
    vmax = float(np.nanmax(np.abs(finite)))
    vmax = max(vmax, 1e-9)

    fig, ax = plt.subplots(figsize=(max(7, len(metric_pairs) * 2.0), max(4, len(systems) * 0.7)))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{benchmark}: MAS vs SAS deltas")
    ax.set_xticks(np.arange(len(metric_pairs)))
    ax.set_xticklabels([label for _, label in metric_pairs], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(systems)))
    ax.set_yticklabels(systems)
    for row_idx, system_label in enumerate(systems):
        for col_idx, _ in enumerate(metric_pairs):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                _format_delta(value),
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("delta vs SAS")
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_vs_sas_delta_heatmap.png"
    return _save_fig(fig, path)


def _save_cost_predictability_chart(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = frame.copy()
    if subset[["avg_cost_per_success", "avg_tokens_cv"]].isna().all().all():
        return None

    systems = _ordered_systems(
        subset,
        success_col="avg_success_rate",
        stability_col="avg_stability",
        tokens_col="avg_tokens_total",
    )
    subset["system_label"] = pd.Categorical(
        subset["system_label"], categories=systems, ordered=True
    )
    subset = subset.sort_values("system_label")
    colors = [_color_for_system(system_label) for system_label in subset["system_label"]]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(systems) * 1.5), 5))
    panels = [
        ("avg_cost_per_success", "cost_per_success"),
        ("avg_tokens_cv", "tokens_cv"),
    ]
    drawn = False
    for ax, (column, label) in zip(axes, panels, strict=True):
        values = subset[column].to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            ax.axis("off")
            ax.text(0.5, 0.5, f"{label} unavailable", ha="center", va="center")
            continue
        drawn = True
        ax.bar(subset["system_label"].astype(str), values, color=colors)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)

    if not drawn:
        plt.close(fig)
        return None

    fig.suptitle(f"{benchmark}: Cost Predictability")
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_cost_predictability.png"
    return _save_fig(fig, path)


def _save_coordination_breakdown_chart(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = frame.copy()
    required = [
        "avg_agent_to_agent_communication_count",
        "avg_system_mediated_communication_count",
        "avg_handoff_count",
    ]
    if not any(column in subset.columns for column in required):
        return None
    available_columns = [column for column in required if column in subset.columns]
    if subset[available_columns].isna().all().all():
        return None

    systems = _ordered_systems(
        subset,
        success_col="avg_success_rate",
        stability_col="avg_stability",
        tokens_col="avg_tokens_total",
    )
    subset["system_label"] = pd.Categorical(
        subset["system_label"], categories=systems, ordered=True
    )
    subset = subset.sort_values("system_label")

    labels = subset["system_label"].astype(str).tolist()
    agent_comm = subset.get(
        "avg_agent_to_agent_communication_count",
        pd.Series(0.0, index=subset.index),
    ).fillna(0.0).to_numpy(dtype=float)
    system_comm = subset.get(
        "avg_system_mediated_communication_count",
        pd.Series(0.0, index=subset.index),
    ).fillna(0.0).to_numpy(dtype=float)
    handoffs = subset.get(
        "avg_handoff_count",
        pd.Series(0.0, index=subset.index),
    ).fillna(0.0).to_numpy(dtype=float)
    colors = [_color_for_system(system_label) for system_label in labels]
    x_values = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.5), 5))
    axes[0].bar(labels, agent_comm, label="agent_to_agent", color="#66c2a5")
    axes[0].bar(
        labels,
        system_comm,
        bottom=agent_comm,
        label="system_mediated",
        color="#fc8d62",
    )
    axes[0].set_title("communication breakdown")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x_values, handoffs, color=colors)
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(labels, rotation=25)
    axes[1].set_title("handoff_count")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"{benchmark}: Coordination Diagnostics")
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_coordination_breakdown.png"
    return _save_fig(fig, path)


def write_report(
    *,
    experiment_root: Path,
    out_dir: Path,
    task_df: pd.DataFrame,
    system_df: pd.DataFrame,
    vs_sas_df: pd.DataFrame,
    plots: dict[str, list[str]],
) -> Path:
    lines = [
        f"# Experiment Analysis: {experiment_root.name}",
        "",
        f"- Experiment Root: `{experiment_root}`",
        f"- Benchmarks: {task_df['benchmark'].nunique()}",
        f"- Topologies: {task_df['system_label'].nunique()}",
        f"- Task rows: {len(task_df)}",
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
            by=["avg_success_rate", "avg_stability", "avg_tokens_total", "system_label"],
            ascending=[False, False, True, True],
        ).iloc[0]
        lines.append(
            f"- `{benchmark}`: strongest paper-aligned system is `{best['system_label']}` "
            f"with success `{best['avg_success_rate']:.3f}`, stability "
            f"`{best['avg_stability']:.3f}` and mean tokens `{best['avg_tokens_total']:.1f}`."
        )
        if "sas" in benchmark_systems["system_label"].values:
            sas_row = benchmark_systems[benchmark_systems["system_label"] == "sas"].iloc[0]
            lines.append(
                f"- `{benchmark}` SAS baseline: success `{sas_row['avg_success_rate']:.3f}`, "
                f"stability `{sas_row['avg_stability']:.3f}`, mean tokens "
                f"`{sas_row['avg_tokens_total']:.1f}`."
            )
        benchmark_vs_sas = vs_sas_df[vs_sas_df["benchmark"] == benchmark]
        if not benchmark_vs_sas.empty:
            leader = benchmark_vs_sas.sort_values(
                by=["mean_success_rate_delta_vs_sas", "mean_stability_delta_vs_sas", "system_label"],
                ascending=[False, False, True],
            ).iloc[0]
            lines.append(
                f"- `{benchmark}` largest success-rate lift vs SAS: `{leader['system_label']}` "
                f"at `{leader['mean_success_rate_delta_vs_sas']:+.3f}`."
            )
        lines.append("")

    display_columns = [
        "benchmark",
        "system_label",
        "task_count",
        "avg_eval_score",
        "avg_success_rate",
        "avg_stability",
        "avg_pass_at_1",
        "avg_pass_at_3",
        "avg_pass_at_5",
        "avg_pass_at_8",
        "avg_tokens_total",
        "avg_cost_per_success",
        "avg_tokens_cv",
        "avg_tool_calls_total",
        "avg_communication_count",
        "avg_handoff_count",
    ]
    lines.extend(["## System Table", ""])
    lines.append(
        render_table(system_df[[column for column in display_columns if column in system_df.columns]].round(3))
    )
    lines.append("")

    if not vs_sas_df.empty:
        lines.extend(["## Delta vs SAS", ""])
        lines.append(render_table(vs_sas_df.round(3)))
        lines.append("")

    lines.extend(["## Plot Inventory", ""])
    for benchmark, benchmark_plots in plots.items():
        for path in benchmark_plots:
            lines.append(f"- `{benchmark}`: `{path}`")
    lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- `stability` and `tokens_cv` are left blank when a task has fewer than two runs.",
            "- `pass_at_k` is left blank when the task has fewer than `k` repeated runs.",
            "- `cost_per_success` is left blank when `success_rate = 0`.",
        ]
    )

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

    plots: dict[str, list[str]] = {}
    for benchmark in sorted(task_df["benchmark"].unique()):
        benchmark_system_df = system_df[system_df["benchmark"] == benchmark].copy()
        benchmark_root = output_dir / benchmark
        benchmark_plot_paths: list[str] = []
        for plot_path in [
            _save_pass_at_k_chart(benchmark_system_df, benchmark=benchmark, out_dir=benchmark_root / "THEORY"),
            _save_success_vs_tokens_frontier(
                benchmark_system_df,
                benchmark=benchmark,
                out_dir=benchmark_root / "RQ1",
            ),
            _save_vs_sas_delta_heatmap(
                vs_sas_system_df,
                benchmark=benchmark,
                out_dir=benchmark_root / "RQ1",
            ),
            _save_cost_predictability_chart(
                benchmark_system_df,
                benchmark=benchmark,
                out_dir=benchmark_root / "THEORY",
            ),
            _save_coordination_breakdown_chart(
                benchmark_system_df,
                benchmark=benchmark,
                out_dir=benchmark_root / "RQ2",
            ),
        ]:
            if plot_path:
                benchmark_plot_paths.append(plot_path)
        plots[benchmark] = benchmark_plot_paths

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
                    by=["avg_success_rate", "avg_stability", "avg_tokens_total", "system_label"],
                    ascending=[False, False, True, True],
                )
                .iloc[0][
                    [
                        "system_label",
                        "avg_eval_score",
                        "avg_success_rate",
                        "avg_stability",
                        "avg_tokens_total",
                        "avg_cost_per_success",
                    ]
                ]
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
    parser = argparse.ArgumentParser(
        description="Analyze a hierarchical experiment root and generate paper-aligned comparison plots."
    )
    parser.add_argument("--experiment-root", required=True, help="Path to artifacts/full_experiment/<experiment-id>")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for analysis artifacts. Defaults to <experiment-root>/Plot",
    )
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else experiment_root / "Plot"
    )
    analyze_experiment(experiment_root, output_dir)
    print(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
