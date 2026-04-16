from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analysis.econ_eval.regime_classification import classify_regime


PASS_AT_K_COLUMNS = ("pass_at_1", "pass_at_3", "pass_at_5", "pass_at_8")
GENERATED_PLOT_SUFFIXES = (
    "_system_scorecard.png",
    "_success_vs_tokens_frontier.png",
    "_vs_sas_tradeoff.png",
    "_pass_at_k.png",
    "_coordination_breakdown.png",
    "_vs_sas_delta_heatmap.png",
    "_cost_predictability.png",
)
SYSTEM_PLOT_STYLES: dict[str, dict[str, str]] = {
    "sas": {
        "short": "SAS",
        "axis": "SAS",
        "color": "#1f77b4",
    },
    "only_voting": {
        "short": "Only Vote",
        "axis": "Only\nVoting",
        "color": "#ff7f0e",
    },
    "fully_linked_debate": {
        "short": "Linked Debate",
        "axis": "Linked\nDebate",
        "color": "#2ca02c",
    },
    "group_chat_debate": {
        "short": "Group Chat",
        "axis": "Group\nChat",
        "color": "#d62728",
    },
    "orchestrator_tree_structure": {
        "short": "Orch Tree",
        "axis": "Orch.\nTree",
        "color": "#9467bd",
    },
    "orchestrator_no_discussion": {
        "short": "Orch No Disc",
        "axis": "Orch.\nNo Disc.",
        "color": "#8c564b",
    },
    "orchestrator_with_discussion": {
        "short": "Orch + Disc",
        "axis": "Orch.\n+ Disc.",
        "color": "#e377c2",
    },
}


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _plot_dir(output_dir: Path, benchmark: str, category: str) -> Path:
    return output_dir / benchmark / category


def _save_fig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.resolve())


def _row_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index)
    values = frame[available].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1, skipna=True)


def _minmax_per_benchmark(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for benchmark, index in out.groupby("benchmark").groups.items():
        benchmark_frame = out.loc[index]
        for column in columns:
            if column not in benchmark_frame.columns:
                continue
            values = pd.to_numeric(benchmark_frame[column], errors="coerce")
            min_v = values.min(skipna=True)
            max_v = values.max(skipna=True)
            norm_col = f"{column}__norm_run"
            if pd.isna(min_v) or pd.isna(max_v):
                out.loc[index, norm_col] = np.nan
            elif max_v - min_v <= 1e-12:
                out.loc[index, norm_col] = 0.0
            else:
                out.loc[index, norm_col] = (values - min_v) / (max_v - min_v)
    return out


def _prepare_run_level_frame(run_df: pd.DataFrame) -> pd.DataFrame:
    frame = run_df.copy()
    frame["quality_proxy"] = _row_mean(frame, ["success", "completion", "score"]).clip(0.0, 1.0)
    frame = _minmax_per_benchmark(
        frame,
        ["tokens_total", "tool_calls_total", "communication_count", "handoff_count"],
    )
    cost_norm_cols = [
        "tokens_total__norm_run",
        "tool_calls_total__norm_run",
        "communication_count__norm_run",
        "handoff_count__norm_run",
    ]
    frame["cost_proxy"] = frame[cost_norm_cols].mean(axis=1, skipna=True).clip(0.0, 1.0)
    frame["utility_proxy"] = frame["quality_proxy"] - frame["cost_proxy"]
    frame["score_effective"] = pd.to_numeric(frame.get("score"), errors="coerce")
    frame["score_effective"] = frame["score_effective"].fillna(pd.to_numeric(frame.get("success"), errors="coerce"))
    return frame


def _load_run_rows(experiment_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        benchmark = benchmark_dir.name
        for system_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
            system_label = system_dir.name
            for task_dir in sorted(path for path in system_dir.iterdir() if path.is_dir()):
                task_id = task_dir.name
                for trace_path in sorted(task_dir.glob("run_*.trace_metrics.json")):
                    with trace_path.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    metrics = payload.get("metrics", {})
                    runtime = payload.get("runtime", {})
                    run_index = payload.get("run_index", runtime.get("run_index"))
                    topology = runtime.get("topology", system_label)
                    system_type = "SAS" if str(topology).lower() == "sas" else "MAS"
                    tool_calls_total = float(metrics.get("tool_calls_total", 0.0) or 0.0)
                    tool_error_count = float(metrics.get("tool_fail_total", 0.0) or 0.0)
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "system_label": system_label,
                            "topology": topology,
                            "system_type": system_type,
                            "task_id": task_id,
                            "run_index": int(run_index) if run_index is not None else None,
                            "success": metrics.get("success"),
                            "completion": metrics.get("completion"),
                            "score": metrics.get("score"),
                            "tokens_total": metrics.get("tokens_total"),
                            "tool_calls_total": tool_calls_total,
                            "communication_count": metrics.get("communication_count"),
                            "communication_count_agent_to_agent": metrics.get(
                                "communication_count_agent_to_agent"
                            ),
                            "communication_count_system_mediated": metrics.get(
                                "communication_count_system_mediated"
                            ),
                            "handoff_count": metrics.get("handoff_count"),
                            "tool_error_count": tool_error_count,
                            "tool_error_rate": (
                                tool_error_count / tool_calls_total if tool_calls_total > 0 else 0.0
                            ),
                        }
                    )
    if not rows:
        raise ValueError(f"No run-level trace_metrics.json files found under {experiment_root}")
    frame = pd.DataFrame(rows)
    for column in [
        "success",
        "completion",
        "score",
        "tokens_total",
        "tool_calls_total",
        "communication_count",
        "communication_count_agent_to_agent",
        "communication_count_system_mediated",
        "handoff_count",
        "tool_error_count",
        "tool_error_rate",
        "run_index",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame

def _style_for_system(system_label: str) -> dict[str, str]:
    return SYSTEM_PLOT_STYLES.get(
        system_label,
        {
            "short": system_label.replace("_", " "),
            "axis": system_label.replace("_", "\n"),
            "color": "#7f7f7f",
        },
    )


def _short_system_label(system_label: str) -> str:
    return _style_for_system(system_label)["short"]


def _axis_system_label(system_label: str) -> str:
    return _style_for_system(system_label)["axis"]


def _format_compact_number(value: float, _: float | None = None) -> str:
    if not math.isfinite(value):
        return "NaN"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 10_000:
        return f"{value / 1_000:.0f}k"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _annotation_offsets() -> list[tuple[int, int]]:
    return [
        (8, 8),
        (8, -8),
        (-8, 8),
        (-8, -8),
        (14, 0),
        (-14, 0),
        (0, 14),
        (0, -14),
        (18, 10),
        (18, -10),
        (-18, 10),
        (-18, -10),
        (24, 0),
        (-24, 0),
        (30, 12),
        (30, -12),
        (-30, 12),
        (-30, -12),
        (36, 0),
        (-36, 0),
    ]


def _bbox_overlap_area(bbox_a: Any, bbox_b: Any) -> float:
    x0 = max(bbox_a.x0, bbox_b.x0)
    y0 = max(bbox_a.y0, bbox_b.y0)
    x1 = min(bbox_a.x1, bbox_b.x1)
    y1 = min(bbox_a.y1, bbox_b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _annotate_scatter_labels(
    ax: Any,
    fig: Any,
    points: list[tuple[float, float, str]],
) -> None:
    if not points:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    display_points = [
        (*point, ax.transData.transform((point[0], point[1])))
        for point in points
    ]
    display_points.sort(
        key=lambda item: sum(
            1.0 / max(np.linalg.norm(item[3] - other[3]), 1.0)
            for other in display_points
            if other is not item
        ),
        reverse=True,
    )

    occupied_bboxes = []
    for x, y, label, anchor_px in display_points:
        best_offset = _annotation_offsets()[0]
        best_bbox = None
        best_score = math.inf
        annotation = ax.annotate(
            label,
            (x, y),
            xytext=best_offset,
            textcoords="offset points",
            fontsize=8.5,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "none",
                "alpha": 0.82,
            },
            arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "#666666", "alpha": 0.65},
        )
        for dx, dy in _annotation_offsets():
            annotation.set_position((dx, dy))
            annotation.set_ha("left" if dx >= 0 else "right")
            annotation.set_va("bottom" if dy >= 0 else "top")
            fig.canvas.draw()
            bbox = annotation.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.03, 1.10)
            overlap_penalty = sum(_bbox_overlap_area(bbox, other) for other in occupied_bboxes)
            out_of_bounds = (
                max(0.0, axes_bbox.x0 - bbox.x0)
                + max(0.0, bbox.x1 - axes_bbox.x1)
                + max(0.0, axes_bbox.y0 - bbox.y0)
                + max(0.0, bbox.y1 - axes_bbox.y1)
            )
            label_center = np.array([(bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0])
            distance_penalty = float(np.linalg.norm(label_center - anchor_px))
            score = overlap_penalty * 1500.0 + out_of_bounds * 180.0 + distance_penalty
            if score < best_score:
                best_score = score
                best_offset = (dx, dy)
                best_bbox = bbox
        annotation.set_position(best_offset)
        annotation.set_ha("left" if best_offset[0] >= 0 else "right")
        annotation.set_va("bottom" if best_offset[1] >= 0 else "top")
        occupied_bboxes.append(best_bbox)


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
    return _style_for_system(system_label)["color"]


def _color_for_benchmark(benchmark: str) -> tuple[float, float, float, float]:
    cmap = plt.cm.get_cmap("tab10")
    idx = abs(hash(str(benchmark).lower())) % 10
    return cmap(idx)


def _marker_for_topology(name: str) -> str:
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


def _jitter(values: np.ndarray, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values + rng.normal(0.0, scale, size=len(values))


def _legend_handles_for_topologies(topologies: list[str]) -> list[Line2D]:
    handles: list[Line2D] = []
    for topology in sorted(set(topologies)):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=_marker_for_topology(topology),
                color="none",
                markerfacecolor="#666666",
                markeredgecolor="#222222",
                markersize=7,
                linestyle="None",
                label=str(topology),
            )
        )
    return handles


def _save_pass_at_k_chart(frame: pd.DataFrame, *, benchmark: str, out_dir: Path) -> str | None:
    pass_columns = [
        (f"avg_{column}", int(column.rsplit("_", 1)[1]))
        for column in PASS_AT_K_COLUMNS
        if f"avg_{column}" in frame.columns and not frame[f"avg_{column}"].dropna().empty
    ]
    if len(pass_columns) < 2:
        return None

    systems = _ordered_systems(
        frame,
        success_col="avg_success_rate",
        stability_col="avg_stability",
        tokens_col="avg_tokens_total",
    )
    fig, ax = plt.subplots(figsize=(9, 5.2))
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
            label=_short_system_label(system_label),
        )
    if not ax.lines:
        plt.close(fig)
        return None

    ax.set_title(f"{benchmark}: pass@k reliability")
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([k for _, k in pass_columns])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_pass_at_k_average.png"
    return _save_fig(fig, path)


def _save_run_utility_comparison(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = _prepare_run_level_frame(frame[frame["benchmark"] == benchmark].copy())
    if subset.empty or subset["utility_proxy"].isna().all():
        return None

    systems = _ordered_systems(
        subset,
        success_col="quality_proxy",
        stability_col="quality_proxy",
        tokens_col="tokens_total",
    )
    subset["system_label"] = pd.Categorical(
        subset["system_label"], categories=systems, ordered=True
    )
    subset = subset.sort_values("system_label")
    rng = np.random.default_rng(7)

    fig, ax = plt.subplots(figsize=(max(10, len(systems) * 1.2), 6))
    benchmark_color = _color_for_benchmark(benchmark)
    for idx, system_label in enumerate(systems):
        system_rows = subset[subset["system_label"] == system_label]
        if system_rows.empty:
            continue
        x = np.full(len(system_rows), idx, dtype=float) + rng.normal(0.0, 0.04, size=len(system_rows))
        y = _jitter(system_rows["utility_proxy"].to_numpy(dtype=float), 0.008, seed=13 + idx)
        ax.scatter(
            x,
            y,
            alpha=0.45,
            s=18,
            color=benchmark_color,
            marker=_marker_for_topology(system_label),
            edgecolors="none",
        )
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, rotation=25)
    ax.set_ylabel("Run-level utility proxy (quality_proxy - cost_proxy)")
    ax.set_title(f"{benchmark}: run-level utility comparison")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="y", alpha=0.3)
    topo_legend = ax.legend(
        handles=_legend_handles_for_topologies(systems),
        title="Topology",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(topo_legend)
    benchmark_handle = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=benchmark_color, markersize=7, label=benchmark)
    ]
    ax.legend(
        handles=benchmark_handle,
        title="Benchmark",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.62),
    )
    return _save_fig(fig, out_dir / f"{_sanitize_filename(benchmark)}_utility_sas_vs_mas_indivudal.png")


def _save_run_success_vs_tokens_frontier(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = frame[frame["benchmark"] == benchmark].copy()
    subset["score_effective"] = pd.to_numeric(subset["score"], errors="coerce")
    subset["score_effective"] = subset["score_effective"].fillna(
        pd.to_numeric(subset["success"], errors="coerce")
    )
    subset = subset.dropna(subset=["tokens_total", "score_effective"])
    if subset.empty:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    benchmark_color = _color_for_benchmark(benchmark)
    topologies: list[str] = []
    for i, (system_label, group) in enumerate(subset.groupby("system_label")):
        topologies.append(str(system_label))
        y = _jitter(group["score_effective"].to_numpy(dtype=float), 0.008, seed=31 + i)
        ax.scatter(
            group["tokens_total"],
            y,
            alpha=0.4,
            s=18,
            color=benchmark_color,
            marker=_marker_for_topology(system_label),
        )
    ax.set_title(f"{benchmark}: run-level success/score vs tokens")
    ax.set_xlabel("tokens_total")
    ax.set_ylabel("success / score")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(alpha=0.3)
    topo_legend = ax.legend(
        handles=_legend_handles_for_topologies(topologies),
        title="Topology",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(topo_legend)
    benchmark_handle = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=benchmark_color, markersize=7, label=benchmark)
    ]
    ax.legend(
        handles=benchmark_handle,
        title="Benchmark",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.62),
    )
    return _save_fig(fig, out_dir / f"{_sanitize_filename(benchmark)}_success_vs_tokens_frontier_indivudal.png")


def _save_run_gain_cost_plane(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = _prepare_run_level_frame(frame[frame["benchmark"] == benchmark].copy())
    sas = subset[subset["system_type"] == "SAS"][
        ["task_id", "run_index", "quality_proxy", "cost_proxy"]
    ].rename(
        columns={
            "quality_proxy": "sas_quality_proxy",
            "cost_proxy": "sas_cost_proxy",
        }
    )
    mas = subset[subset["system_type"] == "MAS"].copy()
    merged = mas.merge(sas, on=["task_id", "run_index"], how="left")
    merged = merged.dropna(subset=["sas_quality_proxy", "sas_cost_proxy"])
    if merged.empty:
        return None
    merged["G"] = merged["quality_proxy"] - merged["sas_quality_proxy"]
    merged["K"] = merged["cost_proxy"] - merged["sas_cost_proxy"]
    merged["collaboration_regime"] = merged.apply(
        lambda row: classify_regime(float(row["G"]), float(row["K"])),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    benchmark_color = _color_for_benchmark(benchmark)
    topologies: list[str] = []
    for i, (system_label, group) in enumerate(merged.groupby("system_label")):
        topologies.append(str(system_label))
        y = _jitter(group["K"].to_numpy(dtype=float), 0.006, seed=51 + i)
        ax.scatter(
            group["G"],
            y,
            alpha=0.4,
            s=18,
            color=benchmark_color,
            marker=_marker_for_topology(system_label),
        )
    lim = float(max(np.nanmax(np.abs(merged["G"])), np.nanmax(np.abs(merged["K"])), 1e-6))
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.2, label="G = K")
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlim(-lim * 1.05, lim * 1.05)
    ax.set_ylim(-lim * 1.05, lim * 1.05)
    ax.set_xlabel("Run-level collaboration gain G")
    ax.set_ylabel("Run-level coordination cost K")
    ax.set_title(f"{benchmark}: run-level gain-cost plane")
    ax.grid(alpha=0.3)
    topo_legend = ax.legend(
        handles=_legend_handles_for_topologies(topologies),
        title="Topology",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(topo_legend)
    benchmark_handle = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=benchmark_color, markersize=7, label=benchmark)
    ]
    ax.legend(
        handles=benchmark_handle,
        title="Benchmark",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.62),
    )
    return _save_fig(fig, out_dir / f"{_sanitize_filename(benchmark)}_gain_cost_plane_indivudal.png")


def _save_run_quality_cost_pareto(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = _prepare_run_level_frame(frame[frame["benchmark"] == benchmark].copy())
    if subset.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    benchmark_color = _color_for_benchmark(benchmark)
    topologies: list[str] = []
    for i, (system_label, group) in enumerate(subset.groupby("system_label")):
        topologies.append(str(system_label))
        y = _jitter(group["quality_proxy"].to_numpy(dtype=float), 0.008, seed=71 + i)
        ax.scatter(
            group["cost_proxy"],
            y,
            alpha=0.4,
            s=18,
            color=benchmark_color,
            marker=_marker_for_topology(system_label),
        )
    ax.set_xlabel("Run-level cost proxy C")
    ax.set_ylabel("Run-level quality proxy Q")
    ax.set_title(f"{benchmark}: run-level quality-cost Pareto view")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(alpha=0.3)
    topo_legend = ax.legend(
        handles=_legend_handles_for_topologies(topologies),
        title="Topology",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(topo_legend)
    benchmark_handle = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=benchmark_color, markersize=7, label=benchmark)
    ]
    ax.legend(
        handles=benchmark_handle,
        title="Benchmark",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.62),
    )
    return _save_fig(fig, out_dir / f"{_sanitize_filename(benchmark)}_quality_cost_pareto_indivudal.png")


def _mahalanobis_distance_matrix(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.zeros(matrix.shape[0], dtype=float)
    cov = np.cov(matrix, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov_inv = np.linalg.pinv(cov)
    diff = matrix - target
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))


def _save_run_mahalanobis_diagnostics(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = frame[frame["benchmark"] == benchmark].copy()
    quality_cols = ["success", "completion", "score"]
    cost_cols = ["tokens_total", "tool_calls_total", "communication_count", "handoff_count"]
    q_matrix = subset[quality_cols].apply(pd.to_numeric, errors="coerce")
    c_matrix = subset[cost_cols].apply(pd.to_numeric, errors="coerce")
    if q_matrix.dropna(how="all").empty or c_matrix.dropna(how="all").empty:
        return None
    q_matrix = q_matrix.fillna(q_matrix.mean(numeric_only=True)).fillna(0.0)
    c_matrix = c_matrix.fillna(c_matrix.mean(numeric_only=True)).fillna(0.0)

    q_arr = q_matrix.to_numpy(dtype=float)
    c_arr = c_matrix.to_numpy(dtype=float)
    q_ideal = np.ones(q_arr.shape[1], dtype=float)
    c_ideal = np.min(c_arr, axis=0)
    q_dist = _mahalanobis_distance_matrix(q_arr, q_ideal)
    c_dist = _mahalanobis_distance_matrix(c_arr, c_ideal)

    plot_frame = subset[["system_label"]].copy()
    plot_frame["q_dist"] = q_dist
    plot_frame["c_dist"] = c_dist

    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    benchmark_color = _color_for_benchmark(benchmark)
    topologies: list[str] = []
    for i, (system_label, group) in enumerate(plot_frame.groupby("system_label")):
        topologies.append(str(system_label))
        y = _jitter(group["q_dist"].to_numpy(dtype=float), 0.004, seed=91 + i)
        ax.scatter(
            group["c_dist"],
            y,
            alpha=0.4,
            s=18,
            color=benchmark_color,
            marker=_marker_for_topology(system_label),
        )
    ax.set_xlabel("Mahalanobis distance to cost ideal")
    ax.set_ylabel("Mahalanobis distance to quality ideal")
    ax.set_title(f"{benchmark}: run-level Mahalanobis diagnostics")
    ax.grid(alpha=0.3)
    topo_legend = ax.legend(
        handles=_legend_handles_for_topologies(topologies),
        title="Topology",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(topo_legend)
    benchmark_handle = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=benchmark_color, markersize=7, label=benchmark)
    ]
    ax.legend(
        handles=benchmark_handle,
        title="Benchmark",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.62),
    )
    return _save_fig(fig, out_dir / f"{_sanitize_filename(benchmark)}_mahalanobis_distance_diagnostics_indivudal.png")


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


    fig, ax = plt.subplots(figsize=(10, 6.5))
    stability_values = subset["avg_stability"].to_numpy(dtype=float)
    finite_stability = stability_values[np.isfinite(stability_values)]
    color_norm = None
    if finite_stability.size:
        color_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        color_map = plt.cm.viridis
    else:
        color_map = None

    benchmark_color = _color_for_benchmark(benchmark)
    topologies: list[str] = []
    mas_labels: list[tuple[float, float, str]] = []

    for _, row in subset.iterrows():
        system_label = str(row["system_label"])
        topologies.append(system_label)
        x = float(row["avg_tokens_total"])
        y = float(row["avg_success_rate"])
        color = _color_for_system(system_label)
        if color_map is not None and not pd.isna(row["avg_stability"]):
            color = color_map(color_norm(float(row["avg_stability"])))
        ax.scatter(
            x,
            y,
            s=220 if system_label == "sas" else 120,
            marker=_marker_for_topology(system_label),
            color=color,
            edgecolors="#111111",
            linewidths=0.8,
        )
        if system_label != "sas":
            mas_labels.append((x, y, _short_system_label(system_label)))

    ax.margins(x=0.16, y=0.12)
    _annotate_scatter_labels(ax, fig, mas_labels)
    if color_map is not None and color_norm is not None:
        mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
        colorbar = fig.colorbar(mappable, ax=ax)
        colorbar.set_label("stability")

    ax.set_title(f"{benchmark}: success_rate vs tokens_total")
    ax.set_xlabel("tokens_total")
    ax.set_ylabel("success_rate")
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_compact_number))
    ax.grid(alpha=0.3)
    topo_legend = ax.legend(
        handles=_legend_handles_for_topologies(topologies),
        title="Topology",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(topo_legend)
    benchmark_handle = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=benchmark_color, markersize=7, label=benchmark)
    ]
    ax.legend(
        handles=benchmark_handle,
        title="Benchmark",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.62),
    )
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_success_vs_tokens_frontier_average.png"
    return _save_fig(fig, path)


def _save_system_scorecard(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = frame.copy()
    if subset.empty:
        return None

    subset = subset.dropna(subset=["avg_success_rate", "avg_tokens_total"], how="all")
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

    labels = [_short_system_label(system_label) for system_label in subset["system_label"].astype(str)]
    colors = [_color_for_system(system_label) for system_label in subset["system_label"].astype(str)]
    success_values = subset["avg_success_rate"].fillna(0.0).to_numpy(dtype=float)
    stability_values = subset.get("avg_stability", pd.Series(np.nan, index=subset.index)).to_numpy(dtype=float)
    token_values = subset["avg_tokens_total"].fillna(0.0).to_numpy(dtype=float)
    y_values = np.arange(len(subset))

    retry_column = None
    retry_label = None
    for column in reversed(PASS_AT_K_COLUMNS):
        avg_column = f"avg_{column}"
        if avg_column in subset.columns and not subset[avg_column].dropna().empty:
            k = int(column.rsplit("_", 1)[1])
            if k > 1:
                retry_column = avg_column
                retry_label = f"pass@{k}"
                break

    fig_height = max(4.8, len(subset) * 0.68)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    quality_ax, cost_ax = axes

    quality_ax.barh(y_values, success_values, color=colors, alpha=0.88)
    quality_ax.set_title("quality")
    quality_ax.set_xlabel("rate")
    quality_ax.set_xlim(0.0, 1.05)
    quality_ax.set_yticks(y_values)
    quality_ax.set_yticklabels(labels)
    quality_ax.invert_yaxis()
    quality_ax.grid(axis="x", alpha=0.3)

    legend_handles = []
    if np.isfinite(stability_values).any():
        legend_handles.append(
            quality_ax.scatter(
                stability_values,
                y_values,
                s=40,
                marker="o",
                facecolors="white",
                edgecolors="#222222",
                linewidths=1.0,
                label="stability",
                zorder=3,
            )
        )
    if retry_column is not None and retry_label is not None:
        retry_values = subset[retry_column].to_numpy(dtype=float)
        legend_handles.append(
            quality_ax.scatter(
                retry_values,
                y_values,
                s=42,
                marker="D",
                color="#111111",
                label=retry_label,
                zorder=3,
            )
        )
    for y, value in zip(y_values, success_values, strict=True):
        quality_ax.text(min(value + 0.02, 1.03), y, f"{value:.2f}", va="center", fontsize=8.5)
    if legend_handles:
        quality_ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8.5)

    cost_ax.barh(y_values, token_values, color=colors, alpha=0.72)
    cost_ax.set_title("cost")
    cost_ax.set_xlabel("avg tokens_total")
    cost_ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_compact_number))
    cost_ax.grid(axis="x", alpha=0.3)
    cost_ax.tick_params(axis="y", left=False, labelleft=False)
    token_padding = max(float(np.nanmax(token_values)) * 0.02, 1.0) if token_values.size else 1.0
    for y, value in zip(y_values, token_values, strict=True):
        cost_ax.text(value + token_padding, y, _format_compact_number(value), va="center", fontsize=8.5)

    fig.suptitle(f"{benchmark}: system scorecard")
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_system_scorecard.png"
    return _save_fig(fig, path)


def _save_vs_sas_tradeoff_chart(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    out_dir: Path,
) -> str | None:
    subset = frame[frame["benchmark"] == benchmark].copy()
    if subset.empty:
        return None
    required_columns = [
        "mean_success_rate_delta_vs_sas",
        "mean_tokens_total_delta_vs_sas",
    ]
    if any(column not in subset.columns for column in required_columns):
        return None
    subset = subset.dropna(subset=required_columns)
    if subset.empty:
        return None
    subset = subset.sort_values(
        by=[
            "mean_success_rate_delta_vs_sas",
            "mean_stability_delta_vs_sas",
            "mean_tokens_total_delta_vs_sas",
            "system_label",
        ],
        ascending=[False, False, True, True],
    )

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.axhline(0.0, color="#666666", linewidth=0.9, alpha=0.7)
    ax.axvline(0.0, color="#666666", linewidth=0.9, alpha=0.7)
    ax.scatter(
        [0.0],
        [0.0],
        s=180,
        marker="*",
        color=_color_for_system("sas"),
        edgecolors="#111111",
        linewidths=0.8,
        zorder=3,
    )
    baseline_label = ax.annotate(
        "SAS",
        (0.0, 0.0),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": "none",
            "alpha": 0.82,
        },
    )

    points: list[tuple[float, float, str]] = []
    for _, row in subset.iterrows():
        system_label = str(row["system_label"])
        x = float(row["mean_tokens_total_delta_vs_sas"])
        y = float(row["mean_success_rate_delta_vs_sas"])
        ax.scatter(
            x,
            y,
            s=110,
            color=_color_for_system(system_label),
            edgecolors="#111111",
            linewidths=0.8,
            zorder=3,
        )
        points.append((x, y, _short_system_label(system_label)))

    ax.margins(x=0.18, y=0.18)
    _annotate_scatter_labels(ax, fig, points)
    baseline_label.set_zorder(4)
    ax.text(
        0.02,
        0.98,
        "Upper-left is better: more success with fewer tokens than SAS.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
    )
    ax.set_title(f"{benchmark}: lift vs extra cost against SAS")
    ax.set_xlabel("mean tokens_total delta vs SAS")
    ax.set_ylabel("mean success_rate delta vs SAS")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_compact_number))
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_vs_sas_tradeoff.png"
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
    axis_labels = [_axis_system_label(label) for label in labels]
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

    fig, axes = plt.subplots(1, 2, figsize=(max(10.5, len(labels) * 1.7), 5.4))
    axes[0].bar(axis_labels, agent_comm, label="agent_to_agent", color="#66c2a5")
    axes[0].bar(
        axis_labels,
        system_comm,
        bottom=agent_comm,
        label="system_mediated",
        color="#fc8d62",
    )
    total_comm = agent_comm + system_comm
    for idx, total in enumerate(total_comm):
        axes[0].text(idx, total + max(total_comm.max() * 0.02, 0.2), f"{total:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("communication count")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x_values, handoffs, color=colors)
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(axis_labels, rotation=0)
    for idx, value in enumerate(handoffs):
        axes[1].text(idx, value + max(handoffs.max() * 0.02, 0.2), f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_title("handoff count")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"{benchmark}: coordination overhead")
    fig.tight_layout()

    path = out_dir / f"{_sanitize_filename(benchmark)}_coordination_breakdown_average.png"
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

    lines.extend(
        [
            "## Plot Guide",
            "",
            "- `system_scorecard`: ranks each topology on success and token cost in one view; markers show stability and best available pass@k retry lift when present.",
            "- `success_vs_tokens_frontier`: absolute quality vs cost tradeoff; point color encodes stability.",
            "- `vs_sas_tradeoff`: how much success each MAS gains relative to SAS and how many extra tokens it costs.",
            "- `coordination_breakdown`: communication and handoff overhead for each topology.",
            "- `pass_at_k`: optional retry curve, only emitted when multiple pass@k values are available.",
            "",
        ]
    )

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
    run_df = _load_run_rows(experiment_root)
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
    for suffix in GENERATED_PLOT_SUFFIXES:
        for stale_plot in output_dir.glob(f"*{suffix}"):
            stale_plot.unlink()

    plots: dict[str, list[str]] = {}
    for benchmark in sorted(task_df["benchmark"].unique()):
        benchmark_system_df = system_df[system_df["benchmark"] == benchmark].copy()
        benchmark_run_df = run_df[run_df["benchmark"] == benchmark].copy()
        benchmark_root = output_dir / benchmark
        avg_rq1 = benchmark_root / "RQ1" / "average"
        ind_rq1 = benchmark_root / "RQ1" / "indivudal"
        avg_rq2 = benchmark_root / "RQ2" / "average"
        ind_rq2 = benchmark_root / "RQ2" / "indivudal"
        avg_theory = benchmark_root / "THEORY" / "average"
        ind_theory = benchmark_root / "THEORY" / "indivudal"
        benchmark_plot_paths: list[str] = []
        for plot_path in [
            _save_system_scorecard(
                benchmark_system_df,
                benchmark=benchmark,
                out_dir=avg_rq1,
            ),
            _save_run_utility_comparison(benchmark_run_df, benchmark=benchmark, out_dir=ind_rq1),
            _save_success_vs_tokens_frontier(
                benchmark_system_df,
                benchmark=benchmark,
                out_dir=avg_rq1,
            ),
            _save_run_success_vs_tokens_frontier(
                benchmark_run_df,
                benchmark=benchmark,
                out_dir=ind_rq1,
            ),
            _save_vs_sas_tradeoff_chart(
                vs_sas_system_df,
                benchmark=benchmark,
                out_dir=avg_rq1,
            ),
            _save_run_gain_cost_plane(
                benchmark_run_df,
                benchmark=benchmark,
                out_dir=ind_rq2,
            ),
            _save_pass_at_k_chart(
                benchmark_system_df,
                benchmark=benchmark,
                out_dir=avg_theory,
            ),
            _save_run_quality_cost_pareto(
                benchmark_run_df,
                benchmark=benchmark,
                out_dir=ind_theory,
            ),
            _save_run_mahalanobis_diagnostics(
                benchmark_run_df,
                benchmark=benchmark,
                out_dir=ind_theory,
            ),
            _save_coordination_breakdown_chart(
                benchmark_system_df,
                benchmark=benchmark,
                out_dir=avg_rq2,
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
