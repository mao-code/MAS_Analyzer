from __future__ import annotations

import csv
import json
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = (
    ROOT
    / "artifacts/full_experiment/20260616T_browsecomp_self_evolved_10sample_comparison_matched_v1"
)

# OpenRouter self-evolved with config matched to the static MAS runtime
# (communication_budget=50, peer_artifact_max_chars=unlimited, termination/final_vote=llm_judge,
#  plus the static BrowseComp tool config).
OPENROUTER_DYNAMIC_ROOT = (
    ROOT
    / "artifacts/full_experiment/"
    "20260616T_browsecomp_self_evolved_openrouter_gpt_oss_120b_10sample_matched_v1"
)
CLAUDE_DYNAMIC_ROOT = (
    ROOT
    / "artifacts/full_experiment/"
    "20260616T_browsecomp_self_evolved_claude_agent_sdk_10sample_fixed_v1"
)
# Single-agent Claude SDK baseline (self-evolved engine constrained to 1 agent => sas layout).
SAS_SDK_ROOT = (
    ROOT
    / "artifacts/full_experiment/"
    "20260616T_browsecomp_sas_claude_agent_sdk_10sample_v1"
)
# Diagnostic: same model + tool budget as MATCHED, but inter-agent comms still starved
# (communication_budget=2, peer_artifact_max_chars=320, lexical/deterministic decisions).
# Used to show config parity did NOT change the GPT-OSS success rate.
OPENROUTER_COMMS_STARVED_ROOT = (
    ROOT
    / "artifacts/full_experiment/"
    "20260616T_browsecomp_self_evolved_openrouter_gpt_oss_120b_10sample_fixed_v1"
)

SYSTEM_ORDER = [
    "sas",
    "only_voting",
    "orchestrator_no_discussion",
    "orchestrator_tree_structure",
    "orchestrator_with_discussion",
    "fully_linked_debate",
    "group_chat_debate",
    "self_evolved",
]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "static": "#A3BEFA",
    "self_evolved": "#F0986E",
    "sas_sdk": "#7FC9A6",
    "diagnostic": "#E2E5EA",
}
MODEL_HATCHES = {
    "GPT-OSS-120B": "///",
    "Gemma-4-31B-IT": "\\\\\\\\",
    "Claude Sonnet 4.6 SDK": "xx",
}


@dataclass(frozen=True)
class ExperimentSpec:
    model_label: str
    backend: str
    run_family: str
    profile: str
    root: Path
    primary: bool = True
    system_label: str = ""  # override the system_dir name in the comparison (e.g. "sas")


STATIC_SPECS = [
    ExperimentSpec(
        model_label="GPT-OSS-120B",
        backend="openrouter",
        run_family="static",
        profile="existing_static_mas_same_10_tasks",
        root=ROOT / "artifacts/full_experiment/20260427T134706Z__openai_gpt_oss_120b",
    ),
    ExperimentSpec(
        model_label="Gemma-4-31B-IT",
        backend="openrouter",
        run_family="static",
        profile="existing_static_mas_same_10_tasks",
        root=ROOT / "artifacts/full_experiment/20260427T134706Z__google_gemma_4_31b_it_nitro",
    ),
]

DYNAMIC_SPECS = [
    ExperimentSpec(
        model_label="GPT-OSS-120B",
        backend="openrouter",
        run_family="self_evolved",
        profile="fixed_tools8_getdoc",
        root=OPENROUTER_DYNAMIC_ROOT,
    ),
    ExperimentSpec(
        model_label="Claude Sonnet 4.6 SDK",
        backend="claude_agent_sdk",
        run_family="self_evolved",
        profile="fixed_tools8_getdoc_finalize",
        root=CLAUDE_DYNAMIC_ROOT,
    ),
    ExperimentSpec(
        model_label="Claude Sonnet 4.6 SDK",
        backend="claude_agent_sdk",
        run_family="sas_sdk",
        profile="single_agent_baseline",
        root=SAS_SDK_ROOT,
        system_label="sas",
    ),
    ExperimentSpec(
        model_label="GPT-OSS-120B",
        backend="openrouter",
        run_family="diagnostic",
        profile="comms_starved_budget2_packet320_lexical",
        root=OPENROUTER_COMMS_STARVED_ROOT,
        primary=False,
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def numeric(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def use_chart_theme() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial"],
            "patch.linewidth": 1.0,
        }
    )


def add_chart_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=76, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=116, break_long_words=False)
    ax.set_title("")
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=13, fontweight="semibold")
    fig.text(left, 0.925, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def display_system(system: str) -> str:
    labels = {
        "sas": "SAS",
        "only_voting": "Only voting",
        "orchestrator_no_discussion": "Orch. no discussion",
        "orchestrator_tree_structure": "Orch. tree",
        "orchestrator_with_discussion": "Orch. + discussion",
        "fully_linked_debate": "Fully linked debate",
        "group_chat_debate": "Group chat debate",
        "self_evolved": "Self-evolved",
    }
    return labels.get(system, system.replace("_", " "))


def task_ids_from_dynamic_root(root: Path) -> list[str]:
    summary_csv = root / "browsecomp/self_evolved/summary.csv"
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [str(row["task_id"]) for row in rows]


def trace_tool_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event_type") != "tool_call":
            continue
        tool_name = str(event.get("payload", {}).get("tool_name", "") or "")
        if tool_name:
            counts[tool_name] += 1
    return counts


def llm_metadata_flags(meta: dict[str, Any]) -> dict[str, int]:
    text = json.dumps(meta, ensure_ascii=False, default=str)
    return {
        "sdk_finalize_skipped_count": text.count("sdk_finalize_skipped"),
        "sdk_finalize_error_count": text.count("sdk_finalize_error"),
    }


def load_run_rows(specs: list[ExperimentSpec], task_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        benchmark_root = spec.root / "browsecomp"
        if not benchmark_root.exists():
            continue
        system_dirs = [p for p in benchmark_root.iterdir() if p.is_dir()]
        if spec.run_family in {"self_evolved", "diagnostic", "sas_sdk"}:
            system_dirs = [benchmark_root / "self_evolved"]
        for system_dir in sorted(system_dirs):
            if not system_dir.exists():
                continue
            system_label = spec.system_label or system_dir.name
            for task_id in task_ids:
                task_dir = system_dir / str(task_id)
                if not task_dir.exists():
                    continue
                for eval_path in sorted(task_dir.glob("run_*.eval.json")):
                    match = re.search(r"run_(\d+)\.eval\.json$", eval_path.name)
                    run_index = int(match.group(1)) if match else 0
                    eval_payload = read_json(eval_path)
                    details = eval_payload.get("details", {})
                    trace_metrics = read_json(task_dir / f"run_{run_index}.trace_metrics.json")
                    metrics = trace_metrics.get("metrics", {})
                    meta = read_json(task_dir / f"run_{run_index}.metadata.json")
                    run_meta = details.get("run_metadata", {}) if isinstance(details, dict) else {}
                    trace_counts = trace_tool_counts(task_dir / f"run_{run_index}.trace.jsonl")
                    flags = llm_metadata_flags(meta)
                    rows.append(
                        {
                            "model_label": spec.model_label,
                            "backend": spec.backend,
                            "run_family": spec.run_family,
                            "profile": spec.profile,
                            "primary": spec.primary,
                            "experiment_root": str(spec.root),
                            "system_label": system_label,
                            "task_id": str(task_id),
                            "run_index": run_index,
                            "success": bool(eval_payload.get("success", metrics.get("success", False))),
                            "completion": bool(
                                eval_payload.get("completion", metrics.get("completion", False))
                            ),
                            "score": numeric(eval_payload.get("score", metrics.get("score"))),
                            "prediction": details.get("prediction", eval_payload.get("prediction", "")),
                            "reference_answer": details.get("reference_answer", ""),
                            "latency_ms": numeric(
                                metrics.get("latency_e2e", metrics.get("C1_latency_p95"))
                            ),
                            "tokens_total": numeric(
                                metrics.get(
                                    "tokens_total",
                                    metrics.get("token_total", metrics.get("C2_tokens_total")),
                                )
                            ),
                            "cost_total": numeric(
                                metrics.get("cost_total", metrics.get("C3_cost_total"))
                            ),
                            "tool_calls_total": numeric(
                                metrics.get("tool_calls_total", metrics.get("C4_tool_calls_total"))
                            ),
                            "search_tool_calls": int(trace_counts.get("search", 0)),
                            "get_document_tool_calls": int(trace_counts.get("get_document", 0)),
                            "inter_agent_tool_calls": int(trace_counts.get("inter_agent_send", 0)),
                            "communication_count": numeric(metrics.get("communication_count")),
                            "handoff_count": numeric(metrics.get("handoff_count")),
                            "steps_total": numeric(metrics.get("steps_total")),
                            "run_status": str(
                                meta.get("run_status") or run_meta.get("run_status") or "completed"
                            ),
                            "fallback": truthy(meta.get("fallback", run_meta.get("fallback", False))),
                            "needs_rerun": truthy(
                                meta.get("needs_rerun", run_meta.get("needs_rerun", False))
                            ),
                            "topology_versions": len(
                                meta.get("self_evolved", {}).get("topology_spec_versions", [])
                            ),
                            "context_state_versions": len(
                                meta.get("self_evolved", {}).get("context_state_versions", [])
                            ),
                            "mutated": meta.get("self_evolved", {}).get("mutation") is not None,
                            **flags,
                        }
                    )
    return pd.DataFrame(rows)


def invalid_mutation_count(root: Path) -> int:
    log_path = root / "run.log"
    if not log_path.exists():
        return 0
    return log_path.read_text(encoding="utf-8", errors="replace").count(
        "Mutation planning failed"
    )


def aggregate_systems(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "model_label",
        "backend",
        "run_family",
        "profile",
        "primary",
        "experiment_root",
        "system_label",
    ]
    for keys, group in run_df.groupby(group_cols, dropna=False):
        (
            model_label,
            backend,
            run_family,
            profile,
            primary,
            experiment_root,
            system_label,
        ) = keys
        root = Path(str(experiment_root))
        rows.append(
            {
                "model_label": model_label,
                "backend": backend,
                "run_family": run_family,
                "profile": profile,
                "primary": bool(primary),
                "system_label": system_label,
                "task_count": group["task_id"].nunique(),
                "run_count": len(group),
                "runs_per_task": len(group) / max(1, group["task_id"].nunique()),
                "success_rate": group["success"].mean(),
                "completion_rate": group["completion"].mean(),
                "avg_score": group["score"].mean(),
                "avg_latency_s": group["latency_ms"].mean() / 1000.0,
                "median_latency_s": group["latency_ms"].median() / 1000.0,
                "avg_tokens_total": group["tokens_total"].mean(),
                "median_tokens_total": group["tokens_total"].median(),
                "avg_cost_total": group["cost_total"].mean(),
                "avg_tool_calls_total": group["tool_calls_total"].mean(),
                "avg_search_tool_calls": group["search_tool_calls"].mean(),
                "avg_inter_agent_tool_calls": group["inter_agent_tool_calls"].mean(),
                "avg_communication_count": group["communication_count"].mean(),
                "avg_handoff_count": group["handoff_count"].mean(),
                "run_failure_count": int((group["run_status"] == "failed").sum()),
                "fallback_count": int(group["fallback"].sum()),
                "needs_rerun_run_count": int(group["needs_rerun"].sum()),
                "needs_rerun_task_count": group.loc[group["needs_rerun"], "task_id"].nunique(),
                "mutated_run_rate": group["mutated"].mean(),
                "avg_topology_versions": group["topology_versions"].mean(),
                "avg_context_state_versions": group["context_state_versions"].mean(),
                "invalid_mutation_count": invalid_mutation_count(root),
                "sdk_finalize_skipped_runs": int((group["sdk_finalize_skipped_count"] > 0).sum()),
                "sdk_finalize_error_runs": int((group["sdk_finalize_error_count"] > 0).sum()),
                "experiment_root": str(root),
            }
        )
    result = pd.DataFrame(rows)
    order = {system: idx for idx, system in enumerate(SYSTEM_ORDER)}
    result["_system_order"] = result["system_label"].map(order).fillna(99)
    return result.sort_values(
        ["run_family", "model_label", "_system_order"], ascending=[True, True, True]
    ).drop(columns=["_system_order"])


def make_dynamic_deltas(system_df: pd.DataFrame) -> pd.DataFrame:
    primary = system_df[system_df["primary"]].copy()
    static = primary[primary["run_family"] == "static"].copy()
    dynamic = primary[primary["run_family"] == "self_evolved"].copy()
    rows: list[dict[str, Any]] = []
    for _, dyn in dynamic.iterrows():
        candidates: list[tuple[str, pd.Series]] = []
        same_model = static[static["model_label"] == dyn["model_label"]]
        same_model_sas = same_model[same_model["system_label"] == "sas"]
        if not same_model_sas.empty:
            candidates.append(("same_model_sas", same_model_sas.iloc[0]))
        if not same_model.empty:
            candidates.append(
                ("same_model_best_static", same_model.loc[same_model["success_rate"].idxmax()])
            )
        if not static.empty:
            candidates.append(("overall_best_static", static.loc[static["success_rate"].idxmax()]))
        for scope, base in candidates:
            rows.append(
                {
                    "dynamic_model": dyn["model_label"],
                    "dynamic_backend": dyn["backend"],
                    "dynamic_profile": dyn["profile"],
                    "baseline_scope": scope,
                    "baseline_model": base["model_label"],
                    "baseline_system": base["system_label"],
                    "dynamic_success_rate": dyn["success_rate"],
                    "baseline_success_rate": base["success_rate"],
                    "success_rate_delta": dyn["success_rate"] - base["success_rate"],
                    "dynamic_completion_rate": dyn["completion_rate"],
                    "baseline_completion_rate": base["completion_rate"],
                    "dynamic_avg_search_calls": dyn["avg_search_tool_calls"],
                    "baseline_avg_search_calls": base["avg_search_tool_calls"],
                    "dynamic_avg_tokens": dyn["avg_tokens_total"],
                    "baseline_avg_tokens": base["avg_tokens_total"],
                }
            )
    return pd.DataFrame(rows)


def sample_output_check(run_df: pd.DataFrame, task_ids: list[str]) -> pd.DataFrame:
    dynamic = run_df[
        (run_df["run_family"].isin(["self_evolved", "diagnostic"]))
        & (run_df["task_id"].isin(task_ids[:3]))
    ].copy()
    dynamic["prediction_preview"] = dynamic["prediction"].astype(str).str.slice(0, 180)
    dynamic["sdk_finalize_skipped"] = dynamic["sdk_finalize_skipped_count"] > 0
    columns = [
        "model_label",
        "backend",
        "profile",
        "task_id",
        "success",
        "completion",
        "reference_answer",
        "prediction_preview",
        "search_tool_calls",
        "inter_agent_tool_calls",
        "topology_versions",
        "context_state_versions",
        "mutated",
        "sdk_finalize_skipped",
    ]
    return dynamic[columns].sort_values(["profile", "task_id"])


def add_pattern_legends(ax: plt.Axes, *, anchor_x: float = 1.01) -> None:
    family_handles = [
        Patch(facecolor=COLORS["static"], edgecolor=TOKENS["ink"], label="Static MAS/SAS"),
        Patch(facecolor=COLORS["self_evolved"], edgecolor=TOKENS["ink"], label="Self-evolved"),
        Patch(facecolor=COLORS["sas_sdk"], edgecolor=TOKENS["ink"], label="SAS baseline (SDK)"),
        Patch(facecolor=COLORS["diagnostic"], edgecolor=TOKENS["ink"], label="Diagnostic"),
    ]
    model_handles = [
        Patch(facecolor=TOKENS["panel"], edgecolor=TOKENS["ink"], hatch=hatch, label=model)
        for model, hatch in MODEL_HATCHES.items()
    ]
    first = ax.legend(
        handles=family_handles,
        title="Run family",
        loc="upper left",
        bbox_to_anchor=(anchor_x, 1.0),
        frameon=False,
    )
    ax.add_artist(first)
    ax.legend(
        handles=model_handles,
        title="Model pattern",
        loc="upper left",
        bbox_to_anchor=(anchor_x, 0.68),
        frameon=False,
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_bar(
    system_df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    subtitle: str,
    xlabel: str,
    path: Path,
    percent: bool = False,
    log_x: bool = False,
) -> None:
    frame = system_df[system_df["primary"]].copy()
    frame["display_label"] = frame.apply(
        lambda row: f"{display_system(row['system_label'])} | {row['model_label']}",
        axis=1,
    )
    frame = frame.sort_values([metric, "completion_rate"], ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, 8.4))
    y = range(len(frame))
    colors = [COLORS.get(str(family), COLORS["diagnostic"]) for family in frame["run_family"]]
    bars = ax.barh(list(y), frame[metric], color=colors, edgecolor=TOKENS["ink"], linewidth=0.9)
    ax.set_yticks(list(y), frame["display_label"])
    for patch, (_, row) in zip(bars, frame.iterrows(), strict=False):
        patch.set_hatch(MODEL_HATCHES.get(row["model_label"], ""))
        value = row[metric]
        if metric in {"success_rate", "completion_rate"}:
            label = f"{value:.0%}"
        elif value >= 1000:
            label = f"{value / 1000:.1f}k"
        else:
            label = f"{value:.1f}"
        ax.text(
            patch.get_width() * (1.03 if log_x else 1.0) + (0 if log_x else 0.01),
            patch.get_y() + patch.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=8,
            color=TOKENS["ink"],
        )
    if percent:
        ax.set_xlim(0, max(0.65, float(frame[metric].max()) + 0.08))
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if log_x:
        lower = max(1.0, float(frame[metric].min()) * 0.75)
        upper = max(10.0, float(frame[metric].max()) * 1.45)
        ax.set_xscale("log")
        ax.set_xlim(lower, upper)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}")
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    add_pattern_legends(ax)
    add_chart_header(fig, ax, title, subtitle)
    save_figure(fig, path)


def plot_quality_cost(system_df: pd.DataFrame) -> None:
    frame = system_df[system_df["primary"]].copy()
    fig, ax = plt.subplots(figsize=(9.8, 6.8))
    marker_by_family = {"static": "o", "self_evolved": "s", "sas_sdk": "^", "diagnostic": "D"}
    color_by_model = {
        "GPT-OSS-120B": "#5477C4",
        "Gemma-4-31B-IT": "#71B436",
        "Claude Sonnet 4.6 SDK": "#CC6F47",
    }
    for _, row in frame.iterrows():
        ax.scatter(
            row["avg_tokens_total"],
            row["success_rate"],
            s=92 if row["run_family"] == "self_evolved" else 58,
            marker=marker_by_family.get(row["run_family"], "o"),
            color=color_by_model.get(row["model_label"], "#464C55"),
            edgecolor=TOKENS["ink"],
            linewidth=0.8,
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}")
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.03, max(0.65, float(frame["success_rate"].max()) + 0.08))
    ax.set_xlabel("Average tokens per run (log scale)")
    ax.set_ylabel("Success rate")
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor=TOKENS["ink"],
            label=model,
        )
        for model, color in color_by_model.items()
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markerfacecolor="#C5CAD3",
            markeredgecolor=TOKENS["ink"],
            label=family.replace("_", " ").title(),
        )
        for family, marker in marker_by_family.items()
    ]
    first = ax.legend(handles=model_handles, title="Model color", loc="upper right", frameon=False)
    ax.add_artist(first)
    ax.legend(handles=family_handles, title="Run marker", loc="center right", frameon=False)
    add_chart_header(
        fig,
        ax,
        "BrowseComp quality-cost plane",
        "Same 10 task IDs. Claude SDK token totals include prompt-cache read/write accounting and are not direct OpenRouter cost equivalents.",
    )
    save_figure(fig, OUT_DIR / "figures/quality_cost_plane.png")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
            )
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    task_ids: list[str],
    system_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> None:
    primary = system_df[system_df["primary"]].copy()
    report_cols = [
        "model_label",
        "backend",
        "run_family",
        "profile",
        "system_label",
        "task_count",
        "run_count",
        "runs_per_task",
        "success_rate",
        "completion_rate",
        "avg_search_tool_calls",
        "avg_tool_calls_total",
        "avg_tokens_total",
        "avg_latency_s",
        "run_failure_count",
        "needs_rerun_task_count",
        "invalid_mutation_count",
        "sdk_finalize_skipped_runs",
    ]
    report_df = primary[report_cols].sort_values(
        ["run_family", "model_label", "success_rate"], ascending=[True, True, False]
    )
    diagnostic = system_df[~system_df["primary"]][report_cols]
    markdown = [
        "# BrowseComp Self-Evolved 10-Sample Comparison",
        "",
        f"Task IDs: {', '.join(task_ids)}.",
        "",
        "Static MAS/SAS rows are recomputed from existing per-run artifacts on the same 10 task IDs; no static runs were rerun. Dynamic rows are the new self-evolved smoke runs.",
        "",
        "## Primary System Metrics",
        "",
        markdown_table(report_df),
        "",
        "## Dynamic Deltas",
        "",
        markdown_table(delta_df),
        "",
        "## First-Sample Output Check",
        "",
        markdown_table(sample_df),
        "",
        "## Diagnostic Rows",
        "",
        markdown_table(diagnostic),
        "",
        "## Notes",
        "",
        "- The OpenRouter/GPT-OSS self-evolved path is now config-matched to the static MAS runtime on every comparable knob: BrowseComp tools (`eval_mode=substring`, `tool_k=5`, `include_get_document=true`, `tool_snippet_max_tokens=512`, `max_tool_iterations=8`) AND the MAS runtime (`communication_budget=50`, `peer_artifact_max_chars=unlimited`, `termination_consensus_mode=llm_judge`, `final_vote_mode=llm_judge`, 4 agents). The earlier `fixed` smoke had matched only the tool budget; it still starved inter-agent communication (`communication_budget=2`, `peer_artifact_max_chars=320`) and used lexical/deterministic decision modes.",
        "- OpenRouter self-evolved uses `openai/gpt-oss-120b` with real BrowseComp `search` and `get_document` calls.",
        "- Claude SDK self-evolved (`claude-sonnet-4-6`) shares the same bridged BrowseComp tools (`search`, `get_document`, snippet 512), but its agent tool-calling loop is managed by the Claude Agent SDK harness itself rather than the static MAS runtime, so its per-agent budget is not forced to match. It runs the no-tools finalizer (`CLAUDE_AGENT_SDK_SKIP_FINALIZE` unset) with `CLAUDE_AGENT_SDK_QUERY_TIMEOUT_S>=600` so each run finalizes a real answer rather than a synthetic fallback.",
        "- SAS baseline (Claude SDK) is a single Claude-SDK agent (self-evolved engine constrained to 1 agent => `sas` layout); it isolates the model's solo browsing ability with no multi-agent collaboration.",
        "- Invalid mutation proposals are counted from `run.log`; they are process observability findings, not run failures.",
        "- Claude SDK token totals include cache read/write accounting and should not be treated as direct cost-equivalent to OpenRouter token totals.",
        "",
        "## Figures",
        "",
        "- `figures/success_rate_by_system.png`",
        "- `figures/search_calls_by_system.png`",
        "- `figures/tokens_by_system.png`",
        "- `figures/quality_cost_plane.png`",
        "",
    ]
    (OUT_DIR / "OBSERVATION_REPORT.md").write_text("\n".join(markdown), encoding="utf-8")


def assert_roots_ready() -> None:
    missing = [
        root
        for root in [OPENROUTER_DYNAMIC_ROOT, CLAUDE_DYNAMIC_ROOT]
        if not (root / "browsecomp/self_evolved/summary.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing completed dynamic summary roots:\n" + "\n".join(str(root) for root in missing)
        )


def main() -> int:
    assert_roots_ready()
    use_chart_theme()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    task_ids = task_ids_from_dynamic_root(OPENROUTER_DYNAMIC_ROOT)
    specs = STATIC_SPECS + DYNAMIC_SPECS
    run_df = load_run_rows(specs, task_ids)
    if run_df.empty:
        raise ValueError("No run rows loaded.")
    system_df = aggregate_systems(run_df)
    delta_df = make_dynamic_deltas(system_df)
    sample_df = sample_output_check(run_df, task_ids)

    run_df.to_csv(OUT_DIR / "run_level_metrics.csv", index=False)
    system_df.to_csv(OUT_DIR / "system_comparison.csv", index=False)
    delta_df.to_csv(OUT_DIR / "dynamic_vs_static_deltas.csv", index=False)
    sample_df.to_csv(OUT_DIR / "first_sample_output_check.csv", index=False)

    plot_bar(
        system_df,
        metric="success_rate",
        title="BrowseComp success by topology and model",
        subtitle="Same 10 task IDs. Bar fills show run family; hatches mark model identity.",
        xlabel="Success rate",
        path=OUT_DIR / "figures/success_rate_by_system.png",
        percent=True,
    )
    plot_bar(
        system_df,
        metric="avg_search_tool_calls",
        title="BrowseComp search calls by topology and model",
        subtitle="Search calls are counted from trace tool_call events; static rows come from existing artifacts.",
        xlabel="Average BrowseComp search calls per run",
        path=OUT_DIR / "figures/search_calls_by_system.png",
    )
    plot_bar(
        system_df,
        metric="avg_tokens_total",
        title="BrowseComp token footprint by topology and model",
        subtitle="Log scale. Claude SDK token accounting includes prompt-cache reads/writes.",
        xlabel="Average tokens per run (log scale)",
        path=OUT_DIR / "figures/tokens_by_system.png",
        log_x=True,
    )
    plot_quality_cost(system_df)
    write_report(task_ids, system_df, delta_df, sample_df)

    print(OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
