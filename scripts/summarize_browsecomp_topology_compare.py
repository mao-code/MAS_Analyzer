#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

SYSTEM_ORDER = [
    "sas",
    "orchestrator_tree_structure",
    "orchestrator_no_discussion",
    "orchestrator_with_discussion",
    "only_voting",
    "fully_linked_debate",
    "group_chat_debate",
    "self_evolved",
]


def _float(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _system_metrics(system_root: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    summary_csv = system_root / "summary.csv"
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    summary_json = _read_json(system_root / "summary.json")
    status = summary_json.get("run_status_summary", {})
    tasks = summary_json.get("tasks", [])
    if not isinstance(tasks, list):
        tasks = []

    success_values = [_float(row, "success_rate", "eval_success_rate") for row in rows]
    completion_values = [_float(row, "Q2_completion_rate", "eval_completion_rate") for row in rows]
    token_values = [_float(row, "tokens_total", "C2_tokens_total") for row in rows]
    tool_values = [_float(row, "tool_calls_total", "C4_tool_calls_total") for row in rows]
    cost_values = [_float(row, "C3_cost_total", "cost_total") for row in rows]
    latency_values = [_float(row, "C1_latency_p95", "latency_total") for row in rows]

    return {
        "system": system_root.name,
        "task_count": len(rows) or len(tasks),
        "success_rate": mean(success_values) if success_values else 0.0,
        "completion_rate": mean(completion_values) if completion_values else 0.0,
        "tokens_total": mean(token_values) if token_values else 0.0,
        "tool_calls_total": mean(tool_values) if tool_values else 0.0,
        "cost_total": mean(cost_values) if cost_values else 0.0,
        "latency_p95": mean(latency_values) if latency_values else 0.0,
        "run_failure_count": int(status.get("run_failure_count", 0) or 0),
        "fallback_count": int(status.get("fallback_count", 0) or 0),
        "needs_rerun_task_count": int(status.get("needs_rerun_task_count", 0) or 0),
        "summary_csv": summary_csv,
        "summary_json": system_root / "summary.json",
    }


def _self_evolved_observations(system_root: Path) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for metadata_path in sorted(system_root.glob("*/run_*.metadata.json")):
        metadata = _read_json(metadata_path)
        se = metadata.get("self_evolved", {})
        if not isinstance(se, dict):
            continue
        spec_versions = se.get("topology_spec_versions", [])
        audit_reports = se.get("audit_reports", [])
        mutation = se.get("mutation")
        short_term = se.get("short_term_playbook_entries", [])
        context_versions = se.get("context_state_versions", [])
        final_synthesis = se.get("final_synthesis", [])
        modes = sorted(
            {
                str(mode.get("mode", ""))
                for report in audit_reports
                if isinstance(report, dict)
                for mode in report.get("detected_modes", [])
                if isinstance(mode, dict) and mode.get("mode")
            }
        )
        observations.append(
            {
                "task_id": metadata.get("task_id", metadata_path.parent.name),
                "turns_executed": metadata.get("turns_executed"),
                "spec_versions": len(spec_versions) if isinstance(spec_versions, list) else 0,
                "mutated": mutation is not None,
                "audit_modes": ", ".join(modes) if modes else "none",
                "short_term_entries": len(short_term) if isinstance(short_term, list) else 0,
                "context_versions": len(context_versions)
                if isinstance(context_versions, list)
                else 0,
                "final_synthesis": len(final_synthesis)
                if isinstance(final_synthesis, list)
                else 0,
                "trajectory": metadata_path.with_name(
                    metadata_path.name.replace(".metadata.json", ".trajectory.md")
                ),
                "trace": metadata_path.with_name(
                    metadata_path.name.replace(".metadata.json", ".trace.jsonl")
                ),
            }
        )
    return observations


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _detect_harness_backend(experiment_root: Path) -> str:
    metadata_paths = (experiment_root / "browsecomp" / "self_evolved").glob(
        "*/run_*.metadata.json"
    )
    for metadata_path in sorted(metadata_paths):
        metadata = _read_json(metadata_path)
        se = metadata.get("self_evolved", {})
        if isinstance(se, dict):
            backend = str(se.get("harness_backend", "")).strip()
            if backend:
                return backend
    name = experiment_root.name.lower()
    if "openrouter" in name:
        return "openrouter"
    if "claude" in name:
        return "claude_agent_sdk"
    return "unknown"


def collect_metrics(benchmark_root: Path) -> list[dict[str, Any]]:
    metrics = []
    for system in SYSTEM_ORDER:
        root = benchmark_root / system
        if root.exists():
            metrics.append(_system_metrics(root))
    return metrics


def build_report(experiment_root: Path) -> str:
    benchmark_root = experiment_root / "browsecomp"
    harness_backend = _detect_harness_backend(experiment_root)
    metrics = collect_metrics(benchmark_root)
    sample_count = max((item["task_count"] for item in metrics), default=0)

    by_system = {item["system"]: item for item in metrics}
    self_metrics = by_system.get("self_evolved")
    static = [item for item in metrics if item["system"] != "self_evolved"]
    best_static = max(static, key=lambda item: item["success_rate"], default=None)

    lines = [
        "# BrowseComp Topology Smoke Test",
        "",
        f"- Experiment root: `{experiment_root}`",
        "- Benchmark: `browsecomp`",
        f"- Samples: {sample_count} tasks per system",
        f"- Harness backend: `{harness_backend}`",
        "- Evaluation: substring smoke-test mode",
        "- Final vote / termination: deterministic or lexical to avoid extra judge calls",
        "",
        "## Summary",
        "",
        "| System | Tasks | Success | Completion | Avg Tokens | Avg Tools | Avg Cost | Failures | Rerun Tasks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        lines.append(
            "| {system} | {task_count} | {success} | {completion} | {tokens} | "
            "{tools} | {cost} | {failures} | {rerun} |".format(
                system=item["system"],
                task_count=item["task_count"],
                success=_fmt(item["success_rate"]),
                completion=_fmt(item["completion_rate"]),
                tokens=_fmt(item["tokens_total"]),
                tools=_fmt(item["tool_calls_total"]),
                cost=_fmt(item["cost_total"]),
                failures=item["run_failure_count"],
                rerun=item["needs_rerun_task_count"],
            )
        )

    lines.extend(["", "## Comparison", ""])
    if self_metrics and best_static:
        delta = self_metrics["success_rate"] - best_static["success_rate"]
        verdict = "better" if delta > 0 else "tied" if delta == 0 else "worse"
        comparator = "than" if verdict in {"better", "worse"} else "with"
        lines.append(
            f"- `self_evolved` success is **{verdict}** {comparator} the best static baseline "
            f"on this smoke sample: `{_fmt(self_metrics['success_rate'])}` vs "
            f"`{_fmt(best_static['success_rate'])}` for `{best_static['system']}` "
            f"(delta `{_fmt(delta)}`)."
        )
        lines.append(
            f"- Treat this as a smoke-test signal, not a benchmark claim: only {sample_count} "
            "samples per system and substring evaluation are used."
        )
    else:
        lines.append("- Comparison incomplete: missing self-evolved or static summary.")

    lines.extend(["", "## Self-Evolved Observability", ""])
    observations = _self_evolved_observations(benchmark_root / "self_evolved")
    if observations:
        lines.extend(
            [
                "| Task | Turns | Spec Versions | Mutated | Audit Modes | Short-Term Memories | Context Versions | Final Synthesis | Trace | Trajectory |",
                "|---|---:|---:|---|---|---:|---:|---:|---|---|",
            ]
        )
        for obs in observations:
            lines.append(
                "| {task} | {turns} | {spec_versions} | {mutated} | {modes} | "
                "{short_term} | {context_versions} | {final_synthesis} | `{trace}` | `{trajectory}` |".format(
                    task=obs["task_id"],
                    turns=obs["turns_executed"],
                    spec_versions=obs["spec_versions"],
                    mutated="yes" if obs["mutated"] else "no",
                    modes=obs["audit_modes"],
                    short_term=obs["short_term_entries"],
                    context_versions=obs["context_versions"],
                    final_synthesis=obs["final_synthesis"],
                    trace=obs["trace"],
                    trajectory=obs["trajectory"],
                )
            )
    else:
        lines.append("- No self-evolved metadata found.")

    lines.extend(
        [
            "",
            "## Debugging Pointers",
            "",
            "- `run_*.trace.jsonl`: ordered structured events.",
            "- `run_*.metadata.json`: topology versions, context-state versions, relay messages, message views, artifacts, audit reports, and playbook memories.",
            "- `run_*.trajectory.md`: compact human-readable workflow and message trajectory.",
            "- `run_*.trace_metrics.json`: cost, tool, communication, handoff, and termination metrics.",
            "- `descriptor.json` and `analysis.json`: task-level metric aggregation.",
            "",
            "Hidden provider chain-of-thought is not stored; the report uses emitted artifacts, summaries, critiques, evidence summaries, tool calls, and controller decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def _family(system: str) -> str:
    if system == "self_evolved":
        return "self_evolved"
    if system == "sas":
        return "sas"
    return "static"


def write_figures(experiment_root: Path, metrics: list[dict[str, Any]]) -> list[Path]:
    """Emit comparison figures across systems. All systems share one model here,
    so color encodes the system family (SAS / static MAS / self-evolved)."""
    metrics = [m for m in metrics if m.get("task_count")]
    if not metrics:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figures_dir = experiment_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    color_by_family = {"sas": "#7FC9A6", "static": "#4C78A8", "self_evolved": "#E4801E"}
    label_by_family = {
        "sas": "SAS baseline",
        "static": "Static MAS",
        "self_evolved": "Self-evolved",
    }
    systems = [m["system"] for m in metrics]
    fams = [_family(s) for s in systems]
    colors = [color_by_family[f] for f in fams]
    x = list(range(len(systems)))
    legend_handles = [
        Patch(facecolor=color_by_family[f], label=label_by_family[f])
        for f in ["sas", "static", "self_evolved"]
        if f in fams
    ]
    written: list[Path] = []

    def _save(fig, fname: str) -> None:
        for ext in ("png", "pdf"):
            path = figures_dir / f"{fname}.{ext}"
            fig.savefig(path, dpi=150)
            written.append(path)
        plt.close(fig)

    def _bar(metric_key: str, title: str, ylabel: str, fname: str, pct: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = [m[metric_key] * (100 if pct else 1) for m in metrics]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.0f}%" if pct else f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.legend(handles=legend_handles, loc="best", fontsize=8)
        fig.tight_layout()
        _save(fig, fname)

    _bar(
        "success_rate",
        "BrowseComp success rate by system (Claude SDK harness)",
        "Success rate (%)",
        "success_rate_by_system",
        pct=True,
    )
    _bar(
        "tool_calls_total",
        "Avg tool calls per task by system (Claude SDK harness)",
        "Avg tool calls",
        "tool_calls_by_system",
    )
    _bar(
        "tokens_total",
        "Avg tokens per task by system (Claude SDK harness)",
        "Avg tokens",
        "tokens_by_system",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for m, c in zip(metrics, colors):
        ax.scatter(
            m["tokens_total"],
            m["success_rate"] * 100,
            color=c,
            s=90,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            m["system"],
            (m["tokens_total"], m["success_rate"] * 100),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Avg tokens per task (cost proxy)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Quality vs cost by system (Claude SDK harness)")
    ax.legend(handles=legend_handles, loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "quality_cost_plane")

    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(experiment_root), encoding="utf-8")
    print(output)
    if not args.no_figures:
        figures = write_figures(experiment_root, collect_metrics(experiment_root / "browsecomp"))
        for path in figures:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
