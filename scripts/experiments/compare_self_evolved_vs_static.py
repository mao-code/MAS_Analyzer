#!/usr/bin/env python3
"""Compare the dynamic self-evolved MAS against the static MAS baselines on BrowseComp.

Reads per-task summary.csv files (one row per task, already aggregated over runs) for
the 7 static systems plus the self-evolved system, restricts to the common task set
(the tasks the self-evolved run actually covered), and emits comparison tables (md/csv)
and figures (png). The static runs are NOT re-executed.

Usage:
  python scripts/experiments/compare_self_evolved_vs_static.py \
    --static-root artifacts/full_experiment/20260427T134706Z__openai_gpt_oss_120b/browsecomp \
    --self-evolved-summary artifacts/selfevo_browsecomp_gptoss/run_main/merged/browsecomp/self_evolved/summary.csv \
    --out-dir artifacts/selfevo_browsecomp_gptoss/run_main/comparison
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Display order: single-agent, then static MAS topologies, dynamic last.
STATIC_SYSTEMS = [
    "sas",
    "only_voting",
    "orchestrator_no_discussion",
    "orchestrator_with_discussion",
    "orchestrator_tree_structure",
    "fully_linked_debate",
    "group_chat_debate",
]
SELF_EVOLVED = "self_evolved"

# Pretty labels for figures/tables.
PRETTY = {
    "sas": "SAS (single)",
    "only_voting": "only_voting",
    "orchestrator_no_discussion": "orch_no_disc",
    "orchestrator_with_discussion": "orch_with_disc",
    "orchestrator_tree_structure": "orch_tree",
    "fully_linked_debate": "fully_linked",
    "group_chat_debate": "group_chat",
    "self_evolved": "self_evolved (dynamic)",
}


def _f(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_system(summary_csv: Path) -> dict[str, dict[str, float | None]]:
    """task_id -> {metric: value} for one system."""
    out: dict[str, dict[str, float | None]] = {}
    with summary_csv.open() as fh:
        for row in csv.DictReader(fh):
            tid = str(row.get("task_id", "")).strip()
            if not tid:
                continue
            out[tid] = {
                "success": _f(row.get("Q1_success_rate")),
                "completion": _f(row.get("Q2_completion_rate")),
                "tokens": _f(row.get("C2_tokens_total")),
                "latency_ms": _f(row.get("latency_e2e")),
                "tool_calls": _f(row.get("C4_tool_calls_total")),
                "steps": _f(row.get("P1_steps_total")),
                "comm": _f(row.get("D2_communication_count")),
                "handoff": _f(row.get("D3_handoff_count")),
                "agents": _f(row.get("agents")),
                "runs": _f(row.get("runs")),
            }
    return out


def mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def aggregate(per_task: dict[str, dict], task_ids: list[str]) -> dict[str, float | None]:
    rows = [per_task[t] for t in task_ids if t in per_task]
    agg: dict[str, float | None] = {}
    for m in ["success", "completion", "tokens", "latency_ms", "tool_calls", "steps", "comm", "handoff", "agents", "runs"]:
        agg[m] = mean([r.get(m) for r in rows])
    agg["latency_s"] = (agg["latency_ms"] / 1000.0) if agg["latency_ms"] is not None else None
    # token cost per successful task (efficiency); None if no successes
    if agg["success"] and agg["success"] > 0 and agg["tokens"] is not None:
        agg["tokens_per_success"] = agg["tokens"] / agg["success"]
    else:
        agg["tokens_per_success"] = None
    agg["n_tasks"] = float(len([t for t in task_ids if t in per_task]))
    return agg


def fmt(v: float | None, nd: int = 3) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


def md_table(systems: list[str], aggs: dict[str, dict], title: str) -> str:
    cols = [
        ("success", "Success", 3),
        ("completion", "Completion", 3),
        ("tokens", "Tokens", 0),
        ("tokens_per_success", "Tok/Succ", 0),
        ("latency_s", "Latency(s)", 1),
        ("tool_calls", "ToolCalls", 2),
        ("steps", "Steps", 2),
        ("comm", "Comm", 2),
        ("handoff", "Handoff", 2),
        ("agents", "Agents", 1),
    ]
    head = "| System | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [f"### {title}", "", head, sep]
    for s in systems:
        a = aggs[s]
        cells = [fmt(a.get(k), nd) for k, _, nd in cols]
        lines.append(f"| {PRETTY.get(s, s)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(systems: list[str], aggs: dict[str, dict], path: Path) -> None:
    cols = ["success", "completion", "tokens", "tokens_per_success", "latency_s",
            "tool_calls", "steps", "comm", "handoff", "agents", "runs", "n_tasks"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system"] + cols)
        for s in systems:
            a = aggs[s]
            w.writerow([s] + [("" if a.get(c) is None else a.get(c)) for c in cols])


# ---------------- figures ----------------

def _bar(ax, systems, values, ylabel, title, highlight=SELF_EVOLVED, fmt_val="{:.2f}"):
    labels = [PRETTY.get(s, s) for s in systems]
    colors = ["#d62728" if s == highlight else "#4c72b0" for s in systems]
    bars = ax.bar(range(len(systems)), [v if v is not None else 0 for v in values], color=colors)
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        if v is not None:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt_val.format(v),
                    ha="center", va="bottom", fontsize=7)


def fig_bars(systems, aggs, out: Path, label: str = "BrowseComp / GPT-OSS-120b"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    succ = [aggs[s]["success"] for s in systems]
    toks = [aggs[s]["tokens"] for s in systems]
    lat = [aggs[s]["latency_s"] for s in systems]
    steps = [aggs[s]["steps"] for s in systems]
    _bar(axes[0, 0], systems, succ, "Success rate", "Quality: success rate", fmt_val="{:.2f}")
    _bar(axes[0, 1], systems, toks, "Mean tokens / task", "Cost: tokens", fmt_val="{:.0f}")
    _bar(axes[1, 0], systems, lat, "Mean latency (s)", "Cost: end-to-end latency", fmt_val="{:.0f}")
    _bar(axes[1, 1], systems, steps, "Mean steps / task", "Process: steps", fmt_val="{:.1f}")
    fig.suptitle(f"Self-evolved (dynamic) vs static MAS — {label}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_pareto(systems, aggs, out: Path, label: str = "BrowseComp / GPT-OSS-120b"):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    xs = [aggs[s]["tokens"] for s in systems]
    ys = [aggs[s]["success"] for s in systems]
    for s, x, y in zip(systems, xs, ys):
        if x is None or y is None:
            continue
        is_se = s == SELF_EVOLVED
        ax.scatter(x, y, s=140 if is_se else 90,
                   color="#d62728" if is_se else "#4c72b0",
                   edgecolor="black", zorder=3, marker="*" if is_se else "o")
        ax.annotate(PRETTY.get(s, s), (x, y), textcoords="offset points",
                    xytext=(7, 5), fontsize=8)
    # Pareto frontier (max success at min tokens): lower tokens + higher success is better
    pts = sorted([(aggs[s]["tokens"], aggs[s]["success"], s) for s in systems
                  if aggs[s]["tokens"] is not None and aggs[s]["success"] is not None])
    frontier, best = [], -1.0
    for x, y, s in pts:  # increasing tokens; keep points improving success
        if y > best:
            frontier.append((x, y))
            best = y
    if len(frontier) >= 2:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                "--", color="gray", alpha=0.7, zorder=1, label="efficiency frontier")
        ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("Mean tokens per task  (cost →)")
    ax.set_ylabel("Success rate  (quality ↑)")
    ax.set_title(f"Cost–quality frontier — {label}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_heatmap(systems, per_task_by_system, task_ids, out: Path):
    import numpy as np
    M = np.full((len(systems), len(task_ids)), np.nan)
    for i, s in enumerate(systems):
        for j, t in enumerate(task_ids):
            v = per_task_by_system[s].get(t, {}).get("success")
            if v is not None:
                M[i, j] = v
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(task_ids), 0.6 * len(systems) + 1.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(task_ids)))
    ax.set_xticklabels(task_ids, fontsize=8)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([PRETTY.get(s, s) for s in systems], fontsize=8)
    ax.set_xlabel("task_id")
    ax.set_title("Per-task success rate (rows=system, cols=task)")
    for i in range(len(systems)):
        for j in range(len(task_ids)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0") if M[i, j] not in (0, 1) else f"{int(M[i, j])}",
                        ha="center", va="center", fontsize=7,
                        color="black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="success")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-root", required=True,
                    help="Path to <experiment>/browsecomp dir holding the 7 static system dirs")
    ap.add_argument("--self-evolved-summary", required=True,
                    help="Path to the merged self_evolved summary.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--benchmark", default="browsecomp", help="Benchmark name (titles only)")
    ap.add_argument("--model", default="GPT-OSS-120b", help="Model label (titles only)")
    args = ap.parse_args()
    run_one(
        static_root=Path(args.static_root),
        self_evolved_summary=Path(args.self_evolved_summary),
        out_dir=Path(args.out_dir),
        benchmark=args.benchmark,
        model=args.model,
        verbose=True,
    )
    return 0


def run_one(
    *,
    static_root: Path,
    self_evolved_summary: Path,
    out_dir: Path,
    benchmark: str = "browsecomp",
    model: str = "GPT-OSS-120b",
    verbose: bool = False,
) -> dict | None:
    """Compare one benchmark; write tables/figures to out_dir.

    Returns {systems, aggs_common, common, per_task_by_system} for cross-benchmark
    rollups, or None if the self_evolved summary is missing/empty.
    """
    label = f"{benchmark} / {model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not self_evolved_summary.exists():
        print(f"WARN: missing self_evolved summary, skipping {benchmark}: {self_evolved_summary}")
        return None

    per_task_by_system: dict[str, dict] = {}
    for s in STATIC_SYSTEMS:
        p = static_root / s / "summary.csv"
        if not p.exists():
            print(f"WARN: missing static system summary: {p}")
            continue
        per_task_by_system[s] = load_system(p)
    per_task_by_system[SELF_EVOLVED] = load_system(self_evolved_summary)
    if not per_task_by_system[SELF_EVOLVED]:
        print(f"WARN: empty self_evolved summary, skipping {benchmark}: {self_evolved_summary}")
        return None

    systems = [s for s in STATIC_SYSTEMS if s in per_task_by_system] + [SELF_EVOLVED]

    # Common task set = tasks the self-evolved run covered (intersection with statics).
    se_tasks = set(per_task_by_system[SELF_EVOLVED])
    common = sorted(
        (t for t in se_tasks if all(t in per_task_by_system[s] for s in systems)),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    se_full = sorted(se_tasks, key=lambda x: int(x) if x.isdigit() else x)
    static_full = sorted(
        set().union(*[set(per_task_by_system[s]) for s in STATIC_SYSTEMS if s in per_task_by_system]),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    if verbose:
        print(f"[{benchmark}] Self-evolved tasks: {se_full}")
        print(f"[{benchmark}] Common tasks (all systems): {common}")

    aggs_common = {s: aggregate(per_task_by_system[s], common) for s in systems}
    aggs_static_full = {s: aggregate(per_task_by_system[s], static_full) for s in STATIC_SYSTEMS if s in per_task_by_system}
    aggs_static_full[SELF_EVOLVED] = aggregate(per_task_by_system[SELF_EVOLVED], se_full)

    # ---- tables ----
    write_csv(systems, aggs_common, out_dir / "comparison_same_tasks.csv")
    write_csv(systems, aggs_static_full, out_dir / "comparison_native_tasks.csv")

    se_runs = aggs_common[SELF_EVOLVED].get("runs")
    static_n = len(static_full)
    md = [
        f"# Self-evolved (dynamic) MAS vs static MAS — {label}",
        "",
        f"- **Benchmark:** {benchmark} · **Model:** {model} (medium reasoning effort) · **Harness:** OpenRouter",
        f"- **Self-evolved run:** new orchestration framework (structural compaction, decision-grade "
        f"consensus, broadened finalize; no hard packet cap), {int(se_runs) if se_runs else 1} run/task.",
        f"- **Static baselines:** pre-existing data in `artifacts/full_experiment`, 3 runs/task (not re-run).",
        f"- **Common task set (apples-to-apples, same {len(common)} tasks):** {', '.join(common)}",
        "",
        f"> Note: static success rates are averaged over 3 runs/task; the self-evolved run is "
        f"{int(se_runs) if se_runs else 1} run/task. Tokens/latency/steps are per-task means.",
        "",
        md_table(systems, aggs_common, f"Same {len(common)} tasks (fair comparison)"),
        "",
        md_table(systems, aggs_static_full,
                 f"Each system on its native task set (static={static_n} tasks, "
                 f"self_evolved={len(se_full)}) — context only"),
        "",
        "## Figures",
        "- `fig_bars.png` — success / tokens / latency / steps by system",
        "- `fig_pareto.png` — cost (tokens) vs quality (success) frontier",
        "- `fig_per_task_heatmap.png` — per-task success grid",
    ]
    (out_dir / "COMPARISON.md").write_text("\n".join(md), encoding="utf-8")

    # ---- figures (on common task set) ----
    fig_bars(systems, aggs_common, out_dir / "fig_bars.png", label=label)
    fig_pareto(systems, aggs_common, out_dir / "fig_pareto.png", label=label)
    fig_heatmap(systems, per_task_by_system, common, out_dir / "fig_per_task_heatmap.png")

    print(f"[{benchmark}] Wrote tables + figures to {out_dir}")
    if verbose:
        print((out_dir / "COMPARISON.md").read_text())
    return {
        "systems": systems,
        "aggs_common": aggs_common,
        "common": common,
        "per_task_by_system": per_task_by_system,
    }


if __name__ == "__main__":
    raise SystemExit(main())
