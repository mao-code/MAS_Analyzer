"""Generate workshop paper figures from workshop_system_metrics.csv.

Usage:
    python scripts/workshop_figures.py --metrics figures/workshop_system_metrics.csv \
        --first-fault figures/first_fault_summary.csv \
        --out figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── canonical order / labels ────────────────────────────────────────────────

TOPOLOGY_ORDER = [
    "sas",
    "only_voting",
    "fully_linked_debate",
    "group_chat_debate",
    "orchestrator_no_discussion",
    "orchestrator_tree_structure",
    "orchestrator_with_discussion",
]
TOPOLOGY_LABELS = {
    "sas": "SAS",
    "only_voting": "Vote",
    "fully_linked_debate": "Full",
    "group_chat_debate": "Chat",
    "orchestrator_no_discussion": "O-no",
    "orchestrator_tree_structure": "O-tree",
    "orchestrator_with_discussion": "O-disc",
}
TOPOLOGY_COLORS = {
    "sas":                        "#1f77b4",
    "only_voting":                "#ff7f0e",
    "fully_linked_debate":        "#2ca02c",
    "group_chat_debate":          "#d62728",
    "orchestrator_no_discussion": "#8c564b",
    "orchestrator_tree_structure":"#9467bd",
    "orchestrator_with_discussion":"#e377c2",
}

BENCHMARK_ORDER = [
    "browsecomp",
    "plancraft",
    "stabletoolbench",
    "workbench",
    "finance_agent",
]
BENCHMARK_LABELS = {
    "browsecomp":     "BrowseComp",
    "plancraft":      "PlanCraft",
    "stabletoolbench":"StableTool",
    "workbench":      "WorkBench",
    "finance_agent":  "Finance",
}

FAILURE_REGIME_COLORS = {
    "Unsupported hypothesis":   "#4878CF",
    "Evidence starvation":      "#F87F13",
    "Premature impossibility":  "#3CB371",
    "Wrong/invalid action":     "#DC143C",
    "Unsupported synthesis":    "#20B2AA",
    "Tool/interface fragility": "#9370DB",
    "Entity grounding failure": "#FF69B4",
    "Workflow blocking":        "#8B6914",
    "Financial support gap":    "#B0A898",
    "Stack/data brittleness":   "#395F8F",
}


def _save(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(prefix.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure 1: grouped bar chart of success rates ────────────────────────────

def plot_success_bars(rows: list[dict], out_prefix: Path) -> None:
    """Grouped bar chart: benchmarks on x-axis, one bar per topology."""
    data: dict[str, dict[str, float]] = {}
    for r in rows:
        bench = r["benchmark"]
        topo  = r["topology"]
        rate  = float(r["avg_success_rate"])
        data.setdefault(bench, {})[topo] = rate

    benchmarks = [b for b in BENCHMARK_ORDER if b in data]
    topos      = [t for t in TOPOLOGY_ORDER if any(t in data[b] for b in benchmarks)]

    n_bench = len(benchmarks)
    n_topo  = len(topos)
    width   = 0.11
    group_w = n_topo * width + 0.12
    x       = np.arange(n_bench) * group_w

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, topo in enumerate(topos):
        heights = [data[b].get(topo, 0.0) for b in benchmarks]
        offset  = (i - n_topo / 2 + 0.5) * width
        ax.bar(
            x + offset, heights, width,
            label=TOPOLOGY_LABELS[topo],
            color=TOPOLOGY_COLORS[topo],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_LABELS[b] for b in benchmarks], fontsize=11)
    ax.set_ylabel("Success rate", fontsize=11)
    ax.set_ylim(0, 0.70)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        title="Topology", title_fontsize=9,
        fontsize=8.5, ncol=4,
        loc="upper right",
        framealpha=0.85,
    )
    ax.set_title("Success rate by benchmark and topology", fontsize=12, pad=8)

    _save(fig, out_prefix)
    print(f"Saved {out_prefix}.pdf / .png")


# ── Figure 2: failure regimes (fixed legend) ────────────────────────────────

def plot_failure_regimes(rows: list[dict], out_prefix: Path) -> None:
    """Stacked bar chart of failure-regime counts, legend outside plot area."""
    # rows expected: benchmark, regime_label, count
    data: dict[str, dict[str, int]] = {}
    for r in rows:
        bench  = r["benchmark"]
        regime = r["regime_label"]
        count  = int(r["count"])
        data.setdefault(bench, {})[regime] = count

    benchmarks = [b for b in BENCHMARK_ORDER if b in data]
    # collect all regimes that appear
    all_regimes = list(FAILURE_REGIME_COLORS.keys())
    present = [reg for reg in all_regimes
               if any(reg in data[b] for b in benchmarks)]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(benchmarks))
    width = 0.55
    bottoms = np.zeros(len(benchmarks))

    handles = []
    for regime in present:
        heights = np.array([data[b].get(regime, 0) for b in benchmarks], dtype=float)
        bars = ax.bar(
            x, heights, width,
            bottom=bottoms,
            color=FAILURE_REGIME_COLORS[regime],
            label=regime,
        )
        bottoms += heights
        handles.append(bars[0])

    ax.set_xticks(x)
    ax.set_xticklabels(
        [BENCHMARK_LABELS[b] for b in benchmarks],
        fontsize=11,
    )
    ax.set_ylabel("Failed runs", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Failure regimes aggregated across topology slices", fontsize=12, pad=8)

    # legend outside to avoid overlap
    ax.legend(
        handles=handles,
        labels=present,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8.5,
        framealpha=0.85,
        title="Failure regime",
        title_fontsize=9,
    )

    _save(fig, out_prefix)
    print(f"Saved {out_prefix}.pdf / .png")


# ── I/O ─────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("figures/workshop_system_metrics.csv"),
    )
    parser.add_argument(
        "--failure-regimes", type=Path,
        default=None,
        help="CSV with columns: benchmark, regime_label, count",
    )
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    metrics = load_csv(args.metrics)
    plot_success_bars(metrics, args.out / "topology_success_bars")

    if args.failure_regimes and args.failure_regimes.exists():
        regime_rows = load_csv(args.failure_regimes)
        plot_failure_regimes(regime_rows, args.out / "failure_regimes_all_topologies")
    else:
        print("No --failure-regimes CSV supplied; skipping failure regime plot.")


if __name__ == "__main__":
    main()
