from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


DEFAULT_ROOT = Path(
    "artifacts/full_experiment/20260426_qwen3_32b_paper_run__qwen_qwen3_32b_nitro"
)
STATES = ("ok", "sparse", "error")
STATE_LABELS = {"ok": "OK", "sparse": "Sparse", "error": "Error"}
TOPOLOGY_ORDER = [
    "sas",
    "only_voting",
    "fully_linked_debate",
    "group_chat_debate",
    "orchestrator_no_discussion",
    "orchestrator_tree_structure",
    "orchestrator_with_discussion",
]


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(errors="ignore") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def eval_success(eval_path: Path) -> bool:
    try:
        data = read_json(eval_path)
    except FileNotFoundError:
        return False
    return bool(data.get("success") or data.get("outcome", {}).get("success"))


def completed_cells(root: Path) -> set[tuple[str, str]]:
    """Benchmark/topology cells present in the aggregate experiment export."""

    summary_path = root / "experiment_summary.csv"
    if not summary_path.exists():
        return set()
    with summary_path.open() as handle:
        return {
            (row["benchmark"], row["topology"])
            for row in csv.DictReader(handle)
            if row.get("benchmark") and row.get("topology")
        }


def matching_eval_for_trace(trace_path: Path) -> Path:
    return trace_path.parent / trace_path.name.replace(".trace.jsonl", ".eval.json")


def classify_tool_result(payload: dict) -> str:
    """Coarse state for a tool result payload.

    The traces do not expose a benchmark-independent semantic success flag, so
    this deliberately uses conservative surface cues. A result is "error" when
    the status or preview reports failure-like behavior, "sparse" when the
    preview is empty or degenerate, and "ok" otherwise.
    """

    status = str(payload.get("status", "")).lower()
    preview = str(payload.get("output_preview", "")).strip()
    text = json.dumps(payload, ensure_ascii=False).lower()
    error_terms = (
        "cache miss",
        "exception",
        "traceback",
        "rate limit",
        "429",
        "missing_information",
        "invalid",
        "failed",
        "error",
        "timeout",
    )
    sparse_terms = (
        "no data",
        "not found",
        "empty",
        "null",
        "none",
        "[]",
        "{}",
    )
    if status in {"error", "failed", "timeout"} or any(term in text for term in error_terms):
        return "error"
    if len(preview) < 20 or any(term in preview.lower() for term in sparse_terms):
        return "sparse"
    return "ok"


def collect_tool_markov(root: Path) -> tuple[list[dict], list[dict], Counter]:
    transitions: dict[str, Counter] = {"success": Counter(), "fail": Counter()}
    row_totals: dict[str, Counter] = {"success": Counter(), "fail": Counter()}
    run_rows: list[dict] = []
    same_tool_loops: Counter = Counter()

    for trace_path in sorted((root / "stabletoolbench").glob("*/*/run_*.trace.jsonl")):
        success = eval_success(matching_eval_for_trace(trace_path))
        outcome = "success" if success else "fail"
        state_seq: list[str] = []
        tool_seq: list[str] = []

        for event in iter_jsonl(trace_path):
            if event.get("event_type") != "tool_result":
                continue
            payload = event.get("payload", {})
            tool_name = payload.get("tool_name") or payload.get("name") or ""
            if tool_name == "inter_agent_send":
                continue
            state_seq.append(classify_tool_result(payload))
            tool_seq.append(tool_name)

        for src, dst in zip(state_seq, state_seq[1:]):
            transitions[outcome][(src, dst)] += 1
            row_totals[outcome][src] += 1

        for prev_tool, next_tool in zip(tool_seq, tool_seq[1:]):
            if prev_tool == next_tool and prev_tool:
                same_tool_loops[(outcome, prev_tool)] += 1

        max_adverse_streak = 0
        current = 0
        for state in state_seq:
            if state == "ok":
                current = 0
            else:
                current += 1
                max_adverse_streak = max(max_adverse_streak, current)

        run_rows.append(
            {
                "benchmark": "stabletoolbench",
                "topology": trace_path.parts[-3],
                "task_id": trace_path.parts[-2],
                "run": trace_path.stem.replace(".trace", ""),
                "outcome": outcome,
                "tool_results": len(state_seq),
                "max_adverse_streak": max_adverse_streak,
                "same_tool_repeats": sum(
                    1 for a, b in zip(tool_seq, tool_seq[1:]) if a == b and a
                ),
            }
        )

    rows: list[dict] = []
    for outcome in ("success", "fail"):
        for src in STATES:
            denom = row_totals[outcome][src]
            for dst in STATES:
                count = transitions[outcome][(src, dst)]
                rows.append(
                    {
                        "outcome": outcome,
                        "from_state": src,
                        "to_state": dst,
                        "count": count,
                        "row_total": denom,
                        "probability": count / denom if denom else 0.0,
                    }
                )

    return rows, run_rows, same_tool_loops


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_vote_entropy(root: Path) -> tuple[list[dict], list[dict]]:
    allowed_cells = completed_cells(root)
    rows: list[dict] = []
    for metadata_path in sorted(root.glob("*/*/*/run_*.metadata.json")):
        rel = metadata_path.relative_to(root)
        benchmark, topology, task_id = rel.parts[:3]
        if allowed_cells and (benchmark, topology) not in allowed_cells:
            continue
        data = read_json(metadata_path)
        vote_tally = data.get("vote_tally") or {}
        if not vote_tally:
            continue
        counts = []
        for value in vote_tally.values():
            try:
                counts.append(float(value))
            except (TypeError, ValueError):
                continue
        total = sum(counts)
        if not counts or total <= 0:
            continue
        probabilities = [count / total for count in counts if count > 0]
        entropy = -sum(p * math.log(p, 2) for p in probabilities)
        normalized_entropy = entropy / math.log(len(probabilities), 2) if len(probabilities) > 1 else 0.0
        eval_path = metadata_path.with_name(metadata_path.name.replace(".metadata.json", ".eval.json"))
        rows.append(
            {
                "benchmark": benchmark,
                "topology": topology,
                "task_id": task_id,
                "run": metadata_path.stem.replace(".metadata", ""),
                "outcome": "success" if eval_success(eval_path) else "fail",
                "vote_options": len(probabilities),
                "vote_total": int(total),
                "entropy_bits": entropy,
                "normalized_entropy": normalized_entropy,
                "max_vote_share": max(probabilities),
                "single_option": int(len(probabilities) == 1),
            }
        )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["topology"], row["outcome"])].append(row)
    summary: list[dict] = []
    for (topology, outcome), group in sorted(grouped.items()):
        n = len(group)
        summary.append(
            {
                "topology": topology,
                "outcome": outcome,
                "runs": n,
                "mean_entropy_bits": sum(row["entropy_bits"] for row in group) / n,
                "mean_normalized_entropy": sum(row["normalized_entropy"] for row in group) / n,
                "mean_max_vote_share": sum(row["max_vote_share"] for row in group) / n,
                "single_option_rate": sum(row["single_option"] for row in group) / n,
            }
        )
    return rows, summary


def plot_vote_entropy(summary: list[dict], output_prefix: Path) -> None:
    topologies = [
        "only_voting",
        "fully_linked_debate",
        "group_chat_debate",
    ]
    labels = ["Voting", "Fully linked", "Group chat"]
    outcomes = ["success", "fail"]
    colors = {"success": "#5B8FF9", "fail": "#E8684A"}
    lookup = {(row["topology"], row["outcome"]): row for row in summary}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    width = 0.36
    x = list(range(len(topologies)))
    metrics = [
        ("mean_normalized_entropy", "Mean normalized entropy"),
        ("single_option_rate", "Single-option tally rate"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for offset, outcome in zip((-width / 2, width / 2), outcomes):
            vals = [lookup.get((topology, outcome), {}).get(metric, 0.0) for topology in topologies]
            bars = ax.bar(
                [pos + offset for pos in x],
                vals,
                width=width,
                color=colors[outcome],
                label=outcome.capitalize(),
                edgecolor="white",
                linewidth=0.7,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(val + 0.025, 1.02),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("Rate / normalized entropy", fontsize=8)
    axes[1].legend(loc="upper left", fontsize=7, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=450)
    plt.close(fig)


def plot_markov(rows: list[dict], output_prefix: Path) -> None:
    lookup = {
        (row["outcome"], row["from_state"], row["to_state"]): row for row in rows
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.1), constrained_layout=True)
    for ax, outcome, title in zip(
        axes, ("success", "fail"), ("Successful runs", "Failed runs")
    ):
        matrix = [
            [
                lookup[(outcome, src, dst)]["probability"]
                for dst in STATES
            ]
            for src in STATES
        ]
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(STATES)), [STATE_LABELS[s] for s in STATES], fontsize=8)
        ax.set_yticks(range(len(STATES)), [STATE_LABELS[s] for s in STATES], fontsize=8)
        ax.set_xlabel("Next tool result state", fontsize=8)
        ax.set_ylabel("Current state", fontsize=8)
        for i, src in enumerate(STATES):
            for j, dst in enumerate(STATES):
                row = lookup[(outcome, src, dst)]
                ax.text(
                    j,
                    i,
                    f"{row['probability']:.2f}\n(n={row['count']})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if row["probability"] < 0.65 else "white",
                )
    fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02, label="Transition probability")
    for ext in ("pdf", "png"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=450)
    plt.close(fig)


def prediction_text(eval_data: dict) -> str:
    details = eval_data.get("details", {})
    details_prediction = details.get("prediction", "")
    if not isinstance(details_prediction, str):
        details_prediction = ""
    judge_raw = details.get("judge_raw", {})
    pieces = [
        eval_data.get("prediction", ""),
        eval_data.get("final_answer", ""),
        details_prediction,
        details.get("reason", ""),
        judge_raw.get("reason", "") if isinstance(judge_raw, dict) else "",
        details.get("error", ""),
    ]
    return " ".join(str(piece) for piece in pieces if piece).lower()


def trace_has_adverse_tool_state(eval_path: Path) -> bool:
    trace_path = eval_path.parent / eval_path.name.replace(".eval.json", ".trace.jsonl")
    if not trace_path.exists():
        return False
    for event in iter_jsonl(trace_path):
        if event.get("event_type") != "tool_result":
            continue
        payload = event.get("payload", {})
        if (payload.get("tool_name") or "") == "inter_agent_send":
            continue
        if classify_tool_result(payload) != "ok":
            return True
    return False


def trace_has_adverse_tool(eval_path: Path, tool_prefix: str) -> bool:
    trace_path = eval_path.parent / eval_path.name.replace(".eval.json", ".trace.jsonl")
    if not trace_path.exists():
        return False
    for event in iter_jsonl(trace_path):
        if event.get("event_type") != "tool_result":
            continue
        payload = event.get("payload", {})
        tool_name = payload.get("tool_name") or ""
        if not tool_name.startswith(tool_prefix):
            continue
        if classify_tool_result(payload) != "ok":
            return True
    return False


def classify_first_fault(eval_path: Path, eval_data: dict, root: Path) -> tuple[str, str, str]:
    benchmark = eval_path.relative_to(root).parts[0]
    text = prediction_text(eval_data)

    entity_terms = (
        "no email",
        "missing participant email",
        "email not found",
        "no email found",
        "email resolution",
        "assigned contact",
        "assignee",
        "sender",
        "recipient",
        "not found in the company directory",
        "not found in company directory",
    )
    tool_terms = (
        "api",
        "tool",
        "cache miss",
        "parameter",
        "endpoint",
        "rate limiting",
        "rate limit",
        "html parsing",
        "edgar",
        "invalid",
        "error",
    )
    evidence_terms = (
        "insufficient",
        "missing",
        "not enough",
        "no direct evidence",
        "not recovered",
        "unspecified",
        "unsupported",
        "unknown",
        "blocked",
        "lack of",
        "lack relevant",
        "lacks",
        "no actionable",
        "no retrieved",
        "could not be retrieved",
        "unable to retrieve",
    )

    if benchmark == "workbench":
        source = (
            "entity_grounding"
            if any(t in text for t in entity_terms)
            or trace_has_adverse_tool(eval_path, "company_directory.find_email_address")
            else "workflow_precondition"
        )
    elif benchmark == "plancraft":
        source = "premature_impossibility" if "impossible" in text else "invalid_or_wrong_action"
    elif benchmark == "stabletoolbench":
        if any(t in text for t in tool_terms) or trace_has_adverse_tool_state(eval_path):
            source = "tool_interface"
        elif any(t in text for t in evidence_terms):
            source = "evidence_gap"
        else:
            source = "unsupported_synthesis"
    elif benchmark == "finance_agent":
        if any(t in text for t in tool_terms):
            source = "data_interface_brittleness"
        else:
            source = "financial_support_gap"
    elif benchmark == "browsecomp":
        if any(t in text for t in evidence_terms):
            source = "retrieval_evidence_gap"
        else:
            source = "unsupported_hypothesis"
    else:
        source = "other"

    propagation = {
        "retrieval_evidence_gap": "contaminated_evidence_state",
        "unsupported_hypothesis": "contaminated_evidence_state",
        "premature_impossibility": "invalid_terminal_state",
        "invalid_or_wrong_action": "invalid_terminal_state",
        "tool_interface": "low_information_tool_state",
        "evidence_gap": "contaminated_evidence_state",
        "unsupported_synthesis": "unsupported_synthesis_state",
        "entity_grounding": "unresolved_action_binding",
        "workflow_precondition": "unresolved_action_binding",
        "data_interface_brittleness": "unsupported_quantitative_context",
        "financial_support_gap": "unsupported_quantitative_context",
    }.get(source, "other_propagation")

    terminal = {
        "browsecomp": "browse_failure",
        "plancraft": "planning_failure",
        "stabletoolbench": "tool_task_failure",
        "workbench": "workflow_failure",
        "finance_agent": "finance_failure",
    }.get(benchmark, "other_failure")
    return source, propagation, terminal


def collect_first_faults(root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    allowed_cells = completed_cells(root)
    for eval_path in sorted(root.glob("*/*/*/run_*.eval.json")):
        rel = eval_path.relative_to(root)
        if allowed_cells and (rel.parts[0], rel.parts[1]) not in allowed_cells:
            continue
        data = read_json(eval_path)
        if data.get("success") or data.get("outcome", {}).get("success"):
            continue
        source, propagation, terminal = classify_first_fault(eval_path, data, root)
        rows.append(
            {
                "benchmark": rel.parts[0],
                "topology": rel.parts[1],
                "task_id": rel.parts[2],
                "run": eval_path.stem.replace(".eval", ""),
                "source": source,
                "propagation": propagation,
                "terminal": terminal,
            }
        )

    counts = Counter((row["benchmark"], row["source"]) for row in rows)
    totals = Counter(row["benchmark"] for row in rows)
    summary = [
        {
            "benchmark": benchmark,
            "source": source,
            "count": count,
            "total": totals[benchmark],
            "share": count / totals[benchmark] if totals[benchmark] else 0.0,
        }
        for (benchmark, source), count in sorted(counts.items())
    ]
    return rows, summary


def plot_topology_fault_heatmaps(rows: list[dict], output_prefix: Path) -> None:
    benchmarks = ["browsecomp", "plancraft", "stabletoolbench", "workbench", "finance_agent"]
    source_order = [
        "retrieval_evidence_gap",
        "unsupported_hypothesis",
        "premature_impossibility",
        "invalid_or_wrong_action",
        "tool_interface",
        "evidence_gap",
        "unsupported_synthesis",
        "entity_grounding",
        "workflow_precondition",
        "data_interface_brittleness",
        "financial_support_gap",
    ]
    counts = Counter((row["benchmark"], row["topology"], row["source"]) for row in rows)
    totals = Counter((row["benchmark"], row["topology"]) for row in rows)

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8), constrained_layout=True)
    axes = axes.flatten()
    last_image = None
    for ax, benchmark in zip(axes, benchmarks):
        sources = [s for s in source_order if any(row["benchmark"] == benchmark and row["source"] == s for row in rows)]
        topologies = [t for t in TOPOLOGY_ORDER if totals[(benchmark, t)]]
        matrix = [
            [
                counts[(benchmark, topology, source)] / totals[(benchmark, topology)]
                if totals[(benchmark, topology)]
                else 0.0
                for source in sources
            ]
            for topology in topologies
        ]
        last_image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_title(benchmark.replace("_", " "), fontsize=10)
        ax.set_xticks(
            range(len(sources)),
            [s.replace("_", "\n") for s in sources],
            fontsize=6.7,
            rotation=0,
        )
        ax.set_yticks(
            range(len(topologies)),
            [t.replace("orchestrator_", "orch_").replace("_", "\n") for t in topologies],
            fontsize=7.0,
        )
        for i, topology in enumerate(topologies):
            for j, source in enumerate(sources):
                val = matrix[i][j]
                if val >= 0.08:
                    ax.text(
                        j,
                        i,
                        f"{val:.0%}",
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color="white" if val > 0.55 else "black",
                    )
    axes[-1].axis("off")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, fraction=0.025, pad=0.01, label="Failure share")
    for ext in ("pdf", "png"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=450)
    plt.close(fig)


FAULT_COLORS = {
    "retrieval_evidence_gap": "#5B8FF9",
    "unsupported_hypothesis": "#8AB6FF",
    "premature_impossibility": "#F6BD16",
    "invalid_or_wrong_action": "#F9D976",
    "tool_interface": "#E8684A",
    "evidence_gap": "#A6CEE3",
    "unsupported_synthesis": "#F28E8C",
    "entity_grounding": "#61DDAA",
    "workflow_precondition": "#9BE7C7",
    "data_interface_brittleness": "#B37FEB",
    "financial_support_gap": "#D3ADF7",
}


def plot_first_fault_distribution(summary: list[dict], output_prefix: Path) -> None:
    benchmarks = ["browsecomp", "plancraft", "stabletoolbench", "workbench", "finance_agent"]
    sources = sorted({row["source"] for row in summary})
    shares = defaultdict(dict)
    labels = defaultdict(dict)
    for row in summary:
        shares[row["benchmark"]][row["source"]] = row["share"]
        labels[row["benchmark"]][row["source"]] = row["count"]

    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    left = [0.0 for _ in benchmarks]
    y = list(range(len(benchmarks)))
    for source in sources:
        vals = [shares[b].get(source, 0.0) for b in benchmarks]
        ax.barh(
            y,
            vals,
            left=left,
            color=FAULT_COLORS.get(source, "#BFBFBF"),
            edgecolor="white",
            height=0.72,
            label=source.replace("_", " "),
        )
        for idx, val in enumerate(vals):
            if val >= 0.12:
                ax.text(
                    left[idx] + val / 2,
                    idx,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(y, [b.replace("_", " ") for b in benchmarks], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of failed runs", fontsize=8)
    ax.set_title("First critical fault localization by benchmark", fontsize=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=6.5,
        frameon=False,
    )
    for ext in ("pdf", "png"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=450)
    plt.close(fig)


def bezier_ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.55):
    cx0 = x0 + (x1 - x0) * 0.45
    cx1 = x0 + (x1 - x0) * 0.55
    verts = [
        (x0, y0a),
        (cx0, y0a),
        (cx1, y1a),
        (x1, y1a),
        (x1, y1b),
        (cx1, y1b),
        (cx0, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def plot_fault_sankey(rows: list[dict], output_prefix: Path) -> None:
    flows = Counter((row["source"], row["propagation"], row["terminal"]) for row in rows)
    total = sum(flows.values())
    if total == 0:
        return

    columns = [
        sorted({s for s, _, _ in flows}),
        sorted({p for _, p, _ in flows}),
        sorted({t for _, _, t in flows}),
    ]
    counts = [
        Counter({node: sum(v for (s, _, _), v in flows.items() if s == node) for node in columns[0]}),
        Counter({node: sum(v for (_, p, _), v in flows.items() if p == node) for node in columns[1]}),
        Counter({node: sum(v for (_, _, t), v in flows.items() if t == node) for node in columns[2]}),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x_positions = [0.08, 0.50, 0.92]
    node_pos: list[dict[str, tuple[float, float]]] = []

    for col_idx, nodes in enumerate(columns):
        y_top = 0.94
        gap = 0.012
        scale = (0.86 - gap * (len(nodes) - 1)) / total
        pos = {}
        for node in nodes:
            height = max(counts[col_idx][node] * scale, 0.012)
            y_bottom = y_top - height
            pos[node] = (y_bottom, y_top)
            color = FAULT_COLORS.get(node, "#D9D9D9")
            if col_idx == 1:
                color = "#D9D9D9"
            if col_idx == 2:
                color = "#F0F0F0"
            ax.add_patch(
                Rectangle(
                    (x_positions[col_idx] - 0.022, y_bottom),
                    0.044,
                    height,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            label = node.replace("_", " ")
            ha = "right" if col_idx == 0 else "left"
            x_text = x_positions[col_idx] - 0.03 if col_idx == 0 else x_positions[col_idx] + 0.03
            ax.text(x_text, (y_bottom + y_top) / 2, label, fontsize=6.3, va="center", ha=ha)
            y_top = y_bottom - gap
        node_pos.append(pos)

    offsets = [defaultdict(float), defaultdict(float), defaultdict(float)]
    for source, propagation, terminal in sorted(flows):
        value = flows[(source, propagation, terminal)]
        h0 = (node_pos[0][source][1] - node_pos[0][source][0]) * value / counts[0][source]
        h1 = (node_pos[1][propagation][1] - node_pos[1][propagation][0]) * value / counts[1][propagation]
        y0a = node_pos[0][source][0] + offsets[0][source]
        y0b = y0a + h0
        y1a = node_pos[1][propagation][0] + offsets[1][propagation]
        y1b = y1a + h1
        bezier_ribbon(
            ax,
            x_positions[0] + 0.022,
            y0a,
            y0b,
            x_positions[1] - 0.022,
            y1a,
            y1b,
            FAULT_COLORS.get(source, "#BFBFBF"),
            alpha=0.35,
        )
        offsets[0][source] += h0
        offsets[1][propagation] += h1

    offsets = [defaultdict(float), defaultdict(float), defaultdict(float)]
    mid_terminal = Counter((p, t) for _, p, t in flows for _ in range(0))
    for propagation, terminal in sorted({(p, t) for _, p, t in flows}):
        value = sum(v for (s, p, t), v in flows.items() if p == propagation and t == terminal)
        h1 = (node_pos[1][propagation][1] - node_pos[1][propagation][0]) * value / counts[1][propagation]
        h2 = (node_pos[2][terminal][1] - node_pos[2][terminal][0]) * value / counts[2][terminal]
        y1a = node_pos[1][propagation][0] + offsets[1][propagation]
        y1b = y1a + h1
        y2a = node_pos[2][terminal][0] + offsets[2][terminal]
        y2b = y2a + h2
        bezier_ribbon(
            ax,
            x_positions[1] + 0.022,
            y1a,
            y1b,
            x_positions[2] - 0.022,
            y2a,
            y2b,
            "#A0A0A0",
            alpha=0.25,
        )
        offsets[1][propagation] += h1
        offsets[2][terminal] += h2

    ax.text(0.08, 0.99, "First fault", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.text(0.50, 0.99, "Contaminated state", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.text(0.92, 0.99, "Terminal regime", ha="center", va="bottom", fontsize=9, weight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=450)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    markov_rows, run_rows, same_tool_loops = collect_tool_markov(args.root)
    write_csv(args.out / "stabletoolbench_tool_markov.csv", markov_rows)
    write_csv(args.out / "stabletoolbench_tool_run_stats.csv", run_rows)
    write_csv(
        args.out / "stabletoolbench_same_tool_loops.csv",
        [
            {"outcome": outcome, "tool_name": tool, "count": count}
            for (outcome, tool), count in same_tool_loops.most_common()
        ],
    )
    plot_markov(markov_rows, args.out / "stabletoolbench_tool_markov")

    fault_rows, fault_summary = collect_first_faults(args.root)
    write_csv(args.out / "first_fault_localization.csv", fault_rows)
    write_csv(args.out / "first_fault_summary.csv", fault_summary)
    plot_first_fault_distribution(fault_summary, args.out / "first_fault_distribution")
    plot_fault_sankey(fault_rows, args.out / "fault_propagation_sankey")
    plot_topology_fault_heatmaps(fault_rows, args.out / "topology_failure_regime_heatmaps")

    vote_rows, vote_summary = collect_vote_entropy(args.root)
    write_csv(args.out / "vote_entropy_runs.csv", vote_rows)
    write_csv(args.out / "vote_entropy_summary.csv", vote_summary)
    plot_vote_entropy(vote_summary, args.out / "vote_entropy_by_topology")

    print("Wrote trace diagnostics to", args.out)


if __name__ == "__main__":
    main()
