"""Topology and workflow graph artifacts (matplotlib PNG rendering)."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any

from cli.common import _write_json
from MAS import MASRunner
from MAS.langgraph_engine import ExperimentSpec, LangGraphMASEngine


def _matplotlib_positions(layout: Any) -> dict[str, tuple[float, float]]:
    topology = str(layout.topology)
    positions: dict[str, tuple[float, float]] = {}

    if topology == "sas":
        positions[layout.agent_ids[0]] = (0.5, 0.5)
        return positions

    if topology == "orchestrator_tree_structure":
        levels = []
        root = [layout.orchestrator_id] if layout.orchestrator_id else []
        if root:
            levels.append(root)
        if layout.managers:
            levels.append(list(layout.managers))
        if layout.leaves:
            levels.append(list(layout.leaves))
        for level_index, agents in enumerate(levels):
            y = 1.0 - (level_index / max(1, len(levels) - 1 or 1))
            for item_index, agent_id in enumerate(agents):
                x = (item_index + 1) / (len(agents) + 1)
                positions[agent_id] = (x, y)
        return positions

    if topology in {"orchestrator_no_discussion", "orchestrator_with_discussion"}:
        if layout.orchestrator_id:
            positions[layout.orchestrator_id] = (0.5, 0.9)
        for index, agent_id in enumerate(layout.specialists):
            positions[agent_id] = ((index + 1) / (len(layout.specialists) + 1), 0.2)
        return positions

    if topology == "group_chat_debate" and layout.groups:
        group_count = len(layout.groups)
        for group_index, group in enumerate(layout.groups):
            x_center = (group_index + 1) / (group_count + 1)
            for member_index, agent_id in enumerate(group):
                y = 0.8 - (member_index * 0.25)
                positions[agent_id] = (x_center, max(0.15, y))
        return positions

    total = max(1, len(layout.agent_ids))
    for index, agent_id in enumerate(layout.agent_ids):
        angle = (2.0 * math.pi * index) / total
        positions[agent_id] = (
            0.5 + 0.34 * math.cos(angle),
            0.5 + 0.34 * math.sin(angle),
        )
    return positions


def _write_matplotlib_graph_png(path: Path, layout: Any) -> None:
    import matplotlib.pyplot as plt

    positions = _matplotlib_positions(layout)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.axis("off")

    drawn: set[tuple[str, str]] = set()
    for source, targets in layout.adjacency.items():
        x1, y1 = positions[source]
        for target in targets:
            key = tuple(sorted((source, target)))
            if key in drawn or target not in positions:
                continue
            drawn.add(key)
            x2, y2 = positions[target]
            ax.plot([x1, x2], [y1, y2], color="#7c8695", linewidth=1.4, alpha=0.8, zorder=1)

    palette = {
        "orchestrator": "#ecb939",
        "root_orchestrator": "#ecb939",
        "manager": "#4a90e2",
        "leaf_worker": "#50c878",
        "specialist": "#50c878",
        "voter": "#f28c8c",
        "debater": "#b38bfa",
        "single_agent": "#ff9f43",
    }
    for agent_id, (x, y) in positions.items():
        role = str(layout.roles.get(agent_id, "agent"))
        color = palette.get(role, "#6cc4c4" if "representative" in role else "#8cbf88")
        ax.scatter([x], [y], s=1800, c=color, edgecolors="#243447", linewidths=1.4, zorder=2)
        ax.text(
            x,
            y,
            f"{agent_id}\n{role}",
            ha="center",
            va="center",
            fontsize=9,
            color="#111827",
            zorder=3,
        )

    ax.set_title(f"MAS Topology: {layout.topology}", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_workflow_matplotlib_graph_png(path: Path, workflow: Any) -> None:
    import matplotlib.pyplot as plt

    node_ids = ["START", *list(workflow.nodes.keys()), "END"]
    positions = {node_id: (index, 0.0) for index, node_id in enumerate(node_ids)}
    edges = LangGraphMASEngine._workflow_edges_from_documentation(workflow)

    fig, ax = plt.subplots(figsize=(max(10.0, len(node_ids) * 1.8), 3.8))
    ax.axis("off")

    for edge in edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": "#7c8695", "linewidth": 1.4},
            zorder=1,
        )

    for node_id, (x, y) in positions.items():
        if node_id in {"START", "END"}:
            color = "#d0d7de"
        elif "dispatch" in node_id:
            color = "#ecb939"
        elif "controller" in node_id or "checker" in node_id:
            color = "#f28c8c"
        elif "judge" in node_id or "voter" in node_id:
            color = "#b38bfa"
        elif node_id == "finalize":
            color = "#50c878"
        else:
            color = "#6cc4c4"
        ax.scatter([x], [y], s=2200, c=color, edgecolors="#243447", linewidths=1.2, zorder=2)
        ax.text(x, y, node_id, ha="center", va="center", fontsize=8.5, color="#111827", zorder=3)

    ax.set_title(f"Workflow: {workflow.topology}", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_system_graph_artifact(
    *,
    runner: MASRunner,
    config: Any,
    run_root: Path,
) -> dict[str, Any]:
    if config.mas.resolved_topology() == "self_evolved":
        # The target topology is planned per run; per-run layouts live in
        # run_*.raw.json under run_metadata.topology_layout.
        payload = {
            "topology": "self_evolved",
            "dynamic": True,
            "render_backend": "none",
            "render_error": "",
            "note": "Topology is planned per run; see run_*.raw.json topology_layout.",
        }
        _write_json(run_root / "mas_graph.json", payload)
        return payload

    spec = ExperimentSpec(
        topology=config.mas.resolved_topology(),
        num_agents=config.mas.total_agents,
        rounds=max(1, int(config.mas.max_turns)),
        discussion_rounds=max(1, int(config.mas.discussion_rounds)),
        communication_budget_per_agent=int(config.mas.communication_count_internally),
        termination_consensus_mode=str(config.mas.termination_consensus_mode),
        peer_artifact_max_chars=int(config.mas.peer_artifact_max_chars),
        agents_per_level=(
            list(config.mas.agents_per_level) if config.mas.agents_per_level is not None else None
        ),
        group_sizes=(list(config.mas.group_sizes) if config.mas.group_sizes is not None else None),
    )

    graph_path = run_root / "mas_graph.png"
    mermaid_path = run_root / "mas_graph.mmd"
    metadata_path = run_root / "mas_graph.json"
    workflow_graph_path = run_root / "workflow_graph.png"
    workflow_mermaid_path = run_root / "workflow_graph.mmd"
    workflow_metadata_path = run_root / "workflow_graph.json"

    layout, visual_graph = runner.engine.build_topology_visual_graph(spec)
    mermaid_text = visual_graph.draw_mermaid()
    mermaid_path.write_text(mermaid_text, encoding="utf-8")

    render_backend = "langgraph_mermaid_api"
    render_error = ""
    try:
        png_bytes = visual_graph.draw_mermaid_png(
            output_file_path=str(graph_path),
            background_color="white",
            max_retries=0,
        )
        with contextlib.suppress(Exception):
            from IPython.display import Image as IPythonImage

            rendered = IPythonImage(data=png_bytes)
            if isinstance(getattr(rendered, "data", None), bytes | bytearray):
                png_bytes = bytes(rendered.data)
        graph_path.write_bytes(png_bytes)
    except Exception as exc:
        render_backend = "matplotlib_fallback"
        render_error = str(exc)
        _write_matplotlib_graph_png(graph_path, layout)

    workflow_definition, workflow_graph = runner.engine.build_workflow_visual_graph(spec)
    workflow_mermaid_text = workflow_graph.draw_mermaid()
    workflow_mermaid_path.write_text(workflow_mermaid_text, encoding="utf-8")

    workflow_render_backend = "langgraph_mermaid_api"
    workflow_render_error = ""
    try:
        workflow_png_bytes = workflow_graph.draw_mermaid_png(
            output_file_path=str(workflow_graph_path),
            background_color="white",
            max_retries=0,
        )
        with contextlib.suppress(Exception):
            from IPython.display import Image as IPythonImage

            rendered = IPythonImage(data=workflow_png_bytes)
            if isinstance(getattr(rendered, "data", None), bytes | bytearray):
                workflow_png_bytes = bytes(rendered.data)
        workflow_graph_path.write_bytes(workflow_png_bytes)
    except Exception as exc:
        workflow_render_backend = "matplotlib_fallback"
        workflow_render_error = str(exc)
        _write_workflow_matplotlib_graph_png(workflow_graph_path, workflow_definition)

    workflow_payload = {
        "topology": workflow_definition.topology,
        "render_backend": workflow_render_backend,
        "render_error": workflow_render_error,
        "png_path": str(workflow_graph_path.resolve()),
        "mermaid_path": str(workflow_mermaid_path.resolve()),
        "workflow": workflow_definition.to_payload(),
    }
    _write_json(workflow_metadata_path, workflow_payload)

    payload = {
        "topology": layout.topology,
        "render_backend": render_backend,
        "render_error": render_error,
        "png_path": str(graph_path.resolve()),
        "mermaid_path": str(mermaid_path.resolve()),
        "layout": layout.to_payload(),
        "workflow": workflow_payload,
    }
    _write_json(metadata_path, payload)
    return payload
