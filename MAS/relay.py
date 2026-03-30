from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .artifacts import ArtifactRecord, answer_signature

TOPOLOGY_SAS = "sas"
TOPOLOGY_ORCHESTRATOR_TREE = "orchestrator_tree_structure"
TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION = "orchestrator_no_discussion"
TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION = "orchestrator_with_discussion"
TOPOLOGY_ONLY_VOTING = "only_voting"
TOPOLOGY_FULLY_LINKED_DEBATE = "fully_linked_debate"
TOPOLOGY_GROUP_CHAT_DEBATE = "group_chat_debate"
TOPOLOGY_AUTO = "auto"

SUPPORTED_TOPOLOGIES = {
    TOPOLOGY_SAS,
    TOPOLOGY_ORCHESTRATOR_TREE,
    TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION,
    TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION,
    TOPOLOGY_ONLY_VOTING,
    TOPOLOGY_FULLY_LINKED_DEBATE,
    TOPOLOGY_GROUP_CHAT_DEBATE,
    TOPOLOGY_AUTO,
}

_TOPOLOGY_ALIASES = {
    "sas": TOPOLOGY_SAS,
    "single_agent": TOPOLOGY_SAS,
    "single_agent_system": TOPOLOGY_SAS,
    "single-agent-system": TOPOLOGY_SAS,
    "orchestrator_tree": TOPOLOGY_ORCHESTRATOR_TREE,
    "orchestrator_tree_structure": TOPOLOGY_ORCHESTRATOR_TREE,
    "orchestrator_tree_struct": TOPOLOGY_ORCHESTRATOR_TREE,
    "orchestrator_no_discussion": TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION,
    "orchestrator_nodiscussion": TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION,
    "orchestrator_with_discussion": TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION,
    "orchestrator_discussion": TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION,
    "only_voting": TOPOLOGY_ONLY_VOTING,
    "voting": TOPOLOGY_ONLY_VOTING,
    "majority_voting": TOPOLOGY_ONLY_VOTING,
    "fully_linked_debate": TOPOLOGY_FULLY_LINKED_DEBATE,
    "fully_connected_debate": TOPOLOGY_FULLY_LINKED_DEBATE,
    "group_chat_debate": TOPOLOGY_GROUP_CHAT_DEBATE,
    "group_chat_plus_debate": TOPOLOGY_GROUP_CHAT_DEBATE,
    "group_chat+debate": TOPOLOGY_GROUP_CHAT_DEBATE,
    "auto": TOPOLOGY_AUTO,
}


@dataclass(frozen=True)
class TopologyLayout:
    """Deterministic communication layout for one named topology."""

    topology: str
    agent_ids: list[str]
    adjacency: dict[str, list[str]]
    roles: dict[str, str]
    level_by_agent: dict[str, int]
    topological_order: list[str]
    orchestrator_id: str | None
    specialists: list[str]
    managers: list[str]
    leaves: list[str]
    groups: list[list[str]]
    representatives: list[str]
    parent_by_agent: dict[str, str]
    children_by_agent: dict[str, list[str]]
    agents_per_level: list[int]

    def to_payload(self) -> dict[str, Any]:
        return {
            "topology": self.topology,
            "agent_ids": list(self.agent_ids),
            "adjacency": {key: list(value) for key, value in self.adjacency.items()},
            "roles": dict(self.roles),
            "level_by_agent": dict(self.level_by_agent),
            "topological_order": list(self.topological_order),
            "orchestrator_id": self.orchestrator_id,
            "specialists": list(self.specialists),
            "managers": list(self.managers),
            "leaves": list(self.leaves),
            "groups": [list(group) for group in self.groups],
            "representatives": list(self.representatives),
            "parent_by_agent": dict(self.parent_by_agent),
            "children_by_agent": {key: list(value) for key, value in self.children_by_agent.items()},
            "agents_per_level": list(self.agents_per_level),
        }


def normalize_topology_name(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    normalized = _TOPOLOGY_ALIASES.get(key, key)
    if normalized not in SUPPORTED_TOPOLOGIES:
        raise ValueError(
            f"Unsupported topology '{name}'. Expected one of: {sorted(SUPPORTED_TOPOLOGIES - {TOPOLOGY_AUTO})}"
        )
    return normalized


def auto_topology_for_agents(num_agents: int) -> str:
    if num_agents <= 1:
        return TOPOLOGY_SAS
    return TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION


def build_layout(
    *,
    topology: str,
    num_agents: int,
    agents_per_level: list[int] | None = None,
    group_sizes: list[int] | None = None,
) -> TopologyLayout:
    """Build the deterministic communication structure for one topology."""

    if num_agents < 1:
        raise ValueError("num_agents must be >= 1")

    topo = normalize_topology_name(topology)
    if topo == TOPOLOGY_AUTO:
        topo = auto_topology_for_agents(num_agents)

    agent_ids = [f"agent_{idx}" for idx in range(num_agents)]
    adjacency: dict[str, set[str]] = {agent_id: set() for agent_id in agent_ids}
    roles = {agent_id: "agent" for agent_id in agent_ids}
    level_by_agent = {agent_id: 0 for agent_id in agent_ids}
    orchestrator_id: str | None = None
    specialists: list[str] = []
    managers: list[str] = []
    leaves: list[str] = []
    groups: list[list[str]] = []
    representatives: list[str] = []
    parent_by_agent: dict[str, str] = {}
    children_by_agent: dict[str, list[str]] = {agent_id: [] for agent_id in agent_ids}
    per_level: list[int] = [num_agents]
    topological_order = list(agent_ids)

    if topo == TOPOLOGY_SAS:
        roles[agent_ids[0]] = "single_agent"

    elif topo == TOPOLOGY_ORCHESTRATOR_TREE:
        orchestrator_id = agent_ids[0]
        levels = _resolve_tree_levels(num_agents, agents_per_level)
        per_level = list(levels)

        manager_count = levels[1]
        managers = agent_ids[1 : 1 + manager_count]
        leaves = agent_ids[1 + manager_count :]
        specialists = list(managers + leaves)

        roles[orchestrator_id] = "root_orchestrator"
        level_by_agent[orchestrator_id] = 0
        for manager in managers:
            roles[manager] = "manager"
            level_by_agent[manager] = 1
        for leaf in leaves:
            roles[leaf] = "leaf_worker"
            level_by_agent[leaf] = 2

        for manager in managers:
            _link(adjacency, orchestrator_id, manager)
            parent_by_agent[manager] = orchestrator_id
            children_by_agent[orchestrator_id].append(manager)

        for index, leaf in enumerate(leaves):
            manager = managers[index % len(managers)]
            _link(adjacency, manager, leaf)
            parent_by_agent[leaf] = manager
            children_by_agent[manager].append(leaf)

        topological_order = [orchestrator_id] + managers + leaves

    elif topo in {TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION, TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION}:
        orchestrator_id = agent_ids[0]
        roles[orchestrator_id] = "orchestrator"
        level_by_agent[orchestrator_id] = 0
        specialists = agent_ids[1:]
        per_level = [1, max(0, num_agents - 1)]
        for specialist in specialists:
            roles[specialist] = "specialist"
            level_by_agent[specialist] = 1
            _link(adjacency, orchestrator_id, specialist)
            parent_by_agent[specialist] = orchestrator_id
            children_by_agent[orchestrator_id].append(specialist)
        topological_order = [orchestrator_id] + specialists

    elif topo == TOPOLOGY_ONLY_VOTING:
        for agent_id in agent_ids:
            roles[agent_id] = "voter"

    elif topo == TOPOLOGY_FULLY_LINKED_DEBATE:
        for idx, src in enumerate(agent_ids):
            roles[src] = "debater"
            for dst in agent_ids[idx + 1 :]:
                _link(adjacency, src, dst)

    elif topo == TOPOLOGY_GROUP_CHAT_DEBATE:
        groups = _resolve_groups(agent_ids, group_sizes)
        representatives = [group[0] for group in groups if group]
        for group_idx, group in enumerate(groups):
            for member_idx, member in enumerate(group):
                level_by_agent[member] = group_idx
                if member_idx == 0:
                    roles[member] = f"group_{group_idx}_representative"
                else:
                    roles[member] = f"group_{group_idx}_member"
            for i, src in enumerate(group):
                for dst in group[i + 1 :]:
                    _link(adjacency, src, dst)
        specialists = list(agent_ids)
        per_level = [len(group) for group in groups]

    else:
        raise ValueError(f"Unhandled topology '{topo}'")

    adjacency_sorted = {
        key: sorted(list(neighbors)) for key, neighbors in adjacency.items()
    }
    children_payload = {
        key: sorted(list(value)) for key, value in children_by_agent.items() if value
    }

    return TopologyLayout(
        topology=topo,
        agent_ids=agent_ids,
        adjacency=adjacency_sorted,
        roles=roles,
        level_by_agent=level_by_agent,
        topological_order=topological_order,
        orchestrator_id=orchestrator_id,
        specialists=specialists,
        managers=managers,
        leaves=leaves,
        groups=groups,
        representatives=representatives,
        parent_by_agent=parent_by_agent,
        children_by_agent=children_payload,
        agents_per_level=per_level,
    )


def vote_majority(candidates: list[str]) -> tuple[str, dict[str, int]]:
    """Deterministic majority vote over raw candidate strings."""

    tally: dict[str, int] = {}
    surface: dict[str, str] = {}
    for text in candidates:
        signature = answer_signature(text)
        if not signature:
            continue
        tally[signature] = tally.get(signature, 0) + 1
        surface.setdefault(signature, str(text))

    if not tally:
        return ("", {})

    ranked = sorted(tally.items(), key=lambda item: (-item[1], item[0]))
    winner = ranked[0][0]
    return (surface[winner], dict(tally))


def vote_artifacts(artifacts: list[ArtifactRecord]) -> tuple[str, dict[str, int]]:
    """Deterministic vote over structured artifacts.

    Tie-break order:
    1. Higher vote count
    2. Higher mean confidence
    3. Lexicographically smaller canonical answer
    """

    groups: dict[str, list[ArtifactRecord]] = {}
    for artifact in artifacts:
        signature = answer_signature(artifact.get("answer", ""))
        if not signature:
            continue
        groups.setdefault(signature, []).append(artifact)

    if not groups:
        return ("", {})

    ranked = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            -(
                sum(float(artifact.get("confidence", 0.5)) for artifact in item[1])
                / len(item[1])
            ),
            item[0],
        ),
    )
    winner_signature, winner_group = ranked[0]
    best_artifact = sorted(
        winner_group,
        key=lambda artifact: (
            -float(artifact.get("confidence", 0.5)),
            str(artifact.get("agent_id", "")),
        ),
    )[0]
    tally = {signature: len(items) for signature, items in groups.items()}
    return (str(best_artifact.get("answer", "")), tally)


def _resolve_tree_levels(num_agents: int, agents_per_level: list[int] | None) -> list[int]:
    if agents_per_level is not None:
        if len(agents_per_level) != 3:
            raise ValueError(
                "Tree topology requires exactly 3 levels in agents_per_level, e.g. [1,2,4]"
            )
        if sum(agents_per_level) != num_agents:
            raise ValueError(
                "sum(agents_per_level) must equal num_agents for tree topology "
                f"({sum(agents_per_level)} != {num_agents})"
            )
        if agents_per_level[0] != 1:
            raise ValueError("Tree topology requires one root orchestrator at level 0")
        if min(agents_per_level[1:]) < 1:
            raise ValueError("Tree topology requires at least one manager and one leaf")
        return list(agents_per_level)

    if num_agents < 3:
        raise ValueError("Tree topology requires at least 3 agents")

    manager_count = 1 if num_agents <= 4 else 2
    max_managers = num_agents - 2
    manager_count = max(1, min(manager_count, max_managers))
    leaves_count = num_agents - 1 - manager_count
    return [1, manager_count, leaves_count]


def _resolve_groups(agent_ids: list[str], group_sizes: list[int] | None) -> list[list[str]]:
    if len(agent_ids) < 2:
        return [list(agent_ids)]

    if group_sizes is None:
        left = max(1, math.floor(len(agent_ids) / 2))
        right = len(agent_ids) - left
        if right == 0:
            return [list(agent_ids)]
        group_sizes = [left, right]

    if sum(group_sizes) != len(agent_ids):
        raise ValueError(
            "sum(group_sizes) must match num_agents "
            f"({sum(group_sizes)} != {len(agent_ids)})"
        )
    if any(size < 1 for size in group_sizes):
        raise ValueError("group_sizes entries must be >= 1")

    groups: list[list[str]] = []
    cursor = 0
    for size in group_sizes:
        groups.append(agent_ids[cursor : cursor + size])
        cursor += size
    return groups


def _link(adjacency: dict[str, set[str]], left: str, right: str) -> None:
    adjacency[left].add(right)
    adjacency[right].add(left)
