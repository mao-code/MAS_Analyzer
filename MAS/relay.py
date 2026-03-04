from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

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
    topology: str
    agent_ids: list[str]
    adjacency: dict[str, list[str]]
    roles: dict[str, str]
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
    if num_agents < 1:
        raise ValueError("num_agents must be >= 1")

    topo = normalize_topology_name(topology)
    if topo == TOPOLOGY_AUTO:
        topo = auto_topology_for_agents(num_agents)

    agent_ids = [f"agent_{idx}" for idx in range(num_agents)]
    adjacency: dict[str, set[str]] = {agent_id: set() for agent_id in agent_ids}
    roles = {agent_id: "agent" for agent_id in agent_ids}
    orchestrator_id: str | None = None
    specialists: list[str] = []
    managers: list[str] = []
    leaves: list[str] = []
    groups: list[list[str]] = []
    representatives: list[str] = []
    parent_by_agent: dict[str, str] = {}
    children_by_agent: dict[str, list[str]] = {agent_id: [] for agent_id in agent_ids}
    per_level: list[int] = [num_agents]

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
        for manager in managers:
            roles[manager] = "manager"
        for leaf in leaves:
            roles[leaf] = "leaf_worker"

        for manager in managers:
            _link(adjacency, orchestrator_id, manager)
            parent_by_agent[manager] = orchestrator_id
            children_by_agent[orchestrator_id].append(manager)

        if not managers and leaves:
            managers = [orchestrator_id]

        for index, leaf in enumerate(leaves):
            manager = managers[index % len(managers)]
            _link(adjacency, manager, leaf)
            parent_by_agent[leaf] = manager
            children_by_agent.setdefault(manager, []).append(leaf)

    elif topo in {TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION, TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION}:
        orchestrator_id = agent_ids[0]
        roles[orchestrator_id] = "orchestrator"
        specialists = agent_ids[1:]
        per_level = [1, max(0, num_agents - 1)]
        for specialist in specialists:
            roles[specialist] = "specialist"
            _link(adjacency, orchestrator_id, specialist)
            parent_by_agent[specialist] = orchestrator_id
            children_by_agent[orchestrator_id].append(specialist)

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
                if member_idx == 0:
                    roles[member] = f"group_{group_idx}_representative"
                else:
                    roles[member] = f"group_{group_idx}_member"
            for i, src in enumerate(group):
                for dst in group[i + 1 :]:
                    _link(adjacency, src, dst)
        specialists = list(agent_ids)

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
    leaves = num_agents - 1 - manager_count
    if leaves < 1:
        leaves = 1
        manager_count = num_agents - 2
    return [1, manager_count, leaves]


def _resolve_groups(agent_ids: list[str], group_sizes: list[int] | None) -> list[list[str]]:
    if len(agent_ids) < 2:
        return [list(agent_ids)]

    if group_sizes is None:
        left = max(1, math.floor(len(agent_ids) / 2))
        right = len(agent_ids) - left
        if right == 0:
            return [list(agent_ids)]
        group_sizes = [left, right]

    if not group_sizes:
        raise ValueError("group_sizes must not be empty")
    if sum(group_sizes) != len(agent_ids):
        raise ValueError(
            f"sum(group_sizes) must equal number of agents ({sum(group_sizes)} != {len(agent_ids)})"
        )
    if any(size < 1 for size in group_sizes):
        raise ValueError("All group_sizes entries must be >= 1")

    groups: list[list[str]] = []
    cursor = 0
    for size in group_sizes:
        groups.append(agent_ids[cursor : cursor + size])
        cursor += size
    return groups


def _link(adjacency: dict[str, set[str]], a: str, b: str) -> None:
    adjacency[a].add(b)
    adjacency[b].add(a)


def extract_topology_messages_for_agent(
    *,
    topology: str,
    phase: str,
    round_index: int,
    agent_id: str,
    messages: list[dict[str, Any]],
    layout: TopologyLayout,
) -> list[dict[str, Any]]:
    topo = normalize_topology_name(topology)
    if topo == TOPOLOGY_AUTO:
        topo = auto_topology_for_agents(len(layout.agent_ids))

    visible = [item for item in messages if agent_id in item.get("recipients", [])]

    if topo == TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION and phase == "specialist_solve":
        # Specialists only receive orchestrator directives, no peer outputs.
        visible = [item for item in visible if item.get("sender") == layout.orchestrator_id]

    if topo == TOPOLOGY_ORCHESTRATOR_TREE and phase == "leaf_work":
        # Leaves only consume direct parent manager instructions.
        parent = layout.parent_by_agent.get(agent_id)
        if parent is not None:
            visible = [item for item in visible if item.get("sender") == parent]

    if topo == TOPOLOGY_GROUP_CHAT_DEBATE and phase == "group_debate":
        group = _group_for_agent(layout, agent_id)
        if group:
            allowed = set(group)
            visible = [item for item in visible if item.get("sender") in allowed]

    if topo == TOPOLOGY_FULLY_LINKED_DEBATE and phase == "debate":
        visible = [
            item
            for item in visible
            if item.get("round", -1) == round_index - 1 or item.get("phase") == "seed"
        ]

    return sorted(visible, key=lambda item: str(item.get("message_id", "")))


def _group_for_agent(layout: TopologyLayout, agent_id: str) -> list[str]:
    for group in layout.groups:
        if agent_id in group:
            return group
    return []


def vote_majority(candidates: list[str]) -> tuple[str, dict[str, int]]:
    tally: dict[str, int] = {}
    canonical_to_original: dict[str, str] = {}

    for text in candidates:
        canonical = _canonical_vote(text)
        tally[canonical] = tally.get(canonical, 0) + 1
        canonical_to_original.setdefault(canonical, text)

    if not tally:
        return "", {}

    winner = sorted(tally.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return canonical_to_original[winner], tally


def _canonical_vote(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return normalized[:280]
