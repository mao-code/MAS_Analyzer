from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from .config import MASConfig


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    level: int
    agent_type: str


@dataclass
class Topology:
    agents: List[AgentSpec]
    adjacency: Dict[str, List[str]]

    def neighbors(self, agent_id: str) -> List[str]:
        return self.adjacency.get(agent_id, [])

    def level_to_agents(self) -> Dict[int, List[str]]:
        mapping: Dict[int, List[str]] = {}
        for agent in self.agents:
            mapping.setdefault(agent.level, []).append(agent.agent_id)
        return mapping


def build_topology(config: MASConfig, seed: int) -> Topology:
    """Build a deterministic topology from config knobs and a seed."""

    rng = random.Random(seed)
    agents_per_level = config.resolved_agents_per_level()

    agents: List[AgentSpec] = []
    level_to_agents: Dict[int, List[str]] = {}
    agent_type_count = len(config.agent_types)
    agent_index = 0

    for level in range(config.levels):
        level_to_agents[level] = []
        for _ in range(agents_per_level[level]):
            agent_id = f"agent_{agent_index}"
            agent_type = config.agent_types[agent_index % agent_type_count]
            spec = AgentSpec(agent_id=agent_id, level=level, agent_type=agent_type)
            agents.append(spec)
            level_to_agents[level].append(agent_id)
            agent_index += 1

    adjacency_sets: Dict[str, set[str]] = {agent.agent_id: set() for agent in agents}

    # Intra-level connectivity.
    for level_agents in level_to_agents.values():
        for i, src in enumerate(level_agents):
            for j in range(i + 1, len(level_agents)):
                dst = level_agents[j]
                if config.full_linked:
                    selected = True
                else:
                    selected = rng.random() <= config.intra_level_link_ratio
                if selected:
                    adjacency_sets[src].add(dst)
                    adjacency_sets[dst].add(src)

    # Cross-level connectivity: adjacent levels only, full bipartite links.
    for level in range(config.levels - 1):
        current_level_agents = level_to_agents.get(level, [])
        next_level_agents = level_to_agents.get(level + 1, [])
        for src in current_level_agents:
            for dst in next_level_agents:
                adjacency_sets[src].add(dst)
                adjacency_sets[dst].add(src)

    adjacency = {
        agent_id: sorted(list(neighbors))
        for agent_id, neighbors in adjacency_sets.items()
    }

    return Topology(agents=agents, adjacency=adjacency)
