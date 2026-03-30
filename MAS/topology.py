from __future__ import annotations

from dataclasses import dataclass

from .config import MASConfig
from .relay import build_layout


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    level: int
    agent_type: str
    role: str = "agent"


@dataclass
class Topology:
    """Topology view derived from the named MAS layout configuration."""

    agents: list[AgentSpec]
    adjacency: dict[str, list[str]]

    def neighbors(self, agent_id: str) -> list[str]:
        return self.adjacency.get(agent_id, [])

    def level_to_agents(self) -> dict[int, list[str]]:
        mapping: dict[int, list[str]] = {}
        for agent in self.agents:
            mapping.setdefault(agent.level, []).append(agent.agent_id)
        return mapping


def build_topology(config: MASConfig, seed: int) -> Topology:
    """Build the deterministic named topology used by the LangGraph runtime.

    `seed` is retained for API compatibility. The current topology layouts are
    declarative and deterministic, so the seed does not affect the result.
    """

    _ = seed
    layout = build_layout(
        topology=config.resolved_topology(),
        num_agents=config.total_agents,
        agents_per_level=(
            list(config.agents_per_level) if config.agents_per_level is not None else None
        ),
        group_sizes=(list(config.group_sizes) if config.group_sizes is not None else None),
    )

    agents: list[AgentSpec] = []
    agent_type_count = len(config.agent_types)
    for index, agent_id in enumerate(layout.agent_ids):
        agents.append(
            AgentSpec(
                agent_id=agent_id,
                level=int(layout.level_by_agent.get(agent_id, 0)),
                agent_type=config.agent_types[index % agent_type_count],
                role=str(layout.roles.get(agent_id, "agent")),
            )
        )

    return Topology(
        agents=agents,
        adjacency={agent_id: list(neighbors) for agent_id, neighbors in layout.adjacency.items()},
    )
