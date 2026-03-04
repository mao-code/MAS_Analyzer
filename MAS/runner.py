from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from descriptor.schema import TraceEvent

from .config import ExperimentConfig
from .langgraph_engine import ExperimentSpec, LangGraphMASEngine
from .llm import OpenRouterLLMClient
from .monitor import DescriptorHook


@dataclass
class MASRunResult:
    final_answer: str
    trace_events: list[TraceEvent]
    run_metadata: dict[str, Any] = field(default_factory=dict)


class MASRunner:
    """MAS runtime backed by LangGraph with topology-aware relay semantics."""

    def __init__(self, config: ExperimentConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client
        self.engine = LangGraphMASEngine(llm_client)

    def run_task(
        self,
        task: Any,
        run_index: int,
        seed: int,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tool_iterations: int = 8,
        descriptor: DescriptorHook | None = None,
        topology: str | None = None,
        agents: int | None = None,
        rounds: int | None = None,
        discussion_rounds: int | None = None,
        group_sizes: list[int] | None = None,
        agents_per_level: list[int] | None = None,
    ) -> MASRunResult:
        mas_cfg = self.config.mas

        resolved_topology = topology or mas_cfg.resolved_topology()
        num_agents = int(agents) if agents is not None else mas_cfg.total_agents
        resolved_rounds = (
            int(rounds)
            if rounds is not None
            else (1 if mas_cfg.turn_mode == "single_turn" else mas_cfg.max_turns)
        )
        resolved_discussion_rounds = (
            int(discussion_rounds)
            if discussion_rounds is not None
            else int(mas_cfg.discussion_rounds)
        )

        resolved_agents_per_level: list[int] | None
        if agents_per_level is not None:
            resolved_agents_per_level = list(agents_per_level)
        elif mas_cfg.agents_per_level is not None and num_agents == mas_cfg.total_agents:
            resolved_agents_per_level = list(mas_cfg.agents_per_level)
        else:
            resolved_agents_per_level = None

        resolved_group_sizes: list[int] | None
        if group_sizes is not None:
            resolved_group_sizes = list(group_sizes)
        elif mas_cfg.group_sizes is not None and num_agents == mas_cfg.total_agents:
            resolved_group_sizes = list(mas_cfg.group_sizes)
        else:
            resolved_group_sizes = None

        spec = ExperimentSpec(
            topology=resolved_topology,
            num_agents=num_agents,
            rounds=max(1, int(resolved_rounds)),
            discussion_rounds=max(1, int(resolved_discussion_rounds)),
            communication_budget_per_agent=int(mas_cfg.communication_count_internally),
            agents_per_level=resolved_agents_per_level,
            group_sizes=resolved_group_sizes,
        )

        run_result = self.engine.run(
            task=task,
            run_index=run_index,
            seed=seed,
            spec=spec,
            agent_types=list(mas_cfg.agent_types),
            tools=list(tools or []),
            max_tool_iterations=max(1, int(max_tool_iterations)),
            descriptor=descriptor,
        )

        return MASRunResult(
            final_answer=run_result.final_answer,
            trace_events=run_result.trace_events,
            run_metadata=run_result.run_metadata,
        )
