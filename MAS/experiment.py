from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from descriptor.experiment import write_run_trace

from .config import ExperimentConfig, ExperimentRuntimeConfig, MASConfig, OpenRouterConfig
from .llm import OpenRouterLLMClient
from .monitor import DescriptorHook
from .runner import MASRunResult, MASRunner


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: Any
    reference_answer: str
    metadata: dict[str, Any]


def build_runtime_config(
    *,
    topology: str,
    agents: int,
    rounds: int,
    discussion_rounds: int = 1,
    communication_budget_per_agent: int = 1,
    termination_consensus_mode: str = "llm_judge",
    peer_artifact_max_chars: int = 320,
    agent_types: list[str] | None = None,
    output_dir: str = "outputs",
    seed: int = 42,
    model: str = "openai/gpt-4o-mini",
) -> ExperimentConfig:
    agent_types = list(agent_types or ["general"])

    return ExperimentConfig(
        openrouter=OpenRouterConfig(api_key=None),
        mas=MASConfig(
            levels=1,
            intra_level_link_ratio=0.0,
            full_linked=False,
            topology=topology,
            number_of_agents=max(1, int(agents)),
            agent_types=agent_types,
            communication_count_internally=max(0, int(communication_budget_per_agent)),
            turn_mode="single_turn" if int(rounds) <= 1 else "multi_turn",
            max_turns=max(1, int(rounds)),
            discussion_rounds=max(1, int(discussion_rounds)),
            termination_consensus_mode=str(termination_consensus_mode),
            peer_artifact_max_chars=max(32, int(peer_artifact_max_chars)),
        ),
        experiment=ExperimentRuntimeConfig(
            output_dir=output_dir,
            runs_per_task=1,
            seed=int(seed),
            task_limit=1,
        ),
        models={"default": model},
    )


def run_experiment(
    *,
    topology: str,
    agents: int,
    prompt: Any = "Solve the task and provide a concise final answer.",
    rounds: int = 1,
    discussion_rounds: int = 1,
    run_index: int = 0,
    seed: int = 42,
    descriptor: DescriptorHook | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_tool_iterations: int = 8,
    group_sizes: list[int] | None = None,
    agents_per_level: list[int] | None = None,
    config: ExperimentConfig | None = None,
    llm_client: OpenRouterLLMClient | None = None,
    task_id: str = "adhoc_task",
    output_trace_path: str | Path | None = None,
) -> MASRunResult:
    cfg = config or build_runtime_config(
        topology=topology,
        agents=agents,
        rounds=rounds,
        discussion_rounds=discussion_rounds,
    )
    cfg.validate()

    client = llm_client or OpenRouterLLMClient(cfg.openrouter, cfg.models)
    runner = MASRunner(cfg, client)

    task = _Task(
        task_id=str(task_id),
        prompt=prompt,
        reference_answer="",
        metadata={},
    )

    result = runner.run_task(
        task=task,
        run_index=run_index,
        seed=seed,
        tools=list(tools or []),
        max_tool_iterations=max_tool_iterations,
        descriptor=descriptor,
        topology=topology,
        agents=agents,
        rounds=rounds,
        discussion_rounds=discussion_rounds,
        group_sizes=group_sizes,
        agents_per_level=agents_per_level,
    )

    if output_trace_path is not None:
        path = Path(output_trace_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_run_trace(result.trace_events, path)

    return result
