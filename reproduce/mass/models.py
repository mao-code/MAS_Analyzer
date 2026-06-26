from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentPromptBundle:
    """Prompt payload for one logical building block."""

    system_instruction: str
    exemplar: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSpec:
    """Concrete workflow sampled from the MASS search space."""

    summarize_rounds: int = 0
    aggregate_width: int = 1
    reflect_rounds: int = 0
    debate_rounds: int = 0
    execute_enabled: bool = False
    order: tuple[str, ...] = ("summarize", "reflect", "debate", "aggregate")

    def active_blocks(self) -> list[str]:
        blocks: list[str] = []
        blocks.append("predictor")
        if self.summarize_rounds > 0:
            blocks.append("summarize")
        if self.reflect_rounds > 0:
            blocks.append("reflect")
        if self.debate_rounds > 0:
            blocks.append("debate")
        if self.aggregate_width > 1:
            blocks.append("aggregate")
        if self.execute_enabled:
            blocks.append("execute")
        return blocks

    @property
    def estimated_agent_count(self) -> int:
        predictor_count = max(1, self.aggregate_width)
        if self.debate_rounds > 0:
            predictor_count = max(predictor_count, 2)
        specialist_count = 0
        if self.summarize_rounds > 0:
            specialist_count += 1
        if self.reflect_rounds > 0:
            specialist_count += 1
        if self.debate_rounds > 0:
            specialist_count += 1
        if self.execute_enabled:
            specialist_count += 1
        aggregator_count = 1 if predictor_count > 1 else 0
        return predictor_count + specialist_count + aggregator_count

    def to_payload(self) -> dict[str, Any]:
        return {
            "summarize_rounds": self.summarize_rounds,
            "aggregate_width": self.aggregate_width,
            "reflect_rounds": self.reflect_rounds,
            "debate_rounds": self.debate_rounds,
            "execute_enabled": self.execute_enabled,
            "order": list(self.order),
            "active_blocks": self.active_blocks(),
            "estimated_agent_count": self.estimated_agent_count,
        }


@dataclass(frozen=True)
class SearchSpace:
    """Task-specific MASS search dimensions."""

    summarize: tuple[int, ...] = (0, 1, 2, 3, 4)
    aggregate: tuple[int, ...] = (1, 3, 5, 7, 9)
    reflect: tuple[int, ...] = (0, 1, 2, 3, 4)
    debate: tuple[int, ...] = (0, 1, 2, 3, 4)
    execute: tuple[bool, ...] = (False, True)
    enabled_blocks: tuple[str, ...] = ("aggregate",)
    max_agent_budget: int = 12
    topology_order: tuple[str, ...] = ("summarize", "reflect", "debate", "aggregate")
    aggregate_minimum_width: int = 3
    debate_minimum_width: int = 2
    reflect_minimum_rounds: int = 1
    summarize_minimum_rounds: int = 1
    debate_minimum_rounds: int = 1

    def block_enabled(self, name: str) -> bool:
        return name in set(self.enabled_blocks)


@dataclass(frozen=True)
class MASSCandidate:
    """One fully specified candidate evaluated by the framework."""

    workflow: WorkflowSpec
    prompts: dict[str, AgentPromptBundle]
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Evaluation result returned by the benchmark adapter."""

    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionTurn:
    """One observable step in the candidate execution trace."""

    step: str
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExampleExecution:
    """Execution artifact for one example under one MASS candidate."""

    example_id: str
    workflow: WorkflowSpec
    final_answer: str
    turns: tuple[ExecutionTurn, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    """Best candidate produced by one MASS stage."""

    stage_name: str
    best_candidate: MASSCandidate
    best_score: float
    explored_candidates: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MASSConfig:
    """Runtime configuration for the standalone MASS-style framework."""

    task_name: str
    search_space: SearchSpace
    candidates_per_stage: int = 8
    random_seed: int = 0
    max_validation_examples: int | None = None
    run_global_prompt_stage: bool = True
    keep_best_after_global_prompt_stage: bool = False
    topology_temperature: float = 1.0
    max_topology_sampling_attempts: int = 256
    prompt_templates: dict[str, AgentPromptBundle] = field(default_factory=dict)
