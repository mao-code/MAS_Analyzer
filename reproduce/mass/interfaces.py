from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import (
    AgentPromptBundle,
    CandidateEvaluation,
    ExampleExecution,
    MASSCandidate,
    WorkflowSpec,
)


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    prompt: Any
    reference_answer: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Validation-driven scorer used by the prompt optimizer (paper Algorithm 1, `O_D`).
# Given a full prompt set, the workflow under optimization, and the validation
# examples, it assembles a candidate and returns its real validation metric.
PromptEvaluator = Callable[
    [dict[str, AgentPromptBundle], "WorkflowSpec", Sequence["BenchmarkExample"]],
    CandidateEvaluation,
]


class BenchmarkAdapter(Protocol):
    """Minimal benchmark hook for the standalone reproduction framework."""

    def validation_examples(self, limit: int | None = None) -> Sequence[BenchmarkExample]: ...

    def evaluate_candidate(
        self,
        candidate: MASSCandidate,
        examples: Sequence[BenchmarkExample],
    ) -> CandidateEvaluation: ...

    def execute_candidate(
        self,
        candidate: MASSCandidate,
        example: BenchmarkExample,
    ) -> ExampleExecution: ...


class OptimizerProtocol(Protocol):
    """Prompt optimizer interface used by the framework."""

    def optimize_block_prompt(
        self,
        *,
        block_name: str,
        seed_prompt: AgentPromptBundle,
        base_prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
        workflow: WorkflowSpec,
        evaluate: PromptEvaluator | None = None,
    ) -> AgentPromptBundle: ...

    def optimize_workflow_prompts(
        self,
        *,
        workflow: WorkflowSpec,
        prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
        evaluate: PromptEvaluator | None = None,
    ) -> dict[str, AgentPromptBundle]: ...
