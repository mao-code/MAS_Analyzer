from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .interfaces import BenchmarkExample, OptimizerProtocol
from .models import AgentPromptBundle, WorkflowSpec


@dataclass(frozen=True)
class MIPROLikeConfig:
    """Lightweight approximation of MIPRO-style prompt optimization."""

    max_bootstrapped_demos: int = 3
    instruction_candidates: int = 10
    rounds_per_agent: int = 10
    include_example_ids: bool = True
    include_reference_answers_when_available: bool = True
    include_block_context: bool = True


class IdentityPromptOptimizer(OptimizerProtocol):
    """A no-op optimizer so the framework can run before real prompt search is wired in."""

    def optimize_block_prompt(
        self,
        *,
        block_name: str,
        seed_prompt: AgentPromptBundle,
        base_prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
        workflow: WorkflowSpec,
    ) -> AgentPromptBundle:
        return AgentPromptBundle(
            system_instruction=seed_prompt.system_instruction,
            exemplar=seed_prompt.exemplar,
            metadata={
                **seed_prompt.metadata,
                "optimizer": "identity",
                "block_name": block_name,
                "conditioned_on": sorted(base_prompts.keys()),
            },
        )

    def optimize_workflow_prompts(
        self,
        *,
        workflow: WorkflowSpec,
        prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
    ) -> dict[str, AgentPromptBundle]:
        return {
            key: AgentPromptBundle(
                system_instruction=value.system_instruction,
                exemplar=value.exemplar,
                metadata={**value.metadata, "optimizer": "identity", "scope": "workflow"},
            )
            for key, value in prompts.items()
        }


class MIPROLikePromptOptimizer(OptimizerProtocol):
    """Heuristic instruction + exemplar optimizer inspired by MIPRO."""

    def __init__(self, config: MIPROLikeConfig | None = None) -> None:
        self.config = config or MIPROLikeConfig()

    def optimize_block_prompt(
        self,
        *,
        block_name: str,
        seed_prompt: AgentPromptBundle,
        base_prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
        workflow: WorkflowSpec,
    ) -> AgentPromptBundle:
        instruction = self._rewrite_instruction(
            block_name=block_name,
            seed_instruction=seed_prompt.system_instruction,
            workflow=workflow,
            scope="block",
        )
        exemplar = self._build_exemplar(
            block_name=block_name,
            examples=examples,
            seed_exemplar=seed_prompt.exemplar,
            workflow=workflow,
        )
        return AgentPromptBundle(
            system_instruction=instruction,
            exemplar=exemplar,
            metadata={
                **seed_prompt.metadata,
                "optimizer": "mipro_like",
                "scope": "block",
                "block_name": block_name,
                "conditioned_on": sorted(base_prompts.keys()),
                "demo_count": min(len(examples), self.config.max_bootstrapped_demos),
                "max_bootstrapped_demos": self.config.max_bootstrapped_demos,
                "instruction_candidates": self.config.instruction_candidates,
                "rounds_per_agent": self.config.rounds_per_agent,
            },
        )

    def optimize_workflow_prompts(
        self,
        *,
        workflow: WorkflowSpec,
        prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
    ) -> dict[str, AgentPromptBundle]:
        return {
            key: AgentPromptBundle(
                system_instruction=self._rewrite_instruction(
                    block_name=key,
                    seed_instruction=value.system_instruction,
                    workflow=workflow,
                    scope="workflow",
                ),
                exemplar=self._build_exemplar(
                    block_name=key,
                    examples=examples,
                    seed_exemplar=value.exemplar,
                    workflow=workflow,
                ),
                metadata={
                    **value.metadata,
                    "optimizer": "mipro_like",
                    "scope": "workflow",
                    "demo_count": min(len(examples), self.config.max_bootstrapped_demos),
                    "max_bootstrapped_demos": self.config.max_bootstrapped_demos,
                    "instruction_candidates": self.config.instruction_candidates,
                    "rounds_per_agent": self.config.rounds_per_agent,
                },
            )
            for key, value in prompts.items()
        }

    def _rewrite_instruction(
        self,
        *,
        block_name: str,
        seed_instruction: str,
        workflow: WorkflowSpec,
        scope: str,
    ) -> str:
        parts = [seed_instruction.strip()]
        if self.config.include_block_context:
            parts.append(f"Role: `{block_name}`.")
            parts.append(f"Optimization scope: `{scope}`.")
            parts.append(
                "Workflow context: "
                f"summarize={workflow.summarize_rounds}, "
                f"reflect={workflow.reflect_rounds}, "
                f"debate={workflow.debate_rounds}, "
                f"aggregate={workflow.aggregate_width}, "
                f"execute={int(workflow.execute_enabled)}."
            )
        parts.append("Be concise, accurate, and consistent with the workflow role.")
        return " ".join(part for part in parts if part)

    def _build_exemplar(
        self,
        *,
        block_name: str,
        examples: Sequence[BenchmarkExample],
        seed_exemplar: str,
        workflow: WorkflowSpec,
    ) -> str:
        demos: list[str] = []
        if seed_exemplar.strip():
            demos.append(seed_exemplar.strip())
        for example in list(examples)[: self.config.max_bootstrapped_demos]:
            demos.append(
                self._render_demo(block_name=block_name, example=example, workflow=workflow)
            )
        return "\n\n".join(demos).strip()

    def _render_demo(
        self,
        *,
        block_name: str,
        example: BenchmarkExample,
        workflow: WorkflowSpec,
    ) -> str:
        lines = []
        if self.config.include_example_ids:
            lines.append(f"Example ID: {example.example_id}")
        lines.append(f"Task: {self._short_text(example.prompt)}")
        lines.append(f"Block role: {block_name}")
        lines.append(f"Active workflow blocks: {', '.join(workflow.active_blocks())}")
        if (
            self.config.include_reference_answers_when_available
            and example.reference_answer not in (None, "")
        ):
            lines.append(f"Reference signal: {self._short_text(example.reference_answer)}")
        else:
            lines.append("Reference signal: unavailable")
        lines.append(
            "Expected behavior: produce an output aligned with this role and preserve useful reasoning signal."
        )
        return "\n".join(lines)

    def _short_text(self, value: Any, *, limit: int = 220) -> str:
        text = str(value).strip().replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."


class DSPyMIPROAdapter(OptimizerProtocol):
    """Optional wrapper point for a future real DSPy MIPRO integration."""

    def __init__(self, delegate: OptimizerProtocol | None = None) -> None:
        self.delegate = delegate or MIPROLikePromptOptimizer()
        try:
            import dspy  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "DSPy is not installed. Install `dspy` before using DSPyMIPROAdapter."
            ) from exc

    def optimize_block_prompt(
        self,
        *,
        block_name: str,
        seed_prompt: AgentPromptBundle,
        base_prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
        workflow: WorkflowSpec,
    ) -> AgentPromptBundle:
        optimized = self.delegate.optimize_block_prompt(
            block_name=block_name,
            seed_prompt=seed_prompt,
            base_prompts=base_prompts,
            examples=examples,
            workflow=workflow,
        )
        return AgentPromptBundle(
            system_instruction=optimized.system_instruction,
            exemplar=optimized.exemplar,
            metadata={**optimized.metadata, "optimizer_backend": "dspy_mipro_adapter"},
        )

    def optimize_workflow_prompts(
        self,
        *,
        workflow: WorkflowSpec,
        prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
    ) -> dict[str, AgentPromptBundle]:
        optimized = self.delegate.optimize_workflow_prompts(
            workflow=workflow,
            prompts=prompts,
            examples=examples,
        )
        return {
            key: AgentPromptBundle(
                system_instruction=value.system_instruction,
                exemplar=value.exemplar,
                metadata={**value.metadata, "optimizer_backend": "dspy_mipro_adapter"},
            )
            for key, value in optimized.items()
        }
