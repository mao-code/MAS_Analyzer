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
        return self._optimize_prompt(
            block_name=block_name,
            seed_instruction=seed_prompt.system_instruction,
            seed_exemplar=seed_prompt.exemplar,
            seed_metadata=seed_prompt.metadata,
            base_metadata={
                "scope": "block",
                "block_name": block_name,
                "conditioned_on": sorted(base_prompts.keys()),
            },
            examples=examples,
            workflow=workflow,
            scope="block",
        )

    def optimize_workflow_prompts(
        self,
        *,
        workflow: WorkflowSpec,
        prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
    ) -> dict[str, AgentPromptBundle]:
        return {
            key: self._optimize_prompt(
                block_name=key,
                seed_instruction=value.system_instruction,
                seed_exemplar=value.exemplar,
                seed_metadata=value.metadata,
                base_metadata={"scope": "workflow"},
                examples=examples,
                workflow=workflow,
                scope="workflow",
            )
            for key, value in prompts.items()
        }

    def _optimize_prompt(
        self,
        *,
        block_name: str,
        seed_instruction: str,
        seed_exemplar: str,
        seed_metadata: dict[str, Any],
        base_metadata: dict[str, Any],
        examples: Sequence[BenchmarkExample],
        workflow: WorkflowSpec,
        scope: str,
    ) -> AgentPromptBundle:
        exemplar = self._build_exemplar(
            block_name=block_name,
            examples=examples,
            seed_exemplar=seed_exemplar,
            workflow=workflow,
        )
        candidates = self._propose_instruction_candidates(
            block_name=block_name,
            seed_instruction=seed_instruction,
            workflow=workflow,
            examples=examples,
            scope=scope,
        )
        scored_candidates = [
            {
                "index": index,
                "score": self._score_instruction_candidate(
                    instruction=instruction,
                    block_name=block_name,
                    examples=examples,
                    workflow=workflow,
                ),
            }
            for index, instruction in enumerate(candidates)
        ]
        selected = max(
            scored_candidates, key=lambda item: (float(item["score"]), -int(item["index"]))
        )
        selected_index = int(selected["index"])
        search_trace = self._build_search_trace(scored_candidates)
        return AgentPromptBundle(
            system_instruction=candidates[selected_index],
            exemplar=exemplar,
            metadata={
                **seed_metadata,
                **base_metadata,
                "optimizer": "mipro_like",
                "demo_count": min(len(examples), self.config.max_bootstrapped_demos),
                "max_bootstrapped_demos": self.config.max_bootstrapped_demos,
                "instruction_candidates": self.config.instruction_candidates,
                "rounds_per_agent": self.config.rounds_per_agent,
                "proposed_instruction_count": len(candidates),
                "selected_instruction_index": selected_index,
                "selected_instruction_score": float(selected["score"]),
                "candidate_search_trace": search_trace,
            },
        )

    def _propose_instruction_candidates(
        self,
        *,
        block_name: str,
        seed_instruction: str,
        workflow: WorkflowSpec,
        examples: Sequence[BenchmarkExample],
        scope: str,
    ) -> list[str]:
        base = self._rewrite_instruction(
            block_name=block_name,
            seed_instruction=seed_instruction,
            workflow=workflow,
            scope=scope,
        )
        dataset_summary = self._dataset_summary(examples)
        hints = self._role_hints(block_name)
        candidates: list[str] = []
        for index in range(max(1, self.config.instruction_candidates)):
            hint = hints[index % len(hints)]
            candidate = (
                f"{base} Candidate strategy {index + 1}: {hint} "
                f"Validation summary: {dataset_summary}"
            )
            candidates.append(candidate)
        return candidates

    def _score_instruction_candidate(
        self,
        *,
        instruction: str,
        block_name: str,
        examples: Sequence[BenchmarkExample],
        workflow: WorkflowSpec,
    ) -> float:
        lowered = instruction.lower()
        score = 0.0
        score += 1.0 if block_name.lower() in lowered else 0.0
        score += 0.5 if "validation summary" in lowered else 0.0
        score += 0.25 * min(len(examples), self.config.max_bootstrapped_demos)
        score += 0.1 * len(set(workflow.active_blocks()))
        if any(example.reference_answer not in (None, "") for example in examples):
            score += 0.5
        role_terms = {
            "predictor": ("solve", "answer", "reason"),
            "aggregate": ("compare", "select", "combine"),
            "summarize": ("summarize", "compress", "context"),
            "reflect": ("critique", "correct", "revise"),
            "debate": ("peer", "argue", "debate"),
            "execute": ("execute", "verify", "feedback"),
        }
        score += sum(0.1 for term in role_terms.get(block_name, ()) if term in lowered)
        return round(score, 6)

    def _build_search_trace(
        self, scored_candidates: list[dict[str, float | int]]
    ) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        best_index = -1
        best_score = float("-inf")
        for round_index in range(self.config.rounds_per_agent):
            candidate = scored_candidates[round_index % len(scored_candidates)]
            candidate_score = float(candidate["score"])
            candidate_index = int(candidate["index"])
            if candidate_score > best_score:
                best_score = candidate_score
                best_index = candidate_index
            trace.append(
                {
                    "round": round_index + 1,
                    "candidate_index": candidate_index,
                    "candidate_score": candidate_score,
                    "best_index": best_index,
                    "best_score": best_score,
                }
            )
        return trace

    def _dataset_summary(self, examples: Sequence[BenchmarkExample]) -> str:
        example_count = len(examples)
        reference_count = sum(
            1 for example in examples if example.reference_answer not in (None, "")
        )
        prompt_preview = (
            self._short_text(examples[0].prompt, limit=120) if examples else "unavailable"
        )
        return (
            f"{example_count} validation examples; {reference_count} with reference signals; "
            f"first task preview: {prompt_preview}"
        )

    def _role_hints(self, block_name: str) -> tuple[str, ...]:
        hints_by_role = {
            "predictor": (
                "reason step by step before giving a concise final answer.",
                "separate assumptions from the final answer.",
                "prefer robust task-specific evidence over generic reasoning.",
            ),
            "aggregate": (
                "compare peer answers and choose the most supported final answer.",
                "resolve disagreements by checking consistency across candidates.",
                "avoid averaging incompatible answers; select one defensible answer.",
            ),
            "summarize": (
                "retain entities, constraints, and evidence needed by later agents.",
                "compress irrelevant context aggressively while preserving task facts.",
                "write a reusable summary for downstream solving agents.",
            ),
            "reflect": (
                "judge whether the current answer is correct before revising it.",
                "name concrete errors and provide actionable correction feedback.",
                "output correct only when no revision is needed.",
            ),
            "debate": (
                "respond to peer answers and defend or update your position.",
                "surface disagreements explicitly before producing an updated answer.",
                "use peer evidence without blindly copying it.",
            ),
            "execute": (
                "produce verifiable feedback from execution or tool-like checking.",
                "focus on concrete failures, outputs, and correction signals.",
                "return feedback that a reflector can use to revise the answer.",
            ),
        }
        return hints_by_role.get(
            block_name,
            (
                "follow the assigned workflow role precisely.",
                "preserve useful reasoning signal for downstream agents.",
                "optimize for correctness under the validation metric.",
            ),
        )

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
