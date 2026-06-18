from __future__ import annotations

import math
import random
from dataclasses import replace

from .interfaces import BenchmarkAdapter, OptimizerProtocol
from .models import (
    AgentPromptBundle,
    MASSCandidate,
    MASSConfig,
    SearchSpace,
    StageResult,
)
from .optimizer import MIPROLikePromptOptimizer
from .topology import build_block_workflow, build_initial_workflow


class MASSFramework:
    """Standalone MASS-style search loop for reproduction-only experiments.

    This implementation mirrors the paper's high-level structure:
    1. block-level prompt warm-up
    2. workflow topology search
    3. workflow-level prompt refinement

    It intentionally avoids any dependency on the production MAS runtime so it
    can be used as a clean research scaffold for custom benchmarks.
    """

    def __init__(
        self,
        config: MASSConfig,
        benchmark: BenchmarkAdapter,
        prompt_optimizer: OptimizerProtocol | None = None,
    ) -> None:
        self.config = config
        self.benchmark = benchmark
        self.prompt_optimizer = prompt_optimizer or MIPROLikePromptOptimizer()
        self._rng = random.Random(config.random_seed)

    def run(self) -> dict[str, StageResult]:
        examples = self.benchmark.validation_examples(limit=self.config.max_validation_examples)
        if not examples:
            raise ValueError("benchmark.validation_examples() returned no examples")

        stage1 = self._run_block_prompt_stage(examples)
        stage2 = self._run_topology_stage(examples, stage1)
        if self.config.run_global_prompt_stage:
            stage3 = self._run_workflow_prompt_stage(examples, stage2)
            return {
                "stage1_block_prompt": stage1,
                "stage2_topology": stage2,
                "stage3_workflow_prompt": stage3,
            }
        return {
            "stage1_block_prompt": stage1,
            "stage2_topology": stage2,
        }

    def _run_block_prompt_stage(self, examples: list) -> StageResult:
        prompts = self._default_prompts(self.config.search_space)
        base_workflow = build_initial_workflow(self.config.search_space)
        predictor_prompt = self.prompt_optimizer.optimize_block_prompt(
            block_name="predictor",
            seed_prompt=prompts["predictor"],
            base_prompts={"predictor": prompts["predictor"]},
            examples=examples,
            workflow=base_workflow,
        )
        base_candidate = MASSCandidate(
            workflow=base_workflow,
            prompts={"predictor": predictor_prompt},
            stage="block_prompt_base",
            metadata={"task_name": self.config.task_name, "block_name": "predictor"},
        )
        base_eval = self.benchmark.evaluate_candidate(base_candidate, examples)
        base_score = float(base_eval.score)

        block_candidates: dict[str, MASSCandidate] = {}
        block_scores: dict[str, float] = {}
        influence_scores: dict[str, float] = {}
        explored = 1
        for block_name in self._search_blocks():
            block_workflow = build_block_workflow(self.config.search_space, block_name)
            optimized_prompt = self.prompt_optimizer.optimize_block_prompt(
                block_name=block_name,
                seed_prompt=prompts[block_name],
                base_prompts={"predictor": predictor_prompt},
                examples=examples,
                workflow=block_workflow,
            )
            candidate = MASSCandidate(
                workflow=block_workflow,
                prompts={
                    "predictor": predictor_prompt,
                    block_name: optimized_prompt,
                },
                stage="block_prompt",
                metadata={"task_name": self.config.task_name, "block_name": block_name},
            )
            evaluation = self.benchmark.evaluate_candidate(candidate, examples)
            score = float(evaluation.score)
            block_candidates[block_name] = candidate
            block_scores[block_name] = score
            influence_scores[block_name] = self._safe_influence(score, base_score)
            explored += 1

        selection_probabilities = self._softmax(influence_scores, self.config.topology_temperature)
        best_block = max(block_scores, key=block_scores.get, default="predictor")
        candidate = block_candidates.get(best_block, base_candidate)
        score = block_scores.get(best_block, base_score)
        return StageResult(
            stage_name="block_prompt",
            best_candidate=candidate,
            best_score=score,
            explored_candidates=explored,
            metadata={
                "base_candidate": base_candidate,
                "base_score": base_score,
                "base_evaluation_details": base_eval.details,
                "block_candidates": block_candidates,
                "block_scores": block_scores,
                "influence_scores": influence_scores,
                "selection_probabilities": selection_probabilities,
            },
        )

    def _run_topology_stage(self, examples: list, stage1: StageResult) -> StageResult:
        base_candidate = stage1.metadata["base_candidate"]
        block_candidates = dict(stage1.metadata["block_candidates"])
        selection_probabilities = dict(stage1.metadata["selection_probabilities"])

        best_candidate = base_candidate
        best_score = float(stage1.metadata["base_score"])
        explored = 0
        sampled_payloads: list[dict[str, object]] = []

        while explored < max(1, self.config.candidates_per_stage):
            workflow, kept_blocks = self._sample_pruned_workflow(selection_probabilities)
            candidate = MASSCandidate(
                workflow=workflow,
                prompts=self._compose_candidate_prompts(
                    base_candidate, block_candidates, kept_blocks
                ),
                stage="topology",
                metadata={
                    "task_name": self.config.task_name,
                    "kept_blocks": kept_blocks,
                    "selection_probabilities": selection_probabilities,
                },
            )
            evaluation = self.benchmark.evaluate_candidate(candidate, examples)
            explored += 1
            sampled_payloads.append(
                {
                    "workflow": workflow.to_payload(),
                    "kept_blocks": list(kept_blocks),
                    "score": float(evaluation.score),
                }
            )
            if evaluation.score > best_score:
                best_candidate = candidate
                best_score = float(evaluation.score)

        return StageResult(
            stage_name="topology",
            best_candidate=best_candidate,
            best_score=best_score,
            explored_candidates=explored,
            metadata={
                "sampled_candidates": sampled_payloads,
                "selection_probabilities": selection_probabilities,
            },
        )

    def _run_workflow_prompt_stage(self, examples: list, stage2: StageResult) -> StageResult:
        optimized_prompts = self.prompt_optimizer.optimize_workflow_prompts(
            workflow=stage2.best_candidate.workflow,
            prompts=stage2.best_candidate.prompts,
            examples=examples,
        )
        candidate = replace(
            stage2.best_candidate, prompts=optimized_prompts, stage="workflow_prompt"
        )
        evaluation = self.benchmark.evaluate_candidate(candidate, examples)
        best_candidate = candidate
        best_score = float(evaluation.score)
        if self.config.keep_best_after_global_prompt_stage and stage2.best_score > best_score:
            best_candidate = stage2.best_candidate
            best_score = stage2.best_score
        return StageResult(
            stage_name="workflow_prompt",
            best_candidate=best_candidate,
            best_score=best_score,
            explored_candidates=1,
            metadata={"evaluation_details": evaluation.details},
        )

    def _default_prompts(self, search_space: SearchSpace) -> dict[str, AgentPromptBundle]:
        prompts = {
            "predictor": AgentPromptBundle(
                system_instruction="Think step by step and produce the strongest candidate solution."
            ),
            "aggregate": AgentPromptBundle(
                system_instruction="Examine candidate solutions from parallel agents and return the strongest final answer."
            ),
        }
        if search_space.block_enabled("summarize"):
            prompts["summarize"] = AgentPromptBundle(
                system_instruction="Summarize the key context and compress the information needed before solving the task."
            )
        if search_space.block_enabled("reflect"):
            prompts["reflect"] = AgentPromptBundle(
                system_instruction="Determine whether the current solution is correct. If not, critique it and propose a corrected revision."
            )
        if search_space.block_enabled("debate"):
            prompts["debate"] = AgentPromptBundle(
                system_instruction="Examine peer solutions, argue for or against them, and finish with an updated answer."
            )
        if search_space.block_enabled("execute"):
            prompts["execute"] = AgentPromptBundle(
                system_instruction="Use available execution or verification tools to produce grounded feedback about the current solution."
            )
        if self.config.prompt_templates:
            prompts.update(self.config.prompt_templates)
        return prompts

    def _search_blocks(self) -> list[str]:
        return [name for name in self.config.search_space.enabled_blocks if name != "predictor"]

    def _safe_influence(self, block_score: float, base_score: float) -> float:
        if abs(base_score) < 1e-8:
            return block_score
        return block_score / base_score

    def _softmax(self, scores: dict[str, float], temperature: float) -> dict[str, float]:
        if not scores:
            return {}
        temp = max(float(temperature), 1e-6)
        max_score = max(scores.values())
        exps = {key: math.exp((value - max_score) / temp) for key, value in scores.items()}
        denom = sum(exps.values()) or 1.0
        return {key: value / denom for key, value in exps.items()}

    def _sample_pruned_workflow(
        self, selection_probabilities: dict[str, float]
    ) -> tuple[object, list[str]]:
        attempts = max(1, self.config.max_topology_sampling_attempts)
        fallback: tuple[object, list[str]] | None = None
        for _ in range(attempts):
            kept_blocks: list[str] = []
            for block_name in self._search_blocks():
                probability = float(selection_probabilities.get(block_name, 0.0))
                if self._rng.random() < probability:
                    kept_blocks.append(block_name)
            workflow = self._workflow_from_kept_blocks(kept_blocks)
            fallback = (workflow, kept_blocks)
            if workflow.estimated_agent_count <= self.config.search_space.max_agent_budget:
                return workflow, kept_blocks
        assert fallback is not None
        return fallback

    def _workflow_from_kept_blocks(self, kept_blocks: list[str]):
        search_space = self.config.search_space
        workflow = build_initial_workflow(search_space)
        if "summarize" in kept_blocks:
            positive = [value for value in search_space.summarize if int(value) > 0]
            workflow = replace(
                workflow,
                summarize_rounds=int(
                    self._rng.choice(positive or [search_space.summarize_minimum_rounds])
                ),
            )
        if "reflect" in kept_blocks:
            positive = [value for value in search_space.reflect if int(value) > 0]
            workflow = replace(
                workflow,
                reflect_rounds=int(
                    self._rng.choice(positive or [search_space.reflect_minimum_rounds])
                ),
            )
        if "debate" in kept_blocks:
            positive = [value for value in search_space.debate if int(value) > 0]
            workflow = replace(
                workflow,
                aggregate_width=max(workflow.aggregate_width, search_space.debate_minimum_width),
                debate_rounds=int(
                    self._rng.choice(positive or [search_space.debate_minimum_rounds])
                ),
            )
        if "aggregate" in kept_blocks:
            positive = [value for value in search_space.aggregate if int(value) > 1]
            workflow = replace(
                workflow,
                aggregate_width=max(
                    int(self._rng.choice(positive or [search_space.aggregate_minimum_width])),
                    search_space.aggregate_minimum_width,
                ),
            )
        if "execute" in kept_blocks:
            workflow = replace(
                workflow,
                execute_enabled=True,
                reflect_rounds=max(workflow.reflect_rounds, search_space.reflect_minimum_rounds),
            )
        return replace(workflow, order=search_space.topology_order)

    def _compose_candidate_prompts(
        self,
        base_candidate: MASSCandidate,
        block_candidates: dict[str, MASSCandidate],
        kept_blocks: list[str],
    ) -> dict[str, AgentPromptBundle]:
        prompts = dict(base_candidate.prompts)
        for block_name in kept_blocks:
            candidate = block_candidates.get(block_name)
            if candidate is None:
                continue
            prompts[block_name] = candidate.prompts[block_name]
        if "aggregate" not in prompts:
            prompts["aggregate"] = self._default_prompts(self.config.search_space)["aggregate"]
        return prompts
