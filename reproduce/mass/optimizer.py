from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .interfaces import BenchmarkExample, OptimizerProtocol, PromptEvaluator
from .models import AgentPromptBundle, WorkflowSpec

InstructionProposalCallback = Callable[
    [str, AgentPromptBundle, Sequence[BenchmarkExample], WorkflowSpec, str, int],
    Sequence[str],
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_payload"):
        return _jsonable(value.to_payload())
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class MIPROLikeConfig:
    """Lightweight approximation of MIPRO-style prompt optimization.

    When the framework supplies a :class:`PromptEvaluator`, candidate selection is
    validation-metric driven (paper Algorithm 1, ``O_D``): few-shot demos are
    bootstrapped from the model's own correct validation predictions, and every
    proposed instruction is scored by actually running the candidate on the
    validation set. Without an evaluator the optimizer falls back to a lexical
    heuristic so unit smoke tests stay offline.
    """

    max_bootstrapped_demos: int = 3
    instruction_candidates: int = 10
    rounds_per_agent: int = 10
    include_example_ids: bool = True
    include_reference_answers_when_available: bool = True
    include_block_context: bool = True
    instruction_proposer: InstructionProposalCallback | None = None
    # Optional cap on how many validation examples each candidate is scored on
    # during prompt search (None = full validation set, the paper default).
    validation_limit: int | None = None
    checkpoint_dir: Path | None = None
    bootstrap_demos: bool = True
    benchmark_name: str = ""


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
        evaluate: PromptEvaluator | None = None,
    ) -> AgentPromptBundle:
        return AgentPromptBundle(
            system_instruction=seed_prompt.system_instruction,
            input_fields=seed_prompt.input_fields,
            output_fields=seed_prompt.output_fields,
            output_contract=seed_prompt.output_contract,
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
        evaluate: PromptEvaluator | None = None,
    ) -> dict[str, AgentPromptBundle]:
        return {
            key: AgentPromptBundle(
                system_instruction=value.system_instruction,
                input_fields=value.input_fields,
                output_fields=value.output_fields,
                output_contract=value.output_contract,
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
        evaluate: PromptEvaluator | None = None,
    ) -> AgentPromptBundle:
        # The full candidate scored on validation = the conditioned base prompts
        # (warmed predictor a0*) with this block's prompt swapped in.
        fixed_prompts = {key: value for key, value in base_prompts.items() if key != block_name}

        def compose(bundle: AgentPromptBundle) -> dict[str, AgentPromptBundle]:
            return {**fixed_prompts, block_name: bundle}

        return self._optimize_prompt(
            block_name=block_name,
            seed_prompt=seed_prompt,
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
            evaluate=evaluate,
            compose=compose,
            seed_prompts=compose(seed_prompt),
        )

    def optimize_workflow_prompts(
        self,
        *,
        workflow: WorkflowSpec,
        prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
        evaluate: PromptEvaluator | None = None,
    ) -> dict[str, AgentPromptBundle]:
        if evaluate is not None and examples:
            return self._optimize_workflow_prompts_joint(
                workflow=workflow,
                prompts=prompts,
                examples=examples,
                evaluate=evaluate,
            )

        # Offline/unit fallback: when no evaluator is supplied, keep the older
        # coordinate pass so direct optimizer calls can still return deterministic
        # heuristic prompts without a benchmark runtime.
        optimized: dict[str, AgentPromptBundle] = dict(prompts)
        for key, value in prompts.items():

            def compose(bundle: AgentPromptBundle, _key: str = key) -> dict[str, AgentPromptBundle]:
                return {**optimized, _key: bundle}

            optimized[key] = self._optimize_prompt(
                block_name=key,
                seed_prompt=value,
                seed_instruction=value.system_instruction,
                seed_exemplar=value.exemplar,
                seed_metadata=value.metadata,
                base_metadata={"scope": "workflow"},
                examples=examples,
                workflow=workflow,
                scope="workflow",
                evaluate=evaluate,
                compose=compose,
                seed_prompts=compose(value),
            )
        return optimized

    def _optimize_workflow_prompts_joint(
        self,
        *,
        workflow: WorkflowSpec,
        prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
        evaluate: PromptEvaluator,
    ) -> dict[str, AgentPromptBundle]:
        scoring_examples = self._scoring_examples(examples)
        if not scoring_examples:
            return dict(prompts)

        candidate_count = max(1, self.config.instruction_candidates)
        candidate_instructions: dict[str, list[str]] = {}
        exemplars: dict[str, str] = {}
        for key, prompt in prompts.items():
            if self.config.bootstrap_demos:
                bootstrapped, demo_source = self._bootstrap_demos(
                    block_name=key,
                    workflow=workflow,
                    examples=scoring_examples,
                    evaluate=evaluate,
                    seed_prompts=prompts,
                )
            else:
                bootstrapped, demo_source = [], "disabled"
            exemplars[key] = self._build_exemplar(
                block_name=key,
                examples=examples,
                seed_exemplar=prompt.exemplar,
                workflow=workflow,
                bootstrapped=bootstrapped,
            )
            candidate_instructions[key] = self._propose_instruction_candidates(
                block_name=key,
                seed_prompt=prompt,
                seed_instruction=prompt.system_instruction,
                workflow=workflow,
                examples=examples,
                scope="workflow_joint",
            )
            print(
                f"MASS_WORKFLOW_PROMPT_PREP block={key} candidates={len(candidate_instructions[key])} "
                f"demo_source={demo_source} demos={len(bootstrapped)}",
                flush=True,
            )

        scored_sets: list[dict[str, Any]] = []
        for index in range(candidate_count):
            candidate_prompts: dict[str, AgentPromptBundle] = {}
            instruction_payload: dict[str, str] = {}
            for key, prompt in prompts.items():
                candidates = candidate_instructions.get(key) or [prompt.system_instruction]
                instruction = candidates[index % len(candidates)]
                instruction_payload[key] = instruction
                base = prompt.with_instruction(instruction)
                candidate_prompts[key] = AgentPromptBundle(
                    system_instruction=base.system_instruction,
                    input_fields=base.input_fields,
                    output_fields=base.output_fields,
                    output_contract=base.output_contract,
                    exemplar=exemplars.get(key, prompt.exemplar),
                    metadata={
                        **prompt.metadata,
                        "optimizer": "mipro_like",
                        "scope": "workflow_joint",
                        "candidate_index": index,
                    },
                )
            print(
                f"MASS_WORKFLOW_PROMPT_CANDIDATE_START index={index + 1}/{candidate_count} "
                f"blocks={sorted(candidate_prompts)} examples={len(scoring_examples)}",
                flush=True,
            )
            try:
                evaluation = evaluate(candidate_prompts, workflow, scoring_examples)
                score = float(evaluation.score)
                error = None
            except Exception as exc:
                score = 0.0
                error = f"{type(exc).__name__}: {exc}"
            scored_sets.append(
                {
                    "index": index,
                    "score": score,
                    "instructions": instruction_payload,
                    **({"error": error} if error else {}),
                }
            )
            print(
                f"MASS_WORKFLOW_PROMPT_CANDIDATE_DONE index={index + 1}/{candidate_count} "
                f"score={score:.6f}"
                + (f" error={error}" if error else ""),
                flush=True,
            )

        selected = max(scored_sets, key=lambda item: (float(item["score"]), -int(item["index"])))
        selected_index = int(selected["index"])
        optimized: dict[str, AgentPromptBundle] = {}
        for key, prompt in prompts.items():
            candidates = candidate_instructions.get(key) or [prompt.system_instruction]
            instruction = candidates[selected_index % len(candidates)]
            base = prompt.with_instruction(instruction)
            optimized[key] = AgentPromptBundle(
                system_instruction=base.system_instruction,
                input_fields=base.input_fields,
                output_fields=base.output_fields,
                output_contract=base.output_contract,
                exemplar=exemplars.get(key, prompt.exemplar),
                metadata={
                    **prompt.metadata,
                    "optimizer": "mipro_like",
                    "scope": "workflow_joint",
                    "selection_mode": "validation_joint",
                    "selected_candidate_index": selected_index,
                    "selected_candidate_score": float(selected["score"]),
                    "validation_example_count": len(scoring_examples),
                    "joint_candidate_search_trace": scored_sets,
                },
            )
        return optimized

    def _optimize_prompt(
        self,
        *,
        block_name: str,
        seed_prompt: AgentPromptBundle,
        seed_instruction: str,
        seed_exemplar: str,
        seed_metadata: dict[str, Any],
        base_metadata: dict[str, Any],
        examples: Sequence[BenchmarkExample],
        workflow: WorkflowSpec,
        scope: str,
        evaluate: PromptEvaluator | None = None,
        compose: Callable[[AgentPromptBundle], dict[str, AgentPromptBundle]] | None = None,
        seed_prompts: dict[str, AgentPromptBundle] | None = None,
    ) -> AgentPromptBundle:
        scoring_examples = self._scoring_examples(examples)
        validation_driven = evaluate is not None and compose is not None and bool(scoring_examples)
        checkpoint_key = self._checkpoint_key(
            block_name=block_name,
            workflow=workflow,
            examples=scoring_examples if validation_driven else examples,
            scope=scope,
            seed_instruction=seed_instruction,
        )
        cached_prompt = self._load_prompt_checkpoint(checkpoint_key)
        if cached_prompt is not None:
            print(
                f"MASS_PROMPT_OPT_RESUME block={block_name} scope={scope} key={checkpoint_key}",
                flush=True,
            )
            return cached_prompt

        # Demos: paper bootstraps few-shot examples from the model's own correct
        # validation predictions. Fall back to reference-answer rendering offline.
        if validation_driven and self.config.bootstrap_demos:
            print(
                f"MASS_PROMPT_BOOTSTRAP_START block={block_name} scope={scope} "
                f"examples={len(scoring_examples)}",
                flush=True,
            )
            bootstrapped, demo_source = self._bootstrap_demos(
                block_name=block_name,
                workflow=workflow,
                examples=scoring_examples,
                evaluate=evaluate,
                seed_prompts=seed_prompts,
            )
        else:
            bootstrapped, demo_source = [], "disabled" if validation_driven else "none"
        if validation_driven and self.config.bootstrap_demos:
            print(
                f"MASS_PROMPT_BOOTSTRAP_DONE block={block_name} scope={scope} "
                f"demo_source={demo_source} demos={len(bootstrapped)}",
                flush=True,
            )
        exemplar = self._build_exemplar(
            block_name=block_name,
            examples=examples,
            seed_exemplar=seed_exemplar,
            workflow=workflow,
            bootstrapped=bootstrapped,
        )
        candidates = self._propose_instruction_candidates(
            block_name=block_name,
            seed_prompt=seed_prompt,
            seed_instruction=seed_instruction,
            workflow=workflow,
            examples=examples,
            scope=scope,
        )

        if validation_driven:
            scored_candidates, selection_mode = self._score_candidates_on_validation(
                candidates=candidates,
                seed_prompt=seed_prompt,
                exemplar=exemplar,
                compose=compose,  # type: ignore[arg-type]
                evaluate=evaluate,  # type: ignore[arg-type]
                workflow=workflow,
                examples=scoring_examples,
                checkpoint_key=checkpoint_key,
            )
        else:
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
            selection_mode = "heuristic"

        selected = max(
            scored_candidates, key=lambda item: (float(item["score"]), -int(item["index"]))
        )
        selected_index = int(selected["index"])
        search_trace = self._build_search_trace(
            scored_candidates,
            candidates=candidates,
            selection_mode=selection_mode,
            validation_example_count=len(scoring_examples) if validation_driven else 0,
        )
        optimized_prompt = AgentPromptBundle(
            system_instruction=candidates[selected_index],
            input_fields=seed_prompt.input_fields,
            output_fields=seed_prompt.output_fields,
            output_contract=seed_prompt.output_contract,
            exemplar=exemplar,
            metadata={
                **seed_metadata,
                **base_metadata,
                "optimizer": "mipro_like",
                "selection_mode": selection_mode,
                "demo_source": demo_source,
                "demo_count": len(bootstrapped)
                if bootstrapped
                else min(len(examples), self.config.max_bootstrapped_demos),
                "max_bootstrapped_demos": self.config.max_bootstrapped_demos,
                "instruction_candidates": self.config.instruction_candidates,
                "rounds_per_agent": self.config.rounds_per_agent,
                "proposed_instruction_count": len(candidates),
                "selected_instruction_index": selected_index,
                "selected_instruction_score": float(selected["score"]),
                "validation_score": float(selected["score"]) if validation_driven else None,
                "validation_example_count": len(scoring_examples) if validation_driven else 0,
                "candidate_search_trace": search_trace,
            },
        )
        self._write_prompt_checkpoint(checkpoint_key, optimized_prompt)
        return optimized_prompt

    def _scoring_examples(self, examples: Sequence[BenchmarkExample]) -> list[BenchmarkExample]:
        limit = self.config.validation_limit
        if limit is None or limit <= 0:
            return list(examples)
        return list(examples)[: int(limit)]

    def _score_candidates_on_validation(
        self,
        *,
        candidates: list[str],
        seed_prompt: AgentPromptBundle,
        exemplar: str,
        compose: Callable[[AgentPromptBundle], dict[str, AgentPromptBundle]],
        evaluate: PromptEvaluator,
        workflow: WorkflowSpec,
        examples: Sequence[BenchmarkExample],
        checkpoint_key: str,
    ) -> tuple[list[dict[str, float | int]], str]:
        scored: list[dict[str, float | int]] = []
        for index, instruction in enumerate(candidates):
            cached = self._load_candidate_checkpoint(
                checkpoint_key=checkpoint_key,
                candidate_index=index,
                instruction=instruction,
            )
            if cached is not None:
                scored.append(cached)
                print(
                    f"MASS_PROMPT_CANDIDATE_RESUME block={workflow.active_blocks()} "
                    f"index={index + 1}/{len(candidates)} score={float(cached['score']):.6f}",
                    flush=True,
                )
                continue
            print(
                f"MASS_PROMPT_CANDIDATE_START block={workflow.active_blocks()} "
                f"index={index + 1}/{len(candidates)} examples={len(examples)}",
                flush=True,
            )
            bundle = seed_prompt.with_instruction(instruction)
            bundle = AgentPromptBundle(
                system_instruction=bundle.system_instruction,
                input_fields=bundle.input_fields,
                output_fields=bundle.output_fields,
                output_contract=bundle.output_contract,
                exemplar=exemplar,
                metadata=dict(bundle.metadata),
            )
            try:
                evaluation = evaluate(compose(bundle), workflow, examples)
                score = float(evaluation.score)
                evaluation_details = getattr(evaluation, "details", {}) or {}
                error = None
            except Exception as exc:
                score = 0.0
                evaluation_details = {}
                error = f"{type(exc).__name__}: {exc}"
            scored_payload: dict[str, float | int | str] = {"index": index, "score": score}
            if error is not None:
                scored_payload["error"] = error
            scored.append(scored_payload)
            self._write_candidate_checkpoint(
                checkpoint_key=checkpoint_key,
                candidate_index=index,
                instruction=instruction,
                scored_payload=scored_payload,
                evaluation_details=evaluation_details,
            )
            print(
                f"MASS_PROMPT_CANDIDATE_DONE index={index + 1}/{len(candidates)} "
                f"score={score:.6f}"
                + (f" error={error}" if error is not None else ""),
                flush=True,
            )
        return scored, "validation"

    def _checkpoint_key(
        self,
        *,
        block_name: str,
        workflow: WorkflowSpec,
        examples: Sequence[BenchmarkExample],
        scope: str,
        seed_instruction: str,
    ) -> str:
        payload = {
            "block_name": block_name,
            "example_ids": [str(example.example_id) for example in examples],
            "scope": scope,
            "seed_instruction": seed_instruction,
            "validation_limit": self.config.validation_limit,
            "workflow": workflow.to_payload(),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        digest = hashlib.sha1(encoded).hexdigest()[:16]
        safe_scope = self._safe_path_part(scope)
        safe_block = self._safe_path_part(block_name)
        return f"{safe_scope}__{safe_block}__{digest}"

    def _load_prompt_checkpoint(self, checkpoint_key: str) -> AgentPromptBundle | None:
        path = self._prompt_checkpoint_path(checkpoint_key)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return AgentPromptBundle(
                system_instruction=str(payload["system_instruction"]),
                input_fields=tuple(str(item) for item in payload.get("input_fields") or ()),
                output_fields=tuple(str(item) for item in payload.get("output_fields") or ()),
                output_contract=str(payload.get("output_contract") or ""),
                exemplar=str(payload.get("exemplar") or ""),
                metadata=dict(payload.get("metadata") or {}),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_prompt_checkpoint(
        self, checkpoint_key: str, prompt: AgentPromptBundle
    ) -> None:
        path = self._prompt_checkpoint_path(checkpoint_key)
        if path is None:
            return
        self._write_json(
            path,
            {
                "system_instruction": prompt.system_instruction,
                "input_fields": list(prompt.input_fields),
                "output_fields": list(prompt.output_fields),
                "output_contract": prompt.output_contract,
                "exemplar": prompt.exemplar,
                "metadata": prompt.metadata,
            },
        )

    def _load_candidate_checkpoint(
        self,
        *,
        checkpoint_key: str,
        candidate_index: int,
        instruction: str,
    ) -> dict[str, float | int | str] | None:
        path = self._candidate_checkpoint_path(checkpoint_key, candidate_index)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if str(payload.get("instruction") or "") != instruction:
                return None
            scored = dict(payload["score_payload"])
            scored["index"] = int(scored.get("index", candidate_index))
            scored["score"] = float(scored.get("score", 0.0))
            return scored
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_candidate_checkpoint(
        self,
        *,
        checkpoint_key: str,
        candidate_index: int,
        instruction: str,
        scored_payload: dict[str, float | int | str],
        evaluation_details: dict[str, Any] | None = None,
    ) -> None:
        path = self._candidate_checkpoint_path(checkpoint_key, candidate_index)
        if path is None:
            return
        self._write_json(
            path,
            {
                "candidate_index": candidate_index,
                "instruction": instruction,
                "score_payload": scored_payload,
                "evaluation_details": evaluation_details or {},
            },
        )

    def _prompt_checkpoint_path(self, checkpoint_key: str) -> Path | None:
        if self.config.checkpoint_dir is None:
            return None
        return Path(self.config.checkpoint_dir) / "optimized_prompts" / f"{checkpoint_key}.json"

    def _candidate_checkpoint_path(
        self, checkpoint_key: str, candidate_index: int
    ) -> Path | None:
        if self.config.checkpoint_dir is None:
            return None
        return (
            Path(self.config.checkpoint_dir)
            / "candidate_scores"
            / checkpoint_key
            / f"candidate_{candidate_index}.json"
        )

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _safe_path_part(self, value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
        return safe or "unknown"

    def _bootstrap_demos(
        self,
        *,
        block_name: str,
        workflow: WorkflowSpec,
        examples: Sequence[BenchmarkExample],
        evaluate: PromptEvaluator | None,
        seed_prompts: dict[str, AgentPromptBundle] | None,
    ) -> tuple[list[str], str]:
        if evaluate is None or seed_prompts is None or not examples:
            return [], "none"
        try:
            evaluation = evaluate(dict(seed_prompts), workflow, examples)
        except Exception:
            return [], "none"
        correct = self._correct_predictions(evaluation, examples)
        if not correct:
            return [], "none"
        demos: list[str] = []
        for example, prediction in correct[: self.config.max_bootstrapped_demos]:
            demos.append(
                f"Input: {self._short_text(example.prompt)}\nOutput: {self._short_text(prediction)}"
            )
        return demos, "bootstrapped_correct_predictions"

    def _correct_predictions(
        self,
        evaluation: Any,
        examples: Sequence[BenchmarkExample],
    ) -> list[tuple[BenchmarkExample, str]]:
        details = dict(getattr(evaluation, "details", {}) or {})
        by_id = {str(example.example_id): example for example in examples}
        seen: set[str] = set()
        correct: list[tuple[BenchmarkExample, str]] = []

        # Preferred: per-example success records from the benchmark-run adapter.
        for record in details.get("benchmark_evaluations") or []:
            if not isinstance(record, dict):
                continue
            example_id = str(record.get("example_id"))
            if example_id in seen or example_id not in by_id:
                continue
            if not bool(record.get("success", float(record.get("score", 0.0) or 0.0) > 0.0)):
                continue
            prediction = self._prediction_for(details, example_id)
            seen.add(example_id)
            correct.append((by_id[example_id], prediction))
        if correct:
            return correct

        # Fallback: scores aligned positionally with the examples list.
        scores = details.get("scores")
        executions = details.get("executions") or []
        if isinstance(scores, list) and len(scores) == len(examples):
            for index, example in enumerate(examples):
                if float(scores[index] or 0.0) <= 0.0:
                    continue
                prediction = ""
                if index < len(executions) and isinstance(executions[index], dict):
                    prediction = str(executions[index].get("final_answer") or "")
                correct.append((example, prediction))
        return correct

    def _prediction_for(self, details: dict[str, Any], example_id: str) -> str:
        for execution in details.get("executions") or []:
            if isinstance(execution, dict) and str(execution.get("example_id")) == example_id:
                return str(execution.get("final_answer") or "")
        return ""

    def _propose_instruction_candidates(
        self,
        *,
        block_name: str,
        seed_prompt: AgentPromptBundle,
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
        requested_count = max(1, self.config.instruction_candidates)
        if self.config.instruction_proposer is not None:
            try:
                proposed = self.config.instruction_proposer(
                    block_name,
                    seed_prompt,
                    examples,
                    workflow,
                    scope,
                    requested_count,
                )
                for instruction in proposed:
                    text = self._sanitize_proposed_instruction(
                        str(instruction).strip(),
                        block_name=block_name,
                    )
                    if text and text not in candidates:
                        candidates.append(text)
                    if len(candidates) >= requested_count:
                        break
            except Exception:
                candidates = []
        for index in range(requested_count):
            if len(candidates) >= requested_count:
                break
            hint = hints[index % len(hints)]
            candidate = (
                f"{base} Candidate strategy {index + 1}: {hint} "
                f"Validation summary: {dataset_summary}"
            )
            candidates.append(candidate)
        return candidates

    def _sanitize_proposed_instruction(self, instruction: str, *, block_name: str) -> str:
        text = " ".join(str(instruction or "").split())
        if not text:
            return ""
        if self.config.benchmark_name.lower() == "plancraft":
            lowered = text.lower()
            forbidden = (
                "reference",
                "ground truth",
                "match",
                "mismatch",
                "aligns with",
                "provided answer",
                "gold answer",
            )
            if any(term in lowered for term in forbidden):
                return ""
            action_terms = ("action", "move:", "smelt:", "impossible", "inventory", "craft")
            if block_name in {"predictor", "aggregate", "debate", "reflect", "execute"} and not any(
                term in lowered for term in action_terms
            ):
                return ""
        return text

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
        self,
        scored_candidates: list[dict[str, float | int]],
        *,
        candidates: list[str],
        selection_mode: str,
        validation_example_count: int,
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
                    "selection_mode": selection_mode,
                    "validation_example_count": validation_example_count,
                    "instruction_preview": self._short_text(
                        candidates[candidate_index] if candidate_index < len(candidates) else "",
                        limit=360,
                    ),
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
        bootstrapped: list[str] | None = None,
    ) -> str:
        demos: list[str] = []
        if seed_exemplar.strip():
            demos.append(seed_exemplar.strip())
        if bootstrapped:
            demos.extend(bootstrapped)
        else:
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
        evaluate: PromptEvaluator | None = None,
    ) -> AgentPromptBundle:
        optimized = self.delegate.optimize_block_prompt(
            block_name=block_name,
            seed_prompt=seed_prompt,
            base_prompts=base_prompts,
            examples=examples,
            workflow=workflow,
            evaluate=evaluate,
        )
        return AgentPromptBundle(
            system_instruction=optimized.system_instruction,
            input_fields=optimized.input_fields,
            output_fields=optimized.output_fields,
            output_contract=optimized.output_contract,
            exemplar=optimized.exemplar,
            metadata={**optimized.metadata, "optimizer_backend": "dspy_mipro_adapter"},
        )

    def optimize_workflow_prompts(
        self,
        *,
        workflow: WorkflowSpec,
        prompts: dict[str, AgentPromptBundle],
        examples: Sequence[BenchmarkExample],
        evaluate: PromptEvaluator | None = None,
    ) -> dict[str, AgentPromptBundle]:
        optimized = self.delegate.optimize_workflow_prompts(
            workflow=workflow,
            prompts=prompts,
            examples=examples,
            evaluate=evaluate,
        )
        return {
            key: AgentPromptBundle(
                system_instruction=value.system_instruction,
                input_fields=value.input_fields,
                output_fields=value.output_fields,
                output_contract=value.output_contract,
                exemplar=value.exemplar,
                metadata={**value.metadata, "optimizer_backend": "dspy_mipro_adapter"},
            )
            for key, value in optimized.items()
        }
