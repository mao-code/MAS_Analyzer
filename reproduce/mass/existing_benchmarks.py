from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from benchmark.base import BenchmarkTask

from .adapters import ScoreCallback, TemplateBenchmarkAdapter
from .executor import ExecutionCallback, MASSCandidateExecutor, ModelCallback
from .interfaces import BenchmarkExample
from .models import CandidateEvaluation, ExampleExecution, MASSCandidate
from .runtime_runner import MASSRuntimeRunner


@dataclass
class ExistingBenchmarkMASSAdapter(TemplateBenchmarkAdapter):
    """Adapter that lets MASS reproduction run against existing repo benchmarks."""

    benchmark: Any = None
    tasks: Sequence[BenchmarkTask] = field(default_factory=list)

    def __init__(
        self,
        *,
        benchmark: Any,
        tasks: Sequence[BenchmarkTask],
        executor: MASSCandidateExecutor | None = None,
        model_callback: ModelCallback | None = None,
        execution_callback: ExecutionCallback | None = None,
        score_callback: ScoreCallback | None = None,
        validation_repeats: int = 1,
        metadata: dict[str, Any] | None = None,
        runtime_llm_client: Any | None = None,
        model_agent_type: str = "default",
        temperature: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.benchmark = benchmark
        self.tasks = list(tasks)
        self._task_by_id = {str(task.task_id): task for task in self.tasks}
        self.validation_repeats = max(1, int(validation_repeats))
        self.runtime_llm_client = runtime_llm_client
        self.model_agent_type = str(model_agent_type)
        self.temperature = float(temperature)
        self.seed = int(seed)
        resolved_executor = executor or MASSCandidateExecutor(
            model_callback=model_callback or MASSCandidateExecutor().model_callback,
            execution_callback=execution_callback,
        )
        super().__init__(
            examples=[self._task_to_example(task) for task in self.tasks],
            executor=resolved_executor,
            score_callback=score_callback or self._score_existing_benchmark,
            metadata=metadata or {},
        )

    def evaluate_candidate(
        self,
        candidate: MASSCandidate,
        examples: Sequence[BenchmarkExample],
    ) -> CandidateEvaluation:
        if self.runtime_llm_client is not None:
            return self._evaluate_candidate_via_benchmark_run(candidate, examples)
        all_scores: list[float] = []
        all_executions: list[dict[str, Any]] = []
        for repeat_index in range(self.validation_repeats):
            evaluation = super().evaluate_candidate(candidate, examples)
            all_scores.extend(float(score) for score in evaluation.details.get("scores", []))
            for execution in evaluation.details.get("executions", []):
                payload = dict(execution)
                payload["validation_repeat"] = repeat_index
                all_executions.append(payload)
        score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        details = {
            "scores": all_scores,
            "executions": all_executions,
            "validation_repeats": self.validation_repeats,
        }
        details["benchmark_scores"] = list(all_scores)
        details["candidate"] = {
            "stage": candidate.stage,
            "workflow": candidate.workflow.to_payload(),
            "prompt_blocks": sorted(candidate.prompts.keys()),
        }
        return CandidateEvaluation(score=score, details=details)

    def _task_to_example(self, task: BenchmarkTask) -> BenchmarkExample:
        return BenchmarkExample(
            example_id=str(task.task_id),
            prompt=task.prompt,
            reference_answer=task.reference_answer,
            metadata=dict(task.metadata or {}),
        )

    def _evaluate_candidate_via_benchmark_run(
        self,
        candidate: MASSCandidate,
        examples: Sequence[BenchmarkExample],
    ) -> CandidateEvaluation:
        all_scores: list[float] = []
        all_executions: list[dict[str, Any]] = []
        all_evaluations: list[dict[str, Any]] = []
        runner = MASSRuntimeRunner(
            candidate=candidate,
            llm_client=self.runtime_llm_client,
            model_agent_type=self.model_agent_type,
            temperature=self.temperature,
        )
        for repeat_index in range(self.validation_repeats):
            for example_index, example in enumerate(examples):
                task = self._task_by_id[str(example.example_id)]
                run_seed = self.seed + repeat_index * 1000 + example_index
                run_result = self.benchmark.run(
                    task=task,
                    runner=runner,
                    run_index=repeat_index,
                    seed=run_seed,
                )
                evaluation = self.benchmark.evaluate(
                    task,
                    run_result.final_answer,
                    run_metadata=dict(run_result.run_metadata),
                )
                quality_score, quality_flags = self._quality_adjusted_score(
                    raw_score=float(evaluation.score),
                    final_answer=str(run_result.final_answer or ""),
                    run_metadata=dict(run_result.run_metadata),
                    evaluation_details=evaluation.details or {},
                )
                all_scores.append(float(quality_score))
                all_evaluations.append(
                    {
                        "example_id": str(example.example_id),
                        "validation_repeat": repeat_index,
                        "score": float(quality_score),
                        "raw_score": float(evaluation.score),
                        "success": bool(evaluation.success) and float(quality_score) > 0.0,
                        "raw_success": bool(evaluation.success),
                        "quality_flags": quality_flags,
                        "details": evaluation.details,
                    }
                )
                all_executions.append(
                    {
                        "example_id": str(example.example_id),
                        "final_answer": run_result.final_answer,
                        "turn_count": int(
                            dict(run_result.run_metadata)
                            .get("execution", {})
                            .get("turn_count", len(run_result.trace_events))
                        ),
                        "trace_events": [event.to_dict() for event in run_result.trace_events],
                        "run_metadata": dict(run_result.run_metadata),
                        "evaluation": evaluation.details,
                        "quality_flags": quality_flags,
                        "raw_score": float(evaluation.score),
                        "validation_repeat": repeat_index,
                    }
                )
        score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        return CandidateEvaluation(
            score=score,
            details={
                "scores": all_scores,
                "benchmark_scores": list(all_scores),
                "benchmark_evaluations": all_evaluations,
                "executions": all_executions,
                "validation_repeats": self.validation_repeats,
                "candidate": {
                    "stage": candidate.stage,
                    "workflow": candidate.workflow.to_payload(),
                    "prompt_blocks": sorted(candidate.prompts.keys()),
                },
                "adapter_metadata": dict(self.metadata),
                "execution_path": "benchmark.run",
            },
        )

    def _quality_adjusted_score(
        self,
        *,
        raw_score: float,
        final_answer: str,
        run_metadata: dict[str, Any],
        evaluation_details: dict[str, Any],
    ) -> tuple[float, list[str]]:
        benchmark_name = str(self.metadata.get("benchmark_name", "") or "").lower()
        flags: list[str] = []
        answer = str(final_answer or "")
        answer_lower = answer.lower()
        refusal_pattern = re.compile(
            r"(unable to determine|cannot answer|can't answer|no context|context (?:text )?"
            r"(?:was )?not provided|insufficient(?:\s+\w+){0,3}\s+"
            r"(?:context|information|evidence)|unable to provide|"
            r"failed to provide|tools? failed|cache miss)",
            re.IGNORECASE,
        )
        if refusal_pattern.search(answer):
            flags.append("refusal_or_no_evidence_answer")
        if re.search(r"\bAnswer\s*:\s*$", answer, re.IGNORECASE | re.DOTALL):
            flags.append("empty_structured_answer")

        if benchmark_name == "browsecomp":
            tool_calls_total = int(run_metadata.get("tool_calls_total", 0) or 0)
            retrieved_docids = list(run_metadata.get("retrieved_docids", []) or [])
            if tool_calls_total <= 0:
                flags.append("browsecomp_no_tool_calls")
            if not retrieved_docids:
                flags.append("browsecomp_no_retrieved_docids")
            if "context: none provided" in answer_lower or "solutions: none provided" in answer_lower:
                flags.append("browsecomp_context_template_leak")
            if flags:
                return 0.0, flags

        if benchmark_name == "stabletoolbench":
            tool_calls_total = int(run_metadata.get("tool_calls_total", 0) or 0)
            payload_text = json.dumps(
                {"answer": answer, "run_metadata": run_metadata, "evaluation": evaluation_details},
                default=str,
            ).lower()
            if tool_calls_total <= 0:
                flags.append("stabletoolbench_no_tool_calls")
            if "stabletoolbench cache miss" in payload_text:
                flags.append("stabletoolbench_cache_miss")
            if any(
                marker in payload_text
                for marker in (
                    "tools did not yield",
                    "toolset did not return",
                    "api tools provided did not yield",
                    "specific api tools provided did not yield",
                    "specific api tools did not yield",
                    "instead, i will provide",
                    "i will provide a comprehensive",
                    "based on general",
                )
            ):
                flags.append("stabletoolbench_generic_fallback")
            if (
                all(
                    marker in payload_text
                    for marker in (
                        "order id",
                        "details of the order",
                        "products",
                        "quantities",
                        "total amount",
                    )
                )
                and any(
                    marker in payload_text
                    for marker in (
                        "unable to find",
                        "unable to retrieve",
                        "cannot provide",
                        "can't provide",
                        "not found",
                        "no record",
                        "please verify",
                        "try again",
                    )
                )
                and not any(
                    marker in answer_lower
                    for marker in (
                        "product:",
                        "products:",
                        "quantity:",
                        "quantities:",
                        "total amount:",
                        "amount:",
                        "$",
                    )
                )
            ):
                flags.append("stabletoolbench_missing_requested_details")
            if "placeholder" in answer_lower or "(refer to" in answer_lower:
                flags.append("stabletoolbench_placeholder_answer")
            if "stabletoolbench_cache_miss" in flags and raw_score < 1.0:
                return 0.0, flags
            if (
                "stabletoolbench_placeholder_answer" in flags
                or "stabletoolbench_generic_fallback" in flags
                or "stabletoolbench_missing_requested_details" in flags
                or "empty_structured_answer" in flags
                or "stabletoolbench_no_tool_calls" in flags
            ):
                return min(float(raw_score), 0.0), flags

        return float(raw_score), flags

    def _score_existing_benchmark(
        self,
        execution: ExampleExecution,
        example: BenchmarkExample,
    ) -> float:
        task = self._task_by_id[str(example.example_id)]
        evaluation = self.benchmark.evaluate(
            task,
            execution.final_answer,
            run_metadata={
                "mass_reproduce": True,
                "workflow": execution.workflow.to_payload(),
                "execution": {
                    "turn_count": len(execution.turns),
                    "metadata": execution.metadata,
                },
            },
        )
        return float(evaluation.score)
