from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from benchmark.base import BenchmarkTask

from .adapters import ScoreCallback, TemplateBenchmarkAdapter
from .executor import ExecutionCallback, MASSCandidateExecutor, ModelCallback
from .interfaces import BenchmarkExample
from .models import CandidateEvaluation, ExampleExecution, MASSCandidate


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
    ) -> None:
        self.benchmark = benchmark
        self.tasks = list(tasks)
        self._task_by_id = {str(task.task_id): task for task in self.tasks}
        self.validation_repeats = max(1, int(validation_repeats))
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
