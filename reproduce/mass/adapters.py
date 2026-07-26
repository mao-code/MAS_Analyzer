from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from .executor import MASSCandidateExecutor
from .interfaces import BenchmarkAdapter, BenchmarkExample
from .models import CandidateEvaluation, ExampleExecution, MASSCandidate

ScoreCallback = Callable[[ExampleExecution, BenchmarkExample], float]


def default_score_callback(execution: ExampleExecution, example: BenchmarkExample) -> float:
    """Very small default scorer for smoke testing and templates."""

    if example.reference_answer in (None, ""):
        return float(bool(execution.final_answer.strip()))
    return float(str(example.reference_answer).strip() in execution.final_answer)


@dataclass
class TemplateBenchmarkAdapter(BenchmarkAdapter):
    """Drop-in benchmark adapter template for custom reproduce experiments."""

    examples: Sequence[BenchmarkExample]
    executor: MASSCandidateExecutor = field(default_factory=MASSCandidateExecutor)
    score_callback: ScoreCallback = default_score_callback
    metadata: dict[str, Any] = field(default_factory=dict)

    def validation_examples(self, limit: int | None = None) -> Sequence[BenchmarkExample]:
        if limit is None:
            return list(self.examples)
        return list(self.examples[:limit])

    def execute_candidate(
        self,
        candidate: MASSCandidate,
        example: BenchmarkExample,
    ) -> ExampleExecution:
        return self.executor.run_candidate(candidate, example)

    def evaluate_candidate(
        self,
        candidate: MASSCandidate,
        examples: Sequence[BenchmarkExample],
    ) -> CandidateEvaluation:
        executions: list[ExampleExecution] = [
            self.execute_candidate(candidate, example) for example in examples
        ]
        scores = [
            float(self.score_callback(execution, example))
            for execution, example in zip(executions, examples)
        ]
        return CandidateEvaluation(
            score=mean(scores) if scores else 0.0,
            details={
                "scores": scores,
                "executions": [
                    {
                        "example_id": execution.example_id,
                        "final_answer": execution.final_answer,
                        "turn_count": len(execution.turns),
                        "metadata": execution.metadata,
                    }
                    for execution in executions
                ],
                "adapter_metadata": dict(self.metadata),
            },
        )
