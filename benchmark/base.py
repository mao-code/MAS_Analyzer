from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence


@dataclass(frozen=True)
class BenchmarkTask:
    """A single benchmark task instance."""

    task_id: str
    prompt: str
    reference_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """Evaluation output for one task prediction."""

    task_id: str
    score: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    """Common interface for benchmark adapters."""

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        ...

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: Dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        ...

    def requirements(self) -> Dict[str, Any]:
        ...
