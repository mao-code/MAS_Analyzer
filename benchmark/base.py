from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from MAS.runner import MASRunResult


@dataclass(frozen=True)
class BenchmarkTask:
    """A single benchmark task instance."""

    task_id: str
    prompt: Any
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
    """Common interface for benchmark adapters.

    Benchmarks own the interaction loop via ``run()``.
    One-shot benchmarks simply delegate to ``runner.run_task()``.
    Interactive benchmarks (e.g. PlanCraft) drive an env step loop internally.
    """

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        ...

    def run(
        self,
        task: BenchmarkTask,
        runner: Any,
        run_index: int,
        seed: int,
    ) -> MASRunResult:
        """Execute a single run and return trace + answer + metadata."""
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
