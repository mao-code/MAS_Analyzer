from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from MAS.runner import MASRunResult


@dataclass(frozen=True)
class BenchmarkTask:
    """A single benchmark task instance."""

    task_id: str
    prompt: Any
    reference_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """Evaluation output for one task prediction."""

    task_id: str
    score: float
    success: bool
    details: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    """Common interface for benchmark adapters.

    Benchmarks own the interaction loop via ``run()``.
    One-shot benchmarks simply delegate to ``runner.run_task()``.
    Interactive benchmarks (e.g. PlanCraft) drive an env step loop internally.
    """

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]: ...

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
        run_metadata: dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation: ...

    def requirements(self) -> dict[str, Any]: ...


def init_run_metadata_aggregate() -> dict[str, Any]:
    return {
        "tool_call_counts": {},
        "tool_calls_total": 0,
        "messages_sent_total": 0,
        "retrieved_docids": [],
        "tool_definitions": [],
        "interaction_logs": [],
        "phase_history": [],
        "step_runs": [],
    }


def merge_step_run_metadata(
    aggregate: dict[str, Any],
    step_metadata: dict[str, Any] | None,
    *,
    outer_step_index: int,
    step_task_id: str,
    final_answer: str,
) -> None:
    if not step_metadata:
        return

    tool_call_counts = aggregate.setdefault("tool_call_counts", {})
    for name, count in dict(step_metadata.get("tool_call_counts", {})).items():
        tool_name = str(name)
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + int(count)

    aggregate["tool_calls_total"] = int(aggregate.get("tool_calls_total", 0)) + int(
        step_metadata.get("tool_calls_total", 0)
    )
    aggregate["messages_sent_total"] = int(aggregate.get("messages_sent_total", 0)) + int(
        step_metadata.get("messages_sent_total", 0)
    )

    retrieved_docids = {
        str(docid)
        for docid in aggregate.get("retrieved_docids", [])
        if str(docid)
    }
    retrieved_docids.update(
        str(docid) for docid in step_metadata.get("retrieved_docids", []) if str(docid)
    )
    aggregate["retrieved_docids"] = sorted(retrieved_docids)

    if not aggregate.get("tool_definitions") and step_metadata.get("tool_definitions"):
        aggregate["tool_definitions"] = list(step_metadata.get("tool_definitions", []))

    for key in ("interaction_logs", "phase_history"):
        entries = list(aggregate.get(key, []))
        for item in step_metadata.get(key, []):
            if isinstance(item, dict):
                entries.append({"outer_step_index": outer_step_index, **item})
            else:
                entries.append({"outer_step_index": outer_step_index, "value": item})
        aggregate[key] = entries

    step_runs = list(aggregate.get("step_runs", []))
    step_runs.append(
        {
            "outer_step_index": outer_step_index,
            "task_id": step_task_id,
            "final_answer": final_answer,
            "run_summary": {
                "topology": step_metadata.get("topology"),
                "turns_executed": step_metadata.get("turns_executed", 0),
                "messages_sent_total": step_metadata.get("messages_sent_total", 0),
                "tool_calls_total": step_metadata.get("tool_calls_total", 0),
                "tool_call_counts": dict(step_metadata.get("tool_call_counts", {})),
                "final_reason": step_metadata.get("final_reason", ""),
            },
        }
    )
    aggregate["step_runs"] = step_runs
