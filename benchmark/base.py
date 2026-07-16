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
        "task_id": None,
        "run_index": None,
        "seed": None,
        "topology": None,
        "rounds_configured": 0,
        "discussion_rounds": 0,
        "turns_executed": 0,
        "messages_sent_by_agent": {},
        "tool_call_counts": {},
        "tool_calls_total": 0,
        "messages_sent_total": 0,
        "remaining_message_budget": {},
        "retrieved_docids": [],
        "tool_definitions": [],
        "interaction_logs": [],
        "phase_history": [],
        "relay_messages": [],
        "message_views": [],
        "descriptor_records": [],
        "termination_history": [],
        "agent_outputs": {},
        "vote_tally": {},
        "selected_artifact_id": "",
        "selected_agent_id": "",
        "selected_source_artifact_ids": [],
        "descriptor_summary": {},
        "topology_layout": {},
        "workflow_definition": {},
        "final_reason": "",
        "self_evolved": {},
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

    for key in ("task_id", "run_index", "seed", "topology"):
        if aggregate.get(key) is None and step_metadata.get(key) is not None:
            aggregate[key] = step_metadata.get(key)

    aggregate["rounds_configured"] = max(
        int(aggregate.get("rounds_configured", 0)),
        int(step_metadata.get("rounds_configured", 0) or 0),
    )
    aggregate["discussion_rounds"] = max(
        int(aggregate.get("discussion_rounds", 0)),
        int(step_metadata.get("discussion_rounds", 0) or 0),
    )
    aggregate["turns_executed"] = int(aggregate.get("turns_executed", 0)) + int(
        step_metadata.get("turns_executed", 0) or 0
    )

    tool_call_counts = aggregate.setdefault("tool_call_counts", {})
    for name, count in dict(step_metadata.get("tool_call_counts", {})).items():
        tool_name = str(name)
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + int(count)

    messages_sent_by_agent = aggregate.setdefault("messages_sent_by_agent", {})
    for agent_id, count in dict(step_metadata.get("messages_sent_by_agent", {})).items():
        agent_key = str(agent_id)
        messages_sent_by_agent[agent_key] = messages_sent_by_agent.get(agent_key, 0) + int(count)

    aggregate["tool_calls_total"] = int(aggregate.get("tool_calls_total", 0)) + int(
        step_metadata.get("tool_calls_total", 0)
    )
    aggregate["messages_sent_total"] = int(aggregate.get("messages_sent_total", 0)) + int(
        step_metadata.get("messages_sent_total", 0)
    )
    if step_metadata.get("remaining_message_budget"):
        aggregate["remaining_message_budget"] = dict(step_metadata.get("remaining_message_budget", {}))

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

    for key in (
        "interaction_logs",
        "phase_history",
        "relay_messages",
        "message_views",
        "descriptor_records",
        "termination_history",
    ):
        entries = list(aggregate.get(key, []))
        for item in step_metadata.get(key, []):
            if isinstance(item, dict):
                entries.append({"outer_step_index": outer_step_index, **item})
            else:
                entries.append({"outer_step_index": outer_step_index, "value": item})
        aggregate[key] = entries

    if step_metadata.get("agent_outputs"):
        aggregate["agent_outputs"] = dict(step_metadata.get("agent_outputs", {}))
    if step_metadata.get("vote_tally"):
        aggregate["vote_tally"] = dict(step_metadata.get("vote_tally", {}))
    if step_metadata.get("selected_artifact_id"):
        aggregate["selected_artifact_id"] = str(step_metadata.get("selected_artifact_id", ""))
    if step_metadata.get("selected_agent_id"):
        aggregate["selected_agent_id"] = str(step_metadata.get("selected_agent_id", ""))
    if step_metadata.get("selected_source_artifact_ids"):
        aggregate["selected_source_artifact_ids"] = list(
            step_metadata.get("selected_source_artifact_ids", [])
        )
    if step_metadata.get("descriptor_summary"):
        aggregate["descriptor_summary"] = dict(step_metadata.get("descriptor_summary", {}))
    if step_metadata.get("topology_layout"):
        aggregate["topology_layout"] = step_metadata.get("topology_layout", {})
    if step_metadata.get("workflow_definition"):
        aggregate["workflow_definition"] = step_metadata.get("workflow_definition", {})
    if step_metadata.get("final_reason"):
        aggregate["final_reason"] = str(step_metadata.get("final_reason", ""))
    if step_metadata.get("self_evolved"):
        # Step-based benchmarks execute MAS once per environment action, but the
        # online skill updater consumes one candidate per completed benchmark run.
        # Preserve the final step's candidate as the run-level learning signal.
        aggregate["self_evolved"] = dict(step_metadata.get("self_evolved", {}))

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
                "messages_sent_by_agent": dict(step_metadata.get("messages_sent_by_agent", {})),
                "tool_calls_total": step_metadata.get("tool_calls_total", 0),
                "tool_call_counts": dict(step_metadata.get("tool_call_counts", {})),
                "final_reason": step_metadata.get("final_reason", ""),
            },
        }
    )
    aggregate["step_runs"] = step_runs
