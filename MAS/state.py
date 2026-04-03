from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from .artifacts import ArtifactRecord, RelayPacket, TerminationDecision
from .llm import OpenRouterLLMClient
from .monitor import DescriptorHook
from .relay import TopologyLayout


class WorkflowState(TypedDict, total=False):
    """Shared LangGraph state used by every topology workflow."""

    task_id: str
    task_prompt: Any
    reference_answer: str
    task_metadata: dict[str, Any]
    run_index: int
    seed: int

    topology: str
    layout: TopologyLayout
    workflow_definition: dict[str, Any]
    rounds: int
    discussion_rounds: int
    termination_consensus_mode: str
    final_vote_mode: str
    peer_artifact_max_chars: int
    round_index: int
    discussion_index: int
    phase: str
    dispatch_id: int
    active_agents: list[str]
    active_agent: str
    next_step: str
    done: bool

    llm_client: OpenRouterLLMClient
    agent_type_by_agent: dict[str, str]
    tools: list[dict[str, Any]]
    max_tool_iterations: int
    descriptor: DescriptorHook

    messages: Annotated[list[RelayPacket], operator.add]
    message_views: Annotated[list[dict[str, Any]], operator.add]
    artifacts: Annotated[list[ArtifactRecord], operator.add]
    trace_payloads: Annotated[list[dict[str, Any]], operator.add]
    phase_history: Annotated[list[dict[str, Any]], operator.add]
    descriptor_records: Annotated[list[dict[str, Any]], operator.add]
    interaction_logs: Annotated[list[dict[str, Any]], operator.add]
    tool_records_log: Annotated[list[dict[str, Any]], operator.add]
    termination_history: Annotated[list[dict[str, Any]], operator.add]

    message_budget: dict[str, int]
    sent_counts: dict[str, int]
    message_seq: int

    final_answer: str
    final_reason: str
    vote_tally: dict[str, int]
    final_vote_source: str
    termination_decision: TerminationDecision
    descriptor_summary: dict[str, Any]
