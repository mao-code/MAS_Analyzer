from __future__ import annotations

import ast
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from answer_utils import classify_answer_mode, extract_substantive_answer, has_substantive_answer
from langchain_core.runnables.graph import Edge, Graph, Node
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from descriptor.schema import TraceEvent

from .artifacts import (
    ArtifactRecord,
    RelayPacket,
    TerminationDecision,
    answer_signature,
    artifacts_by_id,
    average_confidence,
    build_artifact,
    compute_consensus,
    compute_mean_delta,
    latest_artifact_by_agent,
    packet_content,
    packet_payload_from_artifact,
)
from .llm import OpenRouterLLMClient
from .monitor import DescriptorHook, NullDescriptor
from .relay import (
    TOPOLOGY_FULLY_LINKED_DEBATE,
    TOPOLOGY_GROUP_CHAT_DEBATE,
    TOPOLOGY_ONLY_VOTING,
    TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION,
    TOPOLOGY_ORCHESTRATOR_TREE,
    TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION,
    TOPOLOGY_SAS,
    TopologyLayout,
    build_layout,
    normalize_topology_name,
)
from .state import WorkflowState

COMMON_LOG_OUTPUTS = [
    "trace_events",
    "relay_messages",
    "message_views",
    "interaction_logs",
    "descriptor_records",
    "termination_history",
]

SHARED_STATE_FIELDS = [
    "task_id",
    "task_prompt",
    "topology",
    "layout",
    "rounds",
    "discussion_rounds",
    "final_vote_mode",
    "round_index",
    "discussion_index",
    "messages",
    "artifacts",
    "termination_decision",
    "vote_tally",
    "selected_artifact_id",
    "selected_agent_id",
    "selected_source_artifact_ids",
    "descriptor_records",
    "message_budget",
    "sent_counts",
    "budget_sent_counts",
]

UNSUPPORTED_FINAL_ANSWER = "Unable to determine a supported final answer from the available agent outputs."
CONTROL_PACKET_KINDS = frozenset(
    {
        "task_package",
        "orchestrator_feedback",
        "specialist_report",
        "peer_summary",
        "root_task_package",
        "manager_task_package",
        "child_report",
        "manager_report",
    }
)


@dataclass(frozen=True)
class ExperimentSpec:
    topology: str
    num_agents: int
    rounds: int
    discussion_rounds: int = 1
    communication_budget_per_agent: int = 1
    termination_consensus_mode: str = "llm_judge"
    final_vote_mode: str = "llm_judge"
    peer_artifact_max_chars: int = 0
    minimum_discussion_rounds: int = 1
    agents_per_level: list[int] | None = None
    group_sizes: list[int] | None = None
    benchmark_name: str | None = None
    enable_dynamic_roles: bool = True

    def normalized(self) -> ExperimentSpec:
        topology = normalize_topology_name(self.topology)
        rounds = max(1, int(self.rounds))
        discussion_rounds = max(1, int(self.discussion_rounds))
        num_agents = int(self.num_agents)
        if num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        budget = int(self.communication_budget_per_agent)
        if budget < 0:
            raise ValueError("communication_budget_per_agent must be >= 0")
        termination_consensus_mode = str(self.termination_consensus_mode or "llm_judge")
        if termination_consensus_mode not in {"llm_judge", "lexical"}:
            raise ValueError(
                "termination_consensus_mode must be one of: llm_judge, lexical"
            )
        final_vote_mode = str(self.final_vote_mode or "llm_judge")
        if final_vote_mode not in {"llm_judge", "deterministic"}:
            raise ValueError("final_vote_mode must be one of: llm_judge, deterministic")
        peer_artifact_max_chars = int(self.peer_artifact_max_chars)
        if peer_artifact_max_chars < 0:
            raise ValueError("peer_artifact_max_chars must be >= 0 (0 = unlimited)")
        minimum_discussion_rounds = int(self.minimum_discussion_rounds)
        if minimum_discussion_rounds < 0:
            raise ValueError("minimum_discussion_rounds must be >= 0")
        return ExperimentSpec(
            topology=topology,
            num_agents=num_agents,
            rounds=rounds,
            discussion_rounds=discussion_rounds,
            communication_budget_per_agent=budget,
            termination_consensus_mode=termination_consensus_mode,
            final_vote_mode=final_vote_mode,
            peer_artifact_max_chars=peer_artifact_max_chars,
            minimum_discussion_rounds=minimum_discussion_rounds,
            agents_per_level=(
                list(self.agents_per_level) if self.agents_per_level is not None else None
            ),
            group_sizes=(list(self.group_sizes) if self.group_sizes is not None else None),
            benchmark_name=self.benchmark_name,
            enable_dynamic_roles=bool(self.enable_dynamic_roles),
        )


@dataclass
class LangGraphRunResult:
    final_answer: str
    trace_events: list[TraceEvent]
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowDocumentation:
    topology: str
    nodes: dict[str, str]
    edges: list[str]
    conditional_edges: list[str]
    dispatch_logic: str
    aggregation_logic: str
    stopping_criteria: list[str]
    state_fields: list[str]
    logging_outputs: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "topology": self.topology,
            "nodes": dict(self.nodes),
            "edges": list(self.edges),
            "conditional_edges": list(self.conditional_edges),
            "dispatch_logic": self.dispatch_logic,
            "aggregation_logic": self.aggregation_logic,
            "stopping_criteria": list(self.stopping_criteria),
            "state_fields": list(self.state_fields),
            "logging_outputs": list(self.logging_outputs),
        }


class _EventClock:
    def __init__(self) -> None:
        self.cursor = time.time()

    def span(self, latency_ms: float) -> tuple[float, float]:
        duration_s = max(float(latency_ms) / 1000.0, 1e-6)
        start = self.cursor
        end = start + duration_s
        self.cursor = end + 1e-6
        return start, end


class LangGraphMASEngine:
    """LangGraph execution engine with one shared state schema across all MAS topologies."""

    def __init__(self, llm_client: OpenRouterLLMClient) -> None:
        self.llm_client = llm_client
        self._graph_cache: dict[tuple[Any, ...], Any] = {}

    def run(
        self,
        *,
        task: Any,
        run_index: int,
        seed: int,
        spec: ExperimentSpec,
        agent_types: list[str],
        tools: list[dict[str, Any]] | None = None,
        max_tool_iterations: int = 8,
        descriptor: DescriptorHook | None = None,
    ) -> LangGraphRunResult:
        spec = spec.normalized()
        if not agent_types:
            raise ValueError("agent_types must contain at least one entry")

        layout = build_layout(
            topology=spec.topology,
            num_agents=spec.num_agents,
            agents_per_level=spec.agents_per_level,
            group_sizes=spec.group_sizes,
        )
        graph = self._compiled_graph(layout)
        workflow_definition = self._workflow_definition(layout.topology).to_payload()
        agent_type_by_agent = {
            agent_id: agent_types[idx % len(agent_types)] for idx, agent_id in enumerate(layout.agent_ids)
        }

        # -- Dynamic domain-role assignment --------------------------------
        domain_personas: dict[str, dict[str, str]] = {}
        role_assignment_payload: dict[str, Any] = {
            "enabled": bool(spec.enable_dynamic_roles),
            "benchmark_name": str(spec.benchmark_name or ""),
            "used_fallback": False,
            "fallback_reason": "",
            "prompt_messages": [],
            "response": "",
            "llm": {},
            "assignments": {},
        }
        if spec.benchmark_name and spec.enable_dynamic_roles:
            from .role_assigner import RoleAssignmentResult, assign_domain_roles, assign_domain_roles_deterministic

            try:
                assignment_result = assign_domain_roles(
                    benchmark_name=spec.benchmark_name,
                    layout=layout,
                    task_prompt=getattr(task, "prompt", ""),
                    llm_client=self.llm_client,
                )
            except Exception:
                assignments = assign_domain_roles_deterministic(
                    benchmark_name=spec.benchmark_name, layout=layout,
                )
                assignment_result = RoleAssignmentResult(
                    assignments=assignments,
                    prompt_messages=[],
                    response_text="",
                    llm={},
                    used_fallback=True,
                    fallback_reason="engine_role_assignment_error",
                )
            domain_personas = {
                agent_id: {"role_name": a.role_name, "persona": a.persona}
                for agent_id, a in assignment_result.assignments.items()
            }
            role_assignment_payload = {
                "enabled": True,
                "benchmark_name": str(spec.benchmark_name),
                "used_fallback": bool(assignment_result.used_fallback),
                "fallback_reason": str(assignment_result.fallback_reason),
                "prompt_messages": self._serialize_for_json(list(assignment_result.prompt_messages)),
                "response": str(assignment_result.response_text),
                "llm": self._serialize_for_json(dict(assignment_result.llm)),
                "assignments": {
                    agent_id: {"role_name": info.role_name, "persona": info.persona}
                    for agent_id, info in assignment_result.assignments.items()
                },
            }
        elif spec.benchmark_name and not spec.enable_dynamic_roles:
            role_assignment_payload["fallback_reason"] = "disabled"
        elif not spec.benchmark_name:
            role_assignment_payload["fallback_reason"] = "missing_benchmark_name"

        descriptor_hook = descriptor or NullDescriptor()
        state: WorkflowState = {
            "task_id": str(getattr(task, "task_id", "task")),
            "task_prompt": getattr(task, "prompt", ""),
            "benchmark_name": str(spec.benchmark_name or ""),
            "reference_answer": str(getattr(task, "reference_answer", "")),
            "task_metadata": dict(getattr(task, "metadata", {}) or {}),
            "run_index": int(run_index),
            "seed": int(seed),
            "topology": layout.topology,
            "layout": layout,
            "workflow_definition": workflow_definition,
            "rounds": int(spec.rounds),
            "discussion_rounds": int(spec.discussion_rounds),
            "minimum_discussion_rounds": int(spec.minimum_discussion_rounds),
            "termination_consensus_mode": str(spec.termination_consensus_mode),
            "final_vote_mode": str(spec.final_vote_mode),
            "peer_artifact_max_chars": int(spec.peer_artifact_max_chars),
            "round_index": 0,
            "discussion_index": 0,
            "phase": "init",
            "dispatch_id": -1,
            "active_agents": [],
            "next_step": "",
            "done": False,
            "llm_client": self.llm_client,
            "agent_type_by_agent": agent_type_by_agent,
            "tools": list(tools or []),
            "max_tool_iterations": max(1, int(max_tool_iterations)),
            "descriptor": descriptor_hook,
            "messages": [],
            "message_views": [],
            "artifacts": [],
            "trace_payloads": [],
            "phase_history": [],
            "descriptor_records": [],
            "interaction_logs": [],
            "tool_records_log": [],
            "termination_history": [],
            "transcript_summary": {},
            "transcript_compaction_history": [],
            "message_budget": {
                agent_id: int(spec.communication_budget_per_agent) for agent_id in layout.agent_ids
            },
            "sent_counts": {agent_id: 0 for agent_id in layout.agent_ids},
            "budget_sent_counts": {agent_id: 0 for agent_id in layout.agent_ids},
            "message_seq": 0,
            "final_answer": "",
            "final_reason": "",
            "vote_tally": {},
            "final_vote_source": "",
            "selected_artifact_id": "",
            "selected_agent_id": "",
            "selected_source_artifact_ids": [],
            "termination_decision": {},
            "descriptor_summary": {},
            "domain_personas": domain_personas,
            "role_assignment": role_assignment_payload,
        }

        end_state = graph.invoke(state)

        trace_events = self._materialize_trace_events(end_state.get("trace_payloads", []))
        artifacts = list(end_state.get("artifacts", []))
        messages = list(end_state.get("messages", []))
        latest_outputs = {
            agent_id: str(artifact.get("answer", ""))
            for agent_id, artifact in latest_artifact_by_agent(artifacts).items()
        }
        tool_call_counts = self._count_tool_calls(end_state)
        retrieved_docids = self._collect_retrieved_docids(end_state)
        final_answer = str(end_state.get("final_answer") or self._resolve_final_answer(end_state))

        run_metadata = {
            "task_id": end_state.get("task_id"),
            "run_index": run_index,
            "seed": seed,
            "topology": layout.topology,
            "rounds_configured": spec.rounds,
            "discussion_rounds": spec.discussion_rounds,
            "termination_consensus_mode": str(spec.termination_consensus_mode),
            "final_vote_mode": str(spec.final_vote_mode),
            "peer_artifact_max_chars": int(spec.peer_artifact_max_chars),
            "turns_executed": max(1, int(end_state.get("round_index", 0)) + 1),
            "messages_sent_total": sum(end_state.get("sent_counts", {}).values()),
            "messages_sent_by_agent": dict(end_state.get("sent_counts", {})),
            "budget_messages_sent_total": sum(end_state.get("budget_sent_counts", {}).values()),
            "budget_messages_sent_by_agent": dict(end_state.get("budget_sent_counts", {})),
            "tool_call_counts": tool_call_counts,
            "tool_calls_total": int(sum(tool_call_counts.values())),
            "remaining_message_budget": dict(end_state.get("message_budget", {})),
            "tool_definitions": self._serialize_tools(list(end_state.get("tools", []))),
            "retrieved_docids": retrieved_docids,
            "agent_outputs": latest_outputs,
            "vote_tally": dict(end_state.get("vote_tally", {})),
            "final_vote_source": str(end_state.get("final_vote_source", "")),
            "selected_artifact_id": str(end_state.get("selected_artifact_id", "")),
            "selected_agent_id": str(end_state.get("selected_agent_id", "")),
            "selected_source_artifact_ids": list(end_state.get("selected_source_artifact_ids", [])),
            "phase_history": list(end_state.get("phase_history", [])),
            "relay_messages": messages,
            "message_views": list(end_state.get("message_views", [])),
            "interaction_logs": list(end_state.get("interaction_logs", [])),
            "descriptor_records": list(end_state.get("descriptor_records", [])),
            "descriptor_summary": dict(end_state.get("descriptor_summary", {})),
            "termination_history": list(end_state.get("termination_history", [])),
            "transcript_summary": dict(end_state.get("transcript_summary", {})),
            "transcript_compaction_history": list(
                end_state.get("transcript_compaction_history", [])
            ),
            "domain_personas": dict(end_state.get("domain_personas", {})),
            "role_assignment": dict(end_state.get("role_assignment", role_assignment_payload)),
            "topology_layout": layout.to_payload(),
            "workflow_definition": workflow_definition,
            "artifact_records": artifacts,
            "final_reason": str(end_state.get("final_reason", "")),
        }

        return LangGraphRunResult(
            final_answer=final_answer,
            trace_events=trace_events,
            run_metadata=run_metadata,
        )

    def _compiled_graph(self, layout: TopologyLayout):
        cache_key = (
            layout.topology,
            tuple(layout.agent_ids),
            tuple(layout.managers),
            tuple(layout.leaves),
            tuple(tuple(group) for group in layout.groups),
            tuple(layout.representatives),
        )
        compiled = self._graph_cache.get(cache_key)
        if compiled is not None:
            return compiled
        compiled = self._build_graph(layout).compile()
        self._graph_cache[cache_key] = compiled
        return compiled

    def _build_graph(self, layout: TopologyLayout):
        if layout.topology == TOPOLOGY_SAS:
            return self._build_sas_graph(layout)
        if layout.topology == TOPOLOGY_ONLY_VOTING:
            return self._build_only_voting_graph(layout)
        if layout.topology == TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION:
            return self._build_orchestrator_no_discussion_graph(layout)
        if layout.topology == TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION:
            return self._build_orchestrator_with_discussion_graph(layout)
        if layout.topology == TOPOLOGY_ORCHESTRATOR_TREE:
            return self._build_orchestrator_tree_graph(layout)
        if layout.topology == TOPOLOGY_FULLY_LINKED_DEBATE:
            return self._build_fully_linked_debate_graph(layout)
        if layout.topology == TOPOLOGY_GROUP_CHAT_DEBATE:
            return self._build_group_chat_debate_graph(layout)
        raise ValueError(f"Unhandled topology '{layout.topology}'")

    def _new_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("descriptor_monitor", self._descriptor_monitor_node)
        graph.add_node("finalize", self._finalize_node)
        return graph

    def _build_sas_graph(self, layout: TopologyLayout):
        graph = self._new_graph()
        graph.add_node(
            "single_agent",
            self._make_single_agent_stage_node(
                node_name="single_agent",
                stage_role="worker",
                agent_id_resolver=lambda state: layout.agent_ids[0],
                directive_builder=lambda state, agent_id: (
                    "Solve the task end to end. There are no peer agents."
                ),
                message_selector=lambda state, agent_id: [],
            ),
        )
        graph.add_edge(START, "single_agent")
        graph.add_edge("single_agent", "descriptor_monitor")
        graph.add_edge("descriptor_monitor", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def _build_only_voting_graph(self, layout: TopologyLayout):
        graph = self._new_graph()
        graph.add_node(
            "dispatch_independent_agents",
            self._make_dispatch_node(
                node_name="dispatch_independent_agents",
                active_agents_resolver=lambda state: list(layout.agent_ids),
            ),
        )
        graph.add_node(
            "worker",
            self._make_parallel_stage_node(
                node_name="worker",
                stage_role="worker",
                directive_builder=lambda state, agent_id: (
                    "Solve the task independently. Ignore any notion of peer context."
                ),
                message_selector=lambda state, agent_id: [],
            ),
        )
        graph.add_node("voter", self._only_voting_voter_node)
        graph.add_edge(START, "dispatch_independent_agents")
        # Conditional fan-out: one independent worker branch per active agent.
        graph.add_conditional_edges(
            "dispatch_independent_agents",
            self._make_parallel_router(worker_node="worker", fallback_node="voter"),
            ["worker", "voter"],
        )
        graph.add_edge("worker", "voter")
        graph.add_edge("voter", "descriptor_monitor")
        graph.add_edge("descriptor_monitor", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def _build_orchestrator_no_discussion_graph(self, layout: TopologyLayout):
        graph = self._new_graph()
        graph.add_node(
            "orchestrator_plan",
            self._make_single_agent_stage_node(
                node_name="orchestrator_plan",
                stage_role="planner",
                agent_id_resolver=lambda state: layout.orchestrator_id,
                directive_builder=lambda state, agent_id: (
                    "Produce a concise plan and a bounded task package for each specialist."
                ),
                message_selector=lambda state, agent_id: [],
            ),
        )
        graph.add_node(
            "dispatch_specialists",
            self._make_dispatch_node(
                node_name="dispatch_specialists",
                active_agents_resolver=lambda state: list(layout.specialists),
                packet_builder=lambda state, active_agents: self._build_orchestrator_task_packets(
                    state,
                    recipients=active_agents,
                ),
            ),
        )
        graph.add_node(
            "specialist_worker",
            self._make_parallel_stage_node(
                node_name="specialist_worker",
                stage_role="worker",
                directive_builder=lambda state, agent_id: (
                    "Work only from the orchestrator package. Do not infer or request peer outputs."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"task_package", "orchestrator_feedback"},
                    round_index=int(state.get("round_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node(
            "relay_specialist_reports",
            self._make_packet_node(
                node_name="relay_specialist_reports",
                packet_builder=lambda state: self._build_report_packets(
                    state,
                    source_node_names={"specialist_worker"},
                    recipients_by_sender=lambda artifact: [layout.orchestrator_id]
                    if layout.orchestrator_id
                    else [],
                    kind="specialist_report",
                ),
            ),
        )
        graph.add_node(
            "orchestrator_merge",
            self._make_single_agent_stage_node(
                node_name="orchestrator_merge",
                stage_role="aggregator",
                agent_id_resolver=lambda state: layout.orchestrator_id,
                directive_builder=lambda state, agent_id: (
                    "Merge the specialist reports into one best answer. Preserve unresolved issues explicitly."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"specialist_report"},
                    round_index=int(state.get("round_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node(
            "termination_checker",
            self._make_cycle_termination_node(
                node_name="termination_checker",
                topology_label=layout.topology,
                candidate_node_name="orchestrator_merge",
                consensus_node_names={"specialist_worker"},
                expected_count=len(layout.specialists),
                continue_next_step="dispatch_specialists",
                stop_next_step="finalize",
            ),
        )
        graph.add_edge(START, "orchestrator_plan")
        graph.add_edge("orchestrator_plan", "dispatch_specialists")
        # Conditional fan-out: launch the selected specialists, or skip when none are active.
        graph.add_conditional_edges(
            "dispatch_specialists",
            self._make_parallel_router(worker_node="specialist_worker", fallback_node="relay_specialist_reports"),
            ["specialist_worker", "relay_specialist_reports"],
        )
        graph.add_edge("specialist_worker", "relay_specialist_reports")
        graph.add_edge("relay_specialist_reports", "orchestrator_merge")
        graph.add_edge("orchestrator_merge", "descriptor_monitor")
        graph.add_edge("descriptor_monitor", "termination_checker")
        # Conditional loop exit: either run another orchestrator cycle or finalize.
        graph.add_conditional_edges(
            "termination_checker",
            self._route_next_step,
            {
                "dispatch_specialists": "dispatch_specialists",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)
        return graph

    def _build_orchestrator_with_discussion_graph(self, layout: TopologyLayout):
        graph = self._new_graph()
        graph.add_node(
            "orchestrator_plan",
            self._make_single_agent_stage_node(
                node_name="orchestrator_plan",
                stage_role="planner",
                agent_id_resolver=lambda state: layout.orchestrator_id,
                directive_builder=lambda state, agent_id: (
                    "Plan the specialist work and produce bounded task packages."
                ),
                message_selector=lambda state, agent_id: [],
            ),
        )
        graph.add_node(
            "dispatch_specialists",
            self._make_dispatch_node(
                node_name="dispatch_specialists",
                active_agents_resolver=lambda state: list(layout.specialists),
                packet_builder=lambda state, active_agents: self._build_orchestrator_task_packets(
                    state,
                    recipients=active_agents,
                ),
            ),
        )
        graph.add_node(
            "specialists_initial_round",
            self._make_parallel_stage_node(
                node_name="specialists_initial_round",
                stage_role="worker",
                directive_builder=lambda state, agent_id: (
                    "Produce an initial specialist artifact from the orchestrator task package."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"task_package", "orchestrator_feedback"},
                    round_index=int(state.get("round_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node(
            "dispatch_revision_round",
            self._make_dispatch_node(
                node_name="dispatch_revision_round",
                active_agents_resolver=lambda state: list(layout.specialists),
            ),
        )
        graph.add_node(
            "specialists_revision_round",
            self._make_parallel_stage_node(
                node_name="specialists_revision_round",
                stage_role="critic",
                directive_builder=lambda state, agent_id: (
                    "Revise your artifact using the orchestrator-provided peer summary bundle only."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"peer_summary"},
                    round_index=int(state.get("round_index", 0)),
                    discussion_index=int(state.get("discussion_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node(
            "relay_specialist_reports",
            self._make_packet_node(
                node_name="relay_specialist_reports",
                packet_builder=lambda state: self._build_report_packets(
                    state,
                    source_node_names={self._current_specialist_stage_name(state)},
                    recipients_by_sender=lambda artifact: [layout.orchestrator_id]
                    if layout.orchestrator_id
                    else [],
                    kind="specialist_report",
                ),
            ),
        )
        graph.add_node("orchestrator_relay", self._orchestrator_relay_controller)
        graph.add_node(
            "orchestrator_merge",
            self._make_single_agent_stage_node(
                node_name="orchestrator_merge",
                stage_role="aggregator",
                agent_id_resolver=lambda state: layout.orchestrator_id,
                directive_builder=lambda state, agent_id: (
                    "Synthesize the latest specialist artifacts into the current best final answer."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"specialist_report"},
                    round_index=int(state.get("round_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node(
            "cycle_termination_checker",
            self._make_cycle_termination_node(
                node_name="cycle_termination_checker",
                topology_label=layout.topology,
                candidate_node_name="orchestrator_merge",
                consensus_node_names={"specialists_initial_round", "specialists_revision_round"},
                expected_count=len(layout.specialists),
                continue_next_step="dispatch_specialists",
                stop_next_step="finalize",
            ),
        )
        graph.add_edge(START, "orchestrator_plan")
        graph.add_edge("orchestrator_plan", "dispatch_specialists")
        # Conditional fan-out: run the initial specialist round in parallel over the active specialist set.
        graph.add_conditional_edges(
            "dispatch_specialists",
            self._make_parallel_router(
                worker_node="specialists_initial_round",
                fallback_node="relay_specialist_reports",
            ),
            ["specialists_initial_round", "relay_specialist_reports"],
        )
        graph.add_edge("specialists_initial_round", "relay_specialist_reports")
        # Conditional fan-out: each revision round is another parallel specialist pass.
        graph.add_conditional_edges(
            "dispatch_revision_round",
            self._make_parallel_router(
                worker_node="specialists_revision_round",
                fallback_node="orchestrator_merge",
            ),
            ["specialists_revision_round", "orchestrator_merge"],
        )
        graph.add_edge("specialists_revision_round", "relay_specialist_reports")
        graph.add_edge("relay_specialist_reports", "orchestrator_relay")
        # Conditional discussion routing: continue mediated discussion or move to final merge.
        graph.add_conditional_edges(
            "orchestrator_relay",
            self._route_next_step,
            {
                "dispatch_revision_round": "dispatch_revision_round",
                "orchestrator_merge": "orchestrator_merge",
            },
        )
        graph.add_edge("orchestrator_merge", "descriptor_monitor")
        graph.add_edge("descriptor_monitor", "cycle_termination_checker")
        # Conditional outer-cycle routing: launch another orchestration cycle or finalize.
        graph.add_conditional_edges(
            "cycle_termination_checker",
            self._route_next_step,
            {
                "dispatch_specialists": "dispatch_specialists",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)
        return graph

    def _build_orchestrator_tree_graph(self, layout: TopologyLayout):
        graph = self._new_graph()
        graph.add_node(
            "root_plan",
            self._make_single_agent_stage_node(
                node_name="root_plan",
                stage_role="planner",
                agent_id_resolver=lambda state: layout.orchestrator_id,
                directive_builder=lambda state, agent_id: (
                    "Create manager-level task packages for the tree. Communication must remain parent-child only."
                ),
                message_selector=lambda state, agent_id: [],
            ),
        )
        graph.add_node(
            "manager_dispatch",
            self._make_dispatch_node(
                node_name="manager_dispatch",
                active_agents_resolver=lambda state: list(layout.managers),
                packet_builder=lambda state, active_agents: self._build_root_task_packets(
                    state,
                    recipients=active_agents,
                ),
            ),
        )
        graph.add_node(
            "manager_nodes",
            self._make_parallel_stage_node(
                node_name="manager_nodes",
                stage_role="planner",
                directive_builder=lambda state, agent_id: (
                    "Refine the parent task package into child-specific work packages. Do not use sibling information."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"root_task_package"},
                    round_index=int(state.get("round_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node(
            "worker_dispatch",
            self._make_dispatch_node(
                node_name="worker_dispatch",
                active_agents_resolver=lambda state: list(layout.leaves),
                packet_builder=lambda state, active_agents: self._build_manager_task_packets(
                    state,
                    recipients=active_agents,
                ),
            ),
        )
        graph.add_node(
            "worker_nodes",
            self._make_parallel_stage_node(
                node_name="worker_nodes",
                stage_role="worker",
                directive_builder=lambda state, agent_id: (
                    "Work only from your parent manager package. There is no sibling or backflow context."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"manager_task_package"},
                    round_index=int(state.get("round_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node(
            "worker_relay",
            self._make_packet_node(
                node_name="worker_relay",
                packet_builder=lambda state: self._build_report_packets(
                    state,
                    source_node_names={"worker_nodes"},
                    recipients_by_sender=lambda artifact: [
                        layout.parent_by_agent[str(artifact.get("agent_id", ""))]
                    ]
                    if str(artifact.get("agent_id", "")) in layout.parent_by_agent
                    else [],
                    kind="child_report",
                ),
            ),
        )
        graph.add_node(
            "manager_reducers",
            self._make_parallel_stage_node(
                node_name="manager_reducers",
                stage_role="aggregator",
                directive_builder=lambda state, agent_id: (
                    "Aggregate only your child reports into a refined manager artifact."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"child_report"},
                    round_index=int(state.get("round_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node(
            "manager_reduce_dispatch",
            self._make_dispatch_node(
                node_name="manager_reduce_dispatch",
                active_agents_resolver=lambda state: list(layout.managers),
            ),
        )
        graph.add_node(
            "manager_relay",
            self._make_packet_node(
                node_name="manager_relay",
                packet_builder=lambda state: self._build_report_packets(
                    state,
                    source_node_names={"manager_reducers"},
                    recipients_by_sender=lambda artifact: [layout.orchestrator_id]
                    if layout.orchestrator_id
                    else [],
                    kind="manager_report",
                ),
            ),
        )
        graph.add_node(
            "root_reducer",
            self._make_single_agent_stage_node(
                node_name="root_reducer",
                stage_role="aggregator",
                agent_id_resolver=lambda state: layout.orchestrator_id,
                directive_builder=lambda state, agent_id: (
                    "Aggregate manager reports into the root artifact. Follow topological order and prevent backflow."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"manager_report"},
                    round_index=int(state.get("round_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node(
            "termination_checker",
            self._make_cycle_termination_node(
                node_name="termination_checker",
                topology_label=layout.topology,
                candidate_node_name="root_reducer",
                consensus_node_names={"manager_reducers"},
                expected_count=len(layout.managers),
                continue_next_step="manager_dispatch",
                stop_next_step="finalize",
            ),
        )
        graph.add_edge(START, "root_plan")
        graph.add_edge("root_plan", "manager_dispatch")
        # Conditional fan-out: one branch per active manager.
        graph.add_conditional_edges(
            "manager_dispatch",
            self._make_parallel_router(worker_node="manager_nodes", fallback_node="worker_dispatch"),
            ["manager_nodes", "worker_dispatch"],
        )
        graph.add_edge("manager_nodes", "worker_dispatch")
        # Conditional fan-out: one branch per active leaf worker.
        graph.add_conditional_edges(
            "worker_dispatch",
            self._make_parallel_router(worker_node="worker_nodes", fallback_node="worker_relay"),
            ["worker_nodes", "worker_relay"],
        )
        graph.add_edge("worker_nodes", "worker_relay")
        graph.add_edge("worker_relay", "manager_reduce_dispatch")
        graph.add_conditional_edges(
            "manager_reduce_dispatch",
            self._make_parallel_router(worker_node="manager_reducers", fallback_node="manager_relay"),
            ["manager_reducers", "manager_relay"],
        )
        graph.add_edge("manager_reducers", "manager_relay")
        graph.add_edge("manager_relay", "root_reducer")
        graph.add_edge("root_reducer", "descriptor_monitor")
        graph.add_edge("descriptor_monitor", "termination_checker")
        # Conditional tree-cycle routing: descend through the tree again or finalize.
        graph.add_conditional_edges(
            "termination_checker",
            self._route_next_step,
            {
                "manager_dispatch": "manager_dispatch",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)
        return graph

    def _build_fully_linked_debate_graph(self, layout: TopologyLayout):
        graph = self._new_graph()
        graph.add_node("debate_init", self._debate_init_node)
        graph.add_node("debate_controller", self._debate_controller_node)
        graph.add_node(
            "debate_dispatch",
            self._make_dispatch_node(
                node_name="debate_dispatch",
                active_agents_resolver=lambda state: list(layout.agent_ids),
            ),
        )
        graph.add_node(
            "debate_round",
            self._make_parallel_stage_node(
                node_name="debate_round",
                stage_role="critic",
                directive_builder=lambda state, agent_id: (
                    "If no peer summaries are available yet, investigate independently from the task and gather evidence first. "
                    "Otherwise, debate using only bounded peer summaries from the latest round and revise your answer if warranted."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"debate_round"},
                    round_index=int(state.get("round_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node("judge", self._fully_linked_debate_judge_node)
        graph.add_edge(START, "debate_init")
        graph.add_edge("debate_init", "debate_controller")
        # Conditional controller routing: either begin another debate round or hand off to the judge.
        graph.add_conditional_edges(
            "debate_controller",
            self._route_next_step,
            {
                "debate_dispatch": "debate_dispatch",
                "judge": "judge",
            },
        )
        # Conditional fan-out: one debate branch per active debater.
        graph.add_conditional_edges(
            "debate_dispatch",
            self._make_parallel_router(worker_node="debate_round", fallback_node="debate_controller"),
            ["debate_round", "debate_controller"],
        )
        graph.add_edge("debate_round", "debate_controller")
        graph.add_edge("judge", "descriptor_monitor")
        graph.add_edge("descriptor_monitor", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def _build_group_chat_debate_graph(self, layout: TopologyLayout):
        graph = self._new_graph()
        graph.add_node(
            "group_dispatch",
            self._make_dispatch_node(
                node_name="group_dispatch",
                active_agents_resolver=lambda state: list(layout.agent_ids),
            ),
        )
        graph.add_node(
            "group_debate_round",
            self._make_parallel_stage_node(
                node_name="group_debate_round",
                stage_role="critic",
                directive_builder=lambda state, agent_id: (
                    "Debate only inside your group, using bounded group summaries."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"group_debate_round"},
                    round_index=int(state.get("round_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node("group_controller", self._group_controller_node)
        graph.add_node(
            "representative_dispatch",
            self._make_dispatch_node(
                node_name="representative_dispatch",
                active_agents_resolver=lambda state: list(layout.representatives),
            ),
        )
        graph.add_node(
            "representative_merge",
            self._make_parallel_stage_node(
                node_name="representative_merge",
                stage_role="aggregator",
                directive_builder=lambda state, agent_id: (
                    "Merge your group summary with representative-level peer summaries when present."
                ),
                message_selector=lambda state, agent_id: self._messages_for_recipient(
                    state,
                    recipient=agent_id,
                    kinds={"group_summary", "representative_debate_round"},
                    round_index=int(state.get("round_index", 0)),
                    discussion_index=int(state.get("discussion_index", 0)),
                    latest_only=True,
                ),
            ),
        )
        graph.add_node("representative_controller", self._representative_controller_node)
        graph.add_node("final_judge", self._group_chat_final_judge_node)
        graph.add_edge(START, "group_dispatch")
        # Conditional fan-out: one intra-group debate branch per active group member.
        graph.add_conditional_edges(
            "group_dispatch",
            self._make_parallel_router(worker_node="group_debate_round", fallback_node="group_controller"),
            ["group_debate_round", "group_controller"],
        )
        graph.add_edge("group_debate_round", "group_controller")
        # Conditional group routing: continue local debate or promote summaries to representatives.
        graph.add_conditional_edges(
            "group_controller",
            self._route_next_step,
            {
                "group_dispatch": "group_dispatch",
                "representative_dispatch": "representative_dispatch",
            },
        )
        # Conditional fan-out: one representative branch per active representative.
        graph.add_conditional_edges(
            "representative_dispatch",
            self._make_parallel_router(
                worker_node="representative_merge",
                fallback_node="final_judge",
            ),
            ["representative_merge", "final_judge"],
        )
        graph.add_edge("representative_merge", "representative_controller")
        # Conditional representative routing: continue representative debate or hand off to the final judge.
        graph.add_conditional_edges(
            "representative_controller",
            self._route_next_step,
            {
                "representative_dispatch": "representative_dispatch",
                "final_judge": "final_judge",
            },
        )
        graph.add_edge("final_judge", "descriptor_monitor")
        graph.add_edge("descriptor_monitor", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def _make_single_agent_stage_node(
        self,
        *,
        node_name: str,
        stage_role: str,
        agent_id_resolver: Callable[[WorkflowState], str | None],
        directive_builder: Callable[[WorkflowState, str], str],
        message_selector: Callable[[WorkflowState, str], list[RelayPacket]],
    ) -> Callable[[WorkflowState], dict[str, Any]]:
        def node(state: WorkflowState) -> dict[str, Any]:
            agent_id = agent_id_resolver(state)
            if not agent_id:
                return {
                    "phase": node_name,
                    "phase_history": [self._phase_history_entry(state, node_name, active_agents=[])],
                }
            result = self._execute_agent_stage(
                state,
                agent_id=agent_id,
                node_name=node_name,
                stage_role=stage_role,
                directive=directive_builder(state, agent_id),
                visible_messages=message_selector(state, agent_id),
            )
            result["phase"] = node_name
            result["phase_history"] = [self._phase_history_entry(state, node_name, active_agents=[agent_id])]
            if node_name == "single_agent":
                result["final_reason"] = f"{state['topology']}:single_agent"
            return result

        return node

    def _make_parallel_stage_node(
        self,
        *,
        node_name: str,
        stage_role: str,
        directive_builder: Callable[[WorkflowState, str], str],
        message_selector: Callable[[WorkflowState, str], list[RelayPacket]],
    ) -> Callable[[WorkflowState], dict[str, Any]]:
        def node(state: WorkflowState) -> dict[str, Any]:
            agent_id = str(state.get("active_agent", ""))
            return self._execute_agent_stage(
                state,
                agent_id=agent_id,
                node_name=node_name,
                stage_role=stage_role,
                directive=directive_builder(state, agent_id),
                visible_messages=message_selector(state, agent_id),
            )

        return node

    def _make_dispatch_node(
        self,
        *,
        node_name: str,
        active_agents_resolver: Callable[[WorkflowState], list[str]],
        packet_builder: Callable[[WorkflowState, list[str]], list[dict[str, Any]]] | None = None,
    ) -> Callable[[WorkflowState], dict[str, Any]]:
        def node(state: WorkflowState) -> dict[str, Any]:
            dispatch_id = int(state.get("dispatch_id", -1)) + 1
            active_agents = [agent for agent in active_agents_resolver(state) if agent]
            trace = [
                self._draft_event(
                    dispatch_id=dispatch_id,
                    node_order=0,
                    agent_order=-1,
                    event_order=0,
                    actor="system",
                    event_type="revise",
                    payload={
                        "node": node_name,
                        "phase": node_name,
                        "round_index": int(state.get("round_index", 0)),
                        "discussion_index": int(state.get("discussion_index", 0)),
                        "active_agents": list(active_agents),
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=f"run_{state['run_index']}_dispatch_{dispatch_id}_{node_name}",
                )
            ]
            updates: dict[str, Any] = {
                "dispatch_id": dispatch_id,
                "active_agents": active_agents,
                "phase": node_name,
                "phase_history": [self._phase_history_entry(state, node_name, active_agents=active_agents)],
            }
            if packet_builder is not None:
                emitted = self._emit_packets(
                    state,
                    dispatch_id=dispatch_id,
                    phase=node_name,
                    packet_specs=packet_builder(state, active_agents),
                )
                trace.extend(emitted.pop("trace_payloads"))
                updates.update(emitted)
            updates["trace_payloads"] = trace
            return updates

        return node

    def _make_packet_node(
        self,
        *,
        node_name: str,
        packet_builder: Callable[[WorkflowState], list[dict[str, Any]]],
    ) -> Callable[[WorkflowState], dict[str, Any]]:
        def node(state: WorkflowState) -> dict[str, Any]:
            dispatch_id = int(state.get("dispatch_id", -1))
            emitted = self._emit_packets(
                state,
                dispatch_id=dispatch_id,
                phase=node_name,
                packet_specs=packet_builder(state),
            )
            trace = [
                self._draft_event(
                    dispatch_id=dispatch_id,
                    node_order=2,
                    agent_order=-1,
                    event_order=0,
                    actor="system",
                    event_type="verify",
                    payload={
                        "node": node_name,
                        "phase": node_name,
                        "round_index": int(state.get("round_index", 0)),
                        "discussion_index": int(state.get("discussion_index", 0)),
                        "messages_created": len(emitted.get("messages", [])),
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=f"run_{state['run_index']}_{node_name}_{dispatch_id}",
                )
            ]
            trace.extend(emitted.pop("trace_payloads"))
            return {
                "phase": node_name,
                "phase_history": [self._phase_history_entry(state, node_name)],
                "trace_payloads": trace,
                **emitted,
            }

        return node

    def _make_cycle_termination_node(
        self,
        *,
        node_name: str,
        topology_label: str,
        candidate_node_name: str,
        consensus_node_names: set[str],
        expected_count: int,
        continue_next_step: str,
        stop_next_step: str,
    ) -> Callable[[WorkflowState], dict[str, Any]]:
        """Build a reusable round-level termination node for orchestrator and tree cycles."""

        def node(state: WorkflowState) -> dict[str, Any]:
            current_round = int(state.get("round_index", 0))
            candidate_artifacts = self._artifacts_for_nodes(
                state,
                node_names={candidate_node_name},
                round_index=current_round,
                latest_only=True,
            )
            previous_candidates = self._artifacts_for_nodes(
                state,
                node_names={candidate_node_name},
                round_index=current_round - 1,
                latest_only=True,
            )
            consensus_artifacts = self._artifacts_for_nodes(
                state,
                node_names=consensus_node_names,
                round_index=current_round,
                latest_only=True,
            )
            decision = self._termination_decision(
                state,
                stage_name=node_name,
                round_index=current_round,
                discussion_index=int(state.get("discussion_index", 0)),
                minimum_round_index=current_round,
                minimum_required_rounds=0,
                minimum_round_label="round",
                candidate_artifacts=candidate_artifacts,
                previous_candidate_artifacts=previous_candidates,
                consensus_artifacts=consensus_artifacts,
                expected_count=expected_count,
                max_reached=current_round + 1 >= int(state.get("rounds", 1)),
                continue_next_step=continue_next_step,
                stop_next_step=stop_next_step,
            )
            updates: dict[str, Any] = {
                "phase": node_name,
                "termination_decision": decision,
                "termination_history": [dict(decision)],
                "next_step": str(decision.get("next_step", stop_next_step)),
                "phase_history": [self._phase_history_entry(state, node_name)],
                "trace_payloads": [
                    self._termination_event(state, node_name, decision),
                ],
            }
            if not bool(decision.get("should_stop", False)):
                updates["round_index"] = current_round + 1
                updates["discussion_index"] = 0
            else:
                updates["final_reason"] = f"{topology_label}:{decision.get('reason', 'stop')}"
            return updates

        return node

    def _only_voting_voter_node(self, state: WorkflowState) -> dict[str, Any]:
        artifacts = self._artifacts_for_nodes(
            state,
            node_names={"worker"},
            round_index=int(state.get("round_index", 0)),
            latest_only=True,
        )
        vote_result = self._select_final_answer(
            state=state,
            stage_name="voter",
            artifacts=artifacts,
        )
        final_reason = f"{state['topology']}:{self._vote_outcome_reason(vote_result)}"
        return {
            "phase": "voter",
            "vote_tally": dict(vote_result["tally"]),
            "final_vote_source": str(vote_result["source"]),
            "final_answer": self._safe_vote_answer_or_fallback(state, artifacts, vote_result),
            "selected_artifact_id": str(vote_result.get("selected_artifact_id", "")),
            "selected_agent_id": str(vote_result.get("selected_agent_id", "")),
            "selected_source_artifact_ids": list(vote_result.get("selected_source_artifact_ids", [])),
            "final_reason": final_reason,
            "phase_history": [self._phase_history_entry(state, "voter")],
            "trace_payloads": [
                self._draft_event(
                    dispatch_id=int(state.get("dispatch_id", -1)),
                    node_order=3,
                    agent_order=-1,
                    event_order=0,
                    actor="judge",
                    event_type="verify",
                    payload={
                        "node": "voter",
                        "candidate_count": len(artifacts),
                        "vote_tally": dict(vote_result["tally"]),
                        "vote_mode": str(vote_result["mode"]),
                        "vote_source": str(vote_result["source"]),
                        "vote_outcome_reason": self._vote_outcome_reason(vote_result),
                        "vote_explanation": str(vote_result["explanation"]),
                        "selected_artifact_id": str(vote_result.get("selected_artifact_id", "")),
                        "selected_agent_id": str(vote_result.get("selected_agent_id", "")),
                    },
                    token_in=int(vote_result["token_in"]),
                    token_out=int(vote_result["token_out"]),
                    latency_ms=float(vote_result["latency_ms"]),
                    cost_usd=float(vote_result["cost_usd"]),
                    state_id=f"run_{state['run_index']}_voter",
                )
            ],
        }

    def _debate_init_node(self, state: WorkflowState) -> dict[str, Any]:
        return {
            "phase": "debate_init",
            "next_step": "debate_dispatch",
            "phase_history": [self._phase_history_entry(state, "debate_init")],
            "trace_payloads": [
                self._draft_event(
                    dispatch_id=-1,
                    node_order=0,
                    agent_order=-1,
                    event_order=0,
                    actor="system",
                    event_type="plan",
                    payload={
                        "node": "debate_init",
                        "topology": state["topology"],
                        "rounds": state["rounds"],
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=f"run_{state['run_index']}_debate_init",
                )
            ],
        }

    def _debate_controller_node(self, state: WorkflowState) -> dict[str, Any]:
        """Evaluate fully-linked debate stopping criteria and prepare the next bounded peer broadcast."""

        current_round = int(state.get("round_index", 0))
        artifacts = self._artifacts_for_nodes(
            state,
            node_names={"debate_round"},
            round_index=current_round,
            latest_only=True,
        )
        if not artifacts:
            return {
                "phase": "debate_controller",
                "next_step": "debate_dispatch",
                "phase_history": [self._phase_history_entry(state, "debate_controller")],
                "trace_payloads": [
                    self._draft_event(
                        dispatch_id=int(state.get("dispatch_id", -1)),
                        node_order=2,
                        agent_order=-1,
                        event_order=0,
                        actor="system",
                        event_type="verify",
                        payload={"node": "debate_controller", "status": "awaiting_round_zero"},
                        token_in=0,
                        token_out=0,
                        latency_ms=1.0,
                        cost_usd=0.0,
                        state_id=f"run_{state['run_index']}_debate_controller_init",
                    )
                ],
            }

        previous_artifacts = self._artifacts_for_nodes(
            state,
            node_names={"debate_round"},
            round_index=current_round - 1,
            latest_only=True,
        )
        decision = self._termination_decision(
            state,
            stage_name="debate_controller",
            round_index=current_round,
            discussion_index=0,
            minimum_round_index=current_round,
            minimum_required_rounds=int(state.get("minimum_discussion_rounds", 1)),
            minimum_round_label="round",
            candidate_artifacts=artifacts,
            previous_candidate_artifacts=previous_artifacts,
            consensus_artifacts=artifacts,
            expected_count=len(state["layout"].agent_ids),
            max_reached=current_round + 1 >= int(state.get("rounds", 1)),
            continue_next_step="debate_dispatch",
            stop_next_step="judge",
        )
        updates: dict[str, Any] = {
            "phase": "debate_controller",
            "termination_decision": decision,
            "termination_history": [dict(decision)],
            "phase_history": [self._phase_history_entry(state, "debate_controller")],
            "trace_payloads": [self._termination_event(state, "debate_controller", decision)],
        }
        if bool(decision.get("should_stop", False)):
            updates["next_step"] = "judge"
            updates["final_reason"] = f"{state['topology']}:{decision.get('reason', 'judge')}"
            return updates

        compaction = self._compact_transcript_update(
            state,
            stage_name="debate_controller",
            artifacts=artifacts,
            next_round=current_round + 1,
            next_discussion=0,
        )
        self._merge_compaction_update(updates, compaction)
        emitted = self._emit_packets(
            state,
            dispatch_id=int(state.get("dispatch_id", -1)),
            phase="debate_controller",
            packet_specs=self._build_peer_summary_packets(
                state,
                artifacts=artifacts,
                recipients_resolver=lambda artifact: [
                    agent_id
                    for agent_id in state["layout"].agent_ids
                    if agent_id != str(artifact.get("agent_id", ""))
                ],
                kind="debate_round",
                next_round=current_round + 1,
                next_discussion=0,
            ),
        )
        relay_trace = emitted.pop("trace_payloads", [])
        updates.update(emitted)
        updates["trace_payloads"].extend(relay_trace)
        updates["round_index"] = current_round + 1
        updates["next_step"] = "debate_dispatch"
        return updates

    def _fully_linked_debate_judge_node(self, state: WorkflowState) -> dict[str, Any]:
        artifacts = self._artifacts_for_nodes(
            state,
            node_names={"debate_round"},
            round_index=int(state.get("round_index", 0)),
            latest_only=True,
        )
        vote_result = self._select_final_answer(
            state=state,
            stage_name="judge",
            artifacts=artifacts,
        )
        return {
            "phase": "judge",
            "vote_tally": dict(vote_result["tally"]),
            "final_vote_source": str(vote_result["source"]),
            "final_answer": self._safe_vote_answer_or_fallback(state, artifacts, vote_result),
            "selected_artifact_id": str(vote_result.get("selected_artifact_id", "")),
            "selected_agent_id": str(vote_result.get("selected_agent_id", "")),
            "selected_source_artifact_ids": list(vote_result.get("selected_source_artifact_ids", [])),
            "final_reason": str(state.get("final_reason") or f"{state['topology']}:judge_vote"),
            "phase_history": [self._phase_history_entry(state, "judge")],
            "trace_payloads": [
                self._draft_event(
                    dispatch_id=int(state.get("dispatch_id", -1)),
                    node_order=3,
                    agent_order=-1,
                    event_order=0,
                    actor="judge",
                    event_type="verify",
                    payload={
                        "node": "judge",
                        "vote_tally": dict(vote_result["tally"]),
                        "vote_mode": str(vote_result["mode"]),
                        "vote_source": str(vote_result["source"]),
                        "vote_explanation": str(vote_result["explanation"]),
                        "selected_artifact_id": str(vote_result.get("selected_artifact_id", "")),
                        "selected_agent_id": str(vote_result.get("selected_agent_id", "")),
                    },
                    token_in=int(vote_result["token_in"]),
                    token_out=int(vote_result["token_out"]),
                    latency_ms=float(vote_result["latency_ms"]),
                    cost_usd=float(vote_result["cost_usd"]),
                    state_id=f"run_{state['run_index']}_debate_judge",
                )
            ],
        }

    def _group_controller_node(self, state: WorkflowState) -> dict[str, Any]:
        """Control the intra-group debate loop and decide when to promote group summaries upward."""

        current_round = int(state.get("round_index", 0))
        artifacts = self._artifacts_for_nodes(
            state,
            node_names={"group_debate_round"},
            round_index=current_round,
            latest_only=True,
        )
        previous_artifacts = self._artifacts_for_nodes(
            state,
            node_names={"group_debate_round"},
            round_index=current_round - 1,
            latest_only=True,
        )
        decision = self._termination_decision(
            state,
            stage_name="group_controller",
            round_index=current_round,
            discussion_index=0,
            minimum_round_index=current_round,
            minimum_required_rounds=int(state.get("minimum_discussion_rounds", 1)),
            minimum_round_label="round",
            candidate_artifacts=artifacts,
            previous_candidate_artifacts=previous_artifacts,
            consensus_artifacts=artifacts,
            expected_count=len(state["layout"].agent_ids),
            max_reached=current_round + 1 >= int(state.get("rounds", 1)),
            continue_next_step="group_dispatch",
            stop_next_step="representative_dispatch",
        )
        updates: dict[str, Any] = {
            "phase": "group_controller",
            "termination_decision": decision,
            "termination_history": [dict(decision)],
            "phase_history": [self._phase_history_entry(state, "group_controller")],
            "trace_payloads": [self._termination_event(state, "group_controller", decision)],
        }
        if bool(decision.get("should_stop", False)):
            compaction = self._compact_transcript_update(
                state,
                stage_name="group_controller",
                artifacts=artifacts,
                next_round=current_round,
                next_discussion=0,
            )
            self._merge_compaction_update(updates, compaction)
            emitted = self._emit_packets(
                state,
                dispatch_id=int(state.get("dispatch_id", -1)),
                phase="group_controller",
                packet_specs=self._build_group_summary_packets(state, artifacts),
            )
            relay_trace = emitted.pop("trace_payloads", [])
            updates.update(emitted)
            updates["trace_payloads"].extend(relay_trace)
            updates["next_step"] = "representative_dispatch"
            updates["discussion_index"] = 0
            return updates

        compaction = self._compact_transcript_update(
            state,
            stage_name="group_controller",
            artifacts=artifacts,
            next_round=current_round + 1,
            next_discussion=0,
        )
        self._merge_compaction_update(updates, compaction)
        emitted = self._emit_packets(
            state,
            dispatch_id=int(state.get("dispatch_id", -1)),
            phase="group_controller",
            packet_specs=self._build_group_peer_packets(
                state,
                artifacts=artifacts,
                next_round=current_round + 1,
            ),
        )
        relay_trace = emitted.pop("trace_payloads", [])
        updates.update(emitted)
        updates["trace_payloads"].extend(relay_trace)
        updates["round_index"] = current_round + 1
        updates["next_step"] = "group_dispatch"
        return updates

    def _representative_controller_node(self, state: WorkflowState) -> dict[str, Any]:
        """Control the representative-level debate loop before the shared final judge."""

        current_discussion = int(state.get("discussion_index", 0))
        artifacts = self._artifacts_for_nodes(
            state,
            node_names={"representative_merge"},
            round_index=int(state.get("round_index", 0)),
            discussion_index=current_discussion,
            latest_only=True,
        )
        previous_artifacts = self._artifacts_for_nodes(
            state,
            node_names={"representative_merge"},
            round_index=int(state.get("round_index", 0)),
            discussion_index=current_discussion - 1,
            latest_only=True,
        )
        decision = self._termination_decision(
            state,
            stage_name="representative_controller",
            round_index=int(state.get("round_index", 0)),
            discussion_index=current_discussion,
            minimum_round_index=current_discussion,
            minimum_required_rounds=int(state.get("minimum_discussion_rounds", 1)),
            minimum_round_label="discussion round",
            candidate_artifacts=artifacts,
            previous_candidate_artifacts=previous_artifacts,
            consensus_artifacts=artifacts,
            expected_count=len(state["layout"].representatives),
            max_reached=current_discussion + 1 >= int(state.get("discussion_rounds", 1)),
            continue_next_step="representative_dispatch",
            stop_next_step="final_judge",
        )
        updates: dict[str, Any] = {
            "phase": "representative_controller",
            "termination_decision": decision,
            "termination_history": [dict(decision)],
            "phase_history": [self._phase_history_entry(state, "representative_controller")],
            "trace_payloads": [self._termination_event(state, "representative_controller", decision)],
        }
        if bool(decision.get("should_stop", False)) or len(state["layout"].representatives) <= 1:
            updates["next_step"] = "final_judge"
            if bool(decision.get("should_stop", False)):
                updates["final_reason"] = f"{state['topology']}:{decision.get('reason', 'judge')}"
            return updates

        compaction = self._compact_transcript_update(
            state,
            stage_name="representative_controller",
            artifacts=artifacts,
            next_round=int(state.get("round_index", 0)),
            next_discussion=current_discussion + 1,
        )
        self._merge_compaction_update(updates, compaction)
        emitted = self._emit_packets(
            state,
            dispatch_id=int(state.get("dispatch_id", -1)),
            phase="representative_controller",
            packet_specs=self._build_peer_summary_packets(
                state,
                artifacts=artifacts,
                recipients_resolver=lambda artifact: [
                    agent_id
                    for agent_id in state["layout"].representatives
                    if agent_id != str(artifact.get("agent_id", ""))
                ],
                kind="representative_debate_round",
                next_round=int(state.get("round_index", 0)),
                next_discussion=current_discussion + 1,
            ),
        )
        relay_trace = emitted.pop("trace_payloads", [])
        updates.update(emitted)
        updates["trace_payloads"].extend(relay_trace)
        updates["discussion_index"] = current_discussion + 1
        updates["next_step"] = "representative_dispatch"
        return updates

    def _group_chat_final_judge_node(self, state: WorkflowState) -> dict[str, Any]:
        artifacts = self._artifacts_for_nodes(
            state,
            node_names={"representative_merge"},
            round_index=int(state.get("round_index", 0)),
            latest_only=True,
        )
        vote_result = self._select_final_answer(
            state=state,
            stage_name="final_judge",
            artifacts=artifacts,
        )
        return {
            "phase": "final_judge",
            "vote_tally": dict(vote_result["tally"]),
            "final_vote_source": str(vote_result["source"]),
            "final_answer": self._safe_vote_answer_or_fallback(state, artifacts, vote_result),
            "selected_artifact_id": str(vote_result.get("selected_artifact_id", "")),
            "selected_agent_id": str(vote_result.get("selected_agent_id", "")),
            "selected_source_artifact_ids": list(vote_result.get("selected_source_artifact_ids", [])),
            "final_reason": str(state.get("final_reason") or f"{state['topology']}:judge_vote"),
            "phase_history": [self._phase_history_entry(state, "final_judge")],
            "trace_payloads": [
                self._draft_event(
                    dispatch_id=int(state.get("dispatch_id", -1)),
                    node_order=3,
                    agent_order=-1,
                    event_order=0,
                    actor="judge",
                    event_type="verify",
                    payload={
                        "node": "final_judge",
                        "vote_tally": dict(vote_result["tally"]),
                        "vote_mode": str(vote_result["mode"]),
                        "vote_source": str(vote_result["source"]),
                        "vote_explanation": str(vote_result["explanation"]),
                        "selected_artifact_id": str(vote_result.get("selected_artifact_id", "")),
                        "selected_agent_id": str(vote_result.get("selected_agent_id", "")),
                    },
                    token_in=int(vote_result["token_in"]),
                    token_out=int(vote_result["token_out"]),
                    latency_ms=float(vote_result["latency_ms"]),
                    cost_usd=float(vote_result["cost_usd"]),
                    state_id=f"run_{state['run_index']}_group_judge",
                )
            ],
        }

    def _orchestrator_relay_controller(self, state: WorkflowState) -> dict[str, Any]:
        """Run mediated specialist discussion and decide whether another revision round is necessary."""

        current_round = int(state.get("round_index", 0))
        current_discussion = int(state.get("discussion_index", 0))
        artifacts = self._current_specialist_cycle_artifacts(state, round_index=current_round)
        if not artifacts:
            decision: TerminationDecision = {
                "should_stop": True,
                "next_step": "orchestrator_merge",
                "reason": "invalid_or_failed_branch",
                "reason_detail": "No specialist artifacts were available for relay.",
                "stage_name": "orchestrator_relay",
                "round_index": current_round,
                "discussion_index": current_discussion,
                "consensus_ratio": 0.0,
                "average_confidence": 0.0,
                "mean_delta": None,
                "valid_artifact_count": 0,
            }
        else:
            previous_artifacts = self._current_specialist_cycle_artifacts(
                state,
                round_index=current_round,
                discussion_index=current_discussion - 1,
            )
            decision = self._termination_decision(
                state,
                stage_name="orchestrator_relay",
                round_index=current_round,
                discussion_index=current_discussion,
                minimum_round_index=current_discussion,
                minimum_required_rounds=int(state.get("minimum_discussion_rounds", 1)),
                minimum_round_label="discussion round",
                candidate_artifacts=artifacts,
                previous_candidate_artifacts=previous_artifacts,
                consensus_artifacts=artifacts,
                expected_count=len(state["layout"].specialists),
                max_reached=current_discussion >= int(state.get("discussion_rounds", 1)),
                continue_next_step="dispatch_revision_round",
                stop_next_step="orchestrator_merge",
            )

        updates: dict[str, Any] = {
            "phase": "orchestrator_relay",
            "termination_decision": decision,
            "termination_history": [dict(decision)],
            "phase_history": [self._phase_history_entry(state, "orchestrator_relay")],
            "trace_payloads": [self._termination_event(state, "orchestrator_relay", decision)],
        }
        if bool(decision.get("should_stop", False)):
            updates["next_step"] = "orchestrator_merge"
            return updates

        compaction = self._compact_transcript_update(
            state,
            stage_name="orchestrator_relay",
            artifacts=artifacts,
            next_round=current_round,
            next_discussion=current_discussion + 1,
        )
        self._merge_compaction_update(updates, compaction)
        emitted = self._emit_packets(
            state,
            dispatch_id=int(state.get("dispatch_id", -1)),
            phase="orchestrator_relay",
            packet_specs=self._build_orchestrator_peer_packets(state, artifacts),
        )
        relay_trace = emitted.pop("trace_payloads", [])
        updates.update(emitted)
        updates["trace_payloads"].extend(relay_trace)
        updates["discussion_index"] = current_discussion + 1
        updates["next_step"] = "dispatch_revision_round"
        return updates

    def _compact_transcript_update(
        self,
        state: WorkflowState,
        *,
        stage_name: str,
        artifacts: list[ArtifactRecord],
        next_round: int,
        next_discussion: int,
    ) -> dict[str, Any]:
        if not artifacts:
            return {}
        if str(state.get("topology", "")) not in {
            TOPOLOGY_FULLY_LINKED_DEBATE,
            TOPOLOGY_GROUP_CHAT_DEBATE,
            TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION,
            TOPOLOGY_ORCHESTRATOR_TREE,
        }:
            return {}
        if not self._transcript_compaction_enabled():
            return {}

        previous = dict(state.get("transcript_summary", {}) or {})
        prompt = self._build_transcript_compaction_prompt(
            state=state,
            stage_name=stage_name,
            previous_summary=previous,
            artifacts=artifacts,
        )
        source = "llm_summarizer"
        token_in = 0
        token_out = 0
        latency_ms = 1.0
        cost_usd = 0.0
        try:
            t0 = time.perf_counter()
            llm = state["llm_client"].generate(
                prompt=prompt,
                agent_type="judge",
                task_id=str(state.get("task_id", "")),
                run_index=int(state.get("run_index", 0)),
                agent_id=f"{stage_name}_transcript_summarizer",
                tools=None,
                max_tool_iterations=1,
                temperature=0.0,
            )
            latency_ms = max((time.perf_counter() - t0) * 1000.0, 1.0)
            token_in = int(llm.token_in)
            token_out = int(llm.token_out)
            cost_usd = float(llm.cost_usd)
            parsed = self._parse_transcript_summary(str(llm.text or ""))
            if parsed is None or bool(llm.mock_used):
                source = "deterministic_fallback_mock" if bool(llm.mock_used) else "deterministic_fallback_parse_error"
                parsed = self._deterministic_transcript_summary(
                    state=state,
                    previous_summary=previous,
                    artifacts=artifacts,
                    source=source,
                )
        except Exception as exc:
            source = "deterministic_fallback_error"
            parsed = self._deterministic_transcript_summary(
                state=state,
                previous_summary=previous,
                artifacts=artifacts,
                source=source,
            )
            parsed["fallback_reason"] = f"{type(exc).__name__}:{exc}"

        summary = self._bounded_transcript_summary(parsed)
        summary.update(
            {
                "source": source,
                "stage_name": stage_name,
                "round_index": int(state.get("round_index", 0)),
                "discussion_index": int(state.get("discussion_index", 0)),
                "next_round": int(next_round),
                "next_discussion": int(next_discussion),
            }
        )
        history = {
            "stage_name": stage_name,
            "round_index": int(state.get("round_index", 0)),
            "discussion_index": int(state.get("discussion_index", 0)),
            "next_round": int(next_round),
            "next_discussion": int(next_discussion),
            "source": source,
            "artifact_count": len(artifacts),
            "summary_chars": len(json.dumps(summary, ensure_ascii=False)),
            "token_in": token_in,
            "token_out": token_out,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        }
        trace = self._draft_event(
            dispatch_id=int(state.get("dispatch_id", -1)),
            node_order=2,
            agent_order=-1,
            event_order=0,
            actor="system",
            event_type="verify",
            payload={
                "node": stage_name,
                "operation": "transcript_compaction",
                "source": source,
                "artifact_count": len(artifacts),
                "summary_keys": sorted(summary.keys()),
            },
            token_in=token_in,
            token_out=token_out,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            state_id=(
                f"run_{state['run_index']}_{stage_name}_"
                f"compact_{state.get('round_index', 0)}_{state.get('discussion_index', 0)}"
            ),
        )
        return {
            "transcript_summary": summary,
            "transcript_compaction_history": [history],
            "trace_payloads": [trace],
        }

    @staticmethod
    def _merge_compaction_update(updates: dict[str, Any], compaction: dict[str, Any]) -> None:
        if not compaction:
            return
        if "transcript_summary" in compaction:
            updates["transcript_summary"] = compaction["transcript_summary"]
        if compaction.get("transcript_compaction_history"):
            updates.setdefault("transcript_compaction_history", []).extend(
                compaction["transcript_compaction_history"]
            )
        if compaction.get("trace_payloads"):
            updates.setdefault("trace_payloads", []).extend(compaction["trace_payloads"])

    @staticmethod
    def _transcript_compaction_enabled() -> bool:
        raw = str(os.getenv("MAS_TRANSCRIPT_COMPACTION_ENABLED", "1")).strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _build_transcript_compaction_prompt(
        self,
        *,
        state: WorkflowState,
        stage_name: str,
        previous_summary: dict[str, Any],
        artifacts: list[ArtifactRecord],
    ) -> list[dict[str, Any]]:
        payload = {
            "task": state.get("task_prompt"),
            "topology": state.get("topology"),
            "stage_name": stage_name,
            "round_index": int(state.get("round_index", 0)),
            "discussion_index": int(state.get("discussion_index", 0)),
            "previous_summary": previous_summary,
            "latest_artifacts": [
                {
                    "agent_id": str(artifact.get("agent_id", "")),
                    "role": str(artifact.get("role", "")),
                    "answer": self._trim_text(str(artifact.get("answer", "")), max_chars=1200),
                    "summary": self._trim_text(str(artifact.get("summary", "")), max_chars=500),
                    "critique": self._trim_text(str(artifact.get("critique", "")), max_chars=500),
                    "confidence": float(artifact.get("confidence", 0.5)),
                    "unresolved_issues": list(artifact.get("unresolved_issues", []))[:6],
                    "evidence_summary": list(artifact.get("evidence_summary", []))[:6],
                }
                for artifact in artifacts[:8]
            ],
        }
        return [
            {
                "role": "system",
                "content": (
                    "You compact multi-agent debate history for later rounds. "
                    "Return exactly one JSON object with keys: claims, evidence, "
                    "disagreements, unresolved_questions, best_current_answer, confidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    @classmethod
    def _parse_transcript_summary(cls, text: str) -> dict[str, Any] | None:
        payload = cls._extract_json_object(text)
        if not isinstance(payload, dict):
            return None
        return payload

    def _deterministic_transcript_summary(
        self,
        *,
        state: WorkflowState,
        previous_summary: dict[str, Any],
        artifacts: list[ArtifactRecord],
        source: str,
    ) -> dict[str, Any]:
        claims = list(previous_summary.get("claims", [])) if isinstance(previous_summary.get("claims"), list) else []
        evidence = list(previous_summary.get("evidence", [])) if isinstance(previous_summary.get("evidence"), list) else []
        unresolved = (
            list(previous_summary.get("unresolved_questions", []))
            if isinstance(previous_summary.get("unresolved_questions"), list)
            else []
        )
        disagreements = (
            list(previous_summary.get("disagreements", []))
            if isinstance(previous_summary.get("disagreements"), list)
            else []
        )
        best_answer = str(previous_summary.get("best_current_answer", "") or "")
        best_conf = float(previous_summary.get("confidence", 0.0) or 0.0)
        signatures: dict[str, list[str]] = {}
        for artifact in artifacts:
            agent_id = str(artifact.get("agent_id", ""))
            answer = str(artifact.get("answer", "")).strip()
            summary = str(artifact.get("summary", "")).strip()
            if summary:
                claims.append(f"{agent_id}: {summary}")
            for item in list(artifact.get("evidence_summary", []))[:4]:
                evidence.append(f"{agent_id}: {item}")
            for item in list(artifact.get("unresolved_issues", []))[:4]:
                unresolved.append(f"{agent_id}: {item}")
            signature = answer_signature(answer)
            if signature:
                signatures.setdefault(signature, []).append(agent_id)
            confidence = float(artifact.get("confidence", 0.0) or 0.0)
            if answer and confidence >= best_conf:
                best_answer = self._trim_text(answer, max_chars=1200)
                best_conf = confidence
        if len(signatures) > 1:
            disagreements.append(
                "Agents still disagree across answer signatures: "
                + "; ".join(f"{','.join(ids)}" for ids in signatures.values())
            )
        return {
            "claims": claims,
            "evidence": evidence,
            "disagreements": disagreements,
            "unresolved_questions": unresolved,
            "best_current_answer": best_answer,
            "confidence": best_conf,
            "source": source,
        }

    def _bounded_transcript_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        def bounded_list(key: str, *, limit: int = 10, max_chars: int = 360) -> list[str]:
            values = summary.get(key, [])
            if not isinstance(values, list):
                values = [values] if values else []
            return [
                self._trim_text(str(item), max_chars=max_chars)
                for item in values
                if str(item).strip()
            ][:limit]

        try:
            confidence = float(summary.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        return {
            "claims": bounded_list("claims"),
            "evidence": bounded_list("evidence"),
            "disagreements": bounded_list("disagreements", limit=8),
            "unresolved_questions": bounded_list("unresolved_questions", limit=8),
            "best_current_answer": self._trim_text(
                str(summary.get("best_current_answer", "")), max_chars=1200
            ),
            "confidence": max(0.0, min(1.0, confidence)),
        }

    def _descriptor_monitor_node(self, state: WorkflowState) -> dict[str, Any]:
        descriptor = state.get("descriptor")
        dispatch_id = int(state.get("dispatch_id", -1))
        snapshot = {
            "task_id": state.get("task_id"),
            "topology": state.get("topology"),
            "phase": state.get("phase"),
            "round_index": int(state.get("round_index", 0)),
            "discussion_index": int(state.get("discussion_index", 0)),
            "dispatch_id": dispatch_id,
            "message_count": len(state.get("messages", [])),
            "artifact_count": len(state.get("artifacts", [])),
            "done": bool(state.get("done", False)),
        }

        record: dict[str, Any] = {}
        if descriptor is not None and hasattr(descriptor, "on_monitor"):
            emitted = descriptor.on_monitor(dict(snapshot))
            if isinstance(emitted, dict):
                record = dict(emitted)

        event = self._draft_event(
            dispatch_id=dispatch_id,
            node_order=4,
            agent_order=-1,
            event_order=0,
            actor="descriptor",
            event_type="verify",
            payload={"phase": snapshot["phase"], "record": record},
            token_in=0,
            token_out=0,
            latency_ms=1.0,
            cost_usd=0.0,
            state_id=f"run_{state['run_index']}_descriptor_{dispatch_id}",
        )
        if not record:
            record = {
                "phase": snapshot["phase"],
                "round_index": snapshot["round_index"],
                "dispatch_id": dispatch_id,
            }

        return {
            "phase": "descriptor_monitor",
            "descriptor_records": [record],
            "trace_payloads": [event],
            "phase_history": [self._phase_history_entry(state, "descriptor_monitor")],
        }

    def _finalize_node(self, state: WorkflowState) -> dict[str, Any]:
        final_answer = str(state.get("final_answer") or "").strip()
        final_reason = str(state.get("final_reason") or f"{state['topology']}:finalize")
        final_vote_source = str(state.get("final_vote_source", "") or "")
        selected_artifact_id = str(state.get("selected_artifact_id", "") or "")
        selected_agent_id = str(state.get("selected_agent_id", "") or "")
        selected_source_artifact_ids = list(state.get("selected_source_artifact_ids", []))

        if not final_answer:
            selection = self._finalize_selection_for_topology(state)
            if selection is not None:
                candidate_artifacts, vote_result = selection
                final_answer = self._safe_vote_answer_or_fallback(state, candidate_artifacts, vote_result)
                final_vote_source = str(
                    final_vote_source or vote_result.get("source", "topology_finalize")
                )
                selected_artifact_id = str(
                    selected_artifact_id or vote_result.get("selected_artifact_id", "")
                )
                selected_agent_id = str(selected_agent_id or vote_result.get("selected_agent_id", ""))
                if not selected_source_artifact_ids:
                    selected_source_artifact_ids = list(
                        vote_result.get("selected_source_artifact_ids", [])
                    )

        if not final_answer:
            final_answer = str(self._resolve_final_answer(state))

        descriptor_summary: dict[str, Any] = {}
        descriptor = state.get("descriptor")
        if descriptor is not None and hasattr(descriptor, "on_finalize"):
            emitted = descriptor.on_finalize(
                {
                    "task_id": state.get("task_id"),
                    "topology": state.get("topology"),
                    "phase": state.get("phase"),
                    "round_index": state.get("round_index"),
                    "final_answer": final_answer,
                    "message_count": len(state.get("messages", [])),
                    "view_count": len(state.get("message_views", [])),
                    "artifact_count": len(state.get("artifacts", [])),
                }
            )
            if isinstance(emitted, dict):
                descriptor_summary = emitted

        return {
            "phase": "finalize",
            "done": True,
            "final_answer": final_answer,
            "final_reason": final_reason,
            "final_vote_source": final_vote_source,
            "selected_artifact_id": selected_artifact_id,
            "selected_agent_id": selected_agent_id,
            "selected_source_artifact_ids": selected_source_artifact_ids,
            "descriptor_summary": descriptor_summary,
            "phase_history": [self._phase_history_entry(state, "finalize")],
            "trace_payloads": [
                self._draft_event(
                    dispatch_id=int(state.get("dispatch_id", -1)) + 1,
                    node_order=5,
                    agent_order=-1,
                    event_order=0,
                    actor="system",
                    event_type="finalize",
                    payload={
                        "status": "completed",
                        "final_answer": final_answer,
                        "final_reason": final_reason,
                    },
                    token_in=0,
                    token_out=max(1, len(final_answer.split())) if final_answer else 0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=f"run_{state['run_index']}_finalize",
                )
            ],
        }

    def _execute_agent_stage(
        self,
        state: WorkflowState,
        *,
        agent_id: str,
        node_name: str,
        stage_role: str,
        directive: str,
        visible_messages: list[RelayPacket],
    ) -> dict[str, Any]:
        layout = state["layout"]
        role = layout.roles.get(agent_id, "agent")
        dispatch_id = int(state.get("dispatch_id", -1))
        round_index = int(state.get("round_index", 0))
        discussion_index = int(state.get("discussion_index", 0))
        prior_artifact = self._latest_agent_artifact(state, agent_id)
        persona_info = state.get("domain_personas", {}).get(agent_id, {})

        prompt = self._build_agent_prompt(
            state=state,
            task_prompt=state.get("task_prompt"),
            agent_id=agent_id,
            role=role,
            stage_role=stage_role,
            directive=directive,
            round_index=round_index,
            discussion_index=discussion_index,
            visible_messages=visible_messages,
            prior_artifact=prior_artifact,
            persona_info=persona_info,
        )
        prompt_messages = self._normalize_prompt_messages(prompt)
        agent_type = state["agent_type_by_agent"].get(agent_id, "general")
        tools = list(state.get("tools", []))
        opening_critic_round = (
            stage_role == "critic" and not visible_messages and prior_artifact is None
        )

        t0 = time.perf_counter()
        llm = state["llm_client"].generate(
            prompt=prompt,
            agent_type=agent_type,
            task_id=state["task_id"],
            run_index=int(state.get("run_index", 0)),
            agent_id=agent_id,
            tools=tools,
            max_tool_iterations=int(state.get("max_tool_iterations", 8)),
            temperature=0.0,
        )
        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1.0)
        self._raise_for_fatal_tool_failure(llm, agent_id=agent_id, node_name=node_name)
        tool_retry_used = False
        tool_records = list(llm.tool_calls)
        answer_mode = self._answer_mode(str(llm.text or ""))
        if (
            tools
            and self._is_answer_producing_stage(stage_role)
            and not bool(llm.mock_used)
            and answer_mode in {"blocked", "plan", "empty"}
            and (not tool_records or opening_critic_round)
        ):
            retry_prompt = list(prompt_messages)
            if tool_records:
                prior_tool_attempts = ", ".join(
                    self._trim_text(
                        f"{record.get('tool_name', '')}({json.dumps(record.get('arguments', {}), ensure_ascii=False)})",
                        max_chars=180,
                    )
                    for record in tool_records[:3]
                )
                retry_content = (
                    "Your current output is still blocked after limited tool use. "
                    "Reformulate the investigation and make another evidence-gathering attempt before returning blocked. "
                    f"Previous tool attempts: {prior_tool_attempts or 'none recorded'}."
                )
            else:
                retry_content = (
                    "Tools are available for this stage. If evidence is missing, call the relevant tool now "
                    "instead of returning a blocked or planning answer. Return the required JSON only after "
                    "using tools or after deciding the visible evidence is already sufficient. If a required "
                    "identifier is absent, use available list, catalog, search, schedule, or related discovery "
                    "tools to resolve it before asking the user for information."
                )
            retry_prompt.append(
                {
                    "role": "user",
                    "content": retry_content,
                }
            )
            t0 = time.perf_counter()
            llm = state["llm_client"].generate(
                prompt=retry_prompt,
                agent_type=agent_type,
                task_id=state["task_id"],
                run_index=int(state.get("run_index", 0)),
                agent_id=agent_id,
                tools=tools,
                max_tool_iterations=int(state.get("max_tool_iterations", 8)),
                temperature=0.0,
            )
            latency_ms += max((time.perf_counter() - t0) * 1000.0, 1.0)
            self._raise_for_fatal_tool_failure(llm, agent_id=agent_id, node_name=node_name)
            prompt_messages = self._normalize_prompt_messages(retry_prompt)
            tool_records = list(llm.tool_calls)
            tool_retry_used = True

        source_artifact_ids = [
            str(message.get("artifact_id", "")) for message in visible_messages if message.get("artifact_id")
        ]
        if prior_artifact is not None and prior_artifact.get("artifact_id"):
            source_artifact_ids.append(str(prior_artifact.get("artifact_id")))

        artifact = build_artifact(
            text=str(llm.text or ""),
            artifact_id=f"{node_name}:{agent_id}:{round_index}:{discussion_index}:{dispatch_id}",
            dispatch_id=dispatch_id,
            node_name=node_name,
            stage_role=stage_role,
            round_index=round_index,
            discussion_index=discussion_index,
            agent_id=agent_id,
            role=role,
            source_artifact_ids=source_artifact_ids,
            tool_records=tool_records,
            llm_payload={
                "model": llm.model,
                "mock_used": bool(llm.mock_used),
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "metadata": self._serialize_for_json(dict(llm.metadata)),
            },
        )

        tool_log_records: list[dict[str, Any]] = []
        for record in tool_records:
            tool_log_records.append(
                {
                    "agent_id": agent_id,
                    "node_name": node_name,
                    "round_index": round_index,
                    "discussion_index": discussion_index,
                    "tool_name": str(record.get("tool_name", "")),
                    "arguments": self._serialize_for_json(record.get("arguments", {})),
                    "status": str(record.get("status", "")),
                    "error": self._serialize_for_json(record.get("error")),
                    "output": self._serialize_for_json(record.get("output")),
                }
            )

        event_payloads = [
            self._draft_event(
                dispatch_id=dispatch_id,
                node_order=1,
                agent_order=self._agent_order(layout, agent_id),
                event_order=0,
                actor=agent_id,
                event_type="verify",
                payload={
                    "node": node_name,
                    "visible_message_ids": [item.get("message_id") for item in visible_messages],
                    "visible_message_count": len(visible_messages),
                    "prompt_messages": prompt_messages,
                },
                token_in=0,
                token_out=0,
                latency_ms=1.0,
                cost_usd=0.0,
                state_id=f"run_{state['run_index']}_{node_name}_{agent_id}_view",
            )
        ]

        event_order = 1
        for record in tool_records:
            tool_name = str(record.get("tool_name") or "")
            if not tool_name:
                continue
            event_payloads.append(
                self._draft_event(
                    dispatch_id=dispatch_id,
                    node_order=1,
                    agent_order=self._agent_order(layout, agent_id),
                    event_order=event_order,
                    actor=agent_id,
                    event_type="tool_call",
                    payload={
                        "node": node_name,
                        "tool_name": tool_name,
                        "arguments": self._serialize_for_json(record.get("arguments", {})),
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=f"run_{state['run_index']}_{node_name}_{agent_id}_tool_{event_order}",
                )
            )
            event_order += 1
            event_payloads.append(
                self._draft_event(
                    dispatch_id=dispatch_id,
                    node_order=1,
                    agent_order=self._agent_order(layout, agent_id),
                    event_order=event_order,
                    actor="system",
                    event_type="tool_result",
                    payload={
                        "node": node_name,
                        "tool_name": tool_name,
                        "status": str(record.get("status", "")),
                        "output_preview": self._tool_output_preview(record.get("output")),
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=f"run_{state['run_index']}_{node_name}_{agent_id}_tool_{event_order}",
                )
            )
            event_order += 1

        event_payloads.append(
            self._draft_event(
                dispatch_id=dispatch_id,
                node_order=1,
                agent_order=self._agent_order(layout, agent_id),
                event_order=event_order,
                actor=agent_id,
                event_type="act",
                payload={
                    "node": node_name,
                    "stage_role": stage_role,
                    "answer_mode": self._artifact_answer_mode(artifact),
                    "response_preview": self._message_snippet(str(llm.text or "")),
                    "artifact_preview": self._message_snippet(str(artifact.get("summary", ""))),
                },
                token_in=int(llm.token_in),
                token_out=int(llm.token_out),
                latency_ms=latency_ms,
                cost_usd=float(llm.cost_usd),
                state_id=f"run_{state['run_index']}_{node_name}_{agent_id}_act",
            )
        )

        interaction_log = {
            "dispatch_id": dispatch_id,
            "agent_id": agent_id,
            "agent_role": role,
            "domain_role": persona_info.get("role_name", "") if persona_info else "",
            "domain_persona": persona_info.get("persona", "") if persona_info else "",
            "stage_role": stage_role,
            "agent_type": agent_type,
            "phase": node_name,
            "round_index": round_index,
            "discussion_index": discussion_index,
            "tool_use_retry": tool_retry_used,
            "prompt_messages": prompt_messages,
            "visible_messages": self._serialize_for_json(visible_messages),
            "prior_artifact": self._serialize_for_json(prior_artifact),
            "assistant_message": {"role": "assistant", "content": str(llm.text or "")},
            "structured_artifact": self._serialize_for_json(artifact),
            "tool_calls": self._serialize_tool_records(tool_records),
            "llm": {
                "model": llm.model,
                "mock_used": bool(llm.mock_used),
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "metadata": self._serialize_for_json(dict(llm.metadata)),
            },
        }

        return {
            "artifacts": [artifact],
            "message_views": [
                {
                    "dispatch_id": dispatch_id,
                    "viewer": agent_id,
                    "role": role,
                    "phase": node_name,
                    "round_index": round_index,
                    "discussion_index": discussion_index,
                    "visible_message_ids": [str(item.get("message_id", "")) for item in visible_messages],
                    "visible_senders": sorted({str(item.get("sender", "")) for item in visible_messages}),
                    "visible_count": len(visible_messages),
                }
            ],
            "trace_payloads": event_payloads,
            "interaction_logs": [interaction_log],
            "tool_records_log": tool_log_records,
        }

    @staticmethod
    def _raise_for_fatal_tool_failure(
        llm: Any,
        *,
        agent_id: str,
        node_name: str,
    ) -> None:
        metadata = dict(getattr(llm, "metadata", {}) or {})
        if not bool(metadata.get("fatal_tool_failure", False)):
            return
        tool_name = str(metadata.get("tool_failure_tool_name", ""))
        consecutive = int(metadata.get("tool_failure_consecutive_count", 0) or 0)
        raise RuntimeError(
            "fatal_tool_failure: "
            f"tool_failure_circuit_breaker triggered in node={node_name} "
            f"agent_id={agent_id} tool={tool_name} consecutive={consecutive}"
        )

    def _build_orchestrator_task_packets(
        self,
        state: WorkflowState,
        *,
        recipients: list[str],
    ) -> list[dict[str, Any]]:
        artifact = self._latest_artifact_for_nodes(
            state,
            node_names={"orchestrator_merge", "orchestrator_plan"},
        )
        if artifact is None:
            return []
        kind = "task_package" if int(state.get("round_index", 0)) == 0 else "orchestrator_feedback"
        packets: list[dict[str, Any]] = []
        for recipient in recipients:
            payload = self._build_specialist_task_payload(
                state,
                artifact=artifact,
                recipient=recipient,
                recipients=recipients,
            )
            packets.append(
                {
                    "sender": state["layout"].orchestrator_id,
                    "recipients": [recipient],
                    "kind": kind,
                    "round": int(state.get("round_index", 0)),
                    "discussion_index": 0,
                    "artifact_id": artifact.get("artifact_id"),
                    "payload": payload,
                    "content": self._packet_content(state, payload),
                }
            )
        return packets

    @staticmethod
    def _parse_structured_packet_source(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        raw = str(value or "").strip()
        if not raw or raw[0] != "{" or raw[-1] != "}":
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _recipient_focus(role_name: str) -> tuple[str, tuple[str, ...]]:
        lowered = role_name.lower().strip()
        if "crm" in lowered or "customer relationship" in lowered:
            return (
                "Focus on CRM and directory evidence needed to identify the assigned owner.",
                ("customer_relationship_manager", "company_directory", "crm"),
            )
        if "calendar" in lowered:
            return (
                "Focus on time constraints, availability, and scheduling actions.",
                ("calendar",),
            )
        if "email" in lowered or "communication" in lowered:
            return (
                "Focus on contact identity, outreach implications, and communication-side ambiguities.",
                ("email", "company_directory"),
            )
        if "project" in lowered:
            return (
                "Focus on project-management ownership and task linkage evidence.",
                ("project_management",),
            )
        if "analytics" in lowered:
            return (
                "Focus on analytics evidence and quantitative validation.",
                ("analytics",),
            )
        if "verifier" in lowered or "verify" in lowered:
            return (
                "Focus on validating whether the gathered evidence is sufficient to complete the task.",
                tuple(),
            )
        if "integrator" in lowered or "planner" in lowered:
            return (
                "Focus on cross-system dependencies and missing links between tool outputs.",
                tuple(),
            )
        return ("Focus on the most relevant unresolved subproblem for your specialist role.", tuple())

    def _select_task_package_steps(
        self,
        *,
        steps: list[dict[str, Any]],
        role_name: str,
        recipient: str,
        recipients: list[str],
    ) -> list[dict[str, Any]]:
        if not steps:
            return []
        _, tool_prefixes = self._recipient_focus(role_name)
        selected: list[dict[str, Any]] = []
        for step in steps:
            tool_name = str(step.get("tool", "")).lower()
            if tool_prefixes and any(
                tool_name.startswith(f"{prefix}.") or prefix in tool_name for prefix in tool_prefixes
            ):
                selected.append(step)
        if selected:
            return selected
        if not recipients:
            return steps[:]
        try:
            recipient_index = recipients.index(recipient)
        except ValueError:
            recipient_index = 0
        partitioned = [step for idx, step in enumerate(steps) if idx % len(recipients) == recipient_index]
        return partitioned or steps[:1]

    def _build_specialist_task_payload(
        self,
        state: WorkflowState,
        *,
        artifact: ArtifactRecord,
        recipient: str,
        recipients: list[str],
    ) -> dict[str, Any]:
        payload = self._packet_payload_from_artifact(state, artifact)
        persona_info = dict(state.get("domain_personas", {}).get(recipient, {}))
        role_name = str(persona_info.get("role_name", "")).strip()
        focus_text, _ = self._recipient_focus(role_name)
        structured_answer = self._parse_structured_packet_source(artifact.get("answer"))
        raw_steps = structured_answer.get("plan", []) if isinstance(structured_answer, dict) else []
        steps = [step for step in raw_steps if isinstance(step, dict)]
        selected_steps = self._select_task_package_steps(
            steps=steps,
            role_name=role_name,
            recipient=recipient,
            recipients=recipients,
        )
        objective = str(payload.get("summary", "")).strip()
        if not objective:
            objective = "Review the orchestrator plan and execute your assigned subtask."
        if role_name:
            payload["summary"] = (
                f"{objective} Specialist focus for {recipient} ({role_name}): {focus_text}"
            )
            payload["domain_role"] = role_name
        payload["task_package"] = {
            "recipient": recipient,
            "recipient_domain_role": role_name,
            "objective": objective,
            "focus": focus_text,
            "suggested_steps": selected_steps[:4],
            "total_plan_steps": len(steps),
        }
        return payload

    def _build_root_task_packets(
        self,
        state: WorkflowState,
        *,
        recipients: list[str],
    ) -> list[dict[str, Any]]:
        artifact = self._latest_artifact_for_nodes(state, node_names={"root_reducer", "root_plan"})
        if artifact is None:
            return []
        payload = self._packet_payload_from_artifact(state, artifact)
        return [
            {
                "sender": state["layout"].orchestrator_id,
                "recipients": [recipient],
                "kind": "root_task_package",
                "round": int(state.get("round_index", 0)),
                "discussion_index": 0,
                "artifact_id": artifact.get("artifact_id"),
                "payload": payload,
                "content": self._packet_content(state, payload),
            }
            for recipient in recipients
        ]

    def _build_manager_task_packets(
        self,
        state: WorkflowState,
        *,
        recipients: list[str],
    ) -> list[dict[str, Any]]:
        artifacts = self._artifacts_for_nodes(
            state,
            node_names={"manager_nodes"},
            round_index=int(state.get("round_index", 0)),
            latest_only=True,
        )
        specs: list[dict[str, Any]] = []
        recipients_set = set(recipients)
        for artifact in artifacts:
            manager_id = str(artifact.get("agent_id", ""))
            payload = self._packet_payload_from_artifact(state, artifact)
            for child in state["layout"].children_by_agent.get(manager_id, []):
                if child not in recipients_set:
                    continue
                specs.append(
                    {
                        "sender": manager_id,
                        "recipients": [child],
                        "kind": "manager_task_package",
                        "round": int(state.get("round_index", 0)),
                        "discussion_index": 0,
                        "artifact_id": artifact.get("artifact_id"),
                        "payload": payload,
                        "content": self._packet_content(state, payload),
                    }
                )
        return specs

    def _build_report_packets(
        self,
        state: WorkflowState,
        *,
        source_node_names: set[str],
        recipients_by_sender: Callable[[ArtifactRecord], list[str]],
        kind: str,
    ) -> list[dict[str, Any]]:
        artifacts = self._artifacts_for_nodes(
            state,
            node_names=source_node_names,
            round_index=int(state.get("round_index", 0)),
            discussion_index=int(state.get("discussion_index", 0)),
            latest_only=True,
        )
        specs: list[dict[str, Any]] = []
        for artifact in artifacts:
            payload = self._packet_payload_from_artifact(state, artifact)
            specs.append(
                {
                    "sender": str(artifact.get("agent_id", "")),
                    "recipients": recipients_by_sender(artifact),
                    "kind": kind,
                    "round": int(state.get("round_index", 0)),
                    "discussion_index": int(state.get("discussion_index", 0)),
                    "artifact_id": artifact.get("artifact_id"),
                    "payload": payload,
                    "content": self._packet_content(state, payload),
                }
            )
        return specs

    def _build_orchestrator_peer_packets(
        self,
        state: WorkflowState,
        artifacts: list[ArtifactRecord],
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        by_agent = {str(artifact.get("agent_id", "")): artifact for artifact in artifacts}
        for recipient in state["layout"].specialists:
            peer_payloads = []
            peer_ids = []
            for agent_id, artifact in by_agent.items():
                if agent_id == recipient:
                    continue
                payload = self._packet_payload_from_artifact(state, artifact)
                payload["sender"] = agent_id
                peer_payloads.append(payload)
                peer_ids.append(str(artifact.get("artifact_id", "")))
            if not peer_payloads:
                continue
            payload = {
                "summary": self._bundle_peer_summaries(peer_payloads),
                "peer_artifacts": peer_payloads[:6],
            }
            specs.append(
                {
                    "sender": state["layout"].orchestrator_id,
                    "recipients": [recipient],
                    "kind": "peer_summary",
                    "round": int(state.get("round_index", 0)),
                    "discussion_index": int(state.get("discussion_index", 0)) + 1,
                    "artifact_id": ",".join(peer_ids[:6]),
                    "payload": payload,
                    "content": self._packet_content(state, payload),
                }
            )
        return specs

    def _build_peer_summary_packets(
        self,
        state: WorkflowState,
        *,
        artifacts: list[ArtifactRecord],
        recipients_resolver: Callable[[ArtifactRecord], list[str]],
        kind: str,
        next_round: int,
        next_discussion: int,
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for artifact in artifacts:
            payload = self._packet_payload_from_artifact(state, artifact)
            specs.append(
                {
                    "sender": str(artifact.get("agent_id", "")),
                    "recipients": recipients_resolver(artifact),
                    "kind": kind,
                    "round": next_round,
                    "discussion_index": next_discussion,
                    "artifact_id": artifact.get("artifact_id"),
                    "payload": payload,
                    "content": self._packet_content(state, payload),
                }
            )
        return specs

    def _build_group_peer_packets(
        self,
        state: WorkflowState,
        *,
        artifacts: list[ArtifactRecord],
        next_round: int,
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for artifact in artifacts:
            agent_id = str(artifact.get("agent_id", ""))
            group = self._group_for_agent(state["layout"], agent_id)
            recipients = [member for member in group if member != agent_id]
            payload = self._packet_payload_from_artifact(state, artifact)
            specs.append(
                {
                    "sender": agent_id,
                    "recipients": recipients,
                    "kind": "group_debate_round",
                    "round": next_round,
                    "discussion_index": 0,
                    "artifact_id": artifact.get("artifact_id"),
                    "payload": payload,
                    "content": self._packet_content(state, payload),
                }
            )
        return specs

    def _build_group_summary_packets(
        self,
        state: WorkflowState,
        artifacts: list[ArtifactRecord],
    ) -> list[dict[str, Any]]:
        by_agent = {str(artifact.get("agent_id", "")): artifact for artifact in artifacts}
        specs: list[dict[str, Any]] = []
        for group in state["layout"].groups:
            if not group:
                continue
            representative = group[0]
            group_payloads = []
            source_ids = []
            for member in group:
                artifact = by_agent.get(member)
                if artifact is None:
                    continue
                payload = self._packet_payload_from_artifact(state, artifact)
                payload["sender"] = member
                group_payloads.append(payload)
                source_ids.append(str(artifact.get("artifact_id", "")))
            payload = {
                "summary": self._bundle_peer_summaries(group_payloads),
                "group_members": list(group),
                "group_artifacts": group_payloads[:6],
            }
            specs.append(
                {
                    "sender": "system",
                    "recipients": [representative],
                    "kind": "group_summary",
                    "round": int(state.get("round_index", 0)),
                    "discussion_index": 0,
                    "artifact_id": ",".join(source_ids[:6]),
                    "payload": payload,
                    "content": self._packet_content(state, payload),
                }
            )
        return specs

    def _emit_packets(
        self,
        state: WorkflowState,
        *,
        dispatch_id: int,
        phase: str,
        packet_specs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        message_seq = int(state.get("message_seq", 0))
        message_budget = dict(state.get("message_budget", {}))
        sent_counts = dict(state.get("sent_counts", {}))
        budget_sent_counts = dict(state.get("budget_sent_counts", {}))
        messages: list[RelayPacket] = []
        trace_payloads: list[dict[str, Any]] = []

        for spec in packet_specs:
            sender = str(spec.get("sender", "system") or "system")
            kind = str(spec.get("kind", ""))
            recipients = sorted({item for item in spec.get("recipients", []) if item and item != sender})
            if not recipients:
                continue

            consumes_budget = kind not in CONTROL_PACKET_KINDS
            if sender in sent_counts:
                sent_counts[sender] = int(sent_counts.get(sender, 0)) + 1
            if sender in message_budget and consumes_budget:
                remaining = int(message_budget.get(sender, 0))
                if remaining <= 0:
                    sent_counts[sender] = int(sent_counts.get(sender, 0)) - 1
                    continue
                message_budget[sender] = remaining - 1
                budget_sent_counts[sender] = int(budget_sent_counts.get(sender, 0)) + 1

            message_seq += 1
            payload = self._serialize_for_json(spec.get("payload", {}))
            message = RelayPacket(
                message_id=f"m_{message_seq}",
                dispatch_id=dispatch_id,
                sender=sender,
                recipients=recipients,
                kind=kind,
                phase=phase,
                round=int(spec.get("round", state.get("round_index", 0))),
                discussion_index=int(
                    spec.get("discussion_index", state.get("discussion_index", 0))
                ),
                artifact_id=spec.get("artifact_id"),
                content=str(spec.get("content") or self._packet_content(state, payload)),
                payload=payload,
            )
            messages.append(message)
            trace_payloads.append(
                self._draft_event(
                    dispatch_id=dispatch_id,
                    node_order=2,
                    agent_order=self._agent_order(state["layout"], sender),
                    event_order=len(trace_payloads) + 1,
                    actor=sender,
                    event_type="tool_call",
                    payload={
                        "tool_name": "inter_agent_send",
                        "to": recipients,
                        "kind": message["kind"],
                        "phase": phase,
                        "round_index": message["round"],
                        "discussion_index": message["discussion_index"],
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=f"run_{state['run_index']}_{phase}_send_{message_seq}",
                )
            )
            trace_payloads.append(
                self._draft_event(
                    dispatch_id=dispatch_id,
                    node_order=2,
                    agent_order=self._agent_order(state["layout"], sender),
                    event_order=len(trace_payloads) + 1,
                    actor="system",
                    event_type="tool_result",
                    payload={
                        "tool_name": "inter_agent_send",
                        "status": "ok",
                        "message_id": message["message_id"],
                        "visible_to": recipients,
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=f"run_{state['run_index']}_{phase}_send_{message_seq}_result",
                )
            )

        return {
            "messages": messages,
            "message_budget": message_budget,
            "sent_counts": sent_counts,
            "budget_sent_counts": budget_sent_counts,
            "message_seq": message_seq,
            "trace_payloads": trace_payloads,
        }

    def _termination_decision(
        self,
        state: WorkflowState,
        *,
        stage_name: str,
        round_index: int,
        discussion_index: int,
        minimum_round_index: int | None = None,
        minimum_required_rounds: int | None = None,
        minimum_round_label: str = "round",
        candidate_artifacts: list[ArtifactRecord],
        previous_candidate_artifacts: list[ArtifactRecord],
        consensus_artifacts: list[ArtifactRecord],
        expected_count: int,
        max_reached: bool,
        continue_next_step: str,
        stop_next_step: str,
    ) -> TerminationDecision:
        """Compute the explicit code-level stop/continue decision for a looped collaboration stage."""

        available_count = len(
            [
                artifact
                for artifact in consensus_artifacts or candidate_artifacts
                if self._artifact_display_answer(artifact)
            ]
        )
        base_decision: TerminationDecision = {
            "stage_name": stage_name,
            "round_index": round_index,
            "discussion_index": discussion_index,
            "consensus_mode": str(state.get("termination_consensus_mode", "llm_judge")),
            "consensus_source": "uncomputed",
            "consensus_signature": "",
            "consensus_count": 0,
            "consensus_valid_count": 0,
            "consensus_groups": [],
            "consensus_explanation": "",
            "consensus_is_substantive": None,
            "consensus_gate_blocked": False,
            "consensus_gate_reason": "",
            "consensus_ratio": 0.0,
            "progress_source": "uncomputed",
            "progress_status": None,
            "expected_improvement": None,
            "progress_explanation": "",
            "average_confidence": 0.0,
            "mean_delta": None,
            "valid_artifact_count": available_count,
            "control_token_in": 0,
            "control_token_out": 0,
            "control_cost_usd": 0.0,
            "control_latency_ms": 1.0,
        }
        if expected_count > 0 and available_count < max(1, math.ceil(expected_count / 2)):
            return {
                **base_decision,
                "should_stop": True,
                "next_step": stop_next_step,
                "reason": "invalid_or_failed_branch",
                "reason_detail": (
                    f"Only {available_count} branch artifacts were available out of {expected_count} expected branches."
                ),
            }

        assessment = self._compute_termination_assessment(
            state=state,
            stage_name=stage_name,
            round_index=round_index,
            discussion_index=discussion_index,
            candidate_artifacts=candidate_artifacts,
            previous_candidate_artifacts=previous_candidate_artifacts,
            consensus_artifacts=consensus_artifacts or candidate_artifacts,
        )
        avg_conf = average_confidence(candidate_artifacts or consensus_artifacts)
        mean_delta = compute_mean_delta(previous_candidate_artifacts, candidate_artifacts)

        assessment_source = str(assessment.get("source", "lexical"))
        consensus_is_substantive = assessment.get("is_substantive")
        consensus_supported = assessment_source != "llm_judge" or consensus_is_substantive is True

        # Consensus stopping-gate: high lexical/semantic agreement only stops the
        # loop when it is decision-grade. Mirrors the Trace Auditor's
        # premature_consensus predicate so the controller and auditor never
        # disagree; the consensus_ratio metric itself is left untouched.
        consensus_pool = consensus_artifacts or candidate_artifacts
        consensus_quorum = (
            int(assessment.get("valid_count", 0)) > 1
            and float(assessment.get("ratio", 0.0)) >= 0.75
            and consensus_supported
        )
        consensus_decision_grade = self._consensus_is_decision_grade(
            consensus_pool, repair_available=not max_reached
        )
        consensus_gate_blocked = bool(consensus_quorum and not consensus_decision_grade)
        consensus_gate_reason = (
            self._consensus_gate_reason(consensus_pool, float(assessment.get("ratio", 0.0)))
            if consensus_gate_blocked
            else ""
        )

        decision_common: TerminationDecision = {
            **base_decision,
            "consensus_mode": str(assessment.get("mode", base_decision["consensus_mode"])),
            "consensus_source": str(assessment.get("source", "lexical")),
            "consensus_signature": str(assessment.get("signature", "")),
            "consensus_count": int(assessment.get("count", 0)),
            "consensus_valid_count": int(assessment.get("valid_count", 0)),
            "consensus_groups": [
                [int(index) for index in group]
                for group in assessment.get("groups", [])
                if isinstance(group, list)
            ],
            "consensus_explanation": str(assessment.get("explanation", "")),
            "consensus_is_substantive": assessment.get("is_substantive"),
            "consensus_gate_blocked": consensus_gate_blocked,
            "consensus_gate_reason": consensus_gate_reason,
            "consensus_ratio": float(assessment.get("ratio", 0.0)),
            "progress_source": str(assessment.get("progress_source", assessment.get("source", "lexical"))),
            "progress_status": assessment.get("progress_status"),
            "expected_improvement": assessment.get("expected_improvement"),
            "progress_explanation": str(
                assessment.get("progress_explanation", assessment.get("explanation", ""))
            ),
            "average_confidence": avg_conf,
            "mean_delta": mean_delta,
            "valid_artifact_count": available_count,
            "control_token_in": int(assessment.get("token_in", 0)),
            "control_token_out": int(assessment.get("token_out", 0)),
            "control_cost_usd": float(assessment.get("cost_usd", 0.0)),
            "control_latency_ms": float(assessment.get("latency_ms", 1.0)),
        }

        min_rounds = (
            int(state.get("minimum_discussion_rounds", 1))
            if minimum_required_rounds is None
            else int(minimum_required_rounds)
        )
        effective_minimum_round_index = (
            round_index if minimum_round_index is None else int(minimum_round_index)
        )
        if effective_minimum_round_index < min_rounds:
            return {
                **decision_common,
                "should_stop": False,
                "next_step": continue_next_step,
                "reason": "continue",
                "reason_detail": (
                    f"Minimum {min_rounds} discussion round(s) required; "
                    f"currently at {minimum_round_label} {effective_minimum_round_index}."
                ),
            }

        if consensus_quorum and consensus_decision_grade:
            return {
                **decision_common,
                "should_stop": True,
                "next_step": stop_next_step,
                "reason": "consensus_reached",
                "reason_detail": (
                    f"Consensus ratio {float(assessment['ratio']):.2f} met the 0.75 threshold."
                ),
            }

        if mean_delta is not None:
            if assessment_source == "llm_judge":
                if bool(assessment.get("should_stop_for_no_progress", False)):
                    return {
                        **decision_common,
                        "should_stop": True,
                        "next_step": stop_next_step,
                        "reason": "no_meaningful_change",
                        "reason_detail": (
                            str(
                                assessment.get("progress_explanation")
                                or assessment.get("explanation")
                                or "The termination judge determined that another round was unlikely to materially improve correctness."
                            )
                        ),
                    }
            elif mean_delta <= 0.05:
                return {
                    **decision_common,
                    "should_stop": True,
                    "next_step": stop_next_step,
                    "reason": "no_meaningful_change",
                    "reason_detail": f"Mean artifact delta {mean_delta:.3f} stayed below 0.05.",
                }

        if max_reached:
            return {
                **decision_common,
                "should_stop": True,
                "next_step": stop_next_step,
                "reason": "max_rounds_reached",
                "reason_detail": "The configured maximum collaboration rounds were exhausted.",
            }

        return {
            **decision_common,
            "should_stop": False,
            "next_step": continue_next_step,
            "reason": "continue",
            "reason_detail": "No explicit stop condition fired; proceed to the next routed stage.",
        }

    @staticmethod
    def _consensus_is_decision_grade(
        artifacts: list[ArtifactRecord], *, repair_available: bool
    ) -> bool:
        """Whether high answer agreement should actually stop the loop.

        Mirrors the Trace Auditor's ``premature_consensus`` predicate: agreement
        is *not* decision-grade when average confidence is below 0.5 or unresolved
        issues remain — but only while a further step (another round or the single
        repair) is still available. With no step left, agreement always stops the
        loop so a weak-but-stalled turn cannot spin forever.
        """

        if not repair_available:
            return True
        weak = average_confidence(artifacts) < 0.5 or any(
            artifact.get("unresolved_issues") for artifact in artifacts
        )
        return not weak

    @staticmethod
    def _consensus_gate_reason(artifacts: list[ArtifactRecord], ratio: float) -> str:
        unresolved = any(artifact.get("unresolved_issues") for artifact in artifacts)
        return (
            f"Agreement ratio {ratio:.2f} held but consensus is not decision-grade "
            f"(avg confidence {average_confidence(artifacts):.2f}; "
            f"{'open' if unresolved else 'no'} unresolved issues) while a repair/round remains."
        )

    def _compute_termination_assessment(
        self,
        *,
        state: WorkflowState,
        stage_name: str,
        round_index: int,
        discussion_index: int,
        candidate_artifacts: list[ArtifactRecord],
        previous_candidate_artifacts: list[ArtifactRecord],
        consensus_artifacts: list[ArtifactRecord],
    ) -> dict[str, Any]:
        mode = str(state.get("termination_consensus_mode", "llm_judge") or "llm_judge")
        lexical = compute_consensus(consensus_artifacts)
        lexical_result = {
            "mode": mode,
            "source": "lexical",
            "ratio": float(lexical.get("ratio", 0.0)),
            "signature": str(lexical.get("signature", "")),
            "count": int(lexical.get("count", 0)),
            "valid_count": int(lexical.get("valid_count", 0)),
            "groups": [],
            "explanation": "",
            "is_substantive": None,
            "progress_source": "lexical",
            "progress_status": "unclear",
            "expected_improvement": None,
            "progress_explanation": "",
            "should_stop_for_no_progress": False,
            "token_in": 0,
            "token_out": 0,
            "cost_usd": 0.0,
            "latency_ms": 1.0,
        }
        if mode != "llm_judge":
            return lexical_result

        consensus_candidates = self._serialize_termination_assessment_artifacts(
            consensus_artifacts,
            previous_by_agent={},
        )
        if len(consensus_candidates) <= 1:
            return {
                **lexical_result,
                "source": "lexical_singleton",
                "progress_source": "lexical_singleton",
                "groups": [[item["index"]] for item in consensus_candidates],
            }

        previous_by_agent = {
            str(artifact.get("agent_id", "")): artifact
            for artifact in previous_candidate_artifacts
            if str(artifact.get("agent_id", "")).strip()
        }
        current_candidates = self._serialize_termination_assessment_artifacts(
            candidate_artifacts,
            previous_by_agent=previous_by_agent,
        )
        prompt = self._build_termination_assessment_prompt(
            state=state,
            stage_name=stage_name,
            round_index=round_index,
            discussion_index=discussion_index,
            current_candidates=current_candidates,
            consensus_candidates=consensus_candidates,
        )

        t0 = time.perf_counter()
        llm = state["llm_client"].generate(
            prompt=prompt,
            agent_type="judge",
            task_id=str(state.get("task_id", "")),
            run_index=int(state.get("run_index", 0)),
            agent_id=f"{stage_name}_termination_judge",
            tools=None,
            max_tool_iterations=1,
            temperature=0.0,
        )
        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1.0)
        if bool(llm.mock_used):
            return {
                **lexical_result,
                "source": "lexical_fallback_mock",
                "progress_source": "lexical_fallback_mock",
                "explanation": "Termination judge fell back to lexical consensus because the LLM client ran in mock mode.",
                "progress_explanation": "Semantic progress judgment was unavailable because the LLM client ran in mock mode.",
            }

        parsed = self._parse_termination_assessment_judgment(str(llm.text or ""))
        if parsed is None:
            return {
                **lexical_result,
                "source": "lexical_fallback_parse_error",
                "progress_source": "lexical_fallback_parse_error",
                "explanation": "Termination judge returned invalid JSON; lexical consensus was used instead.",
                "progress_explanation": "Semantic progress judgment was unavailable because the termination judge returned invalid JSON.",
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "latency_ms": latency_ms,
            }

        valid_indices = {int(item["index"]) for item in consensus_candidates}
        invalid_indices = {index for index in parsed["invalid_indices"] if index in valid_indices}
        remaining_valid = valid_indices - invalid_indices

        groups: list[list[int]] = []
        seen: set[int] = set()
        for group in parsed["groups"]:
            cleaned = sorted({index for index in group if index in remaining_valid and index not in seen})
            if not cleaned:
                continue
            groups.append(cleaned)
            seen.update(cleaned)
        for index in sorted(remaining_valid - seen):
            groups.append([index])

        if not remaining_valid or not groups:
            return {
                **lexical_result,
                "source": "lexical_fallback_empty_judgment",
                "progress_source": "lexical_fallback_empty_judgment",
                "explanation": "Termination judge did not return any usable valid groups; lexical consensus was used instead.",
                "progress_explanation": "Semantic progress judgment was unavailable because the termination judge did not return usable valid groups.",
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "latency_ms": latency_ms,
            }

        winner_group = max(groups, key=lambda group: (len(group), [-item for item in group]))
        return {
            "mode": mode,
            "source": "llm_judge",
            "ratio": len(winner_group) / len(remaining_valid),
            "signature": ",".join(str(index) for index in winner_group),
            "count": len(winner_group),
            "valid_count": len(remaining_valid),
            "groups": groups,
            "explanation": str(parsed.get("explanation", "")),
            "is_substantive": parsed.get("is_substantive"),
            "progress_source": "llm_judge",
            "progress_status": parsed.get("progress_status"),
            "expected_improvement": parsed.get("expected_improvement"),
            "progress_explanation": str(parsed.get("explanation", "")),
            "should_stop_for_no_progress": bool(parsed.get("should_stop_for_no_progress", False)),
            "token_in": int(llm.token_in),
            "token_out": int(llm.token_out),
            "cost_usd": float(llm.cost_usd),
            "latency_ms": latency_ms,
        }

    def _compute_termination_consensus(
        self,
        *,
        state: WorkflowState,
        stage_name: str,
        round_index: int,
        discussion_index: int,
        artifacts: list[ArtifactRecord],
    ) -> dict[str, Any]:
        return self._compute_termination_assessment(
            state=state,
            stage_name=stage_name,
            round_index=round_index,
            discussion_index=discussion_index,
            candidate_artifacts=artifacts,
            previous_candidate_artifacts=[],
            consensus_artifacts=artifacts,
        )

    def _serialize_termination_assessment_artifacts(
        self,
        artifacts: list[ArtifactRecord],
        *,
        previous_by_agent: dict[str, ArtifactRecord],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for index, artifact in enumerate(artifacts):
            agent_id = str(artifact.get("agent_id", ""))
            previous = previous_by_agent.get(agent_id)
            candidate = self._serialize_judge_candidate(artifact, index=index, previous=previous)
            if candidate is not None:
                serialized.append(candidate)
        return serialized

    def _select_final_answer(
        self,
        *,
        state: WorkflowState,
        stage_name: str,
        artifacts: list[ArtifactRecord],
    ) -> dict[str, Any]:
        mode = str(state.get("final_vote_mode", "llm_judge") or "llm_judge")

        candidates: list[dict[str, Any]] = []
        for index, artifact in enumerate(artifacts):
            candidate = self._serialize_judge_candidate(artifact, index=index)
            if candidate is not None:
                candidates.append(candidate)

        candidate_by_index = {int(item["index"]): item for item in candidates}
        admissible_direct_indices = {
            index
            for index, candidate in candidate_by_index.items()
            if self._candidate_is_admissible_final_winner(state, candidate)[0]
        }
        deterministic_result = self._deterministic_vote_result(
            artifacts,
            mode=mode,
            allowed_indices=admissible_direct_indices,
        )
        singleton_groups = [[index] for index in sorted(candidate_by_index)]

        if mode != "llm_judge":
            if deterministic_result.get("answer"):
                return deterministic_result
            fallback_result = self._non_direct_majority_vote_result(
                state=state,
                artifacts=artifacts,
                candidate_by_index=candidate_by_index,
                groups=singleton_groups,
                mode=mode,
            )
            if fallback_result is not None:
                return {
                    **fallback_result,
                    "source": "deterministic_non_direct_fallback",
                }
            return deterministic_result

        if len(candidates) <= 1:
            if deterministic_result.get("answer"):
                return {
                    **deterministic_result,
                    "source": "deterministic_singleton",
                }
            fallback_result = self._non_direct_majority_vote_result(
                state=state,
                artifacts=artifacts,
                candidate_by_index=candidate_by_index,
                groups=singleton_groups,
                mode=mode,
            )
            if fallback_result is not None:
                return {
                    **fallback_result,
                    "source": "deterministic_singleton_non_direct",
                }
            return {
                **deterministic_result,
                "source": "deterministic_singleton",
            }

        prompt = self._build_final_vote_prompt(
            state=state,
            stage_name=stage_name,
            candidates=candidates,
        )

        t0 = time.perf_counter()
        llm = state["llm_client"].generate(
            prompt=prompt,
            agent_type="judge",
            task_id=str(state.get("task_id", "")),
            run_index=int(state.get("run_index", 0)),
            agent_id=f"{stage_name}_final_vote_judge",
            tools=None,
            max_tool_iterations=1,
            temperature=0.0,
        )
        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1.0)
        if bool(llm.mock_used):
            fallback_result = deterministic_result
            if not fallback_result.get("answer"):
                non_direct_result = self._non_direct_majority_vote_result(
                    state=state,
                    artifacts=artifacts,
                    candidate_by_index=candidate_by_index,
                    groups=singleton_groups,
                    mode=mode,
                )
                if non_direct_result is not None:
                    fallback_result = non_direct_result
            return {
                **fallback_result,
                "source": (
                    "deterministic_non_direct_fallback_mock"
                    if fallback_result is not deterministic_result
                    else "deterministic_fallback_mock"
                ),
                "explanation": "Final vote judge fell back to deterministic voting because the LLM client ran in mock mode.",
            }

        parsed = self._parse_final_vote_judgment(str(llm.text or ""))
        if parsed is None:
            fallback_result = deterministic_result
            if not fallback_result.get("answer"):
                non_direct_result = self._non_direct_majority_vote_result(
                    state=state,
                    artifacts=artifacts,
                    candidate_by_index=candidate_by_index,
                    groups=singleton_groups,
                    mode=mode,
                )
                if non_direct_result is not None:
                    fallback_result = non_direct_result
            return {
                **fallback_result,
                "source": (
                    "deterministic_non_direct_fallback_parse_error"
                    if fallback_result is not deterministic_result
                    else "deterministic_fallback_parse_error"
                ),
                "explanation": "Final vote judge returned invalid JSON; deterministic voting was used instead.",
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "latency_ms": latency_ms,
            }

        valid_indices = set(candidate_by_index)
        invalid_indices = {index for index in parsed["invalid_indices"] if index in valid_indices}
        remaining_valid = valid_indices - invalid_indices

        groups: list[list[int]] = []
        seen: set[int] = set()
        for group in parsed["groups"]:
            cleaned = sorted({index for index in group if index in remaining_valid and index not in seen})
            if not cleaned:
                continue
            groups.append(cleaned)
            seen.update(cleaned)
        for index in sorted(remaining_valid - seen):
            groups.append([index])

        if not remaining_valid or not groups:
            fallback_result = deterministic_result
            if not fallback_result.get("answer"):
                non_direct_result = self._non_direct_majority_vote_result(
                    state=state,
                    artifacts=artifacts,
                    candidate_by_index=candidate_by_index,
                    groups=singleton_groups,
                    mode=mode,
                )
                if non_direct_result is not None:
                    fallback_result = non_direct_result
            return {
                **fallback_result,
                "source": (
                    "deterministic_non_direct_fallback_empty_judgment"
                    if fallback_result is not deterministic_result
                    else "deterministic_fallback_empty_judgment"
                ),
                "explanation": "Final vote judge did not return any usable valid groups; deterministic voting was used instead.",
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "latency_ms": latency_ms,
            }

        winner_index = parsed["winner_index"] if parsed["winner_index"] in remaining_valid else None
        winner_group = next((group for group in groups if winner_index in group), None)
        if winner_group is None:
            winner_group = max(groups, key=lambda group: (len(group), -self._group_mean_confidence(artifacts, group)))
            winner_index = self._best_artifact_index(artifacts, winner_group)

        admissible_direct_indices = {
            index
            for index in remaining_valid
            if self._candidate_is_admissible_final_winner(state, candidate_by_index[index])[0]
        }

        winner_reason = ""
        if winner_index is not None:
            winner_ok, winner_reason = self._candidate_is_admissible_final_winner(
                state,
                candidate_by_index[winner_index],
            )
            if not winner_ok:
                winner_index = None

        if winner_index is None:
            if admissible_direct_indices:
                admissible_result = self._deterministic_vote_result(
                    artifacts,
                    mode=mode,
                    allowed_indices=admissible_direct_indices,
                )
                fallback_reason = (
                    "Final vote judge selected an inadmissible winner; "
                    "deterministic voting over admissible direct answers was used instead."
                    if winner_reason
                    else "Final vote judge did not identify a usable winner; deterministic voting was used instead."
                )
                explanation = self._merge_vote_explanations(
                    str(parsed.get("explanation", "")),
                    winner_reason or fallback_reason,
                )
                return {
                    **admissible_result,
                    "source": (
                        "deterministic_fallback_inadmissible_winner"
                        if winner_reason
                        else "deterministic_fallback_no_winner"
                    ),
                    "explanation": explanation,
                    "token_in": int(llm.token_in),
                    "token_out": int(llm.token_out),
                    "cost_usd": float(llm.cost_usd),
                    "latency_ms": latency_ms,
                }

            fallback_result = self._non_direct_majority_vote_result(
                state=state,
                artifacts=artifacts,
                candidate_by_index=candidate_by_index,
                groups=groups,
                mode=mode,
            )
            if fallback_result is not None:
                explanation = self._merge_vote_explanations(
                    str(parsed.get("explanation", "")),
                    winner_reason or "No admissible direct answer remained after control checks.",
                )
                return {
                    **fallback_result,
                    "source": "llm_judge_non_direct_fallback",
                    "explanation": explanation,
                    "token_in": int(llm.token_in),
                    "token_out": int(llm.token_out),
                    "cost_usd": float(llm.cost_usd),
                    "latency_ms": latency_ms,
                }

            return {
                **deterministic_result,
                "source": (
                    "deterministic_fallback_inadmissible_winner"
                    if winner_reason
                    else "deterministic_fallback_no_winner"
                ),
                "explanation": self._merge_vote_explanations(
                    str(parsed.get("explanation", "")),
                    winner_reason or "Final vote judge did not identify a usable winner.",
                ),
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "latency_ms": latency_ms,
            }

        tally: dict[str, int] = {}
        for group in groups:
            representative_index = self._best_artifact_index(artifacts, group)
            if representative_index is None:
                continue
            signature = answer_signature(str(artifacts[representative_index].get("answer", "")))
            if not signature:
                signature = f"group_{representative_index}"
            tally[signature] = len(group)

        return {
            **self._vote_result_for_artifact(
                artifacts[winner_index],
                mode=mode,
                source="llm_judge",
                tally=tally,
                explanation=str(parsed.get("explanation", "")),
            ),
            "token_in": int(llm.token_in),
            "token_out": int(llm.token_out),
            "cost_usd": float(llm.cost_usd),
            "latency_ms": latency_ms,
        }

    @staticmethod
    def _best_artifact_index(artifacts: list[ArtifactRecord], indices: list[int]) -> int | None:
        ranked: list[tuple[float, str, int]] = []
        for index in indices:
            if index < 0 or index >= len(artifacts):
                continue
            artifact = artifacts[index]
            ranked.append(
                (
                    -float(artifact.get("confidence", 0.5)),
                    str(artifact.get("agent_id", "")),
                    int(index),
                )
            )
        if not ranked:
            return None
        return sorted(ranked)[0][2]

    @staticmethod
    def _group_mean_confidence(artifacts: list[ArtifactRecord], indices: list[int]) -> float:
        scores = [
            float(artifacts[index].get("confidence", 0.5))
            for index in indices
            if 0 <= index < len(artifacts)
        ]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @staticmethod
    def _merge_vote_explanations(primary: str, secondary: str) -> str:
        primary = str(primary or "").strip()
        secondary = str(secondary or "").strip()
        if primary and secondary:
            return f"{primary} {secondary}"
        return primary or secondary

    def _build_control_task_context(self, state: WorkflowState) -> dict[str, Any]:
        task_prompt = state.get("task_prompt", "")
        if isinstance(task_prompt, list):
            normalized = self._normalize_prompt_messages(task_prompt)
            system_constraints: list[str] = []
            latest_user = ""
            latest_non_system = ""
            for item in normalized:
                role = str(item.get("role", "user")).strip().lower()
                content = self._trim_text(str(item.get("content", "")), max_chars=1200)
                if not content:
                    continue
                if role == "system":
                    system_constraints.append(content)
                    continue
                latest_non_system = content
                if role == "user":
                    latest_user = content
            return {
                "current_task": latest_user or latest_non_system,
                "system_constraints": system_constraints[:3],
            }
        return {
            "current_task": self._trim_text(str(task_prompt), max_chars=2000),
            "system_constraints": [],
        }

    @staticmethod
    def _benchmark_guidance_lines(benchmark_name: str, *, audience: str) -> list[str]:
        benchmark = str(benchmark_name or "")
        lines: list[str] = []
        if benchmark == "plancraft":
            lines.extend(
                [
                    "The answer must be the next benchmark action in the adapter grammar.",
                    "Valid action forms include move, craft, smelt, and impossible.",
                    "Treat smelt as a valid benchmark action when the prompt supports it.",
                    "Do not require furnace or fuel inventory unless the task prompt explicitly says so.",
                ]
            )
        if benchmark in {"browsecomp", "finance_agent"}:
            if audience == "agent":
                lines.extend(
                    [
                        "A concrete answer is not sufficient unless the available evidence covers the required criteria.",
                        "If important criteria remain unresolved, say so explicitly instead of overstating certainty.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "A concrete answer is not automatically valid when its own unresolved issues or evidence leave required criteria open.",
                        "Partial synthesis with open unresolved criteria is not a fully supported final answer.",
                    ]
                )
        return lines

    def _serialize_termination_decision_context(self, state: WorkflowState) -> dict[str, Any] | None:
        decision = state.get("termination_decision", {})
        if not isinstance(decision, dict) or not decision:
            return None
        return {
            "stage_name": str(decision.get("stage_name", "")),
            "reason": str(decision.get("reason", "")),
            "reason_detail": str(decision.get("reason_detail", "")),
            "consensus_ratio": float(decision.get("consensus_ratio", 0.0) or 0.0),
            "consensus_count": int(decision.get("consensus_count", 0) or 0),
            "consensus_valid_count": int(decision.get("consensus_valid_count", 0) or 0),
            "consensus_is_substantive": decision.get("consensus_is_substantive"),
            "progress_status": str(decision.get("progress_status", "")),
            "expected_improvement": str(decision.get("expected_improvement", "")),
            "progress_explanation": self._trim_text(
                str(decision.get("progress_explanation", "")),
                max_chars=500,
            ),
        }

    @staticmethod
    def _candidate_support_text(candidate: dict[str, Any]) -> str:
        parts: list[str] = [
            str(candidate.get("summary", "")),
            str(candidate.get("previous_summary", "")),
            str(candidate.get("previous_answer", "")),
        ]
        for key in ("evidence_summary", "unresolved_issues"):
            values = candidate.get(key, [])
            if isinstance(values, list):
                parts.extend(str(item) for item in values)
        return " ".join(part for part in parts if str(part).strip()).strip()

    @staticmethod
    def _text_indicates_open_issue(text: Any) -> bool:
        lowered = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        if not lowered:
            return False
        blockers = (
            "no evidence",
            "insufficient evidence",
            "not enough evidence",
            "need more information",
            "need additional information",
            "needs more information",
            "further research",
            "cannot confirm",
            "cannot verify",
            "could not verify",
            "couldn't verify",
            "not verified",
            "unverified",
            "unknown",
            "unclear",
            "unresolved",
            "missing",
            "not found",
            "criteria remain",
            "criteria remain open",
            "criterion remains open",
            "criteria are open",
            "remain unverified",
        )
        return any(snippet in lowered for snippet in blockers)

    def _candidate_has_blocking_issues(self, candidate: dict[str, Any]) -> bool:
        unresolved_issues = candidate.get("unresolved_issues", [])
        if not isinstance(unresolved_issues, list):
            return False
        return any(self._text_indicates_open_issue(item) for item in unresolved_issues)

    def _candidate_has_weak_evidence(self, state: WorkflowState, candidate: dict[str, Any]) -> bool:
        evidence_summary = candidate.get("evidence_summary", [])
        positive_evidence = int(candidate.get("evidence_count", 0) or 0)
        if isinstance(evidence_summary, list) and evidence_summary:
            if positive_evidence <= 0 and any(
                not self._evidence_entry_is_positive(item) for item in evidence_summary
            ):
                return True
        support_text = self._candidate_support_text(candidate).lower()
        if any(
            snippet in support_text
            for snippet in (
                "no evidence",
                "insufficient evidence",
                "not retrieved",
                "not gathered",
                "unknown",
            )
        ):
            return True
        benchmark_name = str(state.get("benchmark_name", "") or "")
        return benchmark_name in {"browsecomp", "finance_agent"} and positive_evidence <= 0

    def _candidate_violates_benchmark_semantics(
        self,
        state: WorkflowState,
        candidate: dict[str, Any],
    ) -> bool:
        benchmark_name = str(state.get("benchmark_name", "") or "")
        answer = str(candidate.get("answer", "")).strip().lower()
        support_text = self._candidate_support_text(candidate).lower()
        if benchmark_name == "plancraft":
            if answer and not re.match(r"^(move|craft|smelt|impossible)\b", answer):
                return True
            if answer.startswith("impossible") and any(
                snippet in support_text for snippet in ("furnace", "fuel", "coal", "charcoal")
            ):
                return True
        return False

    def _candidate_is_admissible_final_winner(
        self,
        state: WorkflowState,
        candidate: dict[str, Any],
    ) -> tuple[bool, str]:
        if str(candidate.get("answer_mode", "")) != "direct":
            return False, "The selected candidate is not a direct final answer."
        if self._candidate_has_blocking_issues(candidate):
            return False, "The selected candidate still reports unresolved blocking issues."
        if self._candidate_has_weak_evidence(state, candidate):
            return False, "The selected candidate does not have adequate supporting evidence."
        if self._candidate_violates_benchmark_semantics(state, candidate):
            return False, "The selected candidate violates benchmark-visible action semantics."
        return True, ""

    def _vote_result_for_artifact(
        self,
        artifact: ArtifactRecord,
        *,
        mode: str,
        source: str,
        tally: dict[str, int] | None = None,
        explanation: str = "",
        winner_signature: str = "",
    ) -> dict[str, Any]:
        return {
            "mode": mode,
            "source": source,
            "answer": self._artifact_display_answer(artifact, max_chars=6000),
            "tally": dict(tally or {}),
            "explanation": explanation,
            "token_in": 0,
            "token_out": 0,
            "cost_usd": 0.0,
            "latency_ms": 1.0,
            "winner_signature": winner_signature,
            "selected_artifact_id": str(artifact.get("artifact_id", "")),
            "selected_agent_id": str(artifact.get("agent_id", "")),
            "selected_source_artifact_ids": list(artifact.get("source_artifact_ids", [])),
        }

    def _non_direct_majority_vote_result(
        self,
        *,
        state: WorkflowState,
        artifacts: list[ArtifactRecord],
        candidate_by_index: dict[int, dict[str, Any]],
        groups: list[list[int]],
        mode: str,
    ) -> dict[str, Any] | None:
        if not groups:
            return None
        termination_decision = state.get("termination_decision", {})
        preferred_groups = groups
        if (
            isinstance(termination_decision, dict)
            and termination_decision.get("consensus_is_substantive") is False
        ):
            non_direct_groups = [
                group
                for group in groups
                if any(
                    str(candidate_by_index.get(index, {}).get("answer_mode", "")) != "direct"
                    for index in group
                )
            ]
            if non_direct_groups:
                preferred_groups = non_direct_groups

        winner_group = max(
            preferred_groups,
            key=lambda group: (
                len(group),
                sum(
                    1
                    for index in group
                    if str(candidate_by_index.get(index, {}).get("answer_mode", "")) != "direct"
                ),
                self._group_mean_confidence(artifacts, group),
            ),
        )
        winner_index = self._best_artifact_index(artifacts, winner_group)
        if winner_index is None:
            return None

        tally: dict[str, int] = {}
        for group in groups:
            representative_index = self._best_artifact_index(artifacts, group)
            if representative_index is None:
                continue
            signature = answer_signature(str(artifacts[representative_index].get("answer", "")))
            if not signature:
                signature = f"group_{representative_index}"
            tally[signature] = len(group)

        return self._vote_result_for_artifact(
            artifacts[winner_index],
            mode=mode,
            source="llm_judge_non_direct_fallback",
            tally=tally,
            explanation="",
        )

    def _build_final_vote_prompt(
        self,
        *,
        state: WorkflowState,
        stage_name: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        benchmark_name = str(state.get("benchmark_name", "") or "")
        payload = {
            "benchmark_name": benchmark_name,
            "task_context": self._build_control_task_context(state),
            "benchmark_guidance": self._benchmark_guidance_lines(benchmark_name, audience="control"),
            "stage_name": stage_name,
            "termination_decision": self._serialize_termination_decision_context(state),
            "answers": candidates,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are the final judge for a multi-agent workflow. "
                    "Group materially equivalent answers together, mark unusable answers invalid, "
                    "and choose the single best answer index for the task. "
                    "Each candidate includes answer_mode, unresolved_issues, evidence_summary, evidence_count, used_tools, and confidence. "
                    "A direct answer is not automatically valid. Mark candidates invalid when their own unresolved issues, weak evidence, "
                    "or benchmark-visible semantics make them unreliable. "
                    "A singleton concrete answer must not win solely because it is the only concrete answer. "
                    "Use termination_decision as additional controller context when it is provided, especially when the controller already judged the majority cluster non-substantive. "
                    "Do not invent constraints that do not appear in task_context or benchmark_guidance. "
                    "Return strict JSON only with this schema: "
                    '{"groups":[[0,2],[1]],"winner_index":0,"invalid_indices":[3],"explanation":"short reason"}. '
                    "The winner_index must refer to one valid answer."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    def _build_termination_assessment_prompt(
        self,
        *,
        state: WorkflowState,
        stage_name: str,
        round_index: int,
        discussion_index: int,
        current_candidates: list[dict[str, Any]],
        consensus_candidates: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        benchmark_name = str(state.get("benchmark_name", "") or "")
        payload = {
            "benchmark_name": benchmark_name,
            "task_context": self._build_control_task_context(state),
            "benchmark_guidance": self._benchmark_guidance_lines(benchmark_name, audience="control"),
            "stage_name": stage_name,
            "round_index": round_index,
            "discussion_index": discussion_index,
            "candidate_artifacts": current_candidates,
            "consensus_answers": consensus_candidates,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are the termination judge for a multi-agent workflow. "
                    "Use consensus_answers to decide whether multiple answers express the same final answer for task-solving purposes. "
                    "Group answers together only when they are materially equivalent in final claim, decision, or action. "
                    "Ignore wording and formatting differences. "
                    "Each candidate includes answer_mode, unresolved_issues, evidence_summary, evidence_count, used_tools, and confidence. "
                    "Blocked-status, planning-only, and 'no evidence' outputs are not substantive answers. "
                    "Also assess whether the largest group's shared answer is a substantive task solution "
                    "(a concrete claim, decision, or result) rather than merely planning, acknowledging ignorance, "
                    "or restating the task without answering it. "
                    "Do not invent constraints that do not appear in task_context or benchmark_guidance. "
                    "Use candidate_artifacts, including any previous_answer fields, to judge progress. "
                    "Set should_stop_for_no_progress to true only when another round is unlikely to materially improve correctness; "
                    "do not use it merely because wording is similar. "
                    "Planning-only updates, 'need more information', and task restatements are not substantive answers. "
                    "Return strict JSON only with this schema: "
                    '{"groups":[[0,2],[1]],"invalid_indices":[3],"is_substantive":true,"progress_status":"stalled","expected_improvement":"low","should_stop_for_no_progress":true,"explanation":"short reason"}. '
                    "Every non-empty valid consensus_answers index should appear in exactly one group unless it belongs in invalid_indices. "
                    "progress_status must be one of improving, stalled, unclear. "
                    "expected_improvement must be one of high, medium, low. "
                    "Set is_substantive to true only when the majority group provides a definite answer to the task."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    @staticmethod
    def _parse_termination_assessment_judgment(text: str) -> dict[str, Any] | None:
        payload = LangGraphMASEngine._extract_json_object(text)
        if not isinstance(payload, dict):
            return None

        groups_raw = payload.get("groups")
        invalid_raw = payload.get("invalid_indices", [])
        if not isinstance(groups_raw, list):
            return None

        groups: list[list[int]] = []
        for item in groups_raw:
            if not isinstance(item, list):
                continue
            group: list[int] = []
            for value in item:
                try:
                    group.append(int(value))
                except Exception:
                    continue
            if group:
                groups.append(group)

        invalid_indices: list[int] = []
        if isinstance(invalid_raw, list):
            for value in invalid_raw:
                try:
                    invalid_indices.append(int(value))
                except Exception:
                    continue

        is_substantive_raw = payload.get("is_substantive")
        is_substantive: bool | None = None
        if isinstance(is_substantive_raw, bool):
            is_substantive = is_substantive_raw
        elif isinstance(is_substantive_raw, str):
            is_substantive = is_substantive_raw.lower() in ("true", "1", "yes")

        progress_status_raw = str(payload.get("progress_status", "")).strip().lower()
        progress_status = (
            progress_status_raw
            if progress_status_raw in {"improving", "stalled", "unclear"}
            else "unclear"
        )

        expected_improvement_raw = str(payload.get("expected_improvement", "")).strip().lower()
        expected_improvement = (
            expected_improvement_raw
            if expected_improvement_raw in {"high", "medium", "low"}
            else "medium"
        )

        should_stop_raw = payload.get("should_stop_for_no_progress")
        should_stop_for_no_progress = False
        if isinstance(should_stop_raw, bool):
            should_stop_for_no_progress = should_stop_raw
        elif isinstance(should_stop_raw, str):
            should_stop_for_no_progress = should_stop_raw.lower() in ("true", "1", "yes")

        return {
            "groups": groups,
            "invalid_indices": invalid_indices,
            "explanation": str(payload.get("explanation", "")),
            "is_substantive": is_substantive,
            "progress_status": progress_status,
            "expected_improvement": expected_improvement,
            "should_stop_for_no_progress": should_stop_for_no_progress,
        }

    @staticmethod
    def _parse_termination_consensus_judgment(text: str) -> dict[str, Any] | None:
        return LangGraphMASEngine._parse_termination_assessment_judgment(text)

    @staticmethod
    def _parse_final_vote_judgment(text: str) -> dict[str, Any] | None:
        payload = LangGraphMASEngine._extract_json_object(text)
        if not isinstance(payload, dict):
            return None

        groups_raw = payload.get("groups")
        invalid_raw = payload.get("invalid_indices", [])
        if not isinstance(groups_raw, list):
            return None

        try:
            winner_index = int(payload.get("winner_index"))
        except Exception:
            winner_index = None

        groups: list[list[int]] = []
        for item in groups_raw:
            if not isinstance(item, list):
                continue
            group: list[int] = []
            for value in item:
                try:
                    group.append(int(value))
                except Exception:
                    continue
            if group:
                groups.append(group)

        invalid_indices: list[int] = []
        if isinstance(invalid_raw, list):
            for value in invalid_raw:
                try:
                    invalid_indices.append(int(value))
                except Exception:
                    continue

        return {
            "groups": groups,
            "winner_index": winner_index,
            "invalid_indices": invalid_indices,
            "explanation": str(payload.get("explanation", "")),
        }

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        candidates = [str(text or "").strip()]
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", str(text or ""), flags=re.DOTALL)
        candidates.extend(fenced)
        brace_match = re.search(r"\{.*\}", str(text or ""), flags=re.DOTALL)
        if brace_match is not None:
            candidates.append(brace_match.group(0))
        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _trim_text(text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _route_next_step(self, state: WorkflowState) -> str:
        """Route to the explicit next step set by a controller or termination node."""

        return str(state.get("next_step", "finalize"))

    def _make_parallel_router(
        self,
        *,
        worker_node: str,
        fallback_node: str,
    ) -> Callable[[WorkflowState], str | list[Send]]:
        """Fan out one Send per active agent, or jump to the fallback node when empty."""

        def route(state: WorkflowState) -> str | list[Send]:
            active_agents = list(state.get("active_agents", []))
            if not active_agents:
                return fallback_node
            sends: list[Send] = []
            for agent_id in active_agents:
                branch_state = dict(state)
                branch_state["active_agent"] = agent_id
                sends.append(Send(worker_node, branch_state))
            return sends

        return route

    def _messages_for_recipient(
        self,
        state: WorkflowState,
        *,
        recipient: str,
        kinds: set[str] | None = None,
        round_index: int | None = None,
        discussion_index: int | None = None,
        latest_only: bool = False,
    ) -> list[RelayPacket]:
        messages = []
        for message in state.get("messages", []):
            if recipient not in message.get("recipients", []):
                continue
            if kinds is not None and str(message.get("kind", "")) not in kinds:
                continue
            if round_index is not None and int(message.get("round", -1)) != round_index:
                continue
            if discussion_index is not None and int(message.get("discussion_index", -1)) != discussion_index:
                continue
            messages.append(message)

        if not latest_only:
            return sorted(messages, key=self._message_sort_key)

        latest: dict[tuple[str, str], RelayPacket] = {}
        for message in messages:
            key = (str(message.get("sender", "")), str(message.get("kind", "")))
            current = latest.get(key)
            if current is None or self._message_sort_key(message) >= self._message_sort_key(current):
                latest[key] = message
        return sorted(latest.values(), key=self._message_sort_key)

    def _artifacts_for_nodes(
        self,
        state: WorkflowState,
        *,
        node_names: set[str],
        round_index: int | None = None,
        discussion_index: int | None = None,
        latest_only: bool = False,
    ) -> list[ArtifactRecord]:
        artifacts = []
        for artifact in state.get("artifacts", []):
            if str(artifact.get("node_name", "")) not in node_names:
                continue
            if round_index is not None and int(artifact.get("round_index", -1)) != round_index:
                continue
            if discussion_index is not None and int(artifact.get("discussion_index", -1)) != discussion_index:
                continue
            artifacts.append(artifact)

        if not latest_only:
            return sorted(artifacts, key=self._artifact_sort_key)

        latest: dict[str, ArtifactRecord] = {}
        for artifact in artifacts:
            agent_id = str(artifact.get("agent_id", ""))
            current = latest.get(agent_id)
            if current is None or self._artifact_sort_key(artifact) >= self._artifact_sort_key(current):
                latest[agent_id] = artifact
        return sorted(latest.values(), key=self._artifact_sort_key)

    def _latest_artifact_for_nodes(
        self,
        state: WorkflowState,
        *,
        node_names: set[str],
    ) -> ArtifactRecord | None:
        artifacts = self._artifacts_for_nodes(state, node_names=node_names)
        if not artifacts:
            return None
        return sorted(artifacts, key=self._artifact_sort_key)[-1]

    def _latest_substantive_artifact_for_nodes(
        self,
        state: WorkflowState,
        *,
        node_names: set[str],
    ) -> ArtifactRecord | None:
        artifacts = [
            artifact
            for artifact in self._artifacts_for_nodes(state, node_names=node_names)
            if has_substantive_answer(artifact.get("answer", ""))
        ]
        if not artifacts:
            return None
        return sorted(artifacts, key=self._artifact_sort_key)[-1]

    def _latest_agent_artifact(self, state: WorkflowState, agent_id: str) -> ArtifactRecord | None:
        latest = latest_artifact_by_agent(list(state.get("artifacts", [])))
        return latest.get(agent_id)

    def _current_specialist_stage_name(self, state: WorkflowState) -> str:
        artifacts = self._artifacts_for_nodes(
            state,
            node_names={"specialists_revision_round"},
            round_index=int(state.get("round_index", 0)),
            discussion_index=int(state.get("discussion_index", 0)),
        )
        if artifacts:
            return "specialists_revision_round"
        return "specialists_initial_round"

    def _current_specialist_cycle_artifacts(
        self,
        state: WorkflowState,
        *,
        round_index: int,
        discussion_index: int | None = None,
    ) -> list[ArtifactRecord]:
        artifacts = self._artifacts_for_nodes(
            state,
            node_names={"specialists_initial_round", "specialists_revision_round"},
            round_index=round_index,
            latest_only=True,
        )
        if discussion_index is None:
            return artifacts
        filtered = [
            artifact
            for artifact in artifacts
            if int(artifact.get("discussion_index", 0)) == discussion_index
            or (
                discussion_index == 0
                and str(artifact.get("node_name", "")) == "specialists_initial_round"
            )
        ]
        return sorted(filtered, key=self._artifact_sort_key)

    @staticmethod
    def _group_for_agent(layout: TopologyLayout, agent_id: str) -> list[str]:
        for group in layout.groups:
            if agent_id in group:
                return list(group)
        return []

    @staticmethod
    def _message_sort_key(message: RelayPacket) -> tuple[int, int, int]:
        message_id = str(message.get("message_id", "m_0"))
        match = re.search(r"(\d+)$", message_id)
        return (
            int(message.get("round", 0)),
            int(message.get("discussion_index", 0)),
            int(match.group(1)) if match else 0,
        )

    @staticmethod
    def _artifact_sort_key(artifact: ArtifactRecord) -> tuple[int, int, int, str]:
        return (
            int(artifact.get("round_index", 0)),
            int(artifact.get("discussion_index", 0)),
            int(artifact.get("dispatch_id", -1)),
            str(artifact.get("node_name", "")),
        )

    def _build_agent_prompt(
        self,
        *,
        state: WorkflowState,
        task_prompt: Any,
        agent_id: str,
        role: str,
        stage_role: str,
        directive: str,
        round_index: int,
        discussion_index: int,
        visible_messages: list[RelayPacket],
        prior_artifact: ArtifactRecord | None,
        persona_info: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        serialized_tools = self._serialize_tools(list(state.get("tools", [])))
        tools_available = bool(serialized_tools)
        benchmark_name = str(state.get("benchmark_name", "") or "")

        system_lines = [
            "You are one agent in a deterministic multi-agent workflow.",
            f"Agent ID: {agent_id}",
            f"Agent Role: {role}",
            f"Stage Role: {stage_role}",
            (
                "Instruction priority: 1) structural stage contract, 2) tool-use contract, "
                "3) task and benchmark instructions, 4) domain persona."
            ),
        ]
        if persona_info:
            system_lines.append(f"Domain Role: {persona_info.get('role_name', '')}")
            system_lines.append(f"Persona: {persona_info.get('persona', '')}")
        system_lines.append("")
        system_lines.append(
            "Use only the task messages, the prior artifact, and the visible packets provided in this conversation. "
            "Do not invent hidden context."
        )
        system_lines.append(
            "Return exactly one JSON object and do not wrap it in markdown."
        )
        system_lines.append(
            "Required JSON keys: answer_artifact, summary, critique, revision_request, confidence, "
            "unresolved_issues, evidence_summary."
        )
        system_lines.append(
            "Set confidence to reflect confidence in the current answer_artifact only, using this rubric: "
            "0.0 = no answer or pure planning; 0.25 = weak hypothesis; 0.5 = plausible but incomplete; "
            "0.75 = likely correct with remaining gaps; 1.0 = strongly supported final answer. "
            "If a field is unknown, use an empty string, an empty list, or a conservative confidence score."
        )
        system_lines.append(
            "evidence_summary must summarize actual evidence from tool outputs, visible packets, or the prior artifact. "
            "Do not claim evidence that was not actually retrieved or provided."
        )
        if stage_role == "planner":
            system_lines.append(
                "Planner contract: answer_artifact should contain a bounded work plan or task package. "
                "Do not present a speculative final answer as completed work."
            )
        else:
            system_lines.append(
                "Answer-stage contract: answer_artifact must be the current best direct answer or a concise blocked-status explanation. "
                "Do not use answer_artifact for plans, tool lists, search strategies, or sub-question lists."
            )
            if stage_role == "worker":
                system_lines.append(
                    "Worker contract: gather or apply evidence, then state the best supported answer you can defend."
                )
            elif stage_role == "critic":
                system_lines.append(
                    "Critic contract: challenge weak claims, verify against available evidence, and revise toward the best supported answer."
                )
                if not visible_messages and prior_artifact is None:
                    system_lines.append(
                        "Opening-turn critic fallback: there is no peer claim to critique yet. "
                        "Act as an independent investigator this round: gather evidence with tools and propose the strongest supported candidate before switching to critique-oriented revisions."
                    )
            elif stage_role == "aggregator":
                system_lines.append(
                    "Aggregator contract: synthesize peer outputs into one supported answer; do not simply restate unresolved debate."
                )

        if tools_available:
            system_lines.append("Available tools:")
            system_lines.extend(self._compact_tool_summary(serialized_tools))
            if self._is_answer_producing_stage(stage_role):
                system_lines.append(
                    "Tool-use contract: if evidence is missing or weak, call the relevant tool instead of narrating that you need to search."
                )
                system_lines.append(
                    "Do not return a blocked, unknown, or planning answer before at least one tool attempt unless the visible packets or prior artifact already contain sufficient evidence."
                )
                if benchmark_name in {"browsecomp", "finance_agent"}:
                    system_lines.append(
                        "On retrieval tasks, the first useful move is usually a tool call followed by an evidence-backed answer."
                    )

        benchmark_guidance = self._benchmark_guidance_lines(benchmark_name, audience="agent")
        if benchmark_guidance:
            system_lines.append("Benchmark guidance:")
            system_lines.extend(f"- {line}" for line in benchmark_guidance)

        system_message = {
            "role": "system",
            "content": "\n".join(system_lines),
        }

        messages = [system_message]
        if isinstance(task_prompt, list):
            messages.extend(self._normalize_prompt_messages(task_prompt))
        else:
            messages.append({"role": "user", "content": f"Task:\n{task_prompt}"})

        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "agent_role": role,
            "stage_role": stage_role,
            "directive": directive,
            "round_index": round_index,
            "discussion_index": discussion_index,
            "benchmark_name": benchmark_name,
            "prior_artifact": self._packet_payload_from_artifact(state, prior_artifact)
            if prior_artifact
            else None,
            "visible_packets": self._serialize_for_json(visible_messages),
        }
        transcript_summary = state.get("transcript_summary")
        if isinstance(transcript_summary, dict) and transcript_summary:
            payload["rolling_transcript_summary"] = self._serialize_for_json(transcript_summary)
        if serialized_tools:
            payload["available_tools"] = serialized_tools
        if persona_info:
            payload["domain_role"] = persona_info.get("role_name", "")
            payload["persona"] = persona_info.get("persona", "")
        messages.append(
            {
                "role": "user",
                "content": (
                    "Stage context follows as JSON. Treat it as the authoritative runtime state for this step.\n\n"
                    + json.dumps(payload, indent=2, ensure_ascii=False)
                ),
            }
        )
        return messages

    @staticmethod
    def _bundle_peer_summaries(payloads: list[dict[str, Any]]) -> str:
        lines = []
        for index, payload in enumerate(payloads[:6]):
            sender = str(payload.get("sender", f"peer_{index + 1}"))
            summary = str(payload.get("summary", "")) or str(payload.get("answer_artifact", ""))
            lines.append(f"{sender}: {summary}")
        return " | ".join(lines) if lines else "No peer summaries available."

    @staticmethod
    def _peer_artifact_max_chars(state: WorkflowState) -> int:
        return max(0, int(state.get("peer_artifact_max_chars", 0)))

    def _packet_payload_from_artifact(
        self,
        state: WorkflowState,
        artifact: ArtifactRecord,
    ) -> dict[str, Any]:
        return packet_payload_from_artifact(
            artifact,
            max_chars=self._peer_artifact_max_chars(state),
        )

    def _packet_content(self, state: WorkflowState, payload: dict[str, Any]) -> str:
        return packet_content(
            payload,
            max_chars=self._peer_artifact_max_chars(state),
        )

    def _resolve_final_answer(self, state: WorkflowState) -> str:
        artifacts = list(state.get("artifacts", []))
        topology = str(state.get("topology", ""))
        selected_artifact_id = str(state.get("selected_artifact_id", "") or "").strip()

        if selected_artifact_id:
            selected_artifact = artifacts_by_id(artifacts).get(selected_artifact_id)
            if selected_artifact is not None:
                return self._artifact_display_answer(selected_artifact, max_chars=6000)

        if topology == TOPOLOGY_SAS:
            artifact = self._latest_substantive_artifact_for_nodes(state, node_names={"single_agent"})
            if artifact:
                return self._artifact_display_answer(artifact, max_chars=6000)
            return self._fallback_answer_from_artifacts(artifacts)
        if topology in {TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION, TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION}:
            artifact = self._latest_substantive_artifact_for_nodes(
                state,
                node_names={"orchestrator_merge"},
            )
            if artifact:
                return self._artifact_display_answer(artifact, max_chars=6000)
            return self._fallback_answer_from_artifacts(
                self._artifacts_for_nodes(
                    state,
                    node_names={
                        "orchestrator_merge",
                        "specialist_worker",
                        "specialists_initial_round",
                        "specialists_revision_round",
                    },
                    latest_only=True,
                )
            )
        if topology == TOPOLOGY_ORCHESTRATOR_TREE:
            artifact = self._latest_substantive_artifact_for_nodes(
                state,
                node_names={"root_reducer"},
            )
            if artifact:
                return self._artifact_display_answer(artifact, max_chars=6000)
            return self._fallback_answer_from_artifacts(
                self._artifacts_for_nodes(
                    state,
                    node_names={"root_reducer", "manager_reducers", "worker_nodes"},
                    latest_only=True,
                )
            )
        if topology == TOPOLOGY_ONLY_VOTING:
            return self._fallback_answer_from_artifacts(
                self._artifacts_for_nodes(state, node_names={"worker"}, latest_only=True)
            )
        if topology == TOPOLOGY_FULLY_LINKED_DEBATE:
            return self._fallback_answer_from_artifacts(
                self._artifacts_for_nodes(state, node_names={"debate_round"}, latest_only=True)
            )
        if topology == TOPOLOGY_GROUP_CHAT_DEBATE:
            return self._fallback_answer_from_artifacts(
                self._artifacts_for_nodes(state, node_names={"representative_merge"}, latest_only=True)
            )
        return self._fallback_answer_from_artifacts(artifacts)

    def _finalize_selection_for_topology(
        self,
        state: WorkflowState,
    ) -> tuple[list[ArtifactRecord], dict[str, Any]] | None:
        topology = str(state.get("topology", ""))
        round_index = int(state.get("round_index", 0))

        if topology == TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION:
            preferred_nodes = {"orchestrator_merge"}
            candidate_nodes = {"orchestrator_merge", "specialist_worker"}
        elif topology == TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION:
            preferred_nodes = {"orchestrator_merge"}
            candidate_nodes = {
                "orchestrator_merge",
                "specialists_initial_round",
                "specialists_revision_round",
            }
        elif topology == TOPOLOGY_ORCHESTRATOR_TREE:
            preferred_nodes = {"root_reducer"}
            candidate_nodes = {"root_reducer", "manager_reducers"}
        else:
            return None

        candidates = self._artifacts_for_nodes(
            state,
            node_names=candidate_nodes,
            round_index=round_index,
            latest_only=True,
        )
        if not candidates:
            return None

        preferred_candidates = [
            artifact
            for artifact in candidates
            if str(artifact.get("node_name", "")) in preferred_nodes
            and self._artifact_display_answer(artifact)
        ]
        latest_candidate = max(candidates, key=self._artifact_sort_key)
        latest_preferred = (
            max(preferred_candidates, key=self._artifact_sort_key) if preferred_candidates else None
        )
        termination_decision = state.get("termination_decision", {})
        consensus_non_substantive = (
            isinstance(termination_decision, dict)
            and termination_decision.get("consensus_is_substantive") is False
        )

        if (
            latest_preferred is not None
            and self._artifact_sort_key(latest_preferred) >= self._artifact_sort_key(latest_candidate)
            and not consensus_non_substantive
        ):
            return candidates, self._vote_result_for_artifact(
                latest_preferred,
                mode=str(state.get("final_vote_mode", "llm_judge") or "llm_judge"),
                source="topology_finalize_preferred_artifact",
            )

        return candidates, self._select_final_answer(
            state=state,
            stage_name="finalize",
            artifacts=candidates,
        )

    def _safe_vote_answer_or_fallback(
        self,
        state: WorkflowState,
        artifacts: list[ArtifactRecord],
        vote_result: dict[str, Any],
    ) -> str:
        answer = str(vote_result.get("answer", "") or "").strip()
        if answer:
            return answer

        safe_artifacts: list[ArtifactRecord] = []
        for index, artifact in enumerate(artifacts):
            candidate = self._serialize_judge_candidate(artifact, index=index)
            if candidate is None:
                continue
            if str(candidate.get("answer_mode", "")) != "direct":
                safe_artifacts.append(artifact)
                continue
            if self._candidate_is_admissible_final_winner(state, candidate)[0]:
                safe_artifacts.append(artifact)

        if safe_artifacts:
            return self._fallback_answer_from_artifacts(safe_artifacts)
        return UNSUPPORTED_FINAL_ANSWER

    def _fallback_answer_from_artifacts(self, artifacts: list[ArtifactRecord]) -> str:
        deterministic = self._deterministic_vote_result(artifacts, mode="deterministic")
        if deterministic.get("answer"):
            return str(deterministic["answer"])

        recent_text = self._best_recent_nonempty_artifact_text(artifacts)
        if recent_text:
            return recent_text
        return UNSUPPORTED_FINAL_ANSWER

    @staticmethod
    def _vote_outcome_reason(vote_result: dict[str, Any]) -> str:
        tally = vote_result.get("tally", {})
        if not isinstance(tally, dict) or not tally:
            return "no_valid_votes"

        counts = sorted((int(value) for value in tally.values()), reverse=True)
        if len(counts) == 1 or counts[0] > counts[1]:
            return "majority_vote"

        source = str(vote_result.get("source", ""))
        if source.startswith("llm_judge"):
            return "judge_tiebreak"
        if source.startswith("deterministic"):
            return "deterministic_tiebreak"
        return "vote_tiebreak"

    @staticmethod
    def _phase_history_entry(
        state: WorkflowState,
        node_name: str,
        *,
        active_agents: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "phase": node_name,
            "round_index": int(state.get("round_index", 0)),
            "discussion_index": int(state.get("discussion_index", 0)),
            "dispatch_id": int(state.get("dispatch_id", -1)),
            "active_agents": list(active_agents or []),
        }

    def _termination_event(
        self,
        state: WorkflowState,
        node_name: str,
        decision: TerminationDecision,
    ) -> dict[str, Any]:
        return self._draft_event(
            dispatch_id=int(state.get("dispatch_id", -1)),
            node_order=3,
            agent_order=-1,
            event_order=0,
            actor="system",
            event_type="verify",
            payload={
                "node": node_name,
                "termination": dict(decision),
            },
            token_in=int(decision.get("control_token_in", 0)),
            token_out=int(decision.get("control_token_out", 0)),
            latency_ms=float(decision.get("control_latency_ms", 1.0)),
            cost_usd=float(decision.get("control_cost_usd", 0.0)),
            state_id=f"run_{state['run_index']}_{node_name}_termination",
        )

    @staticmethod
    def _draft_event(
        *,
        dispatch_id: int,
        node_order: int,
        agent_order: int,
        event_order: int,
        actor: str,
        event_type: str,
        payload: dict[str, Any],
        token_in: int,
        token_out: int,
        latency_ms: float,
        cost_usd: float,
        state_id: str,
    ) -> dict[str, Any]:
        return {
            "_sort": (dispatch_id, node_order, agent_order, event_order),
            "actor": actor,
            "event_type": event_type,
            "payload": payload,
            "token_in": max(0, int(token_in)),
            "token_out": max(0, int(token_out)),
            "latency_ms": max(0.0, float(latency_ms)),
            "cost_usd": max(0.0, float(cost_usd)),
            "state_id": state_id,
        }

    @staticmethod
    def _normalize_prompt_messages(prompt: Any) -> list[dict[str, Any]]:
        if isinstance(prompt, list):
            normalized = []
            for item in prompt:
                if not isinstance(item, dict):
                    normalized.append({"role": "user", "content": str(item)})
                    continue
                normalized.append(
                    {
                        "role": str(item.get("role", "user")),
                        "content": item.get("content", ""),
                    }
                )
            return normalized
        return [{"role": "user", "content": str(prompt)}]

    @staticmethod
    def _is_answer_producing_stage(stage_role: str) -> bool:
        return stage_role in {"worker", "critic", "aggregator"}

    @staticmethod
    def _answer_mode(value: Any) -> str:
        return classify_answer_mode(value)

    def _artifact_answer_mode(self, artifact: ArtifactRecord) -> str:
        return self._answer_mode(artifact.get("answer", ""))

    def _artifact_display_answer(self, artifact: ArtifactRecord, *, max_chars: int = 1600) -> str:
        direct = extract_substantive_answer(artifact.get("answer", ""))
        if direct:
            return self._trim_text(direct, max_chars=max_chars)
        summary = str(artifact.get("summary", "")).strip()
        if summary:
            return self._trim_text(summary, max_chars=max_chars)
        raw = str(artifact.get("answer", "")).strip()
        return self._trim_text(raw, max_chars=max_chars)

    @staticmethod
    def _evidence_entry_is_positive(value: Any) -> bool:
        text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        if not text:
            return False
        negative_snippets = (
            "no evidence",
            "none",
            "not retrieved",
            "not gathered",
            "no search",
            "not been identified",
            "unknown",
            "insufficient",
        )
        return not any(snippet in text for snippet in negative_snippets)

    def _artifact_evidence_count(self, artifact: ArtifactRecord) -> int:
        evidence = artifact.get("evidence_summary", [])
        if not isinstance(evidence, list):
            return 0
        return sum(1 for item in evidence if self._evidence_entry_is_positive(item))

    def _artifact_used_tools(self, artifact: ArtifactRecord) -> bool:
        records = artifact.get("tool_records", [])
        return isinstance(records, list) and bool(records)

    def _artifact_rank_key(self, artifact: ArtifactRecord) -> tuple[int, int, float, str]:
        return (
            -self._artifact_evidence_count(artifact),
            -int(self._artifact_used_tools(artifact)),
            -float(artifact.get("confidence", 0.5)),
            str(artifact.get("agent_id", "")),
        )

    def _serialize_judge_candidate(
        self,
        artifact: ArtifactRecord,
        *,
        index: int,
        previous: ArtifactRecord | None = None,
    ) -> dict[str, Any] | None:
        answer_text = self._artifact_display_answer(artifact)
        if not answer_text:
            return None

        previous_text = self._artifact_display_answer(previous) if previous is not None else ""
        evidence_summary = artifact.get("evidence_summary", [])
        if not isinstance(evidence_summary, list):
            evidence_summary = []
        unresolved_issues = artifact.get("unresolved_issues", [])
        if not isinstance(unresolved_issues, list):
            unresolved_issues = []

        return {
            "index": index,
            "agent_id": str(artifact.get("agent_id", "")),
            "role": str(artifact.get("role", "")),
            "stage_role": str(artifact.get("stage_role", "")),
            "answer_mode": self._artifact_answer_mode(artifact),
            "answer": answer_text,
            "summary": self._trim_text(str(artifact.get("summary", "")), max_chars=400),
            "confidence": float(artifact.get("confidence", 0.5)),
            "evidence_summary": [self._trim_text(str(item), max_chars=220) for item in evidence_summary[:4]],
            "unresolved_issues": [
                self._trim_text(str(item), max_chars=220) for item in unresolved_issues[:4]
            ],
            "evidence_count": self._artifact_evidence_count(artifact),
            "used_tools": self._artifact_used_tools(artifact),
            "tool_call_count": len(artifact.get("tool_records", []) or []),
            "previous_answer": self._trim_text(previous_text, max_chars=1200) if previous_text else None,
            "previous_summary": (
                self._trim_text(str(previous.get("summary", "")), max_chars=300)
                if previous is not None and str(previous.get("summary", "")).strip()
                else None
            ),
        }

    def _deterministic_vote_result(
        self,
        artifacts: list[ArtifactRecord],
        *,
        mode: str,
        allowed_indices: set[int] | None = None,
    ) -> dict[str, Any]:
        direct_artifacts: list[tuple[int, ArtifactRecord]] = []
        for index, artifact in enumerate(artifacts):
            if allowed_indices is not None and index not in allowed_indices:
                continue
            if self._artifact_answer_mode(artifact) != "direct":
                continue
            direct_artifacts.append((index, artifact))

        groups: dict[str, list[tuple[int, ArtifactRecord]]] = {}
        for index, artifact in direct_artifacts:
            signature = answer_signature(str(artifact.get("answer", "")))
            if not signature:
                continue
            groups.setdefault(signature, []).append((index, artifact))

        if not groups:
            return {
                "mode": mode,
                "source": "deterministic",
                "answer": "",
                "tally": {},
                "explanation": "",
                "token_in": 0,
                "token_out": 0,
                "cost_usd": 0.0,
                "latency_ms": 1.0,
                "selected_artifact_id": "",
                "selected_agent_id": "",
                "selected_source_artifact_ids": [],
            }

        ranked = sorted(
            groups.items(),
            key=lambda item: (
                -len(item[1]),
                -sum(self._artifact_evidence_count(artifact) for _, artifact in item[1]),
                -sum(1 for _, artifact in item[1] if self._artifact_used_tools(artifact)),
                -(
                    sum(float(artifact.get("confidence", 0.5)) for _, artifact in item[1])
                    / len(item[1])
                ),
                item[0],
            ),
        )
        winner_signature, winner_group = ranked[0]
        best_artifact = sorted((artifact for _, artifact in winner_group), key=self._artifact_rank_key)[0]
        return self._vote_result_for_artifact(
            best_artifact,
            mode=mode,
            source="deterministic",
            tally={signature: len(items) for signature, items in groups.items()},
            explanation="",
            winner_signature=winner_signature,
        )

    def _best_recent_nonempty_artifact_text(self, artifacts: list[ArtifactRecord]) -> str:
        ranked: list[tuple[int, int, int, float, str]] = []
        values: dict[tuple[int, int, int, float, str], str] = {}
        for artifact in artifacts:
            text = self._artifact_display_answer(artifact, max_chars=6000)
            if not text:
                continue
            key = (
                int(artifact.get("round_index", -1)),
                int(artifact.get("discussion_index", -1)),
                self._artifact_evidence_count(artifact),
                float(artifact.get("confidence", 0.5)),
                str(artifact.get("agent_id", "")),
            )
            ranked.append(key)
            values[key] = text
        if not ranked:
            return ""
        return values[sorted(ranked, reverse=True)[0]]

    @classmethod
    def _compact_tool_summary(cls, tools: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for tool in tools[:8]:
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            params = tool.get("parameters", {})
            properties = params.get("properties", {}) if isinstance(params, dict) else {}
            required = params.get("required", []) if isinstance(params, dict) else []
            arg_parts: list[str] = []
            if isinstance(properties, dict):
                for key, spec in list(properties.items())[:3]:
                    if not isinstance(spec, dict):
                        continue
                    type_name = str(spec.get("type", "any"))
                    required_marker = "*" if key in required else ""
                    arg_parts.append(f"{key}{required_marker}:{type_name}")
            signature = f"{name}({', '.join(arg_parts)})" if arg_parts else f"{name}()"
            description = re.sub(r"\s+", " ", str(tool.get("description", ""))).strip()
            if description:
                description = description[:120]
                lines.append(f"- {signature}: {description}")
            else:
                lines.append(f"- {signature}")
        return lines

    @classmethod
    def _serialize_tools(cls, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for tool in tools:
            serialized.append(
                {
                    "name": str(tool.get("name", "")),
                    "description": str(tool.get("description", "")),
                    "parameters": cls._serialize_for_json(tool.get("parameters", {})),
                }
            )
        return serialized

    @classmethod
    def _serialize_tool_records(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for record in records:
            serialized.append(
                {
                    "tool_name": str(record.get("tool_name", "")),
                    "arguments": cls._serialize_for_json(record.get("arguments", {})),
                    "status": str(record.get("status", "")),
                    "error": cls._serialize_for_json(record.get("error")),
                    "output_preview": cls._tool_output_preview(record.get("output")),
                }
            )
        return serialized

    @classmethod
    def _serialize_for_json(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._serialize_for_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._serialize_for_json(item) for item in value]
        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)

    @staticmethod
    def _tool_output_preview(value: Any) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:260]

    @staticmethod
    def _agent_order(layout: TopologyLayout, agent_id: str) -> int:
        if agent_id == "system":
            return -1
        if agent_id == "descriptor":
            return -2
        try:
            return layout.agent_ids.index(agent_id)
        except ValueError:
            return len(layout.agent_ids) + 1

    @staticmethod
    def _message_snippet(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())[:260]

    @staticmethod
    def _materialize_trace_events(payloads: list[dict[str, Any]]) -> list[TraceEvent]:
        ordered = sorted(payloads, key=lambda item: tuple(item.get("_sort", (0, 0, 0, 0))))
        clock = _EventClock()
        events: list[TraceEvent] = []
        for item in ordered:
            start, end = clock.span(float(item.get("latency_ms", 1.0)))
            events.append(
                TraceEvent(
                    timestamp_start=start,
                    timestamp_end=end,
                    actor=str(item.get("actor", "system")),
                    event_type=str(item.get("event_type", "act")),
                    payload=dict(item.get("payload", {})),
                    token_in=int(item.get("token_in", 0)),
                    token_out=int(item.get("token_out", 0)),
                    latency_ms=float(item.get("latency_ms", 0.0)),
                    cost_usd=float(item.get("cost_usd", 0.0)),
                    state_id=str(item.get("state_id", "")),
                )
            )
        return events

    def _count_tool_calls(self, state: WorkflowState) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in state.get("tool_records_log", []):
            name = str(record.get("tool_name", ""))
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1
        inter_agent_count = len([message for message in state.get("messages", []) if message.get("sender") != "system"])
        if inter_agent_count:
            counts["inter_agent_send"] = inter_agent_count
        return counts

    def _collect_retrieved_docids(self, state: WorkflowState) -> list[str]:
        docids: set[str] = set()
        for record in state.get("tool_records_log", []):
            tool_name = str(record.get("tool_name", ""))
            arguments = record.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            docids.update(
                self._extract_docids_from_tool_output(
                    output=record.get("output"),
                    arguments=arguments,
                    tool_name=tool_name,
                )
            )
        if not docids:
            docids.update(self._extract_docids(str(state.get("final_answer", ""))))
        return sorted(docid for docid in docids if docid and docid != "None")

    @staticmethod
    def _is_retrieval_tool(tool_name: str) -> bool:
        name = tool_name.lower().strip()
        return "search" in name or "retriev" in name or "document" in name

    @classmethod
    def _extract_docids_from_tool_output(
        cls,
        *,
        output: Any,
        arguments: dict[str, Any],
        tool_name: str,
    ) -> list[str]:
        out: set[str] = set()
        if "docid" in arguments:
            out.add(str(arguments.get("docid")))

        if isinstance(output, dict):
            if "docid" in output:
                out.add(str(output["docid"]))
            payload = [output]
        elif isinstance(output, list):
            payload = output
        elif isinstance(output, str):
            payload = []
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    payload = [parsed]
                elif isinstance(parsed, list):
                    payload = parsed
            except Exception:
                for docid in re.findall(r'"docid"\s*:\s*"([^"]+)"', output):
                    out.add(str(docid))
                for docid in re.findall(r'"docid"\s*:\s*(\d+)', output):
                    out.add(str(docid))
        else:
            payload = []

        for item in payload:
            if isinstance(item, dict) and "docid" in item:
                out.add(str(item["docid"]))

        if not out and cls._is_retrieval_tool(tool_name):
            text = output if isinstance(output, str) else ""
            out.update(cls._extract_docids(text))

        return sorted(docid for docid in out if docid and docid != "None")

    @staticmethod
    def _extract_docids(text: str) -> list[str]:
        single = re.findall(r"\[(\d+)\]", text)
        single_full = re.findall(r"【(\d+)】", text)
        grouped = re.findall(r"\[([^\[\]]+?)\]", text)
        grouped_full = re.findall(r"【([^【】]+?)】", text)

        docids = set(single)
        docids.update(single_full)
        for group in grouped + grouped_full:
            docids.update(re.findall(r"\d+", group))
        return sorted(docids)

    @classmethod
    def build_topology_visual_graph(cls, spec: ExperimentSpec) -> tuple[TopologyLayout, Graph]:
        resolved = spec.normalized()
        layout = build_layout(
            topology=resolved.topology,
            num_agents=resolved.num_agents,
            agents_per_level=resolved.agents_per_level,
            group_sizes=resolved.group_sizes,
        )
        nodes = {
            agent_id: Node(
                id=agent_id,
                name=f"{agent_id}\\n{layout.roles.get(agent_id, 'agent')}",
                data=None,
                metadata={"role": layout.roles.get(agent_id, "agent")},
            )
            for agent_id in layout.agent_ids
        }
        edges: list[Edge] = []
        seen: set[tuple[str, str]] = set()
        for source, neighbors in layout.adjacency.items():
            for target in neighbors:
                key = tuple(sorted((source, target)))
                if key in seen:
                    continue
                seen.add(key)
                label = cls._topology_edge_label(layout, source, target)
                edges.append(Edge(source=source, target=target, data=label))
        return layout, Graph(nodes=nodes, edges=edges)

    @classmethod
    def build_workflow_visual_graph(cls, spec: ExperimentSpec) -> tuple[WorkflowDocumentation, Graph]:
        resolved = spec.normalized()
        workflow = cls._workflow_definition(resolved.topology)
        nodes = {
            "START": Node(
                id="START",
                name="START",
                data=None,
                metadata={"kind": "terminal", "description": "Workflow entrypoint."},
            ),
            "END": Node(
                id="END",
                name="END",
                data=None,
                metadata={"kind": "terminal", "description": "Workflow exit."},
            ),
        }
        for node_id, description in workflow.nodes.items():
            nodes[node_id] = Node(
                id=node_id,
                name=node_id,
                data=None,
                metadata={"kind": "workflow_node", "description": description},
            )
        return workflow, Graph(nodes=nodes, edges=cls._workflow_edges_from_documentation(workflow))

    @classmethod
    def render_topology_mermaid(cls, spec: ExperimentSpec) -> tuple[TopologyLayout, str]:
        layout, graph = cls.build_topology_visual_graph(spec)
        return layout, graph.draw_mermaid()

    @staticmethod
    def _topology_edge_label(layout: TopologyLayout, source: str, target: str) -> str:
        if layout.parent_by_agent.get(source) == target or layout.parent_by_agent.get(target) == source:
            return "hierarchy"
        if source == layout.orchestrator_id or target == layout.orchestrator_id:
            return "relay"
        if layout.groups:
            left_group = LangGraphMASEngine._group_for_agent(layout, source)
            right_group = LangGraphMASEngine._group_for_agent(layout, target)
            if left_group and right_group and left_group == right_group:
                return "group"
        return "peer"

    @classmethod
    def _workflow_edges_from_documentation(cls, workflow: WorkflowDocumentation) -> list[Edge]:
        edges: list[Edge] = []
        seen: set[tuple[str, str, str]] = set()
        for spec in [*workflow.edges, *workflow.conditional_edges]:
            for edge in cls._parse_workflow_edge_spec(spec):
                key = (edge.source, edge.target, str(edge.data or ""))
                if key in seen:
                    continue
                seen.add(key)
                edges.append(edge)
        return edges

    @staticmethod
    def _parse_workflow_edge_spec(spec: str) -> list[Edge]:
        text = str(spec or "").strip()
        if "->" not in text:
            return []

        source, remainder = text.split("->", 1)
        source = source.strip()
        remainder = remainder.strip()
        edges: list[Edge] = []

        if " else " in remainder:
            primary, fallback = remainder.split(" else ", 1)
            primary_target, primary_label = LangGraphMASEngine._split_workflow_target(primary.strip())
            fallback_target, fallback_label = LangGraphMASEngine._split_workflow_target(
                fallback.strip()
            )
            edges.append(
                Edge(
                    source=source,
                    target=primary_target,
                    data=primary_label or "conditional",
                )
            )
            edges.append(
                Edge(
                    source=source,
                    target=fallback_target,
                    data=fallback_label or "else",
                )
            )
            return edges

        if "|" in remainder:
            for chunk in remainder.split("|"):
                target, label = LangGraphMASEngine._split_workflow_target(chunk.strip())
                edges.append(
                    Edge(
                        source=source,
                        target=target,
                        data=label or "conditional",
                    )
                )
            return edges

        target, label = LangGraphMASEngine._split_workflow_target(remainder)
        edges.append(Edge(source=source, target=target, data=label))
        return edges

    @staticmethod
    def _split_workflow_target(text: str) -> tuple[str, str | None]:
        match = re.match(r"(.+?)\s*\((.+)\)\s*$", text)
        if match is None:
            return text.strip(), None
        return match.group(1).strip(), match.group(2).strip()

    @staticmethod
    def _workflow_definition(topology: str) -> WorkflowDocumentation:
        if topology == TOPOLOGY_SAS:
            return WorkflowDocumentation(
                topology=topology,
                nodes={
                    "single_agent": "One end-to-end worker.",
                    "descriptor_monitor": "Shared descriptor hook.",
                    "finalize": "Emit the final answer and final trace event.",
                },
                edges=[
                    "START -> single_agent",
                    "single_agent -> descriptor_monitor",
                    "descriptor_monitor -> finalize",
                    "finalize -> END",
                ],
                conditional_edges=[],
                dispatch_logic="No dispatch. One agent handles the task directly.",
                aggregation_logic="No aggregation. The single artifact becomes the final answer.",
                stopping_criteria=["Single-turn topology. It terminates immediately after `single_agent`."],
                state_fields=SHARED_STATE_FIELDS,
                logging_outputs=COMMON_LOG_OUTPUTS,
            )
        if topology == TOPOLOGY_ONLY_VOTING:
            return WorkflowDocumentation(
                topology=topology,
                nodes={
                    "dispatch_independent_agents": "Selects all workers for independent solving.",
                    "worker": "Independent worker branch.",
                    "voter": "Configurable final voter. Default is an LLM judge with deterministic fallback.",
                    "descriptor_monitor": "Shared descriptor hook.",
                    "finalize": "Finalize the run.",
                },
                edges=[
                    "START -> dispatch_independent_agents",
                    "worker -> voter",
                    "voter -> descriptor_monitor",
                    "descriptor_monitor -> finalize",
                    "finalize -> END",
                ],
                conditional_edges=[
                    "dispatch_independent_agents -> worker (one Send per active agent) else voter",
                ],
                dispatch_logic="Fan out to every worker with no peer packets.",
                aggregation_logic="`voter` either uses an LLM judge to select the best final answer or falls back to deterministic voting with explicit tie-breaks.",
                stopping_criteria=["Single-turn topology. It stops after the voter."],
                state_fields=SHARED_STATE_FIELDS,
                logging_outputs=COMMON_LOG_OUTPUTS,
            )
        if topology == TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION:
            return WorkflowDocumentation(
                topology=topology,
                nodes={
                    "orchestrator_plan": "Initial planner/orchestrator artifact.",
                    "dispatch_specialists": "Build task packages for specialists.",
                    "specialist_worker": "Independent specialist execution.",
                    "relay_specialist_reports": "Bounded specialist-to-orchestrator reports.",
                    "orchestrator_merge": "Merge specialist reports.",
                    "descriptor_monitor": "Shared descriptor hook.",
                    "termination_checker": "Explicit round stopper.",
                    "finalize": "Finalize the run.",
                },
                edges=[
                    "START -> orchestrator_plan",
                    "orchestrator_plan -> dispatch_specialists",
                    "specialist_worker -> relay_specialist_reports",
                    "relay_specialist_reports -> orchestrator_merge",
                    "orchestrator_merge -> descriptor_monitor",
                    "descriptor_monitor -> termination_checker",
                    "finalize -> END",
                ],
                conditional_edges=[
                    "dispatch_specialists -> specialist_worker (parallel Send fan-out) else relay_specialist_reports",
                    "termination_checker -> dispatch_specialists | finalize",
                ],
                dispatch_logic="The orchestrator emits bounded task packages to the activated specialists.",
                aggregation_logic="Specialist artifacts are relayed upward as reports; the orchestrator merges those reports into one artifact.",
                stopping_criteria=[
                    "Max rounds reached",
                    "Consensus across specialist artifacts",
                    "Semantic no-progress across orchestrator merge artifacts",
                    "Invalid or failed branch fallback",
                ],
                state_fields=SHARED_STATE_FIELDS,
                logging_outputs=COMMON_LOG_OUTPUTS,
            )
        if topology == TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION:
            return WorkflowDocumentation(
                topology=topology,
                nodes={
                    "orchestrator_plan": "Initial orchestrator planner.",
                    "dispatch_specialists": "Task package fan-out.",
                    "specialists_initial_round": "Initial specialist artifacts.",
                    "relay_specialist_reports": "Specialist-to-orchestrator reports.",
                    "orchestrator_relay": "Mediated peer-summary relay plus discussion stopping logic.",
                    "dispatch_revision_round": "Revision-round fan-out.",
                    "specialists_revision_round": "Specialist revision stage.",
                    "orchestrator_merge": "Final orchestrator synthesis for the current cycle.",
                    "descriptor_monitor": "Shared descriptor hook.",
                    "cycle_termination_checker": "Outer-cycle stopper.",
                    "finalize": "Finalize the run.",
                },
                edges=[
                    "START -> orchestrator_plan",
                    "orchestrator_plan -> dispatch_specialists",
                    "specialists_initial_round -> relay_specialist_reports",
                    "specialists_revision_round -> relay_specialist_reports",
                    "relay_specialist_reports -> orchestrator_relay",
                    "orchestrator_merge -> descriptor_monitor",
                    "descriptor_monitor -> cycle_termination_checker",
                    "finalize -> END",
                ],
                conditional_edges=[
                    "dispatch_specialists -> specialists_initial_round (parallel Send fan-out) else relay_specialist_reports",
                    "orchestrator_relay -> dispatch_revision_round | orchestrator_merge",
                    "cycle_termination_checker -> dispatch_specialists | finalize",
                ],
                dispatch_logic="The orchestrator sends initial task packages, then bounded peer-summary bundles for revision rounds.",
                aggregation_logic="Specialists never see raw peer transcripts; only orchestrator-generated peer summaries are routed. The orchestrator performs the final synthesis each outer cycle.",
                stopping_criteria=[
                    "Max discussion rounds reached",
                    "Consensus across latest specialist artifacts",
                    "No meaningful change between revisions",
                    "Confidence threshold reached",
                    "Invalid or failed branch fallback",
                    "Max outer cycles reached",
                ],
                state_fields=SHARED_STATE_FIELDS,
                logging_outputs=COMMON_LOG_OUTPUTS,
            )
        if topology == TOPOLOGY_ORCHESTRATOR_TREE:
            return WorkflowDocumentation(
                topology=topology,
                nodes={
                    "root_plan": "Root planner artifact.",
                    "manager_dispatch": "Root-to-manager routing.",
                    "manager_nodes": "Manager decomposition stage.",
                    "worker_dispatch": "Manager-to-leaf routing.",
                    "worker_nodes": "Leaf execution stage.",
                    "worker_relay": "Leaf-to-manager bounded reports.",
                    "manager_reduce_dispatch": "Selects managers for reduction.",
                    "manager_reducers": "Per-manager aggregation.",
                    "manager_relay": "Manager-to-root bounded reports.",
                    "root_reducer": "Root aggregation.",
                    "descriptor_monitor": "Shared descriptor hook.",
                    "termination_checker": "Explicit cycle stopper.",
                    "finalize": "Finalize the run.",
                },
                edges=[
                    "START -> root_plan",
                    "root_plan -> manager_dispatch",
                    "manager_nodes -> worker_dispatch",
                    "worker_nodes -> worker_relay",
                    "worker_relay -> manager_reduce_dispatch",
                    "manager_reducers -> manager_relay",
                    "manager_relay -> root_reducer",
                    "root_reducer -> descriptor_monitor",
                    "descriptor_monitor -> termination_checker",
                    "finalize -> END",
                ],
                conditional_edges=[
                    "manager_dispatch -> manager_nodes (parallel Send fan-out) else worker_dispatch",
                    "worker_dispatch -> worker_nodes (parallel Send fan-out) else worker_relay",
                    "manager_reduce_dispatch -> manager_reducers (parallel Send fan-out) else manager_relay",
                    "termination_checker -> manager_dispatch | finalize",
                ],
                dispatch_logic="Root routes only to managers, managers route only to their own children, and aggregation always follows the parent-child DAG upward.",
                aggregation_logic="Leaf artifacts are refined upward one level at a time. Internal nodes aggregate only their direct children. No sibling or backflow exchange is allowed.",
                stopping_criteria=[
                    "Max rounds reached",
                    "Consensus across manager reducer artifacts",
                    "Semantic no-progress across root artifacts",
                    "Invalid or failed branch fallback",
                ],
                state_fields=SHARED_STATE_FIELDS,
                logging_outputs=COMMON_LOG_OUTPUTS,
            )
        if topology == TOPOLOGY_FULLY_LINKED_DEBATE:
            return WorkflowDocumentation(
                topology=topology,
                nodes={
                    "debate_init": "Initialize the debate controller.",
                    "debate_controller": "Round controller plus explicit stopping logic.",
                    "debate_dispatch": "Fan out the next debate round.",
                    "debate_round": "Parallel debater revision stage.",
                    "judge": "Configurable final judge. Default is an LLM judge with deterministic fallback.",
                    "descriptor_monitor": "Shared descriptor hook.",
                    "finalize": "Finalize the run.",
                },
                edges=[
                    "START -> debate_init",
                    "debate_init -> debate_controller",
                    "debate_round -> debate_controller",
                    "judge -> descriptor_monitor",
                    "descriptor_monitor -> finalize",
                    "finalize -> END",
                ],
                conditional_edges=[
                    "debate_controller -> debate_dispatch | judge",
                    "debate_dispatch -> debate_round (parallel Send fan-out) else debate_controller",
                ],
                dispatch_logic="The controller sends bounded summaries of each debater artifact to all peers for the next round.",
                aggregation_logic="Debaters revise in parallel; the final judge uses an LLM judge by default and falls back to deterministic voting when needed.",
                stopping_criteria=[
                    "Max rounds reached",
                    "Consensus across debater artifacts",
                    "No meaningful change between debate rounds",
                    "Confidence threshold reached",
                    "Invalid or failed branch fallback",
                ],
                state_fields=SHARED_STATE_FIELDS,
                logging_outputs=COMMON_LOG_OUTPUTS,
            )
        if topology == TOPOLOGY_GROUP_CHAT_DEBATE:
            return WorkflowDocumentation(
                topology=topology,
                nodes={
                    "group_dispatch": "Fan out the intra-group debate stage.",
                    "group_debate_round": "Parallel group debate workers.",
                    "group_controller": "Intra-group controller plus stop/merge logic.",
                    "representative_dispatch": "Representative fan-out.",
                    "representative_merge": "Representative merge/debate stage.",
                    "representative_controller": "Representative-level controller plus stopping logic.",
                    "final_judge": "Configurable final judge. Default is an LLM judge with deterministic fallback.",
                    "descriptor_monitor": "Shared descriptor hook.",
                    "finalize": "Finalize the run.",
                },
                edges=[
                    "START -> group_dispatch",
                    "group_debate_round -> group_controller",
                    "representative_merge -> representative_controller",
                    "final_judge -> descriptor_monitor",
                    "descriptor_monitor -> finalize",
                    "finalize -> END",
                ],
                conditional_edges=[
                    "group_dispatch -> group_debate_round (parallel Send fan-out) else group_controller",
                    "group_controller -> group_dispatch | representative_dispatch",
                    "representative_dispatch -> representative_merge (parallel Send fan-out) else final_judge",
                    "representative_controller -> representative_dispatch | final_judge",
                ],
                dispatch_logic="Debate is local to groups until the controller emits representative-only summaries. Inter-group traffic is restricted to representative packets.",
                aggregation_logic="Group members debate locally, representatives merge group summaries, and a shared final judge selects the final answer with an LLM judge by default and deterministic fallback.",
                stopping_criteria=[
                    "Max group rounds reached",
                    "Consensus inside groups",
                    "No meaningful change between group or representative revisions",
                    "Confidence threshold reached",
                    "Invalid or failed branch fallback",
                    "Max representative rounds reached",
                ],
                state_fields=SHARED_STATE_FIELDS,
                logging_outputs=COMMON_LOG_OUTPUTS,
            )
        raise ValueError(f"Unhandled topology '{topology}'")
