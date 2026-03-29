from __future__ import annotations

import json
import operator
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from langchain_core.runnables.graph import Edge, Graph, Node
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from descriptor.schema import TraceEvent

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
    extract_topology_messages_for_agent,
    normalize_topology_name,
    vote_majority,
)


class RuntimeState(TypedDict, total=False):
    task_id: str
    task_prompt: Any
    reference_answer: str
    task_metadata: dict[str, Any]
    run_index: int
    seed: int

    topology: str
    rounds: int
    discussion_rounds: int
    phase: str
    phase_iteration: int
    round_index: int
    done: bool

    layout: TopologyLayout
    dispatch_id: int
    active_agents: list[str]
    active_agent: str

    llm_client: OpenRouterLLMClient
    agent_type_by_agent: dict[str, str]
    tools: list[dict[str, Any]]
    max_tool_iterations: int

    outputs: Annotated[list[dict[str, Any]], operator.add]
    messages: Annotated[list[dict[str, Any]], operator.add]
    message_views: Annotated[list[dict[str, Any]], operator.add]
    trace_payloads: Annotated[list[dict[str, Any]], operator.add]
    phase_history: Annotated[list[dict[str, Any]], operator.add]
    descriptor_records: Annotated[list[dict[str, Any]], operator.add]
    interaction_logs: Annotated[list[dict[str, Any]], operator.add]

    latest_outputs: dict[str, str]
    final_answer: str
    final_reason: str
    vote_tally: dict[str, int]

    message_budget: dict[str, int]
    sent_counts: dict[str, int]
    message_seq: int

    retrieved_docids: list[str]
    tool_call_counts: dict[str, int]

    descriptor: DescriptorHook
    descriptor_summary: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSpec:
    topology: str
    num_agents: int
    rounds: int
    discussion_rounds: int = 1
    communication_budget_per_agent: int = 1
    agents_per_level: list[int] | None = None
    group_sizes: list[int] | None = None

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
        return ExperimentSpec(
            topology=topology,
            num_agents=num_agents,
            rounds=rounds,
            discussion_rounds=discussion_rounds,
            communication_budget_per_agent=budget,
            agents_per_level=(
                list(self.agents_per_level) if self.agents_per_level is not None else None
            ),
            group_sizes=(list(self.group_sizes) if self.group_sizes is not None else None),
        )


@dataclass
class LangGraphRunResult:
    final_answer: str
    trace_events: list[TraceEvent]
    run_metadata: dict[str, Any] = field(default_factory=dict)


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
    """LangGraph-based MAS execution engine with topology-aware message relay."""

    def __init__(self, llm_client: OpenRouterLLMClient) -> None:
        self.llm_client = llm_client
        self.graph = self._build_graph()

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
        agent_type_by_agent = {
            agent_id: agent_types[idx % len(agent_types)] for idx, agent_id in enumerate(layout.agent_ids)
        }

        phase = self._initial_phase(layout.topology)
        descriptor_hook = descriptor or NullDescriptor()
        state: RuntimeState = {
            "task_id": str(getattr(task, "task_id", "task")),
            "task_prompt": getattr(task, "prompt", ""),
            "reference_answer": str(getattr(task, "reference_answer", "")),
            "task_metadata": dict(getattr(task, "metadata", {}) or {}),
            "run_index": int(run_index),
            "seed": int(seed),
            "topology": layout.topology,
            "rounds": int(spec.rounds),
            "discussion_rounds": int(spec.discussion_rounds),
            "phase": phase,
            "phase_iteration": 0,
            "round_index": 0,
            "done": False,
            "layout": layout,
            "dispatch_id": -1,
            "active_agents": [],
            "llm_client": self.llm_client,
            "agent_type_by_agent": agent_type_by_agent,
            "tools": list(tools or []),
            "max_tool_iterations": max(1, int(max_tool_iterations)),
            "outputs": [],
            "messages": [],
            "message_views": [],
            "trace_payloads": [],
            "phase_history": [],
            "descriptor_records": [],
            "interaction_logs": [],
            "latest_outputs": {},
            "final_answer": "",
            "final_reason": "",
            "vote_tally": {},
            "message_budget": {
                agent_id: int(spec.communication_budget_per_agent) for agent_id in layout.agent_ids
            },
            "sent_counts": {agent_id: 0 for agent_id in layout.agent_ids},
            "message_seq": 0,
            "retrieved_docids": [],
            "tool_call_counts": {},
            "descriptor": descriptor_hook,
            "descriptor_summary": {},
        }

        end_state = self.graph.invoke(state)

        trace_events = self._materialize_trace_events(end_state.get("trace_payloads", []))
        final_answer = str(end_state.get("final_answer") or "")
        run_metadata = {
            "task_id": end_state.get("task_id"),
            "run_index": run_index,
            "seed": seed,
            "topology": layout.topology,
            "rounds_configured": spec.rounds,
            "discussion_rounds": spec.discussion_rounds,
            "turns_executed": int(end_state.get("round_index", 0)) + 1,
            "messages_sent_total": sum(end_state.get("sent_counts", {}).values()),
            "messages_sent_by_agent": dict(end_state.get("sent_counts", {})),
            "tool_call_counts": dict(end_state.get("tool_call_counts", {})),
            "tool_calls_total": int(sum(end_state.get("tool_call_counts", {}).values())),
            "remaining_message_budget": dict(end_state.get("message_budget", {})),
            "tool_definitions": self._serialize_tools(list(end_state.get("tools", []))),
            "retrieved_docids": list(end_state.get("retrieved_docids", [])),
            "agent_outputs": dict(end_state.get("latest_outputs", {})),
            "vote_tally": dict(end_state.get("vote_tally", {})),
            "phase_history": list(end_state.get("phase_history", [])),
            "relay_messages": list(end_state.get("messages", [])),
            "message_views": list(end_state.get("message_views", [])),
            "interaction_logs": list(end_state.get("interaction_logs", [])),
            "descriptor_records": list(end_state.get("descriptor_records", [])),
            "descriptor_summary": dict(end_state.get("descriptor_summary", {})),
            "topology_layout": layout.to_payload(),
            "final_reason": str(end_state.get("final_reason", "")),
        }

        return LangGraphRunResult(
            final_answer=final_answer,
            trace_events=trace_events,
            run_metadata=run_metadata,
        )

    def _build_graph(self):
        graph = StateGraph(RuntimeState)

        graph.add_node("plan", self._plan_node)
        graph.add_node("dispatch", self._dispatch_node)
        graph.add_node("agent_step", self._agent_step_node)
        graph.add_node("collect", self._collect_node)
        graph.add_node("descriptor_monitor", self._descriptor_monitor_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "dispatch")
        graph.add_conditional_edges(
            "dispatch",
            self._fan_out_or_collect,
            ["agent_step", "collect"],
        )
        graph.add_edge("agent_step", "collect")
        graph.add_edge("collect", "descriptor_monitor")
        graph.add_conditional_edges(
            "descriptor_monitor",
            self._continue_or_finalize,
            {
                "dispatch": "dispatch",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)

        return graph.compile()

    def _plan_node(self, state: RuntimeState) -> dict[str, Any]:
        event = self._draft_event(
            dispatch_id=-1,
            node_order=0,
            agent_order=-1,
            event_order=0,
            actor="system",
            event_type="plan",
            payload={
                "task_id": state["task_id"],
                "topology": state["topology"],
                "rounds": state["rounds"],
                "discussion_rounds": state["discussion_rounds"],
                "layout": state["layout"].to_payload(),
                "tools": [str(item.get("name", "")) for item in state.get("tools", [])],
                "tool_definitions": self._serialize_tools(list(state.get("tools", []))),
            },
            token_in=0,
            token_out=0,
            latency_ms=1.0,
            cost_usd=0.0,
            state_id=f"run_{state['run_index']}_plan",
        )
        return {"trace_payloads": [event]}

    def _dispatch_node(self, state: RuntimeState) -> dict[str, Any]:
        dispatch_id = int(state.get("dispatch_id", -1)) + 1
        active_agents = self._active_agents_for_phase(state)

        event = self._draft_event(
            dispatch_id=dispatch_id,
            node_order=0,
            agent_order=-1,
            event_order=0,
            actor="system",
            event_type="revise",
            payload={
                "phase": state.get("phase"),
                "phase_iteration": int(state.get("phase_iteration", 0)),
                "round_index": int(state.get("round_index", 0)),
                "active_agents": list(active_agents),
            },
            token_in=0,
            token_out=0,
            latency_ms=1.0,
            cost_usd=0.0,
            state_id=(
                f"run_{state['run_index']}_dispatch_{dispatch_id}_"
                f"{state.get('phase')}_{state.get('round_index')}"
            ),
        )

        return {
            "dispatch_id": dispatch_id,
            "active_agents": active_agents,
            "trace_payloads": [event],
        }

    def _fan_out_or_collect(self, state: RuntimeState):
        active_agents = list(state.get("active_agents", []))
        if not active_agents:
            return "collect"
        dispatch_id = int(state.get("dispatch_id", -1))
        shared = {
            "task_id": state.get("task_id"),
            "task_prompt": state.get("task_prompt"),
            "run_index": state.get("run_index"),
            "topology": state.get("topology"),
            "phase": state.get("phase"),
            "round_index": state.get("round_index"),
            "layout": state.get("layout"),
            "messages": list(state.get("messages", [])),
            "llm_client": state.get("llm_client"),
            "agent_type_by_agent": dict(state.get("agent_type_by_agent", {})),
            "tools": list(state.get("tools", [])),
            "max_tool_iterations": state.get("max_tool_iterations", 8),
        }
        return [
            Send(
                "agent_step",
                {
                    **shared,
                    "active_agent": agent_id,
                    "dispatch_id": dispatch_id,
                },
            )
            for agent_id in active_agents
        ]

    def _agent_step_node(self, state: RuntimeState) -> dict[str, Any]:
        agent_id = str(state.get("active_agent", ""))
        dispatch_id = int(state.get("dispatch_id", -1))
        layout = state["layout"]
        role = layout.roles.get(agent_id, "agent")
        phase = str(state.get("phase", ""))
        round_index = int(state.get("round_index", 0))

        visible_messages = extract_topology_messages_for_agent(
            topology=state["topology"],
            phase=phase,
            round_index=round_index,
            agent_id=agent_id,
            messages=list(state.get("messages", [])),
            layout=layout,
        )

        prompt = self._build_agent_prompt(
            task_prompt=state.get("task_prompt"),
            agent_id=agent_id,
            role=role,
            phase=phase,
            round_index=round_index,
            visible_messages=visible_messages,
        )
        prompt_messages = self._normalize_prompt_messages(prompt)

        agent_type = state["agent_type_by_agent"].get(agent_id, "general")

        t0 = time.perf_counter()
        llm = state["llm_client"].generate(
            prompt=prompt,
            agent_type=agent_type,
            task_id=state["task_id"],
            run_index=int(state.get("run_index", 0)),
            agent_id=agent_id,
            tools=state.get("tools", []),
            max_tool_iterations=int(state.get("max_tool_iterations", 8)),
            temperature=0.0,
        )
        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1.0)

        tool_records = list(llm.tool_calls)
        if state.get("tools") and not tool_records:
            tool_records = self._mock_tool_plan(prompt, list(state.get("tools", [])))

        agent_order = self._agent_order(layout, agent_id)
        event_payloads: list[dict[str, Any]] = []

        view_record = {
            "dispatch_id": dispatch_id,
            "viewer": agent_id,
            "role": role,
            "phase": phase,
            "round_index": round_index,
            "visible_message_ids": [str(item.get("message_id", "")) for item in visible_messages],
            "visible_senders": sorted({str(item.get("sender", "")) for item in visible_messages}),
            "visible_count": len(visible_messages),
        }

        event_payloads.append(
            self._draft_event(
                dispatch_id=dispatch_id,
                node_order=1,
                agent_order=agent_order,
                event_order=0,
                actor=agent_id,
                event_type="verify",
                payload={
                    "phase": phase,
                    "round_index": round_index,
                    "visible_message_count": len(visible_messages),
                    "visible_message_ids": [
                        str(item.get("message_id", "")) for item in visible_messages
                    ][:20],
                    "visible_messages": self._serialize_for_json(visible_messages),
                    "prompt_messages": prompt_messages,
                },
                token_in=0,
                token_out=0,
                latency_ms=1.0,
                cost_usd=0.0,
                state_id=(
                    f"run_{state['run_index']}_dispatch_{dispatch_id}_"
                    f"{agent_id}_view"
                ),
            )
        )

        event_index = 1
        for record in tool_records:
            tool_name = str(record.get("tool_name") or "")
            if not tool_name:
                continue
            arguments = record.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            status = str(record.get("status") or "completed")
            error = record.get("error")
            output = record.get("output")

            event_payloads.append(
                self._draft_event(
                    dispatch_id=dispatch_id,
                    node_order=1,
                    agent_order=agent_order,
                    event_order=event_index,
                    actor=agent_id,
                    event_type="tool_call",
                    payload={
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "status": status,
                        "phase": phase,
                        "round_index": round_index,
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=(
                        f"run_{state['run_index']}_dispatch_{dispatch_id}_"
                        f"{agent_id}_tool_{event_index}_call"
                    ),
                )
            )
            event_index += 1

            event_payloads.append(
                self._draft_event(
                    dispatch_id=dispatch_id,
                    node_order=1,
                    agent_order=agent_order,
                    event_order=event_index,
                    actor=agent_id,
                    event_type="tool_result",
                    payload={
                        "tool_name": tool_name,
                        "status": status,
                        "error": error,
                        "output_preview": self._tool_output_preview(output),
                        "phase": phase,
                        "round_index": round_index,
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=(
                        f"run_{state['run_index']}_dispatch_{dispatch_id}_"
                        f"{agent_id}_tool_{event_index}_result"
                    ),
                )
            )
            event_index += 1

        text = str(llm.text or "")
        event_payloads.append(
            self._draft_event(
                dispatch_id=dispatch_id,
                node_order=1,
                agent_order=agent_order,
                event_order=event_index,
                actor=agent_id,
                event_type="act",
                payload={
                    "phase": phase,
                    "round_index": round_index,
                    "model": llm.model,
                    "mock_used": llm.mock_used,
                    "agent_type": agent_type,
                    "response_preview": self._message_snippet(text),
                    "response_text": text,
                    "metadata": dict(llm.metadata),
                },
                token_in=int(llm.token_in),
                token_out=int(llm.token_out),
                latency_ms=latency_ms,
                cost_usd=float(llm.cost_usd),
                state_id=(
                    f"run_{state['run_index']}_dispatch_{dispatch_id}_"
                    f"{agent_id}_act"
                ),
            )
        )

        retrieved_docids: set[str] = set()
        for record in tool_records:
            tool_name = str(record.get("tool_name") or "")
            arguments = record.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            retrieved_docids.update(
                self._extract_docids_from_tool_output(
                    output=record.get("output"),
                    arguments=arguments,
                    tool_name=tool_name,
                )
            )

        output_record = {
            "dispatch_id": dispatch_id,
            "agent_id": agent_id,
            "phase": phase,
            "round_index": round_index,
            "text": text,
            "tool_records": tool_records,
            "retrieved_docids": sorted(retrieved_docids),
        }
        interaction_log = {
            "dispatch_id": dispatch_id,
            "agent_id": agent_id,
            "agent_role": role,
            "agent_type": agent_type,
            "phase": phase,
            "round_index": round_index,
            "prompt_messages": prompt_messages,
            "visible_messages": self._serialize_for_json(visible_messages),
            "assistant_message": {"role": "assistant", "content": text},
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
            "outputs": [output_record],
            "message_views": [view_record],
            "trace_payloads": event_payloads,
            "interaction_logs": [interaction_log],
        }

    def _collect_node(self, state: RuntimeState) -> dict[str, Any]:
        dispatch_id = int(state.get("dispatch_id", -1))
        phase = str(state.get("phase", ""))
        round_index = int(state.get("round_index", 0))
        layout = state["layout"]

        outputs = [
            item for item in state.get("outputs", []) if int(item.get("dispatch_id", -1)) == dispatch_id
        ]

        latest_outputs = dict(state.get("latest_outputs", {}))
        for item in outputs:
            latest_outputs[str(item.get("agent_id", ""))] = str(item.get("text", ""))

        tool_call_counts = dict(state.get("tool_call_counts", {}))
        retrieved_docids_set = set(str(docid) for docid in state.get("retrieved_docids", []))
        for item in outputs:
            for record in item.get("tool_records", []):
                tool_name = str(record.get("tool_name") or "")
                if not tool_name:
                    continue
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            for docid in item.get("retrieved_docids", []):
                retrieved_docids_set.add(str(docid))

        message_budget = dict(state.get("message_budget", {}))
        sent_counts = dict(state.get("sent_counts", {}))
        message_seq = int(state.get("message_seq", 0))
        new_messages: list[dict[str, Any]] = []
        relay_events: list[dict[str, Any]] = []

        def relay_send(
            *,
            sender: str,
            recipients: list[str],
            content: str,
            kind: str,
            relay_phase: str,
            relay_round: int,
        ) -> None:
            nonlocal message_seq

            deduped = sorted({item for item in recipients if item and item != sender})
            if not deduped:
                return

            if sender in message_budget:
                remaining = int(message_budget.get(sender, 0))
                if remaining <= 0:
                    return
                message_budget[sender] = remaining - 1
                sent_counts[sender] = int(sent_counts.get(sender, 0)) + 1

            message_seq += 1
            message = {
                "message_id": f"m_{message_seq}",
                "sender": sender,
                "recipients": deduped,
                "content": self._message_snippet(content),
                "kind": kind,
                "phase": relay_phase,
                "round": relay_round,
            }
            new_messages.append(message)

            tool_call_counts["inter_agent_send"] = tool_call_counts.get("inter_agent_send", 0) + 1
            relay_events.append(
                self._draft_event(
                    dispatch_id=dispatch_id,
                    node_order=2,
                    agent_order=self._agent_order(layout, sender),
                    event_order=len(relay_events) + 1,
                    actor=sender,
                    event_type="tool_call",
                    payload={
                        "tool_name": "inter_agent_send",
                        "to": deduped,
                        "kind": kind,
                        "phase": relay_phase,
                        "round_index": relay_round,
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=(
                        f"run_{state['run_index']}_dispatch_{dispatch_id}_"
                        f"relay_{message_seq}_call"
                    ),
                )
            )
            relay_events.append(
                self._draft_event(
                    dispatch_id=dispatch_id,
                    node_order=2,
                    agent_order=self._agent_order(layout, sender),
                    event_order=len(relay_events) + 1,
                    actor="system",
                    event_type="tool_result",
                    payload={
                        "tool_name": "inter_agent_send",
                        "status": "ok",
                        "message_id": message["message_id"],
                        "visible_to": deduped,
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=(
                        f"run_{state['run_index']}_dispatch_{dispatch_id}_"
                        f"relay_{message_seq}_result"
                    ),
                )
            )

        round_count = int(state.get("rounds", 1))
        discussion_rounds = int(state.get("discussion_rounds", 1))
        phase_iteration = int(state.get("phase_iteration", 0))
        final_answer = str(state.get("final_answer", ""))
        final_reason = str(state.get("final_reason", ""))
        vote_tally = dict(state.get("vote_tally", {}))
        done = bool(state.get("done", False))

        output_by_agent = {str(item.get("agent_id")): item for item in outputs}
        candidate_texts = [str(item.get("text", "")) for item in outputs]

        topology = state["topology"]
        next_phase = phase
        next_round = round_index
        next_iteration = phase_iteration

        if topology == TOPOLOGY_SAS:
            if candidate_texts:
                final_answer = candidate_texts[-1]
                final_reason = "single_agent"
            done = True

        elif topology == TOPOLOGY_ORCHESTRATOR_TREE:
            root = layout.orchestrator_id
            if phase == "root_delegate":
                root_text = str(output_by_agent.get(str(root), {}).get("text", ""))
                for manager in layout.managers:
                    relay_send(
                        sender=str(root),
                        recipients=[manager],
                        content=root_text,
                        kind="root_to_manager",
                        relay_phase=phase,
                        relay_round=round_index,
                    )
                next_phase = "manager_delegate"

            elif phase == "manager_delegate":
                for manager in layout.managers:
                    manager_text = str(output_by_agent.get(manager, {}).get("text", ""))
                    leaves = [
                        child
                        for child in layout.children_by_agent.get(manager, [])
                        if child in layout.leaves
                    ]
                    relay_send(
                        sender=manager,
                        recipients=leaves,
                        content=manager_text,
                        kind="manager_to_leaf",
                        relay_phase=phase,
                        relay_round=round_index,
                    )
                next_phase = "leaf_work"

            elif phase == "leaf_work":
                for leaf in layout.leaves:
                    leaf_text = str(output_by_agent.get(leaf, {}).get("text", ""))
                    parent = layout.parent_by_agent.get(leaf)
                    if parent:
                        relay_send(
                            sender=leaf,
                            recipients=[parent],
                            content=leaf_text,
                            kind="leaf_to_manager",
                            relay_phase=phase,
                            relay_round=round_index,
                        )
                next_phase = "manager_aggregate"

            elif phase == "manager_aggregate":
                for manager in layout.managers:
                    manager_text = str(output_by_agent.get(manager, {}).get("text", ""))
                    if root:
                        relay_send(
                            sender=manager,
                            recipients=[root],
                            content=manager_text,
                            kind="manager_to_root",
                            relay_phase=phase,
                            relay_round=round_index,
                        )
                if round_index + 1 < round_count:
                    next_phase = "root_delegate"
                    next_round = round_index + 1
                else:
                    next_phase = "root_finalize"

            elif phase == "root_finalize":
                root_text = str(output_by_agent.get(str(root), {}).get("text", ""))
                final_answer = root_text or self._fallback_answer(candidate_texts)
                final_reason = "tree_root_finalize"
                done = True

        elif topology == TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION:
            root = layout.orchestrator_id
            if phase == "specialist_solve":
                for specialist in layout.specialists:
                    text = str(output_by_agent.get(specialist, {}).get("text", ""))
                    if root:
                        relay_send(
                            sender=specialist,
                            recipients=[root],
                            content=text,
                            kind="specialist_to_orchestrator",
                            relay_phase=phase,
                            relay_round=round_index,
                        )
                next_phase = "orchestrator_synthesize"

            elif phase == "orchestrator_synthesize":
                root_text = str(output_by_agent.get(str(root), {}).get("text", ""))
                if round_index + 1 < round_count:
                    for specialist in layout.specialists:
                        relay_send(
                            sender=str(root),
                            recipients=[specialist],
                            content=root_text,
                            kind="orchestrator_feedback",
                            relay_phase=phase,
                            relay_round=round_index,
                        )
                    next_phase = "specialist_solve"
                    next_round = round_index + 1
                else:
                    final_answer = root_text or self._fallback_answer(candidate_texts)
                    final_reason = "orchestrator_no_discussion"
                    done = True

        elif topology == TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION:
            root = layout.orchestrator_id
            if phase == "initial_proposals":
                for specialist in layout.specialists:
                    text = str(output_by_agent.get(specialist, {}).get("text", ""))
                    if root:
                        relay_send(
                            sender=specialist,
                            recipients=[root],
                            content=text,
                            kind="proposal",
                            relay_phase=phase,
                            relay_round=round_index,
                        )

                for specialist in layout.specialists:
                    peers = [
                        str(output_by_agent[item].get("text", ""))
                        for item in layout.specialists
                        if item != specialist and item in output_by_agent
                    ]
                    if peers and root:
                        relay_send(
                            sender=str(root),
                            recipients=[specialist],
                            content=self._render_peer_bundle(peers),
                            kind="peer_bundle",
                            relay_phase=phase,
                            relay_round=round_index,
                        )
                next_phase = "peer_discussion"
                next_iteration = 0

            elif phase == "peer_discussion":
                for specialist in layout.specialists:
                    text = str(output_by_agent.get(specialist, {}).get("text", ""))
                    if root:
                        relay_send(
                            sender=specialist,
                            recipients=[root],
                            content=text,
                            kind="revision",
                            relay_phase=phase,
                            relay_round=round_index,
                        )

                if phase_iteration + 1 < discussion_rounds:
                    for specialist in layout.specialists:
                        peers = [
                            str(output_by_agent[item].get("text", ""))
                            for item in layout.specialists
                            if item != specialist and item in output_by_agent
                        ]
                        if peers and root:
                            relay_send(
                                sender=str(root),
                                recipients=[specialist],
                                content=self._render_peer_bundle(peers),
                                kind="peer_bundle",
                                relay_phase=phase,
                                relay_round=round_index,
                            )
                    next_phase = "peer_discussion"
                    next_iteration = phase_iteration + 1
                else:
                    next_phase = "orchestrator_synthesize"
                    next_iteration = 0

            elif phase == "orchestrator_synthesize":
                root_text = str(output_by_agent.get(str(root), {}).get("text", ""))
                if round_index + 1 < round_count:
                    for specialist in layout.specialists:
                        relay_send(
                            sender=str(root),
                            recipients=[specialist],
                            content=root_text,
                            kind="orchestrator_feedback",
                            relay_phase=phase,
                            relay_round=round_index,
                        )
                    next_phase = "initial_proposals"
                    next_round = round_index + 1
                else:
                    final_answer = root_text or self._fallback_answer(candidate_texts)
                    final_reason = "orchestrator_with_discussion"
                    done = True

        elif topology == TOPOLOGY_ONLY_VOTING:
            final_answer, vote_tally = vote_majority(candidate_texts)
            final_reason = "majority_vote"
            done = True

        elif topology == TOPOLOGY_FULLY_LINKED_DEBATE:
            if round_index + 1 < round_count:
                for item in outputs:
                    sender = str(item.get("agent_id", ""))
                    recipients = [agent_id for agent_id in layout.agent_ids if agent_id != sender]
                    relay_send(
                        sender=sender,
                        recipients=recipients,
                        content=str(item.get("text", "")),
                        kind="debate_round",
                        relay_phase=phase,
                        relay_round=round_index,
                    )
                next_round = round_index + 1
                next_phase = "debate"
            else:
                final_answer, vote_tally = vote_majority(candidate_texts)
                final_reason = "fully_linked_debate_vote"
                done = True

        elif topology == TOPOLOGY_GROUP_CHAT_DEBATE:
            if phase == "group_debate":
                if round_index + 1 < round_count:
                    for item in outputs:
                        sender = str(item.get("agent_id", ""))
                        group = self._group_for_agent(layout, sender)
                        recipients = [agent_id for agent_id in group if agent_id != sender]
                        relay_send(
                            sender=sender,
                            recipients=recipients,
                            content=str(item.get("text", "")),
                            kind="group_debate_round",
                            relay_phase=phase,
                            relay_round=round_index,
                        )
                    next_round = round_index + 1
                    next_phase = "group_debate"
                else:
                    for group in layout.groups:
                        if not group:
                            continue
                        representative = group[0]
                        group_outputs = [
                            str(output_by_agent[item].get("text", ""))
                            for item in group
                            if item in output_by_agent
                        ]
                        relay_send(
                            sender="system",
                            recipients=[representative],
                            content=self._render_group_summary(group_outputs),
                            kind="group_summary",
                            relay_phase=phase,
                            relay_round=round_index,
                        )
                    next_phase = "representative_merge"
                    next_iteration = 0

            elif phase == "representative_merge":
                representatives = list(layout.representatives)
                if phase_iteration + 1 < discussion_rounds and len(representatives) > 1:
                    for item in outputs:
                        sender = str(item.get("agent_id", ""))
                        recipients = [agent_id for agent_id in representatives if agent_id != sender]
                        relay_send(
                            sender=sender,
                            recipients=recipients,
                            content=str(item.get("text", "")),
                            kind="representative_debate_round",
                            relay_phase=phase,
                            relay_round=round_index,
                        )
                    next_phase = "representative_merge"
                    next_iteration = phase_iteration + 1
                else:
                    final_answer, vote_tally = vote_majority(candidate_texts)
                    final_reason = "group_chat_representative_merge"
                    done = True

        else:
            final_answer = self._fallback_answer(candidate_texts)
            final_reason = "fallback"
            done = True

        verify_event = self._draft_event(
            dispatch_id=dispatch_id,
            node_order=2,
            agent_order=-1,
            event_order=0,
            actor="system",
            event_type="verify",
            payload={
                "phase": phase,
                "round_index": round_index,
                "outputs_in_dispatch": len(outputs),
                "messages_created": len(new_messages),
                "next_phase": next_phase,
                "next_round_index": next_round,
            },
            token_in=0,
            token_out=0,
            latency_ms=1.0,
            cost_usd=0.0,
            state_id=(
                f"run_{state['run_index']}_dispatch_{dispatch_id}_collect_verify"
            ),
        )

        history_entry = {
            "dispatch_id": dispatch_id,
            "phase": phase,
            "round_index": round_index,
            "phase_iteration": phase_iteration,
            "active_agents": list(state.get("active_agents", [])),
            "output_agents": [str(item.get("agent_id", "")) for item in outputs],
            "messages_created": len(new_messages),
            "next_phase": next_phase,
            "next_round_index": next_round,
            "done": done,
        }

        updates: dict[str, Any] = {
            "messages": new_messages,
            "trace_payloads": [verify_event] + relay_events,
            "phase_history": [history_entry],
            "latest_outputs": latest_outputs,
            "tool_call_counts": tool_call_counts,
            "retrieved_docids": sorted(retrieved_docids_set),
            "message_budget": message_budget,
            "sent_counts": sent_counts,
            "message_seq": message_seq,
            "vote_tally": vote_tally,
            "phase": next_phase,
            "round_index": next_round,
            "phase_iteration": next_iteration,
            "done": done,
            "final_answer": final_answer,
            "final_reason": final_reason,
        }
        return updates

    def _descriptor_monitor_node(self, state: RuntimeState) -> dict[str, Any]:
        descriptor = state.get("descriptor")
        dispatch_id = int(state.get("dispatch_id", -1))
        snapshot = {
            "task_id": state.get("task_id"),
            "topology": state.get("topology"),
            "phase": state.get("phase"),
            "round_index": int(state.get("round_index", 0)),
            "dispatch_id": dispatch_id,
            "message_count": len(state.get("messages", [])),
            "outputs_seen": len(state.get("outputs", [])),
            "done": bool(state.get("done", False)),
        }

        record: dict[str, Any] = {}
        if descriptor is not None and hasattr(descriptor, "on_monitor"):
            emitted = descriptor.on_monitor(dict(snapshot))
            if isinstance(emitted, dict):
                record = dict(emitted)

        event = self._draft_event(
            dispatch_id=dispatch_id,
            node_order=3,
            agent_order=-1,
            event_order=0,
            actor="descriptor",
            event_type="verify",
            payload={
                "phase": snapshot["phase"],
                "round_index": snapshot["round_index"],
                "record": record,
            },
            token_in=0,
            token_out=0,
            latency_ms=1.0,
            cost_usd=0.0,
            state_id=(
                f"run_{state['run_index']}_dispatch_{dispatch_id}_descriptor"
            ),
        )

        if not record:
            record = {
                "phase": snapshot["phase"],
                "round_index": snapshot["round_index"],
                "dispatch_id": snapshot["dispatch_id"],
            }

        return {
            "descriptor_records": [record],
            "trace_payloads": [event],
        }

    def _continue_or_finalize(self, state: RuntimeState) -> str:
        return "finalize" if bool(state.get("done", False)) else "dispatch"

    def _finalize_node(self, state: RuntimeState) -> dict[str, Any]:
        final_answer = str(state.get("final_answer") or "")
        if not final_answer:
            final_answer = self._fallback_answer(list(state.get("latest_outputs", {}).values()))

        retrieved_docids = set(str(item) for item in state.get("retrieved_docids", []))
        if not retrieved_docids:
            retrieved_docids.update(self._extract_docids(final_answer))

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
                }
            )
            if isinstance(emitted, dict):
                descriptor_summary = emitted

        dispatch_id = int(state.get("dispatch_id", 0)) + 1
        event = self._draft_event(
            dispatch_id=dispatch_id,
            node_order=4,
            agent_order=-1,
            event_order=0,
            actor="system",
            event_type="finalize",
            payload={
                "status": "completed",
                "success": True,
                "final_answer": final_answer,
                "retrieved_docids": sorted(retrieved_docids),
                "final_reason": state.get("final_reason", ""),
            },
            token_in=0,
            token_out=max(1, len(final_answer.split())),
            latency_ms=1.0,
            cost_usd=0.0,
            state_id=f"run_{state['run_index']}_finalize",
        )

        return {
            "final_answer": final_answer,
            "retrieved_docids": sorted(retrieved_docids),
            "descriptor_summary": descriptor_summary,
            "trace_payloads": [event],
            "done": True,
        }

    @staticmethod
    def _active_agents_for_phase(state: RuntimeState) -> list[str]:
        topology = str(state.get("topology", ""))
        phase = str(state.get("phase", ""))
        layout = state["layout"]

        if topology == TOPOLOGY_SAS:
            return list(layout.agent_ids)

        if topology == TOPOLOGY_ORCHESTRATOR_TREE:
            if phase in {"root_delegate", "root_finalize"}:
                return [layout.orchestrator_id] if layout.orchestrator_id else []
            if phase in {"manager_delegate", "manager_aggregate"}:
                return list(layout.managers)
            if phase == "leaf_work":
                return list(layout.leaves)
            return []

        if topology == TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION:
            if phase == "specialist_solve":
                return list(layout.specialists)
            if phase == "orchestrator_synthesize":
                return [layout.orchestrator_id] if layout.orchestrator_id else []
            return []

        if topology == TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION:
            if phase in {"initial_proposals", "peer_discussion"}:
                return list(layout.specialists)
            if phase == "orchestrator_synthesize":
                return [layout.orchestrator_id] if layout.orchestrator_id else []
            return []

        if topology == TOPOLOGY_ONLY_VOTING:
            return list(layout.agent_ids)

        if topology == TOPOLOGY_FULLY_LINKED_DEBATE:
            return list(layout.agent_ids)

        if topology == TOPOLOGY_GROUP_CHAT_DEBATE:
            if phase == "group_debate":
                return list(layout.agent_ids)
            if phase == "representative_merge":
                return list(layout.representatives)
            return []

        return list(layout.agent_ids)

    @staticmethod
    def _initial_phase(topology: str) -> str:
        if topology == TOPOLOGY_SAS:
            return "solve"
        if topology == TOPOLOGY_ORCHESTRATOR_TREE:
            return "root_delegate"
        if topology == TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION:
            return "specialist_solve"
        if topology == TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION:
            return "initial_proposals"
        if topology == TOPOLOGY_ONLY_VOTING:
            return "vote_collect"
        if topology == TOPOLOGY_FULLY_LINKED_DEBATE:
            return "debate"
        if topology == TOPOLOGY_GROUP_CHAT_DEBATE:
            return "group_debate"
        return "solve"

    @staticmethod
    def _build_agent_prompt(
        *,
        task_prompt: Any,
        agent_id: str,
        role: str,
        phase: str,
        round_index: int,
        visible_messages: list[dict[str, Any]],
    ) -> Any:
        if visible_messages:
            message_lines = [
                f"From {item.get('sender')} [{item.get('message_id')}]: {item.get('content')}"
                for item in visible_messages[-8:]
            ]
            inbox_text = "\n".join(message_lines)
        else:
            inbox_text = "None"

        instruction = (
            f"Agent ID: {agent_id}\n"
            f"Role: {role}\n"
            f"Phase: {phase}\n"
            f"Round: {round_index}\n"
            f"Visible relay messages:\n{inbox_text}\n\n"
            "Return your best current answer for this phase."
        )

        if isinstance(task_prompt, list):
            return [{"role": "system", "content": instruction}] + list(task_prompt)

        return f"Task:\n{task_prompt}\n\n{instruction}"

    @staticmethod
    def _fallback_answer(candidates: list[str]) -> str:
        if not candidates:
            return ""
        winner, _ = vote_majority(candidates)
        return winner

    @staticmethod
    def _render_peer_bundle(peer_messages: list[str]) -> str:
        lines = [f"Peer {idx + 1}: {text}" for idx, text in enumerate(peer_messages[:6])]
        return "Peer bundle for critique:\n" + "\n".join(lines)

    @staticmethod
    def _render_group_summary(group_outputs: list[str]) -> str:
        if not group_outputs:
            return "Group summary: no outputs provided."
        lines = [f"- {text}" for text in group_outputs[:6]]
        return "Group summary:\n" + "\n".join(lines)

    @staticmethod
    def _group_for_agent(layout: TopologyLayout, agent_id: str) -> list[str]:
        for group in layout.groups:
            if agent_id in group:
                return list(group)
        return []

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

    @staticmethod
    def _message_snippet(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())[:260]

    @staticmethod
    def _tool_output_preview(output: Any) -> str:
        if isinstance(output, str):
            text = output
        else:
            try:
                text = json.dumps(output, ensure_ascii=False)
            except Exception:
                text = str(output)
        return re.sub(r"\s+", " ", text.strip())[:240]

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
    def _mock_tool_plan(
        cls,
        prompt: Any,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_name: dict[str, dict[str, Any]] = {}
        for tool in tools:
            name = str(tool.get("name", "")).strip()
            if name and callable(tool.get("handler")):
                by_name[name] = tool
        if not by_name:
            return []

        def execute(name: str, args: dict[str, Any]) -> dict[str, Any]:
            handler = by_name[name]["handler"]
            status = "completed"
            error = None
            try:
                output = handler(args)
            except Exception as exc:
                status = "error"
                error = str(exc)
                output = {"error": error}
            return {
                "tool_name": name,
                "arguments": args,
                "status": status,
                "error": error,
                "output": output,
            }

        if isinstance(prompt, list):
            prompt_text = " ".join(
                str(item.get("content", "")) for item in prompt if isinstance(item, dict)
            )
        else:
            prompt_text = str(prompt)
        query = re.sub(r"\s+", " ", prompt_text).strip()[:600]
        if not query:
            query = "Find relevant documents for the task."

        records: list[dict[str, Any]] = []
        if "search" in by_name:
            search_record = execute("search", {"query": query})
            records.append(search_record)

            candidate_docids = cls._extract_docids_from_tool_output(
                output=search_record.get("output"),
                arguments={"query": query},
                tool_name="search",
            )
            if "get_document" in by_name and candidate_docids:
                records.append(execute("get_document", {"docid": candidate_docids[0]}))
        else:
            first_name = next(iter(by_name))
            records.append(execute(first_name, {}))

        return records
