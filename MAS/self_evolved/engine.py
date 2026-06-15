"""Self-evolved topology engine: meta-control loop around the TurnExecutor.

Meta-loop per run (the dynamic target agent system lives inside step 3):
1. PLAN      Topology Planner proposes the initial TopologySpec.
2. SPAWN     Orchestrator (deterministic code) instantiates agents, personas,
             and context policies.
3. EXECUTE   TurnExecutor runs one collaboration turn over the current spec.
4. AUDIT     Trace Auditor inspects the turn (phase 3).
5. CONTROL   Ordered code-level checks decide stop vs. one trace-backed repair.
6. MUTATE    On the single repair path, the Orchestrator applies the planner's
             topology mutation and re-runs one turn (phase 3).
7. FINALIZE  Deterministic/judge vote over the final candidates.

Agents never decide loop termination; the controller logic here does.
"""

from __future__ import annotations

import time
from typing import Any

from answer_utils import has_substantive_answer

from ..artifacts import (
    ArtifactRecord,
    TerminationDecision,
    build_artifact,
    latest_artifact_by_agent,
)
from ..config import SelfEvolvedConfig
from ..langgraph_engine import (
    ExperimentSpec,
    LangGraphMASEngine,
    LangGraphRunResult,
    WorkflowDocumentation,
)
from ..monitor import DescriptorHook, NullDescriptor
from ..relay import TOPOLOGY_SELF_EVOLVED
from .auditor import TraceAuditorAgent
from .context import SharedContextController
from .executor import TurnExecutor, TurnResult, apply_updates
from .harness import AgentHarness
from .planner import TopologyPlannerAgent
from .playbook import PlaybookMaintainer, ShortTermTopologyPlaybook, TopologyPlaybook, task_key
from .spec import TopologySpec


class SelfEvolvedEngine:
    """Drop-in engine with the same ``run`` signature as LangGraphMASEngine."""

    def __init__(self, harness: AgentHarness, se_config: SelfEvolvedConfig) -> None:
        self.llm_client = harness
        self.se_config = se_config
        self._stage = LangGraphMASEngine(harness)
        self.planner = TopologyPlannerAgent(harness, se_config)
        self.auditor = TraceAuditorAgent(harness, se_config)
        self._playbook: TopologyPlaybook | None = None
        self._playbook_loaded = False

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

        num_agents = min(int(spec.num_agents), int(self.se_config.max_initial_agents))

        # 1. PLAN — phase 2 replaces this with the LLM Topology Planner.
        plan_started = time.perf_counter()
        topo_spec, plan_payload = self._propose_initial_spec(
            task, num_agents, str(spec.benchmark_name or ""), bool(tools)
        )
        plan_latency_ms = max((time.perf_counter() - plan_started) * 1000.0, 1.0)
        topo_spec.validate(max_agents=int(self.se_config.max_total_agents))
        layout = topo_spec.to_layout()

        # 2. SPAWN — personas + state initialization.
        agent_type_by_agent = {
            agent_id: agent_types[idx % len(agent_types)]
            for idx, agent_id in enumerate(layout.agent_ids)
        }
        domain_personas, role_assignment_payload = self._assign_personas(
            task=task, spec=spec, layout=layout
        )

        workflow_definition = self._workflow_definition().to_payload()
        state = self._initial_state(
            task=task,
            run_index=run_index,
            seed=seed,
            spec=spec,
            layout=layout,
            agent_type_by_agent=agent_type_by_agent,
            tools=list(tools or []),
            max_tool_iterations=max_tool_iterations,
            descriptor=descriptor or NullDescriptor(),
            domain_personas=domain_personas,
            role_assignment_payload=role_assignment_payload,
            workflow_definition=workflow_definition,
        )

        spec_versions: list[dict[str, Any]] = [topo_spec.to_payload()]
        audit_reports: list[dict[str, Any]] = []
        mutation_payload: dict[str, Any] | None = None
        short_term_playbook = ShortTermTopologyPlaybook()

        self._emit_meta_event(
            state,
            actor="topology_planner",
            event_type="plan",
            node_name="topology_plan",
            payload={
                "topology_spec": layout.to_payload(),
                "spec": topo_spec.to_payload(),
                "rationale": plan_payload.get("rationale", ""),
                "used_fallback": plan_payload.get("used_fallback", True),
                "fallback_reason": plan_payload.get("fallback_reason", ""),
                "playbook_keys": plan_payload.get("playbook_keys", []),
            },
            token_in=int(plan_payload.get("llm", {}).get("token_in", 0)),
            token_out=int(plan_payload.get("llm", {}).get("token_out", 0)),
            latency_ms=plan_latency_ms,
            cost_usd=float(plan_payload.get("llm", {}).get("cost_usd", 0.0)),
        )
        self._emit_meta_event(
            state,
            actor="orchestrator",
            event_type="plan",
            node_name="spawn_agents",
            payload={
                "agents": list(layout.agent_ids),
                "agent_type_by_agent": dict(agent_type_by_agent),
                "context_policies": {
                    node.agent_id: {
                        "visible_kinds": list(node.context.visible_kinds),
                        "visible_from": list(node.context.visible_from),
                        "share_scope": node.context.share_scope,
                        "evidence_access": node.context.evidence_access,
                    }
                    for node in topo_spec.agents
                },
            },
        )
        self._record_context_state(
            state,
            topo_spec,
            layout,
            reason="spawn",
            turn_index=None,
        )

        # 3-6. EXECUTE / AUDIT / CONTROL / MUTATE loop.
        context = SharedContextController(topo_spec)
        executor = TurnExecutor(self._stage, context)
        max_turns = int(self.se_config.max_turns)
        turn_results: list[TurnResult] = []
        decision: TerminationDecision = {}

        for turn_index in range(max_turns):
            result = executor.run_turn(state, topo_spec, turn_index=turn_index)
            turn_results.append(result)
            apply_updates(state, self._stage._descriptor_monitor_node(state))

            audit_report = self._audit_turn(state, topo_spec, turn_index=turn_index)
            if audit_report is not None:
                audit_reports.append(audit_report)

            mutations_used = len(spec_versions) - 1
            repair_available = (
                mutations_used < 1
                and turn_index + 1 < max_turns
                and self._repair_recommended(audit_report)
            )
            decision = self._meta_termination(
                state,
                turn_index=turn_index,
                result=result,
                previous_result=turn_results[-2] if len(turn_results) > 1 else None,
                repair_available=repair_available,
            )
            state["termination_decision"] = decision
            apply_updates(
                state,
                {
                    "termination_history": [dict(decision)],
                    "trace_payloads": [
                        self._stage._termination_event(state, "meta_termination", decision)
                    ],
                },
            )
            self._record_short_term_playbook_turn(
                state=state,
                short_term_playbook=short_term_playbook,
                topo_spec=topo_spec,
                task=task,
                audit_report=audit_report,
                termination_decision=decision,
                turn_index=turn_index,
            )
            if bool(decision.get("should_stop", True)):
                break

            # Single trace-backed repair.
            mutated = self._apply_mutation(
                state,
                topo_spec,
                audit_report,
                task=task,
                agent_types=agent_types,
                short_term_playbook=short_term_playbook,
            )
            if mutated is None:
                # The continue decision is void without a usable mutation;
                # record the corrected stop so the trace stays coherent.
                decision = {
                    **decision,
                    "should_stop": True,
                    "next_step": "finalize",
                    "reason": "max_rounds_reached",
                    "reason_detail": (
                        "Repair was recommended but the planner produced no valid "
                        "mutation; finalizing on the current topology."
                    ),
                }
                state["termination_decision"] = decision
                apply_updates(
                    state,
                    {
                        "termination_history": [dict(decision)],
                        "trace_payloads": [
                            self._stage._termination_event(state, "meta_termination", decision)
                        ],
                    },
                )
                break
            topo_spec, mutation_payload = mutated
            layout = topo_spec.to_layout()
            state["layout"] = layout
            context.set_spec(topo_spec)
            spec_versions.append(topo_spec.to_payload())
            self._record_context_state(
                state,
                topo_spec,
                layout,
                reason="mutation",
                turn_index=turn_index,
            )

        # 7. FINALIZE.
        final_result = turn_results[-1] if turn_results else TurnResult(output_artifact=None)
        final_answer = self._finalize(state, final_result, decision)

        # 8. RECORD — long-term playbook candidate; runs never write the
        # persistent playbook file (see scripts/update_topology_playbook.py).
        playbook_key = task_key(str(spec.benchmark_name or ""), task, tools_available=bool(tools))
        playbook_candidate = PlaybookMaintainer.build_update_candidate(
            key=playbook_key,
            benchmark_name=str(spec.benchmark_name or ""),
            spec_versions=spec_versions,
            audit_reports=audit_reports,
            mutation_payload=mutation_payload,
            termination_decision=dict(decision or {}),
        )
        self._emit_meta_event(
            state,
            actor="playbook_maintainer",
            event_type="act",
            node_name="playbook_candidate",
            payload={"scope": "long_term", "update_candidate": playbook_candidate},
        )

        trace_events = self._stage._materialize_trace_events(state.get("trace_payloads", []))
        run_metadata = self._build_run_metadata(
            state=state,
            spec=spec,
            layout=layout,
            run_index=run_index,
            seed=seed,
            workflow_definition=workflow_definition,
            role_assignment_payload=role_assignment_payload,
            spec_versions=spec_versions,
            audit_reports=audit_reports,
            mutation_payload=mutation_payload,
            plan_payload=plan_payload,
            playbook_candidate=playbook_candidate,
        )
        return LangGraphRunResult(
            final_answer=final_answer,
            trace_events=trace_events,
            run_metadata=run_metadata,
        )

    # -- meta-agent hooks (phases 2-4 replace the stubs) ---------------------

    def _propose_initial_spec(
        self, task: Any, num_agents: int, benchmark_name: str, tools_available: bool
    ) -> tuple[TopologySpec, dict[str, Any]]:
        proposal = self.planner.propose_initial(
            task=task,
            benchmark_name=benchmark_name,
            num_agents=num_agents,
            playbook_entries=self._playbook_entries(task, benchmark_name, tools_available),
        )
        return proposal.spec, proposal.to_payload()

    def _playbook_entries(
        self, task: Any, benchmark_name: str, tools_available: bool
    ) -> list[dict[str, Any]] | None:
        """Long-term playbook priors for the planner; runs never write the file."""

        if not self.se_config.playbook_read:
            return None
        playbook = self._load_playbook()
        if playbook is None or not playbook.entries:
            return None
        key = task_key(benchmark_name, task, tools_available=tools_available)
        entries = playbook.lookup(benchmark_name, key)
        views = [TopologyPlaybook.planner_view(entry) for entry in entries]
        return views or None

    def _load_playbook(self) -> TopologyPlaybook | None:
        if not self._playbook_loaded:
            self._playbook_loaded = True
            self._playbook = TopologyPlaybook.load(self.se_config.playbook_path)
        return self._playbook

    def _audit_turn(
        self, state: dict[str, Any], topo_spec: TopologySpec, *, turn_index: int
    ) -> dict[str, Any] | None:
        audit_started = time.perf_counter()
        report = self.auditor.audit(state, topo_spec, turn_index=turn_index)
        audit_latency_ms = max((time.perf_counter() - audit_started) * 1000.0, 1.0)
        llm_payload = report.get("llm", {}) if isinstance(report.get("llm"), dict) else {}
        self._emit_meta_event(
            state,
            actor="trace_auditor",
            event_type="verify",
            node_name=f"audit_turn_{turn_index}",
            payload={"audit": report},
            token_in=int(llm_payload.get("token_in", 0)),
            token_out=int(llm_payload.get("token_out", 0)),
            latency_ms=audit_latency_ms,
            cost_usd=float(llm_payload.get("cost_usd", 0.0)),
        )
        return report

    @staticmethod
    def _repair_recommended(audit_report: dict[str, Any] | None) -> bool:
        if not audit_report:
            return False
        return bool(audit_report.get("repair_recommended", False))

    def _apply_mutation(
        self,
        state: dict[str, Any],
        topo_spec: TopologySpec,
        audit_report: dict[str, Any] | None,
        *,
        task: Any,
        agent_types: list[str],
        short_term_playbook: ShortTermTopologyPlaybook,
    ) -> tuple[TopologySpec, dict[str, Any]] | None:
        playbook_entries = self._playbook_entries(
            task,
            str(state.get("benchmark_name", "")),
            bool(state.get("tools")),
        ) or []
        playbook_entries = [*short_term_playbook.planner_entries(), *playbook_entries]
        proposal = self.planner.propose_mutation(
            task=task,
            spec=topo_spec,
            audit_report=audit_report or {},
            playbook_entries=playbook_entries,
        )
        mutation = proposal.mutation
        if mutation is None:
            self._emit_meta_event(
                state,
                actor="orchestrator",
                event_type="revise",
                node_name="mutation_skipped",
                payload={
                    "reason": proposal.fallback_reason or "no_mutation_proposed",
                    "used_fallback": bool(proposal.used_fallback),
                },
            )
            return None

        mutated_spec = mutation.apply(topo_spec, max_agents=int(self.se_config.max_total_agents))
        mutated_layout = mutated_spec.to_layout()
        new_agents = [
            agent_id
            for agent_id in mutated_layout.agent_ids
            if agent_id not in state.get("agent_type_by_agent", {})
        ]
        self._register_new_agents(state, new_agents, agent_types, mutated_layout)

        mutation_payload = {
            **proposal.to_payload(),
            "added_agents": list(new_agents),
            "spec_version": int(mutated_spec.version),
        }
        self._emit_meta_event(
            state,
            actor="orchestrator",
            event_type="revise",
            node_name="apply_mutation",
            payload={
                "mutation": mutation.to_payload(),
                "added_agents": list(new_agents),
                "topology_spec": mutated_layout.to_payload(),
            },
        )
        return mutated_spec, mutation_payload

    def _register_new_agents(
        self,
        state: dict[str, Any],
        new_agents: list[str],
        agent_types: list[str],
        layout: Any,
    ) -> None:
        """Spawn bookkeeping for agents created by a mutation: model types,
        message budgets, and deterministic personas (no extra LLM round-trip)."""

        if not new_agents:
            return
        agent_type_by_agent = dict(state.get("agent_type_by_agent", {}))
        message_budget = dict(state.get("message_budget", {}))
        sent_counts = dict(state.get("sent_counts", {}))
        budget_sent_counts = dict(state.get("budget_sent_counts", {}))
        budget = int(state.get("communication_budget_per_agent", 0))
        for idx, agent_id in enumerate(new_agents):
            offset = len(agent_type_by_agent) + idx
            agent_type_by_agent[agent_id] = agent_types[offset % len(agent_types)]
            message_budget.setdefault(agent_id, budget)
            sent_counts.setdefault(agent_id, 0)
            budget_sent_counts.setdefault(agent_id, 0)
        state["agent_type_by_agent"] = agent_type_by_agent
        state["message_budget"] = message_budget
        state["sent_counts"] = sent_counts
        state["budget_sent_counts"] = budget_sent_counts

        benchmark_name = str(state.get("benchmark_name", ""))
        if benchmark_name:
            from ..role_assigner import assign_domain_roles_deterministic

            assignments = assign_domain_roles_deterministic(
                benchmark_name=benchmark_name, layout=layout
            )
            personas = dict(state.get("domain_personas", {}))
            for agent_id in new_agents:
                info = assignments.get(agent_id)
                if info is not None and agent_id not in personas:
                    personas[agent_id] = {
                        "role_name": info.role_name,
                        "persona": info.persona,
            }
            state["domain_personas"] = personas

    def _record_short_term_playbook_turn(
        self,
        *,
        state: dict[str, Any],
        short_term_playbook: ShortTermTopologyPlaybook,
        topo_spec: TopologySpec,
        task: Any,
        audit_report: dict[str, Any] | None,
        termination_decision: dict[str, Any],
        turn_index: int,
    ) -> None:
        key = task_key(
            str(state.get("benchmark_name", "")),
            task,
            tools_available=bool(state.get("tools")),
        )
        entry = short_term_playbook.record_turn(
            key=key,
            benchmark_name=str(state.get("benchmark_name", "")),
            turn_index=turn_index,
            spec_payload=topo_spec.to_payload(),
            audit_report=audit_report,
            termination_decision=termination_decision,
        )
        state.setdefault("short_term_playbook_entries", []).append(entry)
        self._emit_meta_event(
            state,
            actor="playbook_maintainer",
            event_type="revise",
            node_name=f"short_term_playbook_turn_{turn_index}",
            payload={"scope": "short_term", "entry": entry},
        )

    def _record_context_state(
        self,
        state: dict[str, Any],
        topo_spec: TopologySpec,
        layout: Any,
        *,
        reason: str,
        turn_index: int | None,
    ) -> None:
        snapshot = self._context_state_snapshot(
            topo_spec,
            layout,
            reason=reason,
            turn_index=turn_index,
        )
        state.setdefault("context_state_versions", []).append(snapshot)
        self._emit_meta_event(
            state,
            actor="orchestrator",
            event_type="revise",
            node_name=f"context_state_{reason}",
            payload={"context_state": snapshot},
        )

    @staticmethod
    def _context_state_snapshot(
        topo_spec: TopologySpec,
        layout: Any,
        *,
        reason: str,
        turn_index: int | None,
    ) -> dict[str, Any]:
        return {
            "reason": reason,
            "turn_index": turn_index,
            "spec_version": int(topo_spec.version),
            "agents": list(layout.agent_ids),
            "adjacency": {
                agent_id: list(peers) for agent_id, peers in dict(layout.adjacency).items()
            },
            "groups": [
                {
                    "group_id": group.group_id,
                    "pattern": group.pattern,
                    "member_ids": list(group.member_ids),
                    "parent_agent_id": group.parent_agent_id,
                    "leader_id": group.leader_id,
                }
                for group in topo_spec.groups
            ],
            "context_policies": {
                node.agent_id: {
                    "visible_kinds": list(node.context.visible_kinds),
                    "visible_from": list(node.context.visible_from),
                    "share_scope": node.context.share_scope,
                    "evidence_access": node.context.evidence_access,
                    "summary_only": bool(node.context.summary_only),
                    "max_packet_chars": int(node.context.max_packet_chars),
                }
                for node in topo_spec.agents
            },
        }

    # -- control -------------------------------------------------------------

    def _meta_termination(
        self,
        state: dict[str, Any],
        *,
        turn_index: int,
        result: TurnResult,
        previous_result: TurnResult | None,
        repair_available: bool,
    ) -> TerminationDecision:
        candidate = [result.output_artifact] if result.output_artifact else []
        previous: list[ArtifactRecord] = []
        if previous_result is not None and previous_result.output_artifact:
            previous = [previous_result.output_artifact]
        consensus = result.member_artifacts or candidate

        decision = self._stage._termination_decision(
            state,
            stage_name="meta_termination",
            round_index=turn_index,
            discussion_index=int(state.get("discussion_index", 0)),
            minimum_round_index=turn_index,
            minimum_required_rounds=0,
            minimum_round_label="turn",
            candidate_artifacts=candidate or consensus,
            previous_candidate_artifacts=previous,
            consensus_artifacts=consensus,
            expected_count=max(1, int(result.expected_member_count)),
            max_reached=not repair_available,
            continue_next_step="apply_mutation",
            stop_next_step="finalize",
        )
        if (
            bool(decision.get("should_stop", True))
            and str(decision.get("reason")) == "max_rounds_reached"
            and not repair_available
        ):
            decision = {
                **decision,
                "reason_detail": (
                    "No trace-backed repair was available "
                    "(audit found no repair or the single mutation budget was spent)."
                ),
            }
        return decision

    # -- finalize -------------------------------------------------------------

    def _finalize(
        self,
        state: dict[str, Any],
        result: TurnResult,
        decision: TerminationDecision,
    ) -> str:
        candidates: list[ArtifactRecord] = []
        output = result.output_artifact
        if output is not None and has_substantive_answer(str(output.get("answer", ""))):
            candidates = [output]
        elif result.member_artifacts:
            candidates = list(result.member_artifacts)
        elif output is not None:
            candidates = [output]
        else:
            candidates = list(state.get("artifacts", []))

        vote_result = self._stage._select_final_answer(
            state=state, stage_name="finalize", artifacts=candidates
        )
        final_answer = self._stage._safe_vote_answer_or_fallback(state, candidates, vote_result)
        synthesis_artifact = self._maybe_synthesize_final_answer(
            state=state,
            final_answer=final_answer,
            selected_artifact_id=str(vote_result.get("selected_artifact_id", "")),
        )
        if synthesis_artifact is not None:
            candidates = [synthesis_artifact]
            final_answer = str(synthesis_artifact.get("answer", "") or final_answer)
            vote_result = {
                "selected_artifact_id": synthesis_artifact.get("artifact_id", ""),
                "selected_agent_id": synthesis_artifact.get("agent_id", ""),
                "selected_source_artifact_ids": synthesis_artifact.get("source_artifact_ids", []),
                "source": "self_evolved_final_synthesis",
                "tally": {final_answer.lower(): 1} if final_answer else {},
            }

        state["final_answer"] = final_answer
        state["final_reason"] = (
            f"{TOPOLOGY_SELF_EVOLVED}:{decision.get('reason', 'finalize')}"
            if decision
            else f"{TOPOLOGY_SELF_EVOLVED}:finalize"
        )
        state["vote_tally"] = dict(vote_result.get("tally", {}))
        state["final_vote_source"] = str(vote_result.get("source", ""))
        state["selected_artifact_id"] = str(vote_result.get("selected_artifact_id", ""))
        state["selected_agent_id"] = str(vote_result.get("selected_agent_id", ""))
        state["selected_source_artifact_ids"] = list(
            vote_result.get("selected_source_artifact_ids", [])
        )
        apply_updates(state, self._stage._finalize_node(state))
        return str(state.get("final_answer", final_answer))

    def _maybe_synthesize_final_answer(
        self,
        *,
        state: dict[str, Any],
        final_answer: str,
        selected_artifact_id: str,
    ) -> ArtifactRecord | None:
        if not self._needs_final_synthesis(final_answer):
            return None
        evidence = self._collect_tool_evidence_for_synthesis(state)
        if not evidence:
            return None

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are the final synthesizer for a self-evolved multi-agent run. "
                    "Use only the task and retrieved evidence below. Do not call tools. "
                    "Do not say you need to search more. Return exactly one JSON object "
                    "with keys: answer_artifact, summary, critique, revision_request, "
                    "confidence, unresolved_issues, evidence_summary. answer_artifact "
                    "must be only the final answer string when the evidence identifies one."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{state.get('task_prompt', '')}\n\n"
                    f"Previous final answer was non-final:\n{final_answer}\n\n"
                    "Retrieved evidence from this run:\n"
                    + "\n\n".join(evidence)
                ),
            },
        ]
        t0 = time.perf_counter()
        try:
            llm = state["llm_client"].generate(
                prompt=prompt_messages,
                agent_type="general",
                task_id=str(state.get("task_id", "")),
                run_index=int(state.get("run_index", 0)),
                agent_id="final_synthesizer",
                tools=[],
                max_tool_iterations=1,
                temperature=0.0,
            )
        except Exception as exc:
            self._emit_meta_event(
                state,
                actor="final_synthesizer",
                event_type="error",
                node_name="self_evolved_final_synthesis",
                payload={
                    "reason": "synthesis_failed",
                    "error": str(exc),
                    "previous_final_answer": final_answer,
                },
            )
            return None

        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1.0)
        artifact = build_artifact(
            text=str(llm.text or ""),
            artifact_id=f"self_evolved_final_synthesis:final_synthesizer:{state.get('run_index', 0)}",
            dispatch_id=int(state.get("dispatch_id", 0)) + 1,
            node_name="self_evolved_final_synthesis",
            stage_role="aggregator",
            round_index=int(state.get("round_index", 0)),
            discussion_index=int(state.get("discussion_index", 0)),
            agent_id="final_synthesizer",
            role="final_synthesizer",
            source_artifact_ids=[selected_artifact_id] if selected_artifact_id else [],
            tool_records=[],
            llm_payload={
                "model": llm.model,
                "mock_used": bool(llm.mock_used),
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "metadata": self._stage._serialize_for_json(dict(llm.metadata)),
            },
        )
        if not has_substantive_answer(str(artifact.get("answer", ""))):
            self._emit_meta_event(
                state,
                actor="final_synthesizer",
                event_type="verify",
                node_name="self_evolved_final_synthesis",
                payload={
                    "reason": "synthesis_non_substantive",
                    "previous_final_answer": final_answer,
                    "synthesis_preview": str(artifact.get("answer", ""))[:260],
                },
                token_in=int(llm.token_in),
                token_out=int(llm.token_out),
                latency_ms=latency_ms,
                cost_usd=float(llm.cost_usd),
            )
            return None

        apply_updates(
            state,
            {
                "artifacts": [artifact],
                "interaction_logs": [
                    {
                        "dispatch_id": int(state.get("dispatch_id", 0)) + 1,
                        "agent_id": "final_synthesizer",
                        "agent_role": "final_synthesizer",
                        "stage_role": "aggregator",
                        "agent_type": "general",
                        "phase": "self_evolved_final_synthesis",
                        "round_index": int(state.get("round_index", 0)),
                        "discussion_index": int(state.get("discussion_index", 0)),
                        "tool_use_retry": False,
                        "prompt_messages": prompt_messages,
                        "visible_messages": [],
                        "prior_artifact": None,
                        "assistant_message": {"role": "assistant", "content": str(llm.text or "")},
                        "structured_artifact": self._stage._serialize_for_json(artifact),
                        "tool_calls": [],
                        "llm": {
                            "model": llm.model,
                            "mock_used": bool(llm.mock_used),
                            "token_in": int(llm.token_in),
                            "token_out": int(llm.token_out),
                            "cost_usd": float(llm.cost_usd),
                            "metadata": self._stage._serialize_for_json(dict(llm.metadata)),
                        },
                    }
                ],
            },
        )
        self._emit_meta_event(
            state,
            actor="final_synthesizer",
            event_type="act",
            node_name="self_evolved_final_synthesis",
            payload={
                "reason": "non_final_answer_repair",
                "previous_final_answer": final_answer,
                "answer": artifact.get("answer", ""),
                "evidence_items": len(evidence),
                "selected_artifact_id": selected_artifact_id,
            },
            token_in=int(llm.token_in),
            token_out=int(llm.token_out),
            latency_ms=latency_ms,
            cost_usd=float(llm.cost_usd),
        )
        state.setdefault("self_evolved_final_synthesis", []).append(
            {
                "previous_final_answer": final_answer,
                "answer": artifact.get("answer", ""),
                "artifact_id": artifact.get("artifact_id", ""),
                "evidence_items": len(evidence),
            }
        )
        return artifact

    def _needs_final_synthesis(self, final_answer: str) -> bool:
        text = str(final_answer or "").strip()
        if not text:
            return True
        lowered = text.lower()
        planning_markers = (
            "let me search",
            "let me investigate",
            "now let me search",
            "i found a promising lead",
            "looks very promising",
            "need to search",
            "search for the specific evidence",
        )
        return (
            not has_substantive_answer(text)
            or self._stage._answer_mode(text) in {"plan", "blocked", "empty"}
            or any(marker in lowered for marker in planning_markers)
        )

    def _collect_tool_evidence_for_synthesis(self, state: dict[str, Any]) -> list[str]:
        evidence: list[str] = []
        seen: set[str] = set()
        for record in state.get("tool_records_log", []):
            if str(record.get("tool_name", "")) != "search":
                continue
            arguments = record.get("arguments")
            query = str(arguments.get("query", "")) if isinstance(arguments, dict) else ""
            output = record.get("output")
            rows = output if isinstance(output, list) else []
            for row in rows[:3]:
                if not isinstance(row, dict):
                    continue
                docid = str(row.get("docid", "")).strip()
                snippet = str(row.get("snippet", "")).strip()
                if not snippet:
                    continue
                key = f"{docid}:{snippet[:120]}"
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    f"- query={query!r}; docid={docid}; snippet={snippet[:900]}"
                )
                if len(evidence) >= 12:
                    return evidence
        return evidence

    # -- state / metadata ------------------------------------------------------

    def _assign_personas(
        self,
        *,
        task: Any,
        spec: ExperimentSpec,
        layout: Any,
    ) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
        domain_personas: dict[str, dict[str, str]] = {}
        payload: dict[str, Any] = {
            "enabled": bool(spec.enable_dynamic_roles),
            "benchmark_name": str(spec.benchmark_name or ""),
            "used_fallback": False,
            "fallback_reason": "",
            "prompt_messages": [],
            "response": "",
            "llm": {},
            "assignments": {},
        }
        if not spec.benchmark_name:
            payload["fallback_reason"] = "missing_benchmark_name"
            return domain_personas, payload
        if not spec.enable_dynamic_roles:
            payload["fallback_reason"] = "disabled"
            return domain_personas, payload

        from ..role_assigner import (
            RoleAssignmentResult,
            assign_domain_roles,
            assign_domain_roles_deterministic,
        )

        try:
            assignment_result = assign_domain_roles(
                benchmark_name=spec.benchmark_name,
                layout=layout,
                task_prompt=getattr(task, "prompt", ""),
                llm_client=self.llm_client,
            )
        except Exception:
            assignments = assign_domain_roles_deterministic(
                benchmark_name=spec.benchmark_name, layout=layout
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
            agent_id: {"role_name": info.role_name, "persona": info.persona}
            for agent_id, info in assignment_result.assignments.items()
        }
        payload.update(
            {
                "used_fallback": bool(assignment_result.used_fallback),
                "fallback_reason": str(assignment_result.fallback_reason),
                "prompt_messages": self._stage._serialize_for_json(
                    list(assignment_result.prompt_messages)
                ),
                "response": str(assignment_result.response_text),
                "llm": self._stage._serialize_for_json(dict(assignment_result.llm)),
                "assignments": {
                    agent_id: {"role_name": info.role_name, "persona": info.persona}
                    for agent_id, info in assignment_result.assignments.items()
                },
            }
        )
        return domain_personas, payload

    def _initial_state(
        self,
        *,
        task: Any,
        run_index: int,
        seed: int,
        spec: ExperimentSpec,
        layout: Any,
        agent_type_by_agent: dict[str, str],
        tools: list[dict[str, Any]],
        max_tool_iterations: int,
        descriptor: DescriptorHook,
        domain_personas: dict[str, dict[str, str]],
        role_assignment_payload: dict[str, Any],
        workflow_definition: dict[str, Any],
    ) -> dict[str, Any]:
        peer_artifact_max_chars = int(spec.peer_artifact_max_chars)
        if peer_artifact_max_chars <= 0:
            peer_artifact_max_chars = int(self.se_config.default_packet_max_chars)
        return {
            "task_id": str(getattr(task, "task_id", "task")),
            "task_prompt": getattr(task, "prompt", ""),
            "benchmark_name": str(spec.benchmark_name or ""),
            "reference_answer": str(getattr(task, "reference_answer", "")),
            "task_metadata": dict(getattr(task, "metadata", {}) or {}),
            "run_index": int(run_index),
            "seed": int(seed),
            "topology": TOPOLOGY_SELF_EVOLVED,
            "layout": layout,
            "workflow_definition": workflow_definition,
            "rounds": int(self.se_config.max_turns),
            "discussion_rounds": int(spec.discussion_rounds),
            "minimum_discussion_rounds": int(spec.minimum_discussion_rounds),
            "termination_consensus_mode": str(spec.termination_consensus_mode),
            "final_vote_mode": str(spec.final_vote_mode),
            "peer_artifact_max_chars": peer_artifact_max_chars,
            "round_index": 0,
            "discussion_index": 0,
            "phase": "init",
            "dispatch_id": -1,
            "active_agents": [],
            "next_step": "",
            "done": False,
            "llm_client": self.llm_client,
            "agent_type_by_agent": agent_type_by_agent,
            "tools": list(tools),
            "max_tool_iterations": max(1, int(max_tool_iterations)),
            "descriptor": descriptor,
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
            "context_state_versions": [],
            "short_term_playbook_entries": [],
            "communication_budget_per_agent": int(spec.communication_budget_per_agent),
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

    def _emit_meta_event(
        self,
        state: dict[str, Any],
        *,
        actor: str,
        event_type: str,
        node_name: str,
        payload: dict[str, Any],
        token_in: int = 0,
        token_out: int = 0,
        latency_ms: float = 1.0,
        cost_usd: float = 0.0,
    ) -> None:
        dispatch_id = int(state.get("dispatch_id", -1)) + 1
        state["dispatch_id"] = dispatch_id
        state["phase"] = node_name
        event = self._stage._draft_event(
            dispatch_id=dispatch_id,
            node_order=0,
            agent_order=-1,
            event_order=0,
            actor=actor,
            event_type=event_type,
            payload={"node": node_name, **self._stage._serialize_for_json(payload)},
            token_in=token_in,
            token_out=token_out,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            state_id=f"run_{state['run_index']}_{node_name}",
        )
        apply_updates(state, {"trace_payloads": [event]})

    @staticmethod
    def _workflow_definition() -> WorkflowDocumentation:
        return WorkflowDocumentation(
            topology=TOPOLOGY_SELF_EVOLVED,
            nodes={
                "topology_plan": "Topology Planner proposes the initial TopologySpec",
                "spawn_agents": "Orchestrator instantiates agents, personas, context policies",
                "execute_turn": "TurnExecutor interprets the current TopologySpec",
                "audit_turn": "Trace Auditor inspects the turn for failure modes",
                "meta_termination": "Ordered code-level stop/repair decision",
                "short_term_playbook_turn": "Playbook Maintainer records turn-level process memory",
                "apply_mutation": "Orchestrator applies the single planner mutation",
                "finalize": "Vote over the final candidates",
            },
            edges=[
                "START -> topology_plan",
                "topology_plan -> spawn_agents",
                "spawn_agents -> execute_turn",
                "execute_turn -> audit_turn",
                "audit_turn -> meta_termination",
                "meta_termination -> short_term_playbook_turn",
                "apply_mutation -> execute_turn",
                "finalize -> END",
            ],
            conditional_edges=[
                "meta_termination -> apply_mutation (single repair available)",
                "meta_termination -> finalize (stop)",
            ],
            dispatch_logic=(
                "Group-tree interpretation: leader plan -> member contributions "
                "(expansions recurse) -> relay -> aggregate."
            ),
            aggregation_logic=(
                "Root output preferred when substantive; otherwise vote over member artifacts."
            ),
            stopping_criteria=[
                "invalid_or_failed_branch",
                "consensus_reached",
                "no_meaningful_change",
                "max_rounds_reached (no repair available or mutation budget spent)",
            ],
            state_fields=[
                "topology_spec_versions",
                "context_state_versions",
                "short_term_playbook_entries",
                "audit_reports",
                "mutation",
            ],
            logging_outputs=["trace_events", "relay_messages", "interaction_logs"],
        )

    def _build_run_metadata(
        self,
        *,
        state: dict[str, Any],
        spec: ExperimentSpec,
        layout: Any,
        run_index: int,
        seed: int,
        workflow_definition: dict[str, Any],
        role_assignment_payload: dict[str, Any],
        spec_versions: list[dict[str, Any]],
        audit_reports: list[dict[str, Any]],
        mutation_payload: dict[str, Any] | None,
        plan_payload: dict[str, Any],
        playbook_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifacts = list(state.get("artifacts", []))
        latest_outputs = {
            agent_id: str(artifact.get("answer", ""))
            for agent_id, artifact in latest_artifact_by_agent(artifacts).items()
        }
        tool_call_counts = self._stage._count_tool_calls(state)
        return {
            "task_id": state.get("task_id"),
            "run_index": run_index,
            "seed": seed,
            "topology": TOPOLOGY_SELF_EVOLVED,
            "rounds_configured": int(self.se_config.max_turns),
            "discussion_rounds": spec.discussion_rounds,
            "termination_consensus_mode": str(spec.termination_consensus_mode),
            "final_vote_mode": str(spec.final_vote_mode),
            "peer_artifact_max_chars": int(state.get("peer_artifact_max_chars", 0)),
            "turns_executed": max(1, int(state.get("round_index", 0)) + 1),
            "messages_sent_total": sum(state.get("sent_counts", {}).values()),
            "messages_sent_by_agent": dict(state.get("sent_counts", {})),
            "budget_messages_sent_total": sum(state.get("budget_sent_counts", {}).values()),
            "budget_messages_sent_by_agent": dict(state.get("budget_sent_counts", {})),
            "tool_call_counts": tool_call_counts,
            "tool_calls_total": int(sum(tool_call_counts.values())),
            "remaining_message_budget": dict(state.get("message_budget", {})),
            "tool_definitions": self._stage._serialize_tools(list(state.get("tools", []))),
            "retrieved_docids": self._stage._collect_retrieved_docids(state),
            "agent_outputs": latest_outputs,
            "vote_tally": dict(state.get("vote_tally", {})),
            "final_vote_source": str(state.get("final_vote_source", "")),
            "selected_artifact_id": str(state.get("selected_artifact_id", "")),
            "selected_agent_id": str(state.get("selected_agent_id", "")),
            "selected_source_artifact_ids": list(state.get("selected_source_artifact_ids", [])),
            "phase_history": list(state.get("phase_history", [])),
            "relay_messages": list(state.get("messages", [])),
            "message_views": list(state.get("message_views", [])),
            "interaction_logs": list(state.get("interaction_logs", [])),
            "descriptor_records": list(state.get("descriptor_records", [])),
            "descriptor_summary": dict(state.get("descriptor_summary", {})),
            "termination_history": list(state.get("termination_history", [])),
            "transcript_summary": dict(state.get("transcript_summary", {})),
            "transcript_compaction_history": list(state.get("transcript_compaction_history", [])),
            "domain_personas": dict(state.get("domain_personas", {})),
            "role_assignment": dict(state.get("role_assignment", role_assignment_payload)),
            "topology_layout": layout.to_payload(),
            "workflow_definition": workflow_definition,
            "artifact_records": artifacts,
            "final_reason": str(state.get("final_reason", "")),
            "self_evolved": {
                "harness_backend": str(self.se_config.harness_backend),
                "topology_spec_versions": list(spec_versions),
                "context_state_versions": list(state.get("context_state_versions", [])),
                "mutation": mutation_payload,
                "audit_reports": list(audit_reports),
                "planner": dict(plan_payload),
                "audit_mode": str(self.se_config.audit_mode),
                "playbook_memory": {
                    "short_term": "updated in memory after each audited turn",
                    "long_term": "persistent file updated post-hoc from eval-joined candidates",
                },
                "short_term_playbook_entries": list(
                    state.get("short_term_playbook_entries", [])
                ),
                "final_synthesis": list(state.get("self_evolved_final_synthesis", [])),
                "playbook_update_candidate": playbook_candidate,
            },
        }
