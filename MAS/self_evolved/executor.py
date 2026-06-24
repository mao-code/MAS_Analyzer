"""Turn executor: interprets a TopologySpec for one collaboration turn.

Instead of compiling a static LangGraph per topology, the executor walks the
group tree directly. Every agent stage still runs through
``LangGraphMASEngine._execute_agent_stage`` so prompts, artifact coercion,
tool retry, and trace events are identical to the fixed-topology systems.

Pattern semantics (uniform across nesting levels):
- singleton: the member runs its contribution stage.
- star:      leader plans -> task packets -> members contribute -> reports
             relayed to leader -> leader aggregates.
- chain:     members contribute sequentially, each seeing the predecessor's
             bounded report packet.
- debate:    members contribute, exchange peer summaries along group edges,
             then run one critic revision round; output by deterministic vote.
- voting:    members contribute independently; output by deterministic vote.

An agent with an attached subgroup contributes by expansion: it plans, the
subgroup executes its own pattern, reports relay back, and the agent
aggregates — its aggregate artifact is its report to the enclosing group.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..artifacts import ArtifactRecord, answer_signature, packet_payload_from_artifact
from ..langgraph_engine import LangGraphMASEngine
from .context import SharedContextController
from .spec import GroupSpec, TopologySpec

# State keys accumulated across stages (mirrors the LangGraph reducers).
APPEND_STATE_KEYS = frozenset(
    {
        "messages",
        "message_views",
        "artifacts",
        "trace_payloads",
        "phase_history",
        "descriptor_records",
        "interaction_logs",
        "tool_records_log",
        "termination_history",
        "transcript_compaction_history",
        "context_state_versions",
        "short_term_playbook_entries",
    }
)

TASK_PACKET_KINDS = {"task_package", "orchestrator_feedback"}

# Self-evolved retrieval directive. When a run exposes a get_document tool
# (retrieval benchmarks such as browsecomp), search results are only pointers;
# the dominant failure mode is agents that search repeatedly but never open a
# document and then give up with an "insufficient evidence" answer. Every
# self-evolved agent stage is told to read before concluding. Inert for runs
# without get_document (plancraft / workbench / stabletoolbench / mock mode).
_READ_FIRST_DIRECTIVE = (
    " This task provides a get_document tool. Search returns only short snippets, which "
    "are NOT sufficient to answer — you MUST call get_document to read the full text of "
    "the most relevant documents. As soon as a search returns a relevant docid, "
    "immediately call get_document on the top 1-3 docids; do NOT run more than two rounds "
    "of search before opening a document. Answer only from documents you have opened with "
    "get_document — never answer from snippets alone, and never return an 'insufficient "
    "evidence' or blocked answer without having opened at least one document."
)


# Self-evolved tool-chaining directive for domain-tool tasks (stabletoolbench,
# workbench) that expose API tools but no get_document. Failure mode observed: an
# agent runs one entity "search", the backend returns a cache miss for a non-canonical
# argument (e.g. name="Messi"), the search circuit breaker trips, and the agent answers
# from incomplete data. The fix is behavioral: use canonical arguments and chain the
# follow-up tools to gather the specific facts the question asks for.
_TOOL_CHAIN_DIRECTIVE = (
    " Use full, official, canonical names and identifiers in tool arguments (e.g. "
    "'Lionel Messi', not 'Messi'); a non-canonical argument often returns nothing. One "
    "lookup rarely answers the question: after locating the relevant entity, chain the "
    "follow-up tools (details / profile / related lookups) to gather the specific facts "
    "the question asks for before answering. If a tool returns no result, retry once with "
    "a more canonical argument instead of abandoning the tool."
)


def _run_has_get_document(state: dict[str, Any]) -> bool:
    return any(
        isinstance(tool, dict) and str(tool.get("name", "")) == "get_document"
        for tool in (state.get("tools") or [])
    )


def _run_has_tools(state: dict[str, Any]) -> bool:
    return bool(state.get("tools"))


def apply_updates(state: dict[str, Any], updates: dict[str, Any]) -> None:
    """Merge stage-node updates into the plain-dict state."""

    for key, value in updates.items():
        if key in APPEND_STATE_KEYS and isinstance(value, list):
            state.setdefault(key, []).extend(value)
        else:
            state[key] = value


@dataclass
class TurnResult:
    output_artifact: ArtifactRecord | None
    member_artifacts: list[ArtifactRecord] = field(default_factory=list)
    expected_member_count: int = 0


class TurnExecutor:
    def __init__(self, stage_engine: LangGraphMASEngine, context: SharedContextController) -> None:
        self._engine = stage_engine
        self._context = context
        # Per-run ledger of executed tool-call signatures. Guarantees an identical
        # (tool, args) call is recorded once even if the planner spawns parallel
        # workers that each invoke it — without this, side-effecting tools (e.g.
        # calendar.create_event) get replayed and corrupt the final state. Scoped
        # to one run (the executor is created per run) and shared across turns.
        self._executed_signatures: set[str] = set()

    def run_turn(self, state: dict[str, Any], spec: TopologySpec, *, turn_index: int) -> TurnResult:
        state["round_index"] = int(turn_index)
        state["discussion_index"] = 0
        root = spec.group(spec.root_group_id)
        return self._execute_root_group(state, spec, root)

    # -- group execution ----------------------------------------------------

    def _execute_root_group(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        group: GroupSpec,
    ) -> TurnResult:
        if group.pattern == "singleton":
            artifact = self._contribute(state, spec, group.member_ids[0], group)
            members = [artifact] if artifact else []
            return TurnResult(
                output_artifact=artifact, member_artifacts=members, expected_member_count=1
            )

        if group.pattern == "star":
            leader = str(group.leader_id)
            workers = [member for member in group.member_ids if member != leader]
            plan_artifact = self._run_leader_stage(
                state,
                spec,
                group,
                agent_id=leader,
                stage_role="planner",
                node_suffix="plan",
                directive=(
                    "Produce a concise plan and a bounded task package for each group member."
                ),
            )
            self._emit_task_packets(
                state, sender=leader, recipients=workers, artifact=plan_artifact
            )
            member_artifacts = self._run_member_contributions(state, spec, group, workers)
            self._relay_reports(
                state,
                group=group,
                artifacts=member_artifacts,
                recipients=[leader],
                kind="specialist_report",
            )
            merge_artifact = self._run_leader_stage(
                state,
                spec,
                group,
                agent_id=leader,
                stage_role="aggregator",
                node_suffix="merge",
                directive=(
                    "Merge the member reports into one best answer. "
                    "Preserve unresolved issues explicitly."
                ),
                visible_kinds={"specialist_report"},
                # Final synthesis always sees the global evidence digest.
                evidence_scope="global",
            )
            return TurnResult(
                output_artifact=merge_artifact,
                member_artifacts=member_artifacts,
                expected_member_count=len(workers),
            )

        if group.pattern == "chain":
            member_artifacts = self._run_member_contributions(
                state, spec, group, list(group.member_ids), sequential_relay=True
            )
            output = member_artifacts[-1] if member_artifacts else None
            return TurnResult(
                output_artifact=output,
                member_artifacts=member_artifacts,
                expected_member_count=len(group.member_ids),
            )

        # debate / voting roots
        member_artifacts = self._run_member_contributions(
            state, spec, group, list(group.member_ids)
        )
        if group.pattern == "debate" and member_artifacts:
            member_artifacts = self._run_debate_revision(state, spec, group, member_artifacts)
        output = self._select_group_output(member_artifacts)
        return TurnResult(
            output_artifact=output,
            member_artifacts=member_artifacts,
            expected_member_count=len(group.member_ids),
        )

    def _execute_expansion(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        *,
        parent_agent_id: str,
        subgroup: GroupSpec,
    ) -> ArtifactRecord | None:
        """Run an attached subgroup as the parent agent's contribution."""

        plan_artifact = self._run_leader_stage(
            state,
            spec,
            subgroup,
            agent_id=parent_agent_id,
            stage_role="planner",
            node_suffix="plan",
            directive=(
                "Decompose your assigned subtask into bounded task packages "
                "for your subgroup members."
            ),
            visible_kinds=TASK_PACKET_KINDS | {"peer_summary"},
        )
        self._emit_task_packets(
            state,
            sender=parent_agent_id,
            recipients=list(subgroup.member_ids),
            artifact=plan_artifact,
        )
        member_artifacts = self._run_member_contributions(
            state,
            spec,
            subgroup,
            list(subgroup.member_ids),
            sequential_relay=subgroup.pattern == "chain",
        )
        if subgroup.pattern == "debate" and member_artifacts:
            member_artifacts = self._run_debate_revision(state, spec, subgroup, member_artifacts)
        self._relay_reports(
            state,
            group=subgroup,
            artifacts=member_artifacts,
            recipients=[parent_agent_id],
            kind="child_report",
        )
        parent_scope = (
            "global"
            if spec.agent(parent_agent_id).context.evidence_access == "global"
            else "branch"
        )
        return self._run_leader_stage(
            state,
            spec,
            subgroup,
            agent_id=parent_agent_id,
            stage_role="aggregator",
            node_suffix="merge",
            directive=(
                "Synthesize your subgroup reports into one supported answer for "
                "your branch; this artifact is your report to the enclosing group."
            ),
            visible_kinds={"child_report"},
            # Branch aggregators see at least their own branch's evidence.
            evidence_scope=parent_scope,
        )

    # -- member contributions -----------------------------------------------

    def _run_member_contributions(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        group: GroupSpec,
        members: list[str],
        *,
        sequential_relay: bool = False,
    ) -> list[ArtifactRecord]:
        if not members:
            return []
        self._dispatch(state, node_name=f"se_{group.group_id}_work_dispatch", active_agents=members)
        artifacts: list[ArtifactRecord] = []
        previous: ArtifactRecord | None = None
        for member in members:
            if sequential_relay and previous is not None:
                self._relay_reports(
                    state,
                    group=group,
                    artifacts=[previous],
                    recipients=[member],
                    kind="peer_summary",
                )
            artifact = self._contribute(state, spec, member, group)
            if artifact is not None:
                artifacts.append(artifact)
                previous = artifact
        return artifacts

    def _contribute(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        agent_id: str,
        group: GroupSpec,
    ) -> ArtifactRecord | None:
        subgroup = spec.subgroup_of(agent_id)
        if subgroup is not None:
            return self._execute_expansion(
                state,
                spec,
                parent_agent_id=agent_id,
                subgroup=subgroup,
            )

        node = spec.agent(agent_id)
        directive = (
            "Work only from the task and the visible packets. "
            "Gather or apply evidence, then state the best supported answer."
            if node.stage_role == "worker"
            else "Verify the visible claims against evidence and revise toward "
            "the best supported answer."
        )
        visible = self._context.visible_packets(
            state,
            agent_id=agent_id,
            kinds=TASK_PACKET_KINDS | {"peer_summary"},
            round_index=int(state.get("round_index", 0)),
        )
        digest = self._context.evidence_digest_packet(state, agent_id=agent_id)
        if digest is not None:
            visible = [digest, *visible]
        return self._run_stage(
            state,
            agent_id=agent_id,
            node_name=f"se_{group.group_id}_work",
            stage_role=node.stage_role,
            directive=directive,
            visible_messages=visible,
        )

    def _run_debate_revision(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        group: GroupSpec,
        member_artifacts: list[ArtifactRecord],
    ) -> list[ArtifactRecord]:
        state["discussion_index"] = int(state.get("discussion_index", 0)) + 1
        members = list(group.member_ids)
        peers_by_sender = {
            str(artifact.get("agent_id", "")): artifact for artifact in member_artifacts
        }
        packet_specs: list[dict[str, Any]] = []
        for sender, artifact in peers_by_sender.items():
            recipients = [member for member in members if member != sender]
            payload = packet_payload_from_artifact(
                artifact, max_chars=self._packet_max_chars(state)
            )
            packet_specs.append(
                {
                    "sender": sender,
                    "recipients": recipients,
                    "kind": "peer_summary",
                    "round": int(state.get("round_index", 0)),
                    "discussion_index": int(state.get("discussion_index", 0)),
                    "artifact_id": artifact.get("artifact_id"),
                    "payload": payload,
                }
            )
        self._dispatch(
            state,
            node_name=f"se_{group.group_id}_debate_dispatch",
            active_agents=members,
            packet_specs=packet_specs,
        )

        revised: list[ArtifactRecord] = []
        for member in members:
            visible = self._context.visible_packets(
                state,
                agent_id=member,
                kinds={"peer_summary"},
                round_index=int(state.get("round_index", 0)),
                discussion_index=int(state.get("discussion_index", 0)),
            )
            digest = self._context.evidence_digest_packet(state, agent_id=member)
            if digest is not None:
                visible = [digest, *visible]
            artifact = self._run_stage(
                state,
                agent_id=member,
                node_name=f"se_{group.group_id}_debate",
                stage_role="critic",
                directive=(
                    "Challenge weak peer claims using the visible debate packets and "
                    "revise toward the best supported answer."
                ),
                visible_messages=visible,
            )
            if artifact is not None:
                revised.append(artifact)
        return revised

    # -- shared stage / packet helpers ---------------------------------------

    def _run_leader_stage(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        group: GroupSpec,
        *,
        agent_id: str,
        stage_role: str,
        node_suffix: str,
        directive: str,
        visible_kinds: set[str] | None = None,
        evidence_scope: str | None = None,
    ) -> ArtifactRecord | None:
        self._dispatch(
            state,
            node_name=f"se_{group.group_id}_{node_suffix}_dispatch",
            active_agents=[agent_id],
        )
        visible: list[Any] = []
        if visible_kinds:
            visible = self._context.visible_packets(
                state,
                agent_id=agent_id,
                kinds=visible_kinds,
                round_index=int(state.get("round_index", 0)),
            )
        if evidence_scope is not None:
            digest = self._context.evidence_digest_packet(
                state, agent_id=agent_id, scope=evidence_scope
            )
            if digest is not None:
                visible = [digest, *visible]
        return self._run_stage(
            state,
            agent_id=agent_id,
            node_name=f"se_{group.group_id}_{node_suffix}",
            stage_role=stage_role,
            directive=directive,
            visible_messages=visible,
        )

    def _run_stage(
        self,
        state: dict[str, Any],
        *,
        agent_id: str,
        node_name: str,
        stage_role: str,
        directive: str,
        visible_messages: list[Any],
    ) -> ArtifactRecord | None:
        if _run_has_get_document(state):
            directive = f"{directive}{_READ_FIRST_DIRECTIVE}"
        elif _run_has_tools(state):
            directive = f"{directive}{_TOOL_CHAIN_DIRECTIVE}"
        try:
            updates = self._engine._execute_agent_stage(
                state,
                agent_id=agent_id,
                node_name=node_name,
                stage_role=stage_role,
                directive=directive,
                visible_messages=visible_messages,
            )
        except RuntimeError as exc:
            # A flaky tool tripped the per-agent circuit breaker (the shared stage
            # raises fatal_tool_failure). Static topologies finalize from their other
            # agents; the dynamic orchestrator must not let one agent's tool flakiness
            # kill the whole task. Skip this agent's contribution and continue — the
            # remaining agents / finalize synthesis still produce an answer.
            if "fatal_tool_failure" not in str(exc):
                raise
            print(
                f"[self_evolved] RECOVERED node={node_name} agent_id={agent_id} "
                f"reason=fatal_tool_failure recovery=skip_agent_continue: {exc}",
                flush=True,
            )
            return None
        artifacts = updates.get("artifacts") or []
        apply_updates(state, updates)
        artifact = artifacts[0] if artifacts else None
        self._dedupe_tool_records(artifact)
        self._context.record_evidence(state, artifact, agent_id=agent_id)
        return artifact

    def _dedupe_tool_records(self, artifact: ArtifactRecord | None) -> None:
        """Drop tool records whose (tool, args) already ran earlier this run.

        The artifact dict (and its ``tool_records`` list) is shared by reference
        with ``state['artifacts']`` and the run metadata the benchmark evaluates,
        so stripping the redundant record in place ensures each external action is
        replayed exactly once. Distinct calls (different args) are always kept, so
        diverse parallel search is unaffected — only exact duplicates collapse.
        """

        if not isinstance(artifact, dict):
            return
        records = artifact.get("tool_records")
        if not isinstance(records, list) or not records:
            return
        kept: list[Any] = []
        for record in records:
            if not isinstance(record, dict):
                kept.append(record)
                continue
            name = str(record.get("tool_name", "")).strip()
            if not name or name == "inter_agent_send":
                kept.append(record)
                continue
            arguments = record.get("arguments")
            try:
                arg_sig = json.dumps(arguments, sort_keys=True, default=str)
            except Exception:
                arg_sig = str(arguments)
            signature = f"{name}({arg_sig})"
            if signature in self._executed_signatures:
                continue
            self._executed_signatures.add(signature)
            kept.append(record)
        artifact["tool_records"] = kept

    def _dispatch(
        self,
        state: dict[str, Any],
        *,
        node_name: str,
        active_agents: list[str],
        packet_specs: list[dict[str, Any]] | None = None,
    ) -> None:
        dispatch_id = int(state.get("dispatch_id", -1)) + 1
        state["dispatch_id"] = dispatch_id
        state["active_agents"] = list(active_agents)
        state["phase"] = node_name
        trace = [
            self._engine._draft_event(
                dispatch_id=dispatch_id,
                node_order=0,
                agent_order=-1,
                event_order=0,
                actor="orchestrator",
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
        apply_updates(
            state,
            {
                "trace_payloads": trace,
                "phase_history": [
                    {
                        "phase": node_name,
                        "round_index": int(state.get("round_index", 0)),
                        "discussion_index": int(state.get("discussion_index", 0)),
                        "dispatch_id": dispatch_id,
                        "active_agents": list(active_agents),
                    }
                ],
            },
        )
        if packet_specs:
            emitted = self._engine._emit_packets(
                state, dispatch_id=dispatch_id, phase=node_name, packet_specs=packet_specs
            )
            apply_updates(state, emitted)

    def _emit_task_packets(
        self,
        state: dict[str, Any],
        *,
        sender: str,
        recipients: list[str],
        artifact: ArtifactRecord | None,
    ) -> None:
        if artifact is None or not recipients:
            return
        kind = "task_package" if int(state.get("round_index", 0)) == 0 else "orchestrator_feedback"
        payload = packet_payload_from_artifact(artifact, max_chars=self._packet_max_chars(state))
        packet_specs = [
            {
                "sender": sender,
                "recipients": [recipient],
                "kind": kind,
                "round": int(state.get("round_index", 0)),
                "discussion_index": 0,
                "artifact_id": artifact.get("artifact_id"),
                "payload": payload,
            }
            for recipient in recipients
        ]
        emitted = self._engine._emit_packets(
            state,
            dispatch_id=int(state.get("dispatch_id", -1)),
            phase=str(state.get("phase", "se_task_packets")),
            packet_specs=packet_specs,
        )
        apply_updates(state, emitted)

    def _relay_reports(
        self,
        state: dict[str, Any],
        *,
        group: GroupSpec,
        artifacts: list[ArtifactRecord],
        recipients: list[str],
        kind: str,
    ) -> None:
        if not artifacts or not recipients:
            return
        packet_specs = []
        for artifact in artifacts:
            payload = packet_payload_from_artifact(
                artifact, max_chars=self._packet_max_chars(state)
            )
            packet_specs.append(
                {
                    "sender": str(artifact.get("agent_id", "")),
                    "recipients": list(recipients),
                    "kind": kind,
                    "round": int(state.get("round_index", 0)),
                    "discussion_index": int(state.get("discussion_index", 0)),
                    "artifact_id": artifact.get("artifact_id"),
                    "payload": payload,
                }
            )
        emitted = self._engine._emit_packets(
            state,
            dispatch_id=int(state.get("dispatch_id", -1)),
            phase=f"se_{group.group_id}_relay",
            packet_specs=packet_specs,
        )
        apply_updates(state, emitted)

    # -- output selection -----------------------------------------------------

    @staticmethod
    def _select_group_output(artifacts: list[ArtifactRecord]) -> ArtifactRecord | None:
        """Deterministic group output: majority answer signature, then evidence,
        tool use, and confidence as tie-breakers."""

        if not artifacts:
            return None
        counts: dict[str, int] = {}
        for artifact in artifacts:
            signature = answer_signature(str(artifact.get("answer", "")))
            if signature:
                counts[signature] = counts.get(signature, 0) + 1

        def rank(artifact: ArtifactRecord) -> tuple[int, int, float, str]:
            signature = answer_signature(str(artifact.get("answer", "")))
            evidence = artifact.get("evidence_summary", [])
            evidence_count = len(evidence) if isinstance(evidence, list) else 0
            used_tools = bool(artifact.get("tool_records"))
            return (
                counts.get(signature, 0),
                evidence_count + int(used_tools),
                float(artifact.get("confidence", 0.5)),
                str(artifact.get("agent_id", "")),
            )

        return sorted(artifacts, key=rank, reverse=True)[0]

    @staticmethod
    def _packet_max_chars(state: dict[str, Any]) -> int:
        return max(0, int(state.get("peer_artifact_max_chars", 0)))
