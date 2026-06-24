"""Topology Planner meta-agent.

Reads the query (plus playbook priors, phase 4) and proposes the initial
topology for the dynamic target agent system. The LLM returns a compact
plan DSL rather than a full spec; the planner expands it deterministically
into a validated ``TopologySpec``. Offline/mock runs fall back to a
deterministic spec, mirroring ``MAS.role_assigner``.

Plan DSL (single JSON object):
{
  "rationale": "why this topology fits the task",
  "pattern": "star",            // singleton | star | chain | debate | voting
  "num_agents": 4,              // root group size, coordinator included
  "verifier": true,             // optional: one root worker becomes a critic
  "expansions": [               // optional nested subgroups
    {"member_index": 1, "pattern": "star", "num_subagents": 3}
  ]
}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..artifacts import _extract_json_payload
from ..config import SelfEvolvedConfig
from ..relay import build_layout
from .spec import (
    GROUP_PATTERNS,
    MUTATION_OPS,
    AgentNode,
    GroupSpec,
    MutationOp,
    TopologyMutation,
    TopologySpec,
    spec_from_layout,
)

logger = logging.getLogger(__name__)


def _principles_section(principles: list[str] | None) -> str:
    """Render benchmark-agnostic playbook principles for a planner prompt."""

    items = [str(p).strip() for p in (principles or []) if str(p).strip()]
    if not items:
        return ""
    lines = "\n".join(f"  - {item}" for item in items)
    return "## Standing principles (always apply)\n" + lines + "\n\n"


@dataclass(frozen=True)
class MutationProposal:
    mutation: TopologyMutation | None
    used_fallback: bool
    fallback_reason: str
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    response_text: str = ""
    llm: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "mutation": self.mutation.to_payload() if self.mutation else None,
            "used_fallback": bool(self.used_fallback),
            "fallback_reason": self.fallback_reason,
            "prompt_messages": list(self.prompt_messages),
            "response": self.response_text,
            "llm": dict(self.llm),
        }


@dataclass(frozen=True)
class PlannerProposal:
    spec: TopologySpec
    rationale: str
    used_fallback: bool
    fallback_reason: str
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    response_text: str = ""
    llm: dict[str, Any] = field(default_factory=dict)
    playbook_keys: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "rationale": self.rationale,
            "used_fallback": bool(self.used_fallback),
            "fallback_reason": self.fallback_reason,
            "prompt_messages": list(self.prompt_messages),
            "response": self.response_text,
            "llm": dict(self.llm),
            "playbook_keys": list(self.playbook_keys),
        }


class TopologyPlannerAgent:
    def __init__(self, llm_client: Any, se_config: SelfEvolvedConfig) -> None:
        self.llm_client = llm_client
        self.se_config = se_config

    # -- initial proposal ----------------------------------------------------

    def propose_initial(
        self,
        *,
        task: Any,
        benchmark_name: str,
        num_agents: int,
        playbook_entries: list[dict[str, Any]] | None = None,
        principles: list[str] | None = None,
        skill_text: str | None = None,
    ) -> PlannerProposal:
        max_agents = max(1, int(num_agents))
        prompt_messages = self._build_initial_prompt(
            task=task,
            benchmark_name=benchmark_name,
            max_agents=max_agents,
            playbook_entries=playbook_entries or [],
            principles=principles or [],
            skill_text=skill_text or "",
        )
        response_text = ""
        llm_payload: dict[str, Any] = {}
        fallback_reason = ""

        try:
            result = self.llm_client.generate(
                prompt=prompt_messages,
                agent_type="planner",
                task_id=str(getattr(task, "task_id", "task")),
                run_index=0,
                agent_id="topology_planner",
                temperature=0.0,
            )
            response_text = str(result.text or "")
            llm_payload = {
                "model": str(result.model),
                "mock_used": bool(result.mock_used),
                "token_in": int(result.token_in),
                "token_out": int(result.token_out),
                "cost_usd": float(result.cost_usd),
                "metadata": dict(result.metadata or {}),
            }
            spec, rationale = self._spec_from_plan_response(response_text, max_agents=max_agents)
            if spec is not None:
                spec.validate(max_agents=int(self.se_config.max_total_agents))
                return PlannerProposal(
                    spec=spec,
                    rationale=rationale,
                    used_fallback=False,
                    fallback_reason="",
                    prompt_messages=prompt_messages,
                    response_text=response_text,
                    llm=llm_payload,
                    playbook_keys=[str(entry.get("key", "")) for entry in (playbook_entries or [])],
                )
            fallback_reason = "invalid_or_unparseable_plan"
        except Exception as exc:
            logger.warning("Topology planning failed; using fallback", exc_info=True)
            fallback_reason = str(exc) or "llm_error"

        fallback = self.fallback_initial_spec(max_agents)
        return PlannerProposal(
            spec=fallback,
            rationale=fallback.rationale,
            used_fallback=True,
            fallback_reason=fallback_reason,
            prompt_messages=prompt_messages,
            response_text=response_text,
            llm=llm_payload,
            playbook_keys=[str(entry.get("key", "")) for entry in (playbook_entries or [])],
        )

    @staticmethod
    def fallback_initial_spec(num_agents: int) -> TopologySpec:
        fallback_topology = "sas" if num_agents <= 1 else "orchestrator_no_discussion"
        layout = build_layout(topology=fallback_topology, num_agents=num_agents)
        return spec_from_layout(layout, rationale=f"deterministic_fallback:{fallback_topology}")

    # -- mutation proposal (the single trace-backed repair) ---------------------

    def propose_mutation(
        self,
        *,
        task: Any,
        spec: TopologySpec,
        audit_report: dict[str, Any],
        playbook_entries: list[dict[str, Any]] | None = None,
        principles: list[str] | None = None,
    ) -> MutationProposal:
        """Ask the planner LLM for one topology mutation addressing the audit
        findings. There is no deterministic fallback mutation: an unusable
        response means no repair (the run finalizes on the current topology)."""

        prompt_messages = self._build_mutation_prompt(
            task=task,
            spec=spec,
            audit_report=audit_report,
            playbook_entries=playbook_entries or [],
            principles=principles or [],
        )
        response_text = ""
        llm_payload: dict[str, Any] = {}
        fallback_reason = ""

        try:
            result = self.llm_client.generate(
                prompt=prompt_messages,
                agent_type="planner",
                task_id=str(getattr(task, "task_id", "task")),
                run_index=0,
                agent_id="topology_planner",
                temperature=0.0,
            )
            response_text = str(result.text or "")
            llm_payload = {
                "model": str(result.model),
                "mock_used": bool(result.mock_used),
                "token_in": int(result.token_in),
                "token_out": int(result.token_out),
                "cost_usd": float(result.cost_usd),
                "metadata": dict(result.metadata or {}),
            }
            mutation = self._mutation_from_response(response_text, audit_report)
            if mutation is not None:
                # Validate by applying against the current spec.
                mutation.apply(spec, max_agents=int(self.se_config.max_total_agents))
                return MutationProposal(
                    mutation=mutation,
                    used_fallback=False,
                    fallback_reason="",
                    prompt_messages=prompt_messages,
                    response_text=response_text,
                    llm=llm_payload,
                )
            fallback_reason = "invalid_or_unparseable_mutation"
        except Exception as exc:
            logger.warning("Mutation planning failed; no repair applied", exc_info=True)
            fallback_reason = str(exc) or "llm_error"

        return MutationProposal(
            mutation=None,
            used_fallback=True,
            fallback_reason=fallback_reason,
            prompt_messages=prompt_messages,
            response_text=response_text,
            llm=llm_payload,
        )

    def _build_mutation_prompt(
        self,
        *,
        task: Any,
        spec: TopologySpec,
        audit_report: dict[str, Any],
        playbook_entries: list[dict[str, Any]],
        principles: list[str] | None = None,
    ) -> list[dict[str, str]]:
        groups_summary = "\n".join(
            f"  - {group.group_id}: pattern={group.pattern}, "
            f"members=[{', '.join(group.member_ids)}]"
            + (f", parent_agent={group.parent_agent_id}" if group.parent_agent_id else "")
            + (f", leader={group.leader_id}" if group.leader_id else "")
            for group in spec.groups
        )
        findings = (
            "\n".join(
                f"  - {mode.get('mode')} [{mode.get('severity')}] "
                f"agents={mode.get('agent_ids', [])}: {mode.get('detail', '')}"
                for mode in audit_report.get("detected_modes", [])
            )
            or "  - (no findings)"
        )

        playbook_section = ""
        if playbook_entries:
            lines = [
                f"  - scope={entry.get('scope', 'long_term')} key={entry.get('key', '')}: "
                f"support={entry.get('support', '')} mutation={entry.get('mutation', '')} "
                f"notes={entry.get('notes', '')}"
                for entry in playbook_entries[:5]
            ]
            playbook_section = (
                "## Playbook repair memories (long-term and turn-level)\n"
                + "\n".join(lines)
                + "\n\n"
            )

        budget_left = int(self.se_config.max_total_agents) - len(spec.agents)
        principles_section = _principles_section(principles)
        system_msg = (
            "You are a topology planner performing the single trace-backed repair of "
            "a multi-agent system. Given the current topology and audit findings, "
            "propose ONE small mutation that addresses the strongest failure signal. "
            "You return only a compact JSON mutation; deterministic code applies and "
            'validates it. If no mutation is clearly useful, return {"ops": []}. '
            "Weigh the standing principles and playbook memories below as accumulated "
            "experience when choosing the repair."
        )
        user_msg = (
            f"## Current topology (version {spec.version})\n{groups_summary}\n\n"
            f"## Audit findings\n{findings}\n"
            f"Auditor recommendation: {audit_report.get('recommendation', '')}\n\n"
            f"{principles_section}"
            f"{playbook_section}"
            f"## Available ops (at most 3 per mutation)\n"
            f'- expand_agent_to_group: {{"op": "expand_agent_to_group", '
            f'"agent_id": "agent_1", "pattern": "star|chain|debate|voting", '
            f'"num_subagents": 3}} — the agent becomes the hub of a new subgroup.\n'
            f'- set_group_pattern: {{"op": "set_group_pattern", "group_id": "g_root", '
            f'"pattern": "debate"}}\n'
            f'- add_agent: {{"op": "add_agent", "group_id": "g_root", '
            f'"structural_role": "verifier", "stage_role": "critic"}}\n'
            f'- add_edge / remove_edge: {{"op": "add_edge", "src": "agent_1", '
            f'"dst": "agent_2"}}\n'
            f'- set_context_policy: {{"op": "set_context_policy", "agent_id": "agent_0", '
            f'"evidence_access": "global"}}\n\n'
            f"## Common repairs (symptom -> op)\n"
            f"- insufficient search coverage: add_agent (worker) to the root group, or "
            f"set_group_pattern root -> voting, so more searchers cover different facets.\n"
            f"- duplicate state-changing tool calls: set_group_pattern root -> chain so "
            f"exactly one agent executes the write (never parallelize a write tool).\n"
            f"- unverified / low-confidence answer: add_agent (structural_role verifier, "
            f"stage_role critic), or set_group_pattern root -> debate.\n"
            f"- blocked / failing branch: expand_agent_to_group on the stuck agent with "
            f"focused subtasks.\n\n"
            f"## Constraints\n"
            f"- At most {budget_left} new agents may be added in total.\n"
            f"- Keep the mutation minimal: target the audited failure, nothing else.\n\n"
            f"## Output format\n"
            f"Return ONLY one JSON object:\n"
            f'{{"rationale": "...", "ops": [{{"op": "expand_agent_to_group", '
            f'"agent_id": "agent_1", "pattern": "star", "num_subagents": 3}}]}}'
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def _mutation_from_response(
        self, text: str, audit_report: dict[str, Any]
    ) -> TopologyMutation | None:
        payload = _extract_json_payload(text or "")
        if not isinstance(payload, dict):
            return None
        raw_ops = payload.get("ops", [])
        if not isinstance(raw_ops, list) or not raw_ops:
            return None

        ops: list[MutationOp] = []
        for raw in raw_ops[:3]:
            if not isinstance(raw, dict):
                continue
            op_name = str(raw.get("op", "")).strip()
            if op_name not in MUTATION_OPS:
                continue
            args = {key: value for key, value in raw.items() if key != "op"}
            ops.append(MutationOp(op=op_name, args=args))
        if not ops:
            return None

        target_modes = tuple(
            str(mode.get("mode", ""))
            for mode in audit_report.get("detected_modes", [])
            if mode.get("mode")
        )
        return TopologyMutation(
            rationale=str(payload.get("rationale", "")).strip()[:600],
            target_failure_modes=target_modes,
            ops=tuple(ops),
        )

    # -- prompt ----------------------------------------------------------------

    def _build_initial_prompt(
        self,
        *,
        task: Any,
        benchmark_name: str,
        max_agents: int,
        playbook_entries: list[dict[str, Any]],
        principles: list[str] | None = None,
        skill_text: str = "",
    ) -> list[dict[str, str]]:
        task_preview = str(getattr(task, "prompt", "") or "")[:800]

        playbook_section = ""
        if playbook_entries:
            lines = []
            for entry in playbook_entries[:5]:
                lines.append(
                    f"  - scope={entry.get('scope', 'long_term')} "
                    f"key={entry.get('key', '')}: pattern={entry.get('pattern', '')} "
                    f"support={entry.get('support', '')} notes={entry.get('notes', '')}"
                )
            playbook_section = (
                "## Historical experience (reuse like a skill)\n"
                "Topologies that worked or failed on tasks like this in prior runs. "
                "An entry whose key names a different benchmark is transferred by task "
                "shape (same tool availability + size); weight it by its support.\n"
                + "\n".join(lines)
                + "\n\n"
            )

        principles_section = _principles_section(principles)

        # The agent-maintained markdown skill, when present, is the planner's primary
        # long-term memory: it already carries the standing principles, the how-to-choose
        # guidance, and lessons from prior runs, so it replaces the inline guidance and the
        # JSON priors. Without it, fall back to the JSON principles + entries plus the
        # built-in guidance block (keeps offline/fresh setups and tests working).
        guidance_block = (
            "## How to choose the topology (match it to your analysis)\n"
            "  - Broad retrieval / search (gather facts from many sources): provision "
            "several searcher workers (a star with workers, or voting) that each search "
            "a DIFFERENT facet or query; do not rely on a single searcher. If the clues "
            "form a dependent chain (each step needs the previous answer), use chain or "
            "debate so the reasoning assembles in shared context instead of fragmenting.\n"
            "  - Factuality / high hallucination risk: include a verifier or critic — "
            "but only one that re-checks evidence (a debate, or a critic that re-derives "
            "the answer), never a passive agent that just waits.\n"
            "  - External state mutation (create / update / delete / send / schedule / "
            "pay): exactly ONE agent may execute the mutating tool. Prefer singleton or "
            "chain; never put the same write tool behind parallel workers, debate, or "
            "voting — it double-applies and corrupts state. Reading and planning may "
            "still parallelize.\n"
            "  - Ambiguous reasoning with several defensible answers: debate or voting "
            "to surface and resolve disagreement.\n"
            "  - Complex, multi-part tasks: a star (coordinator + workers), or "
            "expansions (tree) when sub-parts themselves decompose.\n"
            "  - Simple single-step lookup or transform: a singleton.\n\n"
        )
        if skill_text and skill_text.strip():
            experience_block = (
                "## Topology planning skill (accumulated experience — consult before choosing)\n"
                + skill_text.strip()
                + "\n\n"
            )
            guidance_block = ""
        else:
            experience_block = f"{principles_section}{playbook_section}"

        system_msg = (
            "You are an expert topology planner (architect) for a multi-agent system. "
            "Given one task, you design a small, query-conditioned topology of "
            "specialized agents. Work in three steps: (1) ANALYZE the task, "
            "(2) choose the topology its analysis implies, (3) justify the choice and "
            "say what each agent does. You return only a compact JSON plan; "
            "deterministic code expands and validates it.\n"
            "Analyze along three axes:\n"
            "  - task type: retrieval/search, multi-step reasoning, coding, external "
            "tool use, state mutation, verification, planning, comparison, "
            "summarization, etc.;\n"
            "  - attributes: ambiguity, need for breadth/parallelism, need for debate, "
            "need for verification, hallucination risk, whether external state is "
            "mutated, whether tools are required, whether outputs must be aggregated;\n"
            "  - failure risks: duplicated writes, thin search coverage, premature "
            "consensus, weak verification, poor decomposition.\n"
            "Prefer the smallest topology that covers the work — extra agents cost "
            "tokens and can conflict — but DO provision enough agents to cover the task "
            "(for example, several searchers for broad retrieval). Consult the topology "
            "planning skill / accumulated experience below (standing principles, "
            "how-to-choose guidance, and lessons from prior runs); follow it unless this "
            "task clearly calls for otherwise."
        )
        user_msg = (
            f"Plan the topology for one task from the **{benchmark_name or 'unknown'}** "
            f"benchmark.\n\n"
            f"## Task preview\n{task_preview or '(no task prompt)'}\n\n"
            f"{experience_block}"
            f"{guidance_block}"
            f"## Constraints\n"
            f"- Total agents (root group plus all subgroup members) must be <= {max_agents}.\n"
            f"- Root patterns: singleton | star | chain | debate | voting.\n"
            f"- star requires num_agents >= 2 (one coordinator plus workers).\n"
            f"- debate and voting require num_agents >= 2.\n"
            f"- Optional expansions turn one root worker into the hub of a nested "
            f"subgroup (member_index counts root workers from 0, excluding the "
            f"coordinator; num_subagents >= 1, or >= 2 for debate/voting).\n"
            f'- Optional "verifier": true makes the last root worker a critic that '
            f"verifies instead of producing parallel work.\n\n"
            f"## Output format\n"
            f"Return ONLY one JSON object, no markdown fences, of the form:\n"
            f'{{"task_analysis": {{"task_type": "...", "attributes": ["..."], '
            f'"failure_risks": ["..."]}}, "rationale": "why this topology fits and what '
            f'each agent does", "pattern": "star", "num_agents": 3, "verifier": false, '
            f'"expansions": []}}'
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    # -- plan parsing / expansion -----------------------------------------------

    def _spec_from_plan_response(
        self, text: str, *, max_agents: int
    ) -> tuple[TopologySpec | None, str]:
        payload = _extract_json_payload(text or "")
        if not isinstance(payload, dict):
            return None, ""
        pattern = str(payload.get("pattern", "")).strip().lower()
        if pattern not in GROUP_PATTERNS:
            return None, ""

        rationale = str(payload.get("rationale", "")).strip()
        # Surface the planner's own task analysis in the rationale so it is visible in
        # the plan trace and the long-term playbook candidate (the model reasons first,
        # then chooses — the analysis is the chain-of-thought behind the topology).
        analysis = payload.get("task_analysis")
        if isinstance(analysis, dict):
            ttype = str(analysis.get("task_type", "")).strip()
            risks = analysis.get("failure_risks", [])
            risks_text = (
                ", ".join(str(r) for r in risks) if isinstance(risks, list) else str(risks)
            ).strip()
            prefix = "; ".join(
                part
                for part in (
                    f"type={ttype}" if ttype else "",
                    f"risks={risks_text}" if risks_text else "",
                )
                if part
            )
            if prefix:
                rationale = f"{prefix} | {rationale}" if rationale else prefix
        rationale = rationale[:600]
        try:
            requested = int(payload.get("num_agents", 1))
        except Exception:
            return None, ""

        expansions = payload.get("expansions", [])
        if not isinstance(expansions, list):
            expansions = []
        verifier = bool(payload.get("verifier", False))

        try:
            spec = self._build_spec(
                pattern=pattern,
                num_agents=requested,
                verifier=verifier,
                expansions=expansions,
                max_agents=max_agents,
                rationale=rationale or f"planner:{pattern}",
            )
        except Exception:
            logger.warning("Failed to expand topology plan", exc_info=True)
            return None, rationale
        return spec, rationale

    def _build_spec(
        self,
        *,
        pattern: str,
        num_agents: int,
        verifier: bool,
        expansions: list[Any],
        max_agents: int,
        rationale: str,
    ) -> TopologySpec:
        min_agents = 2 if pattern in {"star", "debate", "voting"} else 1
        root_size = max(min_agents, min(int(num_agents), max_agents))
        if pattern == "singleton":
            root_size = 1

        agent_ids = [f"agent_{idx}" for idx in range(root_size)]
        leader_id = agent_ids[0] if pattern in {"star", "chain"} else None
        root_workers = agent_ids[1:] if leader_id else list(agent_ids)

        structural_role = {
            "singleton": "worker",
            "star": "worker",
            "chain": "worker",
            "debate": "debater",
            "voting": "voter",
        }[pattern]

        agents: list[AgentNode] = []
        for agent_id in agent_ids:
            if agent_id == leader_id:
                agents.append(
                    AgentNode(agent_id=agent_id, group_id="g_root", structural_role="coordinator")
                )
            else:
                agents.append(
                    AgentNode(agent_id=agent_id, group_id="g_root", structural_role=structural_role)
                )
        if verifier and root_workers:
            verifier_id = root_workers[-1]
            agents = [
                AgentNode(
                    agent_id=node.agent_id,
                    group_id=node.group_id,
                    structural_role="verifier",
                    stage_role="critic",
                )
                if node.agent_id == verifier_id
                else node
                for node in agents
            ]

        groups: list[GroupSpec] = [
            GroupSpec(
                group_id="g_root",
                pattern=pattern,
                member_ids=tuple(agent_ids),
                leader_id=leader_id,
            )
        ]

        next_index = root_size
        expanded_parents: set[str] = set()
        for raw in expansions:
            if not isinstance(raw, dict):
                continue
            sub_pattern = str(raw.get("pattern", "")).strip().lower()
            if sub_pattern not in GROUP_PATTERNS or sub_pattern == "singleton":
                continue
            try:
                member_index = int(raw.get("member_index", -1))
                sub_count = int(raw.get("num_subagents", 0))
            except Exception:
                continue
            if not 0 <= member_index < len(root_workers):
                continue
            parent_id = root_workers[member_index]
            if parent_id in expanded_parents:
                continue
            min_sub = 2 if sub_pattern in {"debate", "voting"} else 1
            sub_count = max(min_sub, min(sub_count, max_agents - next_index))
            if next_index + sub_count > max_agents:
                continue
            group_id = f"g_{parent_id}"
            member_ids = tuple(f"agent_{next_index + offset}" for offset in range(sub_count))
            sub_role = {
                "star": "worker",
                "chain": "worker",
                "debate": "debater",
                "voting": "voter",
            }[sub_pattern]
            for member_id in member_ids:
                agents.append(
                    AgentNode(agent_id=member_id, group_id=group_id, structural_role=sub_role)
                )
            groups.append(
                GroupSpec(
                    group_id=group_id,
                    pattern=sub_pattern,
                    member_ids=member_ids,
                    parent_agent_id=parent_id,
                )
            )
            expanded_parents.add(parent_id)
            next_index += sub_count

        spec = TopologySpec(
            version=0,
            agents=tuple(agents),
            groups=tuple(groups),
            root_group_id="g_root",
            rationale=rationale,
        )
        spec.validate(max_agents=max_agents)
        return spec
