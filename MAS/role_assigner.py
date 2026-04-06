"""LLM-based dynamic role assignment for multi-agent topologies.

Before the main workflow executes, this module asks an LLM to map each
agent in the topology to a domain-specific role drawn from the benchmark
role pool.  A deterministic round-robin fallback is provided for offline
or mock-mode runs.

Reference
---------
Inspired by the communication-aware agent design in
*Cut the Crap: An Economical Communication Pipeline for LLM-based
Multi-Agent Systems* (arXiv 2410.02506).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .relay import TopologyLayout
from .role_pools import DomainRole, get_role_pool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonaAssignment:
    """The domain role assigned to one agent."""

    role_name: str
    persona: str


@dataclass(frozen=True)
class RoleAssignmentResult:
    """Role-assignment output plus the planner trace used to produce it."""

    assignments: dict[str, PersonaAssignment]
    prompt_messages: list[dict[str, str]]
    response_text: str
    llm: dict[str, Any]
    used_fallback: bool
    fallback_reason: str


# ── public API ────────────────────────────────────────────────────────

def assign_domain_roles(
    *,
    benchmark_name: str,
    layout: TopologyLayout,
    task_prompt: Any,
    llm_client: Any,
) -> RoleAssignmentResult:
    """Use an LLM to assign domain-specific personas to agents.

    Falls back to :func:`assign_domain_roles_deterministic` when the LLM
    call fails or the response cannot be parsed.
    """
    role_pool = get_role_pool(benchmark_name)
    if not role_pool:
        return RoleAssignmentResult(
            assignments={},
            prompt_messages=[],
            response_text="",
            llm={},
            used_fallback=False,
            fallback_reason="no_role_pool",
        )

    prompt_messages = _build_assignment_prompt(
        benchmark_name=benchmark_name,
        layout=layout,
        role_pool=role_pool,
        task_prompt=task_prompt,
    )
    response_text = ""
    llm_payload: dict[str, Any] = {}
    fallback_reason = ""

    try:
        result = llm_client.generate(
            prompt=prompt_messages,
            agent_type="default",
            task_id="role_assignment",
            run_index=0,
            agent_id="role_assigner",
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
        assignments = _parse_assignment_response(
            response_text, layout.agent_ids, role_pool,
        )
        if assignments:
            logger.info("Dynamic role assignment succeeded for %s", benchmark_name)
            return RoleAssignmentResult(
                assignments=assignments,
                prompt_messages=prompt_messages,
                response_text=response_text,
                llm=llm_payload,
                used_fallback=False,
                fallback_reason="",
            )
        fallback_reason = "invalid_or_incomplete_response"
    except Exception as exc:
        logger.warning(
            "LLM role assignment failed for %s; using deterministic fallback",
            benchmark_name,
            exc_info=True,
        )
        fallback_reason = str(exc) or "llm_error"

    return RoleAssignmentResult(
        assignments=assign_domain_roles_deterministic(
            benchmark_name=benchmark_name, layout=layout,
        ),
        prompt_messages=prompt_messages,
        response_text=response_text,
        llm=llm_payload,
        used_fallback=True,
        fallback_reason=fallback_reason,
    )


def assign_domain_roles_deterministic(
    *,
    benchmark_name: str,
    layout: TopologyLayout,
) -> dict[str, PersonaAssignment]:
    """Deterministic fallback: round-robin assign roles without an LLM."""
    role_pool = get_role_pool(benchmark_name)
    if not role_pool:
        return {}

    assignments: dict[str, PersonaAssignment] = {}
    for idx, agent_id in enumerate(layout.agent_ids):
        role = role_pool[idx % len(role_pool)]
        assignments[agent_id] = PersonaAssignment(
            role_name=role.name,
            persona=role.persona,
        )
    return assignments


# ── prompt construction ───────────────────────────────────────────────

def _build_assignment_prompt(
    *,
    benchmark_name: str,
    layout: TopologyLayout,
    role_pool: list[DomainRole],
    task_prompt: Any,
) -> list[dict[str, str]]:
    """Build the chat-format messages sent to the assigner LLM."""

    # Summarise agent structural roles
    agent_lines = "\n".join(
        f"  - {aid}: structural_role={layout.roles.get(aid, 'agent')}, "
        f"level={layout.level_by_agent.get(aid, 0)}"
        for aid in layout.agent_ids
    )

    # Summarise adjacency
    adj_lines = "\n".join(
        f"  - {sender} -> {', '.join(receivers)}"
        for sender, receivers in layout.adjacency.items()
        if receivers
    )
    if not adj_lines:
        adj_lines = "  (no inter-agent links)"

    # Format role pool
    pool_lines = "\n".join(
        f"  {idx + 1}. **{role.name}** — {role.persona}"
        for idx, role in enumerate(role_pool)
    )

    # Truncate task prompt preview
    task_preview = str(task_prompt)[:600] if task_prompt else "(no task prompt)"

    system_msg = (
        "You are a multi-agent system architect. Your task is to assign "
        "domain-specific roles to agents in a multi-agent topology. "
        "Structural workflow contracts remain authoritative; personas should "
        "specialize how each agent approaches its assigned stage, especially "
        "for tool use, evidence gathering, verification, and synthesis."
    )

    user_msg = (
        f"Assign domain-specific roles to the agents in a "
        f"**{layout.topology}** topology for the **{benchmark_name}** "
        f"benchmark.\n\n"
        f"## Topology Structure\n"
        f"- Topology type: {layout.topology}\n"
        f"- Number of agents: {len(layout.agent_ids)}\n"
        f"- Agents (id, structural role, hierarchy level):\n{agent_lines}\n"
        f"- Communication links:\n{adj_lines}\n\n"
        f"## Task Preview\n{task_preview}\n\n"
        f"## Available Domain Roles\n{pool_lines}\n\n"
        f"## Instructions\n"
        f"Assign exactly one domain role to each agent. Consider:\n"
        f"1. The agent's structural position — orchestrators / root nodes "
        f"benefit from coordinator or planner roles; leaf workers benefit "
        f"from specialist or executor roles.\n"
        f"2. The specific task requirements — which expertise areas are "
        f"most needed for this task.\n"
        f"2a. The likely tool workflow — retrieval-heavy tasks should map "
        f"querying, evidence-reading, verification, and synthesis roles to "
        f"different agents when the topology supports specialization.\n"
        f"3. Diversity — avoid assigning the same role to all agents "
        f"unless the topology requires uniform roles (e.g. voting).\n"
        f"4. For hierarchical topologies: higher-level agents should have "
        f"broader, coordinating roles; lower-level agents should have "
        f"focused, execution-oriented roles.\n"
        f"5. You may assign the same role to multiple agents if the "
        f"topology has more agents than available roles.\n\n"
        f"Return a JSON object mapping each agent_id to a role name from "
        f"the pool above. Example:\n"
        f'{{"agent_0": "Role Name A", "agent_1": "Role Name B"}}\n\n'
        f"Return ONLY the JSON object, no other text."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ── response parsing ─────────────────────────────────────────────────

def _parse_assignment_response(
    text: str,
    agent_ids: list[str],
    role_pool: list[DomainRole],
) -> dict[str, PersonaAssignment]:
    """Parse the LLM JSON response into validated PersonaAssignment mappings."""
    if not text:
        return {}

    # Extract JSON object from potentially noisy LLM output
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        logger.warning("No JSON object found in role assignment response")
        return {}

    try:
        raw: dict[str, str] = json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("Failed to parse role assignment JSON")
        return {}

    # Build lookup: role_name (case-insensitive) -> DomainRole
    pool_lookup: dict[str, DomainRole] = {
        role.name.lower(): role for role in role_pool
    }

    assignments: dict[str, PersonaAssignment] = {}
    for agent_id in agent_ids:
        role_name = raw.get(agent_id, "")
        matched_role = pool_lookup.get(role_name.lower().strip())
        if matched_role:
            assignments[agent_id] = PersonaAssignment(
                role_name=matched_role.name,
                persona=matched_role.persona,
            )

    # If we failed to assign any agent, signal failure so caller can
    # fall back to deterministic assignment.
    if len(assignments) < len(agent_ids):
        missing = set(agent_ids) - set(assignments)
        logger.warning(
            "Role assignment incomplete — missing agents: %s; "
            "falling back to deterministic",
            missing,
        )
        return {}

    return assignments
