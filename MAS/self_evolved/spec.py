"""General topology specification for the self-evolved system.

A ``TopologySpec`` is a superset of the seven uniform layouts in ``MAS.relay``:
agents are organized into pattern groups (singleton/star/chain/debate/voting),
and any agent may own an attached subgroup that performs its contribution
(the doc's "leaf becomes a star-orchestrator with 3 subagents" mutation).
``TopologySpec.to_layout()`` projects the spec onto a regular ``TopologyLayout``
so the existing stage executor, trace writers, and role assigner work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..relay import (
    TOPOLOGY_FULLY_LINKED_DEBATE,
    TOPOLOGY_ONLY_VOTING,
    TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION,
    TOPOLOGY_ORCHESTRATOR_TREE,
    TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION,
    TOPOLOGY_SAS,
    TOPOLOGY_SELF_EVOLVED,
    TopologyLayout,
)

GROUP_PATTERNS = frozenset({"singleton", "star", "chain", "debate", "voting"})
CONTRIBUTION_STAGE_ROLES = frozenset({"worker", "critic"})


@dataclass(frozen=True)
class ContextPolicy:
    """Per-agent visibility policy applied by the SharedContextController.

    Empty tuples mean "no extra restriction" (the structural packet routing
    already bounds what an agent can receive).
    """

    visible_kinds: tuple[str, ...] = ()
    visible_from: tuple[str, ...] = ()  # agent ids or tokens: parent, children, group
    share_scope: str = "group"  # parent_only | group | global
    evidence_access: str = "own"  # own | branch | global
    summary_only: bool = False
    max_packet_chars: int = 0  # 0 = use the engine default


DEFAULT_POLICY = ContextPolicy()


@dataclass(frozen=True)
class AgentNode:
    agent_id: str
    group_id: str
    structural_role: str = "worker"  # coordinator | worker | verifier | debater | voter
    stage_role: str = "worker"  # contribution stage contract: worker | critic
    persona_hint: str = ""
    allowed_tools: tuple[str, ...] | None = None  # None = all run tools
    context: ContextPolicy = DEFAULT_POLICY


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    pattern: str  # singleton | star | chain | debate | voting
    member_ids: tuple[str, ...]
    parent_agent_id: str | None = None  # attachment point in another group; None = root
    leader_id: str | None = None  # required for a root star group


@dataclass(frozen=True)
class TopologySpec:
    """Versioned description of the dynamic target agent system."""

    version: int
    agents: tuple[AgentNode, ...]
    groups: tuple[GroupSpec, ...]
    root_group_id: str
    extra_edges: tuple[tuple[str, str], ...] = field(default=())
    rationale: str = ""

    # -- lookups -----------------------------------------------------------

    def agent(self, agent_id: str) -> AgentNode:
        for node in self.agents:
            if node.agent_id == agent_id:
                return node
        raise KeyError(f"Unknown agent_id '{agent_id}'")

    def group(self, group_id: str) -> GroupSpec:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(f"Unknown group_id '{group_id}'")

    def subgroup_of(self, agent_id: str) -> GroupSpec | None:
        """Return the subgroup attached to this agent, if any."""

        for group in self.groups:
            if group.parent_agent_id == agent_id:
                return group
        return None

    def group_depth(self, group_id: str) -> int:
        depth = 0
        group = self.group(group_id)
        while group.parent_agent_id is not None:
            group = self.group(self.agent(group.parent_agent_id).group_id)
            depth += 1
        return depth

    # -- validation --------------------------------------------------------

    def validate(self, *, max_agents: int) -> None:
        if not self.agents:
            raise ValueError("TopologySpec must contain at least one agent")
        if len(self.agents) > max_agents:
            raise ValueError(f"TopologySpec has {len(self.agents)} agents; the cap is {max_agents}")

        agent_ids = [node.agent_id for node in self.agents]
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("TopologySpec agent ids must be unique")
        group_ids = [group.group_id for group in self.groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("TopologySpec group ids must be unique")

        groups_by_id = {group.group_id: group for group in self.groups}
        root = groups_by_id.get(self.root_group_id)
        if root is None:
            raise ValueError(f"root_group_id '{self.root_group_id}' is not a declared group")
        if root.parent_agent_id is not None:
            raise ValueError("The root group must not have a parent_agent_id")

        agent_id_set = set(agent_ids)
        membership: dict[str, str] = {}
        parents_seen: set[str] = set()
        for group in self.groups:
            if group.pattern not in GROUP_PATTERNS:
                raise ValueError(
                    f"Group '{group.group_id}' has unsupported pattern '{group.pattern}'"
                )
            if not group.member_ids:
                raise ValueError(f"Group '{group.group_id}' has no members")
            if group.pattern == "singleton" and len(group.member_ids) != 1:
                raise ValueError(f"Singleton group '{group.group_id}' must have exactly 1 member")
            if group.pattern in {"debate", "voting"} and len(group.member_ids) < 2:
                raise ValueError(
                    f"{group.pattern} group '{group.group_id}' needs at least 2 members"
                )
            for member in group.member_ids:
                if member not in agent_id_set:
                    raise ValueError(
                        f"Group '{group.group_id}' references unknown agent '{member}'"
                    )
                if member in membership:
                    raise ValueError(f"Agent '{member}' belongs to more than one group")
                membership[member] = group.group_id
            if group.parent_agent_id is not None:
                parent = group.parent_agent_id
                if parent not in agent_id_set:
                    raise ValueError(
                        f"Group '{group.group_id}' attaches to unknown agent '{parent}'"
                    )
                if parent in group.member_ids:
                    raise ValueError(
                        f"Group '{group.group_id}' cannot contain its own parent agent"
                    )
                if parent in parents_seen:
                    raise ValueError(f"Agent '{parent}' owns more than one subgroup")
                parents_seen.add(parent)
            is_root_star = group.group_id == self.root_group_id and group.pattern == "star"
            if is_root_star and (
                group.leader_id is None or group.leader_id not in group.member_ids
            ):
                raise ValueError("A root star group requires leader_id among its members")
            if group.leader_id is not None and group.leader_id not in group.member_ids:
                raise ValueError(
                    f"Group '{group.group_id}' leader '{group.leader_id}' is not a member"
                )

        for node in self.agents:
            if membership.get(node.agent_id) != node.group_id:
                raise ValueError(
                    f"Agent '{node.agent_id}' declares group '{node.group_id}' but group "
                    f"membership says '{membership.get(node.agent_id)}'"
                )
            if node.stage_role not in CONTRIBUTION_STAGE_ROLES:
                raise ValueError(
                    f"Agent '{node.agent_id}' stage_role must be one of "
                    f"{sorted(CONTRIBUTION_STAGE_ROLES)}"
                )

        # Attachment graph must be a tree rooted at the root group.
        for group in self.groups:
            seen: set[str] = set()
            cursor = group
            while cursor.parent_agent_id is not None:
                if cursor.group_id in seen:
                    raise ValueError("Group attachment graph contains a cycle")
                seen.add(cursor.group_id)
                cursor = groups_by_id[membership[cursor.parent_agent_id]]
            if cursor.group_id != self.root_group_id:
                raise ValueError(f"Group '{group.group_id}' is not attached to the root group")

        for src, dst in self.extra_edges:
            if src not in agent_id_set or dst not in agent_id_set:
                raise ValueError(f"extra_edges references unknown agents: ({src}, {dst})")
            if src == dst:
                raise ValueError("extra_edges must connect two distinct agents")

    # -- projections -------------------------------------------------------

    def ordered_agent_ids(self) -> list[str]:
        """Deterministic execution-order traversal: leader first, then members,
        recursing into each member's attached subgroup."""

        ordered: list[str] = []

        def visit_group(group: GroupSpec) -> None:
            members = list(group.member_ids)
            if group.leader_id and group.leader_id in members:
                members.remove(group.leader_id)
                members.insert(0, group.leader_id)
            for member in members:
                ordered.append(member)
            for member in members:
                subgroup = self.subgroup_of(member)
                if subgroup is not None:
                    visit_group(subgroup)

        visit_group(self.group(self.root_group_id))
        return ordered

    def to_layout(self) -> TopologyLayout:
        ordered_ids = self.ordered_agent_ids()
        adjacency: dict[str, set[str]] = {agent_id: set() for agent_id in ordered_ids}
        roles: dict[str, str] = {}
        level_by_agent: dict[str, int] = {}
        parent_by_agent: dict[str, str] = {}
        children_by_agent: dict[str, list[str]] = {agent_id: [] for agent_id in ordered_ids}
        layout_groups: list[list[str]] = []
        representatives: list[str] = []

        def link(a: str, b: str) -> None:
            adjacency[a].add(b)
            adjacency[b].add(a)

        root = self.group(self.root_group_id)
        orchestrator_id = root.leader_id if root.pattern in {"star", "chain"} else None
        if root.pattern == "singleton":
            orchestrator_id = None

        def base_level(group: GroupSpec) -> int:
            if group.parent_agent_id is None:
                return 0
            return level_by_agent[group.parent_agent_id] + 1

        def visit_group(group: GroupSpec) -> None:
            level = base_level(group)
            members = list(group.member_ids)
            if group.leader_id and group.leader_id in members:
                members.remove(group.leader_id)
                members.insert(0, group.leader_id)

            for member in members:
                node = self.agent(member)
                roles[member] = node.structural_role
                is_leader = member == group.leader_id
                level_by_agent[member] = (
                    level if is_leader else (level + 1 if group.leader_id else level)
                )

            if group.pattern in {"star", "chain"} and group.leader_id:
                for member in members[1:]:
                    link(group.leader_id, member)
                    parent_by_agent[member] = group.leader_id
                    children_by_agent[group.leader_id].append(member)
            if group.pattern == "chain":
                chain_members = members[1:] if group.leader_id else members
                for prev, nxt in zip(chain_members, chain_members[1:]):
                    link(prev, nxt)
            if group.pattern == "debate":
                for idx, src in enumerate(members):
                    for dst in members[idx + 1 :]:
                        link(src, dst)
                layout_groups.append(list(members))

            if group.parent_agent_id is not None:
                parent = group.parent_agent_id
                representatives.append(parent)
                for member in members:
                    link(parent, member)
                    parent_by_agent.setdefault(member, parent)
                    children_by_agent[parent].append(member)

            for member in members:
                subgroup = self.subgroup_of(member)
                if subgroup is not None:
                    visit_group(subgroup)

        visit_group(root)

        for src, dst in self.extra_edges:
            link(src, dst)

        if orchestrator_id:
            roles[orchestrator_id] = self.agent(orchestrator_id).structural_role

        specialists = [agent_id for agent_id in ordered_ids if agent_id != orchestrator_id]
        managers = sorted({group.parent_agent_id for group in self.groups if group.parent_agent_id})
        leaves = [
            agent_id
            for agent_id in ordered_ids
            if self.subgroup_of(agent_id) is None
            and agent_id != orchestrator_id
            and agent_id not in managers
        ]

        max_level = max(level_by_agent.values(), default=0)
        agents_per_level = [0] * (max_level + 1)
        for level in level_by_agent.values():
            agents_per_level[level] += 1

        return TopologyLayout(
            topology=TOPOLOGY_SELF_EVOLVED,
            agent_ids=ordered_ids,
            adjacency={key: sorted(value) for key, value in adjacency.items()},
            roles=roles,
            level_by_agent=level_by_agent,
            topological_order=list(ordered_ids),
            orchestrator_id=orchestrator_id,
            specialists=specialists,
            managers=managers,
            leaves=leaves,
            groups=layout_groups,
            representatives=sorted(set(representatives)),
            parent_by_agent=parent_by_agent,
            children_by_agent={
                key: sorted(value) for key, value in children_by_agent.items() if value
            },
            agents_per_level=agents_per_level,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "version": int(self.version),
            "root_group_id": self.root_group_id,
            "rationale": self.rationale,
            "agents": [
                {
                    "agent_id": node.agent_id,
                    "group_id": node.group_id,
                    "structural_role": node.structural_role,
                    "stage_role": node.stage_role,
                    "persona_hint": node.persona_hint,
                    "allowed_tools": (
                        list(node.allowed_tools) if node.allowed_tools is not None else None
                    ),
                    "context": {
                        "visible_kinds": list(node.context.visible_kinds),
                        "visible_from": list(node.context.visible_from),
                        "share_scope": node.context.share_scope,
                        "evidence_access": node.context.evidence_access,
                        "summary_only": bool(node.context.summary_only),
                        "max_packet_chars": int(node.context.max_packet_chars),
                    },
                }
                for node in self.agents
            ],
            "groups": [
                {
                    "group_id": group.group_id,
                    "pattern": group.pattern,
                    "member_ids": list(group.member_ids),
                    "parent_agent_id": group.parent_agent_id,
                    "leader_id": group.leader_id,
                }
                for group in self.groups
            ],
            "extra_edges": [list(edge) for edge in self.extra_edges],
        }


def spec_from_layout(layout: TopologyLayout, *, rationale: str = "") -> TopologySpec:
    """Convert a uniform ``TopologyLayout`` into an equivalent ``TopologySpec``.

    Used for deterministic planner fallbacks so offline runs stay reproducible.
    """

    topo = layout.topology
    agent_ids = list(layout.agent_ids)

    if topo == TOPOLOGY_SAS or len(agent_ids) == 1:
        root = GroupSpec(group_id="g_root", pattern="singleton", member_ids=(agent_ids[0],))
        agents = (AgentNode(agent_id=agent_ids[0], group_id="g_root", structural_role="worker"),)
        return TopologySpec(
            version=0,
            agents=agents,
            groups=(root,),
            root_group_id="g_root",
            rationale=rationale or f"fallback:{topo}",
        )

    if topo in {TOPOLOGY_ORCHESTRATOR_NO_DISCUSSION, TOPOLOGY_ORCHESTRATOR_WITH_DISCUSSION}:
        leader = layout.orchestrator_id or agent_ids[0]
        root = GroupSpec(
            group_id="g_root",
            pattern="star",
            member_ids=tuple(agent_ids),
            leader_id=leader,
        )
        agents = tuple(
            AgentNode(
                agent_id=agent_id,
                group_id="g_root",
                structural_role="coordinator" if agent_id == leader else "worker",
            )
            for agent_id in agent_ids
        )
        return TopologySpec(
            version=0,
            agents=agents,
            groups=(root,),
            root_group_id="g_root",
            rationale=rationale or f"fallback:{topo}",
        )

    if topo == TOPOLOGY_ONLY_VOTING:
        root = GroupSpec(group_id="g_root", pattern="voting", member_ids=tuple(agent_ids))
        agents = tuple(
            AgentNode(agent_id=agent_id, group_id="g_root", structural_role="voter")
            for agent_id in agent_ids
        )
        return TopologySpec(
            version=0,
            agents=agents,
            groups=(root,),
            root_group_id="g_root",
            rationale=rationale or f"fallback:{topo}",
        )

    if topo == TOPOLOGY_FULLY_LINKED_DEBATE:
        root = GroupSpec(group_id="g_root", pattern="debate", member_ids=tuple(agent_ids))
        agents = tuple(
            AgentNode(agent_id=agent_id, group_id="g_root", structural_role="debater")
            for agent_id in agent_ids
        )
        return TopologySpec(
            version=0,
            agents=agents,
            groups=(root,),
            root_group_id="g_root",
            rationale=rationale or f"fallback:{topo}",
        )

    if topo == TOPOLOGY_ORCHESTRATOR_TREE:
        leader = layout.orchestrator_id or agent_ids[0]
        managers = list(layout.managers)
        root_members = tuple([leader] + managers)
        groups: list[GroupSpec] = [
            GroupSpec(
                group_id="g_root",
                pattern="star",
                member_ids=root_members,
                leader_id=leader,
            )
        ]
        agents: list[AgentNode] = [
            AgentNode(agent_id=leader, group_id="g_root", structural_role="coordinator")
        ]
        for manager in managers:
            agents.append(
                AgentNode(agent_id=manager, group_id="g_root", structural_role="coordinator")
            )
            children = tuple(layout.children_by_agent.get(manager, []))
            if not children:
                continue
            group_id = f"g_{manager}"
            groups.append(
                GroupSpec(
                    group_id=group_id,
                    pattern="star",
                    member_ids=children,
                    parent_agent_id=manager,
                )
            )
            for child in children:
                agents.append(
                    AgentNode(agent_id=child, group_id=group_id, structural_role="worker")
                )
        return TopologySpec(
            version=0,
            agents=tuple(agents),
            groups=tuple(groups),
            root_group_id="g_root",
            rationale=rationale or f"fallback:{topo}",
        )

    raise ValueError(f"spec_from_layout does not support topology '{topo}'")


# ── topology mutations ──────────────────────────────────────────────────

MUTATION_OPS = frozenset(
    {
        "expand_agent_to_group",
        "set_group_pattern",
        "add_agent",
        "add_edge",
        "remove_edge",
        "set_context_policy",
    }
)

_ROLE_BY_PATTERN = {
    "singleton": "worker",
    "star": "worker",
    "chain": "worker",
    "debate": "debater",
    "voting": "voter",
}


@dataclass(frozen=True)
class MutationOp:
    op: str
    args: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {"op": self.op, "args": dict(self.args)}


@dataclass(frozen=True)
class TopologyMutation:
    """One trace-backed repair: an ordered list of validated mutation ops."""

    rationale: str
    target_failure_modes: tuple[str, ...]
    ops: tuple[MutationOp, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "rationale": self.rationale,
            "target_failure_modes": list(self.target_failure_modes),
            "ops": [op.to_payload() for op in self.ops],
        }

    def apply(self, spec: TopologySpec, *, max_agents: int) -> TopologySpec:
        if not self.ops:
            raise ValueError("TopologyMutation has no ops")
        agents = list(spec.agents)
        groups = list(spec.groups)
        extra_edges = list(spec.extra_edges)

        for op in self.ops:
            if op.op not in MUTATION_OPS:
                raise ValueError(f"Unsupported mutation op '{op.op}'")
            handler = getattr(self, f"_apply_{op.op}")
            agents, groups, extra_edges = handler(op.args, agents, groups, extra_edges)

        mutated = TopologySpec(
            version=spec.version + 1,
            agents=tuple(agents),
            groups=tuple(groups),
            root_group_id=spec.root_group_id,
            extra_edges=tuple(extra_edges),
            rationale=self.rationale,
        )
        mutated.validate(max_agents=max_agents)
        return mutated

    # -- op handlers; each returns the updated (agents, groups, extra_edges) --

    @staticmethod
    def _next_agent_index(agents: list[AgentNode]) -> int:
        highest = -1
        for node in agents:
            tail = node.agent_id.rsplit("_", 1)[-1]
            if tail.isdigit():
                highest = max(highest, int(tail))
        return highest + 1

    def _apply_expand_agent_to_group(self, args, agents, groups, extra_edges):
        agent_id = str(args.get("agent_id", ""))
        pattern = str(args.get("pattern", "star")).strip().lower()
        count = int(args.get("num_subagents", 2))
        if pattern not in GROUP_PATTERNS or pattern == "singleton":
            raise ValueError(f"expand_agent_to_group: unsupported pattern '{pattern}'")
        if not any(node.agent_id == agent_id for node in agents):
            raise ValueError(f"expand_agent_to_group: unknown agent '{agent_id}'")
        if any(group.parent_agent_id == agent_id for group in groups):
            raise ValueError(f"expand_agent_to_group: agent '{agent_id}' already owns a subgroup")
        count = max(2 if pattern in {"debate", "voting"} else 1, count)

        start = self._next_agent_index(agents)
        group_id = f"g_{agent_id}"
        member_ids = tuple(f"agent_{start + offset}" for offset in range(count))
        role = _ROLE_BY_PATTERN[pattern]
        for member_id in member_ids:
            agents.append(AgentNode(agent_id=member_id, group_id=group_id, structural_role=role))
        groups.append(
            GroupSpec(
                group_id=group_id,
                pattern=pattern,
                member_ids=member_ids,
                parent_agent_id=agent_id,
            )
        )
        return agents, groups, extra_edges

    @staticmethod
    def _apply_set_group_pattern(args, agents, groups, extra_edges):
        group_id = str(args.get("group_id", ""))
        pattern = str(args.get("pattern", "")).strip().lower()
        if pattern not in GROUP_PATTERNS:
            raise ValueError(f"set_group_pattern: unsupported pattern '{pattern}'")
        updated = []
        found = False
        for group in groups:
            if group.group_id != group_id:
                updated.append(group)
                continue
            found = True
            leader = group.leader_id
            if pattern in {"star", "chain"}:
                leader = leader if leader in group.member_ids else group.member_ids[0]
            else:
                leader = None
            updated.append(
                GroupSpec(
                    group_id=group.group_id,
                    pattern=pattern,
                    member_ids=group.member_ids,
                    parent_agent_id=group.parent_agent_id,
                    leader_id=leader,
                )
            )
        if not found:
            raise ValueError(f"set_group_pattern: unknown group '{group_id}'")
        return agents, updated, extra_edges

    def _apply_add_agent(self, args, agents, groups, extra_edges):
        group_id = str(args.get("group_id", ""))
        structural_role = str(args.get("structural_role", "worker"))
        stage_role = str(args.get("stage_role", "worker"))
        target = next((group for group in groups if group.group_id == group_id), None)
        if target is None:
            raise ValueError(f"add_agent: unknown group '{group_id}'")
        agent_id = f"agent_{self._next_agent_index(agents)}"
        agents.append(
            AgentNode(
                agent_id=agent_id,
                group_id=group_id,
                structural_role=structural_role,
                stage_role=stage_role,
            )
        )
        updated = [
            GroupSpec(
                group_id=group.group_id,
                pattern=group.pattern,
                member_ids=group.member_ids + (agent_id,),
                parent_agent_id=group.parent_agent_id,
                leader_id=group.leader_id,
            )
            if group.group_id == group_id
            else group
            for group in groups
        ]
        return agents, updated, extra_edges

    @staticmethod
    def _apply_add_edge(args, agents, groups, extra_edges):
        src = str(args.get("src", ""))
        dst = str(args.get("dst", ""))
        edge = (src, dst)
        if edge not in extra_edges and (dst, src) not in extra_edges:
            extra_edges.append(edge)
        return agents, groups, extra_edges

    @staticmethod
    def _apply_remove_edge(args, agents, groups, extra_edges):
        src = str(args.get("src", ""))
        dst = str(args.get("dst", ""))
        extra_edges = [edge for edge in extra_edges if edge not in {(src, dst), (dst, src)}]
        return agents, groups, extra_edges

    @staticmethod
    def _apply_set_context_policy(args, agents, groups, extra_edges):
        agent_id = str(args.get("agent_id", ""))
        target = next((node for node in agents if node.agent_id == agent_id), None)
        if target is None:
            raise ValueError(f"set_context_policy: unknown agent '{agent_id}'")
        policy = target.context
        updated_policy = ContextPolicy(
            visible_kinds=tuple(args.get("visible_kinds", policy.visible_kinds)),
            visible_from=tuple(args.get("visible_from", policy.visible_from)),
            share_scope=str(args.get("share_scope", policy.share_scope)),
            evidence_access=str(args.get("evidence_access", policy.evidence_access)),
            summary_only=bool(args.get("summary_only", policy.summary_only)),
            max_packet_chars=int(args.get("max_packet_chars", policy.max_packet_chars)),
        )
        agents = [
            AgentNode(
                agent_id=node.agent_id,
                group_id=node.group_id,
                structural_role=node.structural_role,
                stage_role=node.stage_role,
                persona_hint=node.persona_hint,
                allowed_tools=node.allowed_tools,
                context=updated_policy,
            )
            if node.agent_id == agent_id
            else node
            for node in agents
        ]
        return agents, groups, extra_edges
