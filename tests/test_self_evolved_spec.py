import unittest

from MAS.relay import build_layout
from MAS.self_evolved.spec import (
    AgentNode,
    GroupSpec,
    TopologySpec,
    spec_from_layout,
)


def _heterogeneous_spec() -> TopologySpec:
    """Root star whose leaf agent_1 expands into a star subgroup and whose
    leaf agent_2 expands into a fully-linked debate subgroup (the doc's
    mutation example, expressed as a static spec)."""

    agents = (
        AgentNode(agent_id="agent_0", group_id="g_root", structural_role="coordinator"),
        AgentNode(agent_id="agent_1", group_id="g_root", structural_role="coordinator"),
        AgentNode(agent_id="agent_2", group_id="g_root", structural_role="coordinator"),
        AgentNode(agent_id="agent_3", group_id="g_root", structural_role="worker"),
        AgentNode(agent_id="agent_4", group_id="g_star", structural_role="worker"),
        AgentNode(agent_id="agent_5", group_id="g_star", structural_role="worker"),
        AgentNode(agent_id="agent_6", group_id="g_star", structural_role="worker"),
        AgentNode(agent_id="agent_7", group_id="g_debate", structural_role="debater"),
        AgentNode(agent_id="agent_8", group_id="g_debate", structural_role="debater"),
    )
    groups = (
        GroupSpec(
            group_id="g_root",
            pattern="star",
            member_ids=("agent_0", "agent_1", "agent_2", "agent_3"),
            leader_id="agent_0",
        ),
        GroupSpec(
            group_id="g_star",
            pattern="star",
            member_ids=("agent_4", "agent_5", "agent_6"),
            parent_agent_id="agent_1",
        ),
        GroupSpec(
            group_id="g_debate",
            pattern="debate",
            member_ids=("agent_7", "agent_8"),
            parent_agent_id="agent_2",
        ),
    )
    return TopologySpec(version=0, agents=agents, groups=groups, root_group_id="g_root")


class TestTopologySpecValidation(unittest.TestCase):
    def test_heterogeneous_nested_spec_validates(self) -> None:
        spec = _heterogeneous_spec()
        spec.validate(max_agents=10)

    def test_agent_cap_enforced(self) -> None:
        spec = _heterogeneous_spec()
        with self.assertRaises(ValueError):
            spec.validate(max_agents=3)

    def test_debate_group_needs_two_members(self) -> None:
        agents = (AgentNode(agent_id="a", group_id="g_root", structural_role="debater"),)
        groups = (GroupSpec(group_id="g_root", pattern="debate", member_ids=("a",)),)
        spec = TopologySpec(version=0, agents=agents, groups=groups, root_group_id="g_root")
        with self.assertRaises(ValueError):
            spec.validate(max_agents=10)

    def test_root_star_requires_leader(self) -> None:
        agents = (
            AgentNode(agent_id="a", group_id="g_root"),
            AgentNode(agent_id="b", group_id="g_root"),
        )
        groups = (GroupSpec(group_id="g_root", pattern="star", member_ids=("a", "b")),)
        spec = TopologySpec(version=0, agents=agents, groups=groups, root_group_id="g_root")
        with self.assertRaises(ValueError):
            spec.validate(max_agents=10)

    def test_duplicate_membership_rejected(self) -> None:
        agents = (
            AgentNode(agent_id="a", group_id="g_root"),
            AgentNode(agent_id="b", group_id="g_root"),
        )
        groups = (
            GroupSpec(group_id="g_root", pattern="voting", member_ids=("a", "b")),
            GroupSpec(
                group_id="g_other",
                pattern="voting",
                member_ids=("a", "b"),
                parent_agent_id="a",
            ),
        )
        spec = TopologySpec(version=0, agents=agents, groups=groups, root_group_id="g_root")
        with self.assertRaises(ValueError):
            spec.validate(max_agents=10)

    def test_subgroup_cannot_contain_its_parent(self) -> None:
        agents = (
            AgentNode(agent_id="a", group_id="g_root"),
            AgentNode(agent_id="b", group_id="g_sub"),
            AgentNode(agent_id="c", group_id="g_sub"),
        )
        groups = (
            GroupSpec(group_id="g_root", pattern="singleton", member_ids=("a",)),
            GroupSpec(
                group_id="g_sub",
                pattern="voting",
                member_ids=("b", "c"),
                parent_agent_id="b",
            ),
        )
        spec = TopologySpec(version=0, agents=agents, groups=groups, root_group_id="g_root")
        with self.assertRaises(ValueError):
            spec.validate(max_agents=10)


class TestTopologySpecLayout(unittest.TestCase):
    def test_to_layout_projection(self) -> None:
        spec = _heterogeneous_spec()
        spec.validate(max_agents=10)
        layout = spec.to_layout()

        self.assertEqual(layout.topology, "self_evolved")
        self.assertEqual(len(layout.agent_ids), 9)
        self.assertEqual(layout.orchestrator_id, "agent_0")

        # Root star edges.
        self.assertIn("agent_1", layout.adjacency["agent_0"])
        self.assertIn("agent_3", layout.adjacency["agent_0"])
        # Expansion attachment edges.
        self.assertIn("agent_4", layout.adjacency["agent_1"])
        self.assertIn("agent_7", layout.adjacency["agent_2"])
        # Debate subgroup is fully linked internally.
        self.assertIn("agent_8", layout.adjacency["agent_7"])
        # Star subgroup spokes are not linked to each other.
        self.assertNotIn("agent_5", layout.adjacency["agent_4"])

        self.assertEqual(layout.parent_by_agent["agent_4"], "agent_1")
        self.assertEqual(layout.parent_by_agent["agent_7"], "agent_2")
        self.assertEqual(sorted(layout.managers), ["agent_1", "agent_2"])
        self.assertIn("agent_3", layout.leaves)
        self.assertNotIn("agent_1", layout.leaves)
        self.assertEqual(layout.groups, [["agent_7", "agent_8"]])

        # Nested members sit one level below their parent agent.
        self.assertEqual(layout.level_by_agent["agent_0"], 0)
        self.assertEqual(layout.level_by_agent["agent_1"], 1)
        self.assertEqual(layout.level_by_agent["agent_4"], 2)

    def test_ordered_agent_ids_leader_first(self) -> None:
        spec = _heterogeneous_spec()
        ordered = spec.ordered_agent_ids()
        self.assertEqual(ordered[0], "agent_0")
        self.assertEqual(set(ordered), {f"agent_{i}" for i in range(9)})
        # Subgroup members appear after the root members.
        self.assertGreater(ordered.index("agent_4"), ordered.index("agent_3"))


class TestSpecFromLayout(unittest.TestCase):
    def test_sas_round_trip(self) -> None:
        spec = spec_from_layout(build_layout(topology="sas", num_agents=1))
        spec.validate(max_agents=5)
        root = spec.group(spec.root_group_id)
        self.assertEqual(root.pattern, "singleton")
        self.assertEqual(len(spec.agents), 1)

    def test_orchestrator_round_trip(self) -> None:
        spec = spec_from_layout(build_layout(topology="orchestrator_no_discussion", num_agents=4))
        spec.validate(max_agents=5)
        root = spec.group(spec.root_group_id)
        self.assertEqual(root.pattern, "star")
        self.assertEqual(root.leader_id, "agent_0")
        layout = spec.to_layout()
        self.assertEqual(layout.orchestrator_id, "agent_0")
        self.assertEqual(len(layout.specialists), 3)

    def test_voting_and_debate_round_trip(self) -> None:
        for topology in ("only_voting", "fully_linked_debate"):
            spec = spec_from_layout(build_layout(topology=topology, num_agents=3))
            spec.validate(max_agents=5)
            self.assertEqual(len(spec.agents), 3)

    def test_tree_round_trip(self) -> None:
        spec = spec_from_layout(
            build_layout(
                topology="orchestrator_tree_structure",
                num_agents=6,
                agents_per_level=[1, 2, 3],
            )
        )
        spec.validate(max_agents=10)
        layout = spec.to_layout()
        self.assertEqual(layout.orchestrator_id, "agent_0")
        self.assertEqual(len(layout.agent_ids), 6)
        # Managers own subgroups of leaves.
        self.assertTrue(spec.subgroup_of("agent_1") is not None)


if __name__ == "__main__":
    unittest.main()
