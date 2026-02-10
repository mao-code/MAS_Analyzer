import unittest

from MAS.config import MASConfig
from MAS.topology import build_topology


class TestTopology(unittest.TestCase):
    def _cfg(self) -> MASConfig:
        cfg = MASConfig(
            levels=3,
            intra_level_link_ratio=0.5,
            full_linked=False,
            agents_per_level=[2, 2, 2],
            number_of_agents=6,
            agent_types=["a", "b"],
            communication_count_internally=1,
            turn_mode="multi_turn",
            max_turns=3,
        )
        cfg.validate()
        return cfg

    def test_deterministic_given_seed(self) -> None:
        cfg = self._cfg()
        t1 = build_topology(cfg, seed=123)
        t2 = build_topology(cfg, seed=123)
        self.assertEqual(t1.adjacency, t2.adjacency)
        self.assertEqual([a.agent_id for a in t1.agents], [a.agent_id for a in t2.agents])

    def test_adjacent_level_connectivity_only(self) -> None:
        cfg = MASConfig(
            levels=3,
            intra_level_link_ratio=0.0,
            full_linked=False,
            agents_per_level=[1, 1, 1],
            number_of_agents=3,
            agent_types=["x"],
            communication_count_internally=1,
            turn_mode="single_turn",
            max_turns=1,
        )
        cfg.validate()

        topology = build_topology(cfg, seed=1)
        # agent_0(level0) <-> agent_1(level1), agent_2(level2) <-> agent_1(level1)
        self.assertIn("agent_1", topology.adjacency["agent_0"])
        self.assertIn("agent_1", topology.adjacency["agent_2"])
        # no direct edge between level 0 and level 2
        self.assertNotIn("agent_2", topology.adjacency["agent_0"])
        self.assertNotIn("agent_0", topology.adjacency["agent_2"])


if __name__ == "__main__":
    unittest.main()
