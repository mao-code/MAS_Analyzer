import unittest

from MAS import run_experiment


class TestLangGraphTopologies(unittest.TestCase):
    def test_run_all_required_topologies(self) -> None:
        scenarios = [
            ("sas", 1, 1),
            ("orchestrator_tree_structure", 3, 1),
            ("orchestrator_no_discussion", 4, 2),
            ("orchestrator_with_discussion", 4, 2),
            ("only_voting", 4, 1),
            ("fully_linked_debate", 4, 2),
            ("group_chat_debate", 4, 2),
        ]

        for topology, agents, rounds in scenarios:
            with self.subTest(topology=topology):
                result = run_experiment(
                    topology=topology,
                    agents=agents,
                    rounds=rounds,
                    prompt="What is 2 + 2?",
                    seed=7,
                )
                self.assertTrue(result.trace_events)
                self.assertIn("topology", result.run_metadata)
                self.assertEqual(result.run_metadata["topology"], topology)
                self.assertIn("relay_messages", result.run_metadata)
                self.assertIn("message_views", result.run_metadata)

    def test_orchestrator_no_discussion_visibility(self) -> None:
        result = run_experiment(
            topology="orchestrator_no_discussion",
            agents=4,
            rounds=2,
            prompt="Provide one sentence answer.",
            seed=11,
        )

        views = result.run_metadata["message_views"]
        specialist_views = [
            view
            for view in views
            if view["phase"] == "specialist_solve" and view["viewer"].startswith("agent_")
        ]
        self.assertTrue(specialist_views)
        for view in specialist_views:
            senders = set(view.get("visible_senders", []))
            self.assertTrue(senders.issubset({"agent_0"}))

    def test_fully_linked_debate_broadcasts_to_all_peers(self) -> None:
        result = run_experiment(
            topology="fully_linked_debate",
            agents=5,
            rounds=2,
            prompt="Provide a final answer.",
            seed=3,
        )

        messages = result.run_metadata["relay_messages"]
        self.assertTrue(messages)
        for message in messages:
            self.assertEqual(len(message["recipients"]), 4)

    def test_group_chat_debate_stays_inside_groups(self) -> None:
        result = run_experiment(
            topology="group_chat_debate",
            agents=5,
            group_sizes=[2, 3],
            rounds=2,
            prompt="Provide your best answer.",
            seed=5,
        )

        layout = result.run_metadata["topology_layout"]
        messages = result.run_metadata["relay_messages"]
        groups = [set(group) for group in layout["groups"]]

        for message in messages:
            if message["kind"] != "group_debate_round":
                continue
            sender = message["sender"]
            recipients = set(message["recipients"])
            sender_group = next(group for group in groups if sender in group)
            self.assertTrue(recipients.issubset(sender_group - {sender}))


if __name__ == "__main__":
    unittest.main()
