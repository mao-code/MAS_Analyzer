import unittest

from MAS import ExperimentSpec, LangGraphMASEngine, run_experiment
from MAS.llm import LLMResult, OpenRouterLLMClient


class _TerminationJudgeLLM(OpenRouterLLMClient):
    def __init__(self, *, mock_used: bool = False, text: str = "") -> None:
        self._mock_used = mock_used
        self._text = text

    def generate(
        self,
        *,
        prompt,
        agent_type,
        task_id,
        run_index,
        agent_id,
        tools=None,
        max_tool_iterations=8,
        temperature=0.0,
    ) -> LLMResult:
        return LLMResult(
            text=self._text,
            token_in=11,
            token_out=7,
            cost_usd=0.0,
            model="judge-model",
            mock_used=self._mock_used,
            metadata={},
        )


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
            if view["phase"] in {"specialist_worker", "specialist_solve"}
            and view["viewer"].startswith("agent_")
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

    def test_workflow_visual_graph_includes_control_flow_nodes(self) -> None:
        workflow, graph = LangGraphMASEngine.build_workflow_visual_graph(
            ExperimentSpec(
                topology="fully_linked_debate",
                num_agents=4,
                rounds=2,
                discussion_rounds=1,
            )
        )

        self.assertIn("debate_controller", workflow.nodes)
        self.assertIn("judge", workflow.nodes)
        self.assertIn("finalize", workflow.nodes)

        mermaid = graph.draw_mermaid()
        self.assertIn("debate_controller", mermaid)
        self.assertIn("judge", mermaid)
        self.assertIn("finalize", mermaid)

    def test_termination_consensus_uses_llm_judge_when_available(self) -> None:
        engine = LangGraphMASEngine(
            _TerminationJudgeLLM(
                text='{"groups":[[0,1],[2]],"invalid_indices":[],"explanation":"0 and 1 match"}'
            )
        )
        state = {
            "termination_consensus_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which city is correct?",
        }
        artifacts = [
            {"agent_id": "agent_0", "answer": "Paris is the capital of France."},
            {"agent_id": "agent_1", "answer": "The capital of France is Paris."},
            {"agent_id": "agent_2", "answer": "London"},
        ]

        consensus = engine._compute_termination_consensus(
            state=state,
            stage_name="debate_controller",
            round_index=0,
            discussion_index=0,
            artifacts=artifacts,
        )

        self.assertEqual(consensus["source"], "llm_judge")
        self.assertAlmostEqual(consensus["ratio"], 2 / 3)
        self.assertEqual(consensus["groups"], [[0, 1], [2]])
        self.assertEqual(consensus["token_in"], 11)
        self.assertEqual(consensus["token_out"], 7)

    def test_termination_consensus_falls_back_to_lexical_in_mock_mode(self) -> None:
        engine = LangGraphMASEngine(
            _TerminationJudgeLLM(
                mock_used=True,
                text='{"groups":[[0,1]],"invalid_indices":[],"explanation":"unused in mock"}',
            )
        )
        state = {
            "termination_consensus_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which city is correct?",
        }
        artifacts = [
            {"agent_id": "agent_0", "answer": "Paris is the capital of France."},
            {"agent_id": "agent_1", "answer": "The capital of France is Paris."},
        ]

        consensus = engine._compute_termination_consensus(
            state=state,
            stage_name="debate_controller",
            round_index=0,
            discussion_index=0,
            artifacts=artifacts,
        )

        self.assertEqual(consensus["source"], "lexical_fallback_mock")
        self.assertEqual(consensus["mode"], "llm_judge")
        self.assertEqual(consensus["valid_count"], 2)


if __name__ == "__main__":
    unittest.main()
