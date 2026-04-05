import unittest

from MAS import ExperimentSpec, LangGraphMASEngine, run_experiment
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.relay import build_layout


class _JudgeLLM(OpenRouterLLMClient):
    def __init__(
        self,
        *,
        mock_used: bool = False,
        text: str = "",
        text_by_agent_id: dict[str, str] | None = None,
    ) -> None:
        self._mock_used = mock_used
        self._text = text
        self._text_by_agent_id = dict(text_by_agent_id or {})

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
            text=self._text_by_agent_id.get(agent_id, self._text),
            token_in=11,
            token_out=7,
            cost_usd=0.0,
            model="judge-model",
            mock_used=self._mock_used,
            metadata={},
        )


class TestLangGraphTopologies(unittest.TestCase):
    def test_experiment_spec_normalized_preserves_role_assignment_fields(self) -> None:
        spec = ExperimentSpec(
            topology="sas",
            num_agents=1,
            rounds=1,
            benchmark_name="browsecomp",
            enable_dynamic_roles=False,
        )

        normalized = spec.normalized()

        self.assertEqual(normalized.benchmark_name, "browsecomp")
        self.assertFalse(normalized.enable_dynamic_roles)

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

    def test_orchestrator_tree_manager_reducers_execute_per_manager(self) -> None:
        valid_worker_json = (
            '{"answer_artifact":"Paris","summary":"Paris","critique":"","revision_request":"",'
            '"confidence":1.0,"unresolved_issues":[],"evidence_summary":[]}'
        )
        result = run_experiment(
            topology="orchestrator_tree_structure",
            agents=5,
            rounds=1,
            prompt="Which city is correct?",
            seed=13,
            llm_client=_JudgeLLM(text=valid_worker_json),
        )

        reducer_logs = [
            log for log in result.run_metadata["interaction_logs"] if log["phase"] == "manager_reducers"
        ]
        self.assertTrue(reducer_logs)
        self.assertTrue(all(str(log.get("agent_id", "")).strip() for log in reducer_logs))
        self.assertTrue(all(log.get("visible_messages") for log in reducer_logs))

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
            _JudgeLLM(
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
            _JudgeLLM(
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

    def test_non_substantive_consensus_does_not_stop(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text=(
                    '{"groups":[[0,1]],"invalid_indices":[],"is_substantive":false,'
                    '"progress_status":"improving","expected_improvement":"high",'
                    '"should_stop_for_no_progress":false,"explanation":"Both answers are still planning."}'
                )
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
            {"agent_id": "agent_0", "answer": "Need more searching before I can answer.", "confidence": 0.95},
            {"agent_id": "agent_1", "answer": "I still need more information before answering.", "confidence": 0.9},
        ]

        decision = engine._termination_decision(
            state,
            stage_name="debate_controller",
            round_index=0,
            discussion_index=0,
            candidate_artifacts=artifacts,
            previous_candidate_artifacts=[],
            consensus_artifacts=artifacts,
            expected_count=2,
            max_reached=False,
            continue_next_step="debate_dispatch",
            stop_next_step="judge",
        )

        self.assertFalse(decision["should_stop"])
        self.assertEqual(decision["reason"], "continue")
        self.assertEqual(decision["consensus_source"], "llm_judge")
        self.assertFalse(decision["consensus_is_substantive"])

    def test_semantic_no_progress_triggers_stop(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text=(
                    '{"groups":[[0],[1]],"invalid_indices":[],"is_substantive":false,'
                    '"progress_status":"stalled","expected_improvement":"low",'
                    '"should_stop_for_no_progress":true,'
                    '"explanation":"Another round is unlikely to materially improve correctness."}'
                )
            )
        )
        state = {
            "termination_consensus_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which city is correct?",
        }
        previous_artifacts = [
            {"agent_id": "agent_0", "answer": "Paris is probably right but unverified."},
            {"agent_id": "agent_1", "answer": "London is probably right but unverified."},
        ]
        artifacts = [
            {"agent_id": "agent_0", "answer": "My current guess remains Paris.", "confidence": 0.55},
            {"agent_id": "agent_1", "answer": "My current guess remains London.", "confidence": 0.6},
        ]

        decision = engine._termination_decision(
            state,
            stage_name="debate_controller",
            round_index=1,
            discussion_index=0,
            candidate_artifacts=artifacts,
            previous_candidate_artifacts=previous_artifacts,
            consensus_artifacts=artifacts,
            expected_count=2,
            max_reached=False,
            continue_next_step="debate_dispatch",
            stop_next_step="judge",
        )

        self.assertTrue(decision["should_stop"])
        self.assertEqual(decision["reason"], "no_meaningful_change")
        self.assertEqual(decision["progress_source"], "llm_judge")
        self.assertEqual(decision["progress_status"], "stalled")
        self.assertEqual(decision["expected_improvement"], "low")

    def test_high_confidence_is_diagnostic_only(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text=(
                    '{"groups":[[0],[1]],"invalid_indices":[],"is_substantive":true,'
                    '"progress_status":"improving","expected_improvement":"high",'
                    '"should_stop_for_no_progress":false,"explanation":"The agents still disagree."}'
                )
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
            {"agent_id": "agent_0", "answer": "Paris", "confidence": 0.99},
            {"agent_id": "agent_1", "answer": "London", "confidence": 0.97},
        ]

        decision = engine._termination_decision(
            state,
            stage_name="debate_controller",
            round_index=0,
            discussion_index=0,
            candidate_artifacts=artifacts,
            previous_candidate_artifacts=[],
            consensus_artifacts=artifacts,
            expected_count=2,
            max_reached=False,
            continue_next_step="debate_dispatch",
            stop_next_step="judge",
        )

        self.assertFalse(decision["should_stop"])
        self.assertEqual(decision["reason"], "continue")
        self.assertGreaterEqual(decision["average_confidence"], 0.95)

    def test_termination_uses_lexical_delta_on_parse_error(self) -> None:
        engine = LangGraphMASEngine(_JudgeLLM(text="not json"))
        state = {
            "termination_consensus_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which city is correct?",
        }
        previous_artifacts = [
            {"agent_id": "agent_0", "answer": "Paris"},
            {"agent_id": "agent_1", "answer": "London"},
        ]
        artifacts = [
            {"agent_id": "agent_0", "answer": "Paris", "confidence": 0.8},
            {"agent_id": "agent_1", "answer": "London", "confidence": 0.8},
        ]

        decision = engine._termination_decision(
            state,
            stage_name="debate_controller",
            round_index=1,
            discussion_index=0,
            candidate_artifacts=artifacts,
            previous_candidate_artifacts=previous_artifacts,
            consensus_artifacts=artifacts,
            expected_count=2,
            max_reached=False,
            continue_next_step="debate_dispatch",
            stop_next_step="judge",
        )

        self.assertTrue(decision["should_stop"])
        self.assertEqual(decision["reason"], "no_meaningful_change")
        self.assertEqual(decision["consensus_source"], "lexical_fallback_parse_error")

    def test_final_vote_uses_llm_judge_when_available(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text_by_agent_id={
                    "judge_final_vote_judge": (
                        '{"groups":[[0,1],[2]],"winner_index":1,"invalid_indices":[],"explanation":"1 is clearer and supported by 0"}'
                    )
                }
            )
        )
        state = {
            "final_vote_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which city is correct?",
        }
        artifacts = [
            {"agent_id": "agent_0", "answer": "Paris is the capital of France.", "confidence": 0.6},
            {"agent_id": "agent_1", "answer": "The capital of France is Paris.", "confidence": 0.8},
            {"agent_id": "agent_2", "answer": "London", "confidence": 0.2},
        ]

        vote = engine._select_final_answer(
            state=state,
            stage_name="judge",
            artifacts=artifacts,
        )

        self.assertEqual(vote["source"], "llm_judge")
        self.assertEqual(vote["answer"], "The capital of France is Paris.")
        self.assertEqual(sorted(vote["tally"].values()), [1, 2])
        self.assertEqual(vote["token_in"], 11)
        self.assertEqual(vote["token_out"], 7)

    def test_final_vote_falls_back_to_deterministic_in_mock_mode(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                mock_used=True,
                text_by_agent_id={
                    "judge_final_vote_judge": (
                        '{"groups":[[0,1]],"winner_index":1,"invalid_indices":[],"explanation":"unused in mock"}'
                    )
                },
            )
        )
        state = {
            "final_vote_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which city is correct?",
        }
        artifacts = [
            {"agent_id": "agent_0", "answer": "Paris", "confidence": 0.6},
            {"agent_id": "agent_1", "answer": "Paris", "confidence": 0.7},
            {"agent_id": "agent_2", "answer": "London", "confidence": 0.2},
        ]

        vote = engine._select_final_answer(
            state=state,
            stage_name="judge",
            artifacts=artifacts,
        )

        self.assertEqual(vote["source"], "deterministic_fallback_mock")
        self.assertEqual(vote["answer"], "Paris")
        self.assertEqual(vote["tally"]["paris"], 2)

    def test_group_chat_representative_controller_respects_discussion_round_limit(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text=(
                    '{"groups":[[0],[1]],"invalid_indices":[],"is_substantive":true,'
                    '"progress_status":"improving","expected_improvement":"medium",'
                    '"should_stop_for_no_progress":false,"explanation":"Representatives still disagree."}'
                )
            )
        )
        state = {
            "topology": "group_chat_debate",
            "discussion_rounds": 2,
            "termination_consensus_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which city is correct?",
            "round_index": 1,
            "discussion_index": 1,
            "dispatch_id": 3,
            "layout": build_layout(topology="group_chat_debate", num_agents=4),
            "artifacts": [
                {
                    "node_name": "representative_merge",
                    "agent_id": "agent_0",
                    "round_index": 1,
                    "discussion_index": 0,
                    "dispatch_id": 1,
                    "answer": "Paris",
                    "confidence": 0.7,
                },
                {
                    "node_name": "representative_merge",
                    "agent_id": "agent_2",
                    "round_index": 1,
                    "discussion_index": 0,
                    "dispatch_id": 1,
                    "answer": "London",
                    "confidence": 0.6,
                },
                {
                    "node_name": "representative_merge",
                    "agent_id": "agent_0",
                    "round_index": 1,
                    "discussion_index": 1,
                    "dispatch_id": 2,
                    "answer": "Paris",
                    "confidence": 0.8,
                },
                {
                    "node_name": "representative_merge",
                    "agent_id": "agent_2",
                    "round_index": 1,
                    "discussion_index": 1,
                    "dispatch_id": 2,
                    "answer": "London",
                    "confidence": 0.75,
                },
            ],
        }

        updates = engine._representative_controller_node(state)

        self.assertEqual(updates["next_step"], "final_judge")
        self.assertTrue(updates["termination_decision"]["should_stop"])
        self.assertEqual(updates["termination_decision"]["reason"], "max_rounds_reached")

    def test_group_chat_representative_controller_stops_on_discussion_limit_in_round_zero(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text=(
                    '{"groups":[[0],[1]],"invalid_indices":[],"is_substantive":true,'
                    '"progress_status":"improving","expected_improvement":"medium",'
                    '"should_stop_for_no_progress":false,"explanation":"Representatives still disagree."}'
                )
            )
        )
        state = {
            "topology": "group_chat_debate",
            "discussion_rounds": 2,
            "termination_consensus_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which city is correct?",
            "round_index": 0,
            "discussion_index": 2,
            "dispatch_id": 3,
            "layout": build_layout(topology="group_chat_debate", num_agents=4),
            "artifacts": [
                {
                    "node_name": "representative_merge",
                    "agent_id": "agent_0",
                    "round_index": 0,
                    "discussion_index": 1,
                    "dispatch_id": 2,
                    "answer": "Paris",
                    "confidence": 0.8,
                },
                {
                    "node_name": "representative_merge",
                    "agent_id": "agent_2",
                    "round_index": 0,
                    "discussion_index": 1,
                    "dispatch_id": 2,
                    "answer": "London",
                    "confidence": 0.75,
                },
                {
                    "node_name": "representative_merge",
                    "agent_id": "agent_0",
                    "round_index": 0,
                    "discussion_index": 2,
                    "dispatch_id": 3,
                    "answer": "Paris",
                    "confidence": 0.82,
                },
                {
                    "node_name": "representative_merge",
                    "agent_id": "agent_2",
                    "round_index": 0,
                    "discussion_index": 2,
                    "dispatch_id": 3,
                    "answer": "London",
                    "confidence": 0.77,
                },
            ],
        }

        updates = engine._representative_controller_node(state)

        self.assertEqual(updates["next_step"], "final_judge")
        self.assertTrue(updates["termination_decision"]["should_stop"])
        self.assertEqual(updates["termination_decision"]["reason"], "max_rounds_reached")

    def test_orchestrator_relay_stops_on_discussion_limit_in_round_zero(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text=(
                    '{"groups":[[0],[1],[2]],"invalid_indices":[],"is_substantive":true,'
                    '"progress_status":"improving","expected_improvement":"medium",'
                    '"should_stop_for_no_progress":false,"explanation":"Specialists still disagree."}'
                )
            )
        )
        state = {
            "topology": "orchestrator_with_discussion",
            "discussion_rounds": 2,
            "minimum_discussion_rounds": 1,
            "termination_consensus_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which city is correct?",
            "round_index": 0,
            "discussion_index": 2,
            "dispatch_id": 4,
            "layout": build_layout(topology="orchestrator_with_discussion", num_agents=4),
            "artifacts": [
                {
                    "node_name": "specialists_revision_round",
                    "agent_id": "agent_1",
                    "round_index": 0,
                    "discussion_index": 1,
                    "dispatch_id": 3,
                    "answer": "Paris",
                    "confidence": 0.7,
                },
                {
                    "node_name": "specialists_revision_round",
                    "agent_id": "agent_2",
                    "round_index": 0,
                    "discussion_index": 1,
                    "dispatch_id": 3,
                    "answer": "London",
                    "confidence": 0.65,
                },
                {
                    "node_name": "specialists_revision_round",
                    "agent_id": "agent_3",
                    "round_index": 0,
                    "discussion_index": 1,
                    "dispatch_id": 3,
                    "answer": "Berlin",
                    "confidence": 0.6,
                },
                {
                    "node_name": "specialists_revision_round",
                    "agent_id": "agent_1",
                    "round_index": 0,
                    "discussion_index": 2,
                    "dispatch_id": 4,
                    "answer": "Paris",
                    "confidence": 0.72,
                },
                {
                    "node_name": "specialists_revision_round",
                    "agent_id": "agent_2",
                    "round_index": 0,
                    "discussion_index": 2,
                    "dispatch_id": 4,
                    "answer": "London",
                    "confidence": 0.68,
                },
                {
                    "node_name": "specialists_revision_round",
                    "agent_id": "agent_3",
                    "round_index": 0,
                    "discussion_index": 2,
                    "dispatch_id": 4,
                    "answer": "Berlin",
                    "confidence": 0.62,
                },
            ],
        }

        updates = engine._orchestrator_relay_controller(state)

        self.assertEqual(updates["next_step"], "orchestrator_merge")
        self.assertTrue(updates["termination_decision"]["should_stop"])
        self.assertEqual(updates["termination_decision"]["reason"], "max_rounds_reached")

    def test_only_voting_ignores_placeholder_worker_outputs(self) -> None:
        engine = LangGraphMASEngine(_JudgeLLM(text="unused"))
        state = {
            "topology": "only_voting",
            "final_vote_mode": "deterministic",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which city is correct?",
            "round_index": 0,
            "discussion_index": 0,
            "dispatch_id": 0,
            "artifacts": [
                {"node_name": "worker", "round_index": 0, "agent_id": "agent_0", "answer": "thought:"},
                {"node_name": "worker", "round_index": 0, "agent_id": "agent_1", "answer": "Paris"},
            ],
        }

        vote = engine._only_voting_voter_node(state)

        self.assertEqual(vote["final_answer"], "Paris")
        self.assertEqual(vote["vote_tally"], {"paris": 1})
        self.assertEqual(vote["final_reason"], "only_voting:majority_vote")

    def test_final_vote_ignores_plan_artifacts_and_extracts_structured_answer(self) -> None:
        engine = LangGraphMASEngine(_JudgeLLM(text="unused"))
        state = {
            "final_vote_mode": "deterministic",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Who is the correct person?",
        }
        artifacts = [
            {"agent_id": "agent_0", "answer": "{'plan': ['search', 'verify']}", "confidence": 0.9},
            {
                "agent_id": "agent_1",
                "answer": (
                    "{'individual_name': 'Laura Lojo-Rodriguez', "
                    "'verification_details': {'book': 'Routledge 2018'}}"
                ),
                "confidence": 0.6,
            },
        ]

        vote = engine._select_final_answer(
            state=state,
            stage_name="judge",
            artifacts=artifacts,
        )

        self.assertEqual(vote["source"], "deterministic")
        self.assertEqual(vote["answer"], "Laura Lojo-Rodriguez")
        self.assertEqual(vote["tally"], {"laura lojo rodriguez": 1})

    def test_resolve_final_answer_skips_planner_fallback(self) -> None:
        engine = LangGraphMASEngine(_JudgeLLM(text="unused"))
        state = {
            "topology": "orchestrator_tree_structure",
            "artifacts": [
                {
                    "node_name": "root_plan",
                    "agent_id": "agent_0",
                    "round_index": 0,
                    "discussion_index": 0,
                    "dispatch_id": 0,
                    "answer": "{'plan': ['break into subquestions']}",
                },
                {
                    "node_name": "root_reducer",
                    "agent_id": "agent_0",
                    "round_index": 0,
                    "discussion_index": 0,
                    "dispatch_id": 1,
                    "answer": "{'sub_questions': ['verify candidate']}",
                },
                {
                    "node_name": "worker_nodes",
                    "agent_id": "agent_2",
                    "round_index": 0,
                    "discussion_index": 0,
                    "dispatch_id": 2,
                    "answer": "Queen Arwa University",
                    "confidence": 0.8,
                },
            ],
        }

        self.assertEqual(engine._resolve_final_answer(state), "Queen Arwa University")

    def test_only_voting_reports_judge_tiebreak_when_tally_is_tied(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text_by_agent_id={
                    "voter_final_vote_judge": (
                        '{"groups":[[0,1],[2,3]],"winner_index":2,"invalid_indices":[],"explanation":"2 and 3 are better supported"}'
                    )
                }
            )
        )
        state = {
            "topology": "only_voting",
            "final_vote_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which brand is correct?",
            "round_index": 0,
            "discussion_index": 0,
            "dispatch_id": 0,
            "artifacts": [
                {"node_name": "worker", "round_index": 0, "agent_id": "agent_0", "answer": "VKO", "confidence": 0.4},
                {"node_name": "worker", "round_index": 0, "agent_id": "agent_1", "answer": "VKO", "confidence": 0.5},
                {"node_name": "worker", "round_index": 0, "agent_id": "agent_2", "answer": "Vakkorama", "confidence": 0.9},
                {"node_name": "worker", "round_index": 0, "agent_id": "agent_3", "answer": "Vakkorama", "confidence": 0.8},
            ],
        }

        vote = engine._only_voting_voter_node(state)

        self.assertEqual(vote["final_answer"], "Vakkorama")
        self.assertEqual(vote["vote_tally"], {"vko": 2, "vakkorama": 2})
        self.assertEqual(vote["final_reason"], "only_voting:judge_tiebreak")


if __name__ == "__main__":
    unittest.main()
