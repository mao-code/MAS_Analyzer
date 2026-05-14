import json
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from MAS import ExperimentSpec, LangGraphMASEngine, build_runtime_config, run_experiment
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.relay import build_layout
from answer_utils import extract_substantive_answer


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


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: Any
    reference_answer: str = ""
    metadata: dict[str, Any] | None = None


class _SequencedLLM(OpenRouterLLMClient):
    def __init__(self, responses: list[LLMResult]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "prompt": prompt,
                "agent_type": agent_type,
                "task_id": task_id,
                "run_index": run_index,
                "agent_id": agent_id,
                "tools": tools,
            }
        )
        index = min(len(self.calls), len(self._responses)) - 1
        return self._responses[index]


class _RoleAwareLLM(OpenRouterLLMClient):
    def __init__(self) -> None:
        self.calls: list[str] = []

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
        self.calls.append(agent_id)
        if agent_id == "role_assigner":
            return LLMResult(
                text='{"agent_0":"Web Search Strategist"}',
                token_in=9,
                token_out=6,
                cost_usd=0.0,
                model="judge-model",
                mock_used=False,
                metadata={},
            )
        return LLMResult(
            text=(
                '{"answer_artifact":"Queen Arwa University","summary":"Queen Arwa University",'
                '"critique":"","revision_request":"","confidence":0.8,'
                '"unresolved_issues":[],"evidence_summary":["Document 82002 confirms the graduation ceremony date."]}'
            ),
            token_in=13,
            token_out=8,
            cost_usd=0.0,
            model="judge-model",
            mock_used=False,
            metadata={},
        )


class _WorkbenchRoleAwareLLM(OpenRouterLLMClient):
    def __init__(self) -> None:
        pass

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
        if agent_id == "role_assigner":
            text = (
                '{"agent_0":"Workflow Planner","agent_1":"CRM Data Analyst",'
                '"agent_2":"Calendar Operations Specialist","agent_3":"Email and Communication Expert"}'
            )
        else:
            prompt_text = json.dumps(prompt) if isinstance(prompt, list) else str(prompt)
            if "Stage Role: planner" in prompt_text:
                text = (
                    '{"answer_artifact":{"plan":['
                    '{"step":1,"tool":"company_directory.find_email_address","parameters":{"name":"Riley Brown"},"description":"Identify the owner contact."},'
                    '{"step":2,"tool":"calendar.search_events","parameters":{"query":"Riley Brown","time_min":"2023-11-16 00:00:00"},"description":"Check recent meetings."},'
                    '{"step":3,"tool":"calendar.create_event","parameters":{"event_name":"Update on Riley Brown"},"description":"Schedule the follow-up if conditions are met."}'
                    ']},"summary":"Plan the work for specialists.","critique":"","revision_request":"","confidence":0.8,"unresolved_issues":[],"evidence_summary":["Need owner lookup and calendar verification."]}'
                )
            else:
                text = (
                    '{"answer_artifact":"Blocked until the assignee is verified.","summary":"Blocked until the assignee is verified.",'
                    '"critique":"","revision_request":"","confidence":0.5,'
                    '"unresolved_issues":["Need assignee"],"evidence_summary":["Owner information not yet verified."]}'
                )
        return LLMResult(
            text=text,
            token_in=11,
            token_out=7,
            cost_usd=0.0,
            model="judge-model",
            mock_used=False,
            metadata={},
        )


class _CompactionLLM(OpenRouterLLMClient):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"agent_id": agent_id, "prompt": prompt})
        if agent_id.endswith("_transcript_summarizer"):
            text = (
                '{"claims":["rolling claim"],"evidence":["rolling evidence"],'
                '"disagreements":["rolling disagreement"],'
                '"unresolved_questions":["rolling unresolved"],'
                '"best_current_answer":"42","confidence":0.75}'
            )
        else:
            text = (
                '{"answer_artifact":"42","summary":"answer 42",'
                '"critique":"","revision_request":"","confidence":0.8,'
                '"unresolved_issues":[],"evidence_summary":["local evidence"]}'
            )
        return LLMResult(
            text=text,
            token_in=11,
            token_out=7,
            cost_usd=0.0,
            model="mock-model",
            mock_used=False,
            metadata={},
        )


class _FatalToolFailureLLM(OpenRouterLLMClient):
    def __init__(self) -> None:
        pass

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
            text='{"answer_artifact":"blocked","summary":"blocked","critique":"","revision_request":"","confidence":0.0,"unresolved_issues":["tool failed"],"evidence_summary":[]}',
            token_in=3,
            token_out=2,
            cost_usd=0.0,
            model="mock-model",
            mock_used=False,
            metadata={
                "fatal_tool_failure": True,
                "tool_failure_tool_name": "google_web_search",
                "tool_failure_consecutive_count": 2,
            },
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

    def test_transcript_compaction_summary_is_added_to_later_round_prompts(self) -> None:
        llm = _CompactionLLM()
        engine = LangGraphMASEngine(llm)

        with patch.dict("os.environ", {"MAS_TRANSCRIPT_COMPACTION_ENABLED": "1"}, clear=False):
            result = engine.run(
                task=_Task(task_id="t1", prompt="What is the answer?"),
                run_index=0,
                seed=7,
                spec=ExperimentSpec(
                    topology="fully_linked_debate",
                    num_agents=2,
                    rounds=2,
                    communication_budget_per_agent=5,
                    termination_consensus_mode="lexical",
                    final_vote_mode="deterministic",
                    enable_dynamic_roles=False,
                ),
                agent_types=["general"],
            )

        self.assertIn("rolling claim", result.run_metadata["transcript_summary"]["claims"])
        self.assertGreaterEqual(len(result.run_metadata["transcript_compaction_history"]), 1)
        second_round_prompts = [
            json.dumps(call["prompt"])
            for call in llm.calls
            if call["agent_id"] in {"agent_0", "agent_1"}
            and "rolling_transcript_summary" in json.dumps(call["prompt"])
        ]
        self.assertTrue(second_round_prompts)

    def test_fatal_search_tool_failure_raises_for_run_level_rerun(self) -> None:
        engine = LangGraphMASEngine(_FatalToolFailureLLM())

        with self.assertRaisesRegex(RuntimeError, "fatal_tool_failure"):
            engine.run(
                task=_Task(task_id="t1", prompt="Find the evidence."),
                run_index=0,
                seed=7,
                spec=ExperimentSpec(
                    topology="sas",
                    num_agents=1,
                    rounds=1,
                    enable_dynamic_roles=False,
                ),
                agent_types=["general"],
                tools=[{"name": "google_web_search", "description": "Search"}],
            )

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

    def test_orchestrator_with_discussion_revision_receives_peer_summaries_under_low_budget(self) -> None:
        valid_json = (
            '{"answer_artifact":"Paris","summary":"Paris","critique":"","revision_request":"",'
            '"confidence":1.0,"unresolved_issues":[],"evidence_summary":["Document support."]}'
        )
        config = build_runtime_config(
            topology="orchestrator_with_discussion",
            agents=4,
            rounds=2,
            discussion_rounds=2,
            communication_budget_per_agent=2,
        )

        result = run_experiment(
            topology="orchestrator_with_discussion",
            agents=4,
            rounds=2,
            discussion_rounds=2,
            prompt="Which city is correct?",
            seed=17,
            config=config,
            llm_client=_JudgeLLM(text=valid_json),
        )

        peer_summaries = [
            message for message in result.run_metadata["relay_messages"] if message["kind"] == "peer_summary"
        ]
        self.assertEqual(len(peer_summaries), 3)

        revision_views = [
            view for view in result.run_metadata["message_views"] if view["phase"] == "specialists_revision_round"
        ]
        self.assertEqual(len(revision_views), 3)
        for view in revision_views:
            self.assertEqual(view["visible_count"], 1)
            self.assertEqual(set(view.get("visible_senders", [])), {"agent_0"})

    def test_orchestrator_message_counts_match_emitted_packets(self) -> None:
        engine = LangGraphMASEngine(_WorkbenchRoleAwareLLM())
        task = _Task(
            task_id="workbench_role_task",
            prompt=[
                {"role": "system", "content": "Use workplace tools to complete the task."},
                {"role": "user", "content": "Book the follow-up meeting with the assigned owner."},
            ],
            metadata={},
        )

        result = engine.run(
            task=task,
            run_index=0,
            seed=5,
            spec=ExperimentSpec(
                topology="orchestrator_with_discussion",
                num_agents=4,
                rounds=1,
                discussion_rounds=1,
                benchmark_name="workbench",
                enable_dynamic_roles=True,
            ),
            agent_types=["general"],
            tools=[],
            max_tool_iterations=1,
        )

        relay_messages = result.run_metadata["relay_messages"]
        self.assertEqual(result.run_metadata["messages_sent_total"], len(relay_messages))
        self.assertEqual(
            sum(result.run_metadata["messages_sent_by_agent"].values()),
            len(relay_messages),
        )

    def test_orchestrator_task_packages_are_specialist_specific(self) -> None:
        engine = LangGraphMASEngine(_WorkbenchRoleAwareLLM())
        task = _Task(
            task_id="workbench_specialist_task",
            prompt=[
                {"role": "system", "content": "Use workplace tools to complete the task."},
                {"role": "user", "content": "Book the follow-up meeting with the assigned owner."},
            ],
            metadata={},
        )

        result = engine.run(
            task=task,
            run_index=0,
            seed=6,
            spec=ExperimentSpec(
                topology="orchestrator_with_discussion",
                num_agents=4,
                rounds=1,
                discussion_rounds=1,
                benchmark_name="workbench",
                enable_dynamic_roles=True,
            ),
            agent_types=["general"],
            tools=[],
            max_tool_iterations=1,
        )

        task_packages = [
            message for message in result.run_metadata["relay_messages"] if message["kind"] == "task_package"
        ]
        self.assertEqual(len(task_packages), 3)
        summaries = {message["payload"]["summary"] for message in task_packages}
        self.assertEqual(len(summaries), 3)
        for message in task_packages:
            packet = message["payload"]["task_package"]
            self.assertEqual(packet["recipient"], message["recipients"][0])
            self.assertTrue(packet["recipient_domain_role"])
            self.assertTrue(packet["suggested_steps"])

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

    def test_orchestrator_tree_control_packets_do_not_starve_child_visibility(self) -> None:
        valid_json = (
            '{"answer_artifact":"Paris","summary":"Paris","critique":"","revision_request":"",'
            '"confidence":1.0,"unresolved_issues":[],"evidence_summary":["Document support."]}'
        )
        config = build_runtime_config(
            topology="orchestrator_tree_structure",
            agents=5,
            rounds=1,
            communication_budget_per_agent=1,
        )

        result = run_experiment(
            topology="orchestrator_tree_structure",
            agents=5,
            rounds=1,
            prompt="Which city is correct?",
            seed=19,
            config=config,
            llm_client=_JudgeLLM(text=valid_json),
        )

        manager_views = [view for view in result.run_metadata["message_views"] if view["phase"] == "manager_nodes"]
        worker_views = [view for view in result.run_metadata["message_views"] if view["phase"] == "worker_nodes"]
        self.assertEqual(len(manager_views), 2)
        self.assertEqual(len(worker_views), 2)
        self.assertTrue(all(view["visible_count"] == 1 for view in manager_views))
        self.assertTrue(all(view["visible_count"] == 1 for view in worker_views))

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

    def test_final_vote_prefers_direct_answer_over_blocked_candidate(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text_by_agent_id={
                    "judge_final_vote_judge": (
                        '{"groups":[[0],[1]],"winner_index":1,"invalid_indices":[],"explanation":"wrong winner"}'
                    )
                }
            )
        )
        state = {
            "final_vote_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which institution is correct?",
        }
        artifacts = [
            {
                "agent_id": "agent_0",
                "answer": "Queen Arwa University",
                "summary": "Queen Arwa University",
                "confidence": 0.6,
                "evidence_summary": ["Document 82002 confirms the graduation ceremony date."],
            },
            {
                "agent_id": "agent_1",
                "answer": "The requested information cannot be determined because no evidence has been retrieved.",
                "summary": "Blocked",
                "confidence": 0.95,
                "evidence_summary": ["No evidence has been retrieved."],
            },
        ]

        vote = engine._select_final_answer(
            state=state,
            stage_name="judge",
            artifacts=artifacts,
        )

        self.assertEqual(vote["answer"], "Queen Arwa University")

    def test_final_vote_empty_judgment_uses_non_direct_fallback_when_needed(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text_by_agent_id={
                    "judge_final_vote_judge": (
                        '{"groups":[],"winner_index":null,"invalid_indices":[0,1],'
                        '"explanation":"All candidates leave required criteria unresolved."}'
                    )
                }
            )
        )
        state = {
            "final_vote_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which institution is best supported?",
        }
        artifacts = [
            {
                "artifact_id": "candidate_0",
                "agent_id": "agent_0",
                "answer": "Queen Arwa University",
                "summary": "Queen Arwa University",
                "confidence": 0.9,
                "unresolved_issues": ["Criteria C and D remain unresolved."],
                "evidence_summary": ["Document 82002 confirms the graduation ceremony date."],
            },
            {
                "artifact_id": "candidate_1",
                "agent_id": "agent_1",
                "answer": "Queen Arwa University",
                "summary": "Queen Arwa University",
                "confidence": 0.8,
                "unresolved_issues": ["The 2022 events remain unverified."],
                "evidence_summary": ["Document 5412 confirms the 2002 event."],
            },
        ]

        vote = engine._select_final_answer(
            state=state,
            stage_name="judge",
            artifacts=artifacts,
        )

        self.assertEqual(vote["source"], "deterministic_non_direct_fallback_empty_judgment")
        self.assertEqual(vote["answer"], "Queen Arwa University")
        self.assertEqual(vote["selected_artifact_id"], "candidate_0")

    def test_plancraft_control_prompts_use_current_turn_not_few_shot_examples(self) -> None:
        engine = LangGraphMASEngine(_JudgeLLM(text="unused"))
        state = {
            "benchmark_name": "plancraft",
            "task_prompt": [
                {"role": "system", "content": "Use the benchmark action grammar exactly."},
                {"role": "user", "content": "Example target: andesite"},
                {"role": "assistant", "content": "craft: from [I1] to [A1] with quantity 1"},
                {"role": "user", "content": "Example target: iron_ingot"},
                {"role": "assistant", "content": "smelt: from [I2] to [I3] with quantity 1"},
                {"role": "user", "content": "Target item: quartz\nInventory: nether quartz ore in [I19]."},
            ],
        }
        candidates = [
            {
                "index": 0,
                "agent_id": "agent_0",
                "answer_mode": "direct",
                "answer": "smelt: from [I19] to [I18] with quantity 1",
                "summary": "Smelt the ore.",
                "confidence": 0.8,
                "evidence_summary": ["Quartz can be smelted from the provided ore."],
                "unresolved_issues": [],
                "evidence_count": 1,
                "used_tools": False,
            }
        ]

        final_prompt = engine._build_final_vote_prompt(
            state=state,
            stage_name="judge",
            candidates=candidates,
        )
        termination_prompt = engine._build_termination_assessment_prompt(
            state=state,
            stage_name="debate_controller",
            round_index=0,
            discussion_index=0,
            current_candidates=candidates,
            consensus_candidates=candidates,
        )

        final_payload = json.loads(final_prompt[1]["content"])
        termination_payload = json.loads(termination_prompt[1]["content"])

        for payload in (final_payload, termination_payload):
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertIn("quartz", serialized)
            self.assertNotIn("andesite", serialized)
            self.assertNotIn("iron_ingot", serialized)
            self.assertEqual(
                payload["task_context"]["current_task"],
                "Target item: quartz\nInventory: nether quartz ore in [I19].",
            )

    def test_plancraft_final_vote_rejects_furnace_fuel_impossible_answer(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text_by_agent_id={
                    "voter_final_vote_judge": (
                        '{"groups":[[0,1],[2,3]],"winner_index":2,"invalid_indices":[],"explanation":"The impossible answer is more cautious."}'
                    )
                }
            )
        )
        state = {
            "benchmark_name": "plancraft",
            "final_vote_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "VAL0491",
            "run_index": 0,
            "task_prompt": [
                {"role": "system", "content": "Use benchmark actions only."},
                {"role": "user", "content": "Target item: quartz"},
            ],
        }
        artifacts = [
            {
                "artifact_id": "a0",
                "agent_id": "agent_0",
                "answer": "smelt: from [I19] to [I18] with quantity 1",
                "summary": "Smelt the ore into quartz.",
                "confidence": 0.7,
                "unresolved_issues": [],
                "evidence_summary": ["The benchmark action grammar allows smelt here."],
                "source_artifact_ids": [],
            },
            {
                "artifact_id": "a1",
                "agent_id": "agent_1",
                "answer": "smelt: from [I19] to [I18] with quantity 1",
                "summary": "Smelt the ore into quartz.",
                "confidence": 0.75,
                "unresolved_issues": [],
                "evidence_summary": ["Quartz is obtained by smelting the provided ore."],
                "source_artifact_ids": [],
            },
            {
                "artifact_id": "a2",
                "agent_id": "agent_2",
                "answer": "impossible: no furnace or fuel is present",
                "summary": "Blocked because no furnace is in inventory.",
                "confidence": 0.8,
                "unresolved_issues": ["No furnace or fuel is available in inventory."],
                "evidence_summary": ["No furnace or fuel is available in inventory."],
                "source_artifact_ids": [],
            },
            {
                "artifact_id": "a3",
                "agent_id": "agent_3",
                "answer": "impossible: cannot smelt without a furnace or fuel",
                "summary": "Smelting is impossible without furnace inventory.",
                "confidence": 0.78,
                "unresolved_issues": ["Smelting would require furnace or fuel ownership."],
                "evidence_summary": ["The inventory does not contain a furnace or fuel."],
                "source_artifact_ids": [],
            },
        ]

        vote = engine._select_final_answer(
            state=state,
            stage_name="voter",
            artifacts=artifacts,
        )

        self.assertEqual(vote["answer"], "smelt: from [I19] to [I18] with quantity 1")
        self.assertEqual(vote["source"], "deterministic_fallback_inadmissible_winner")
        self.assertEqual(vote["selected_artifact_id"], "a1")

    def test_retrieval_singleton_with_open_criteria_does_not_override_non_substantive_majority(self) -> None:
        engine = LangGraphMASEngine(
            _JudgeLLM(
                text_by_agent_id={
                    "judge_final_vote_judge": (
                        '{"groups":[[1,2],[0]],"winner_index":0,"invalid_indices":[],"explanation":"Agent 0 gives the only concrete answer."}'
                    )
                }
            )
        )
        state = {
            "benchmark_name": "browsecomp",
            "final_vote_mode": "llm_judge",
            "llm_client": engine.llm_client,
            "task_id": "769",
            "run_index": 0,
            "task_prompt": "Which institution matches all criteria?",
            "termination_decision": {
                "consensus_is_substantive": False,
                "progress_status": "stalled",
                "expected_improvement": "low",
            },
        }
        artifacts = [
            {
                "artifact_id": "direct_guess",
                "agent_id": "agent_0",
                "answer": "Lingnan University",
                "summary": "Possible match, but criteria remain open.",
                "confidence": 0.92,
                "unresolved_issues": ["I could not verify that all degree criteria are satisfied."],
                "evidence_summary": ["One source names Lingnan University, but the required criteria remain unverified."],
                "source_artifact_ids": [],
            },
            {
                "artifact_id": "blocked_1",
                "agent_id": "agent_1",
                "answer": "I cannot determine the institution because the retrieved evidence does not resolve all criteria.",
                "summary": "Criteria remain unresolved.",
                "confidence": 0.65,
                "unresolved_issues": ["The available evidence is incomplete."],
                "evidence_summary": ["No evidence retrieved so far resolves all criteria."],
                "source_artifact_ids": [],
            },
            {
                "artifact_id": "blocked_2",
                "agent_id": "agent_2",
                "answer": "The answer cannot be determined from the currently retrieved evidence.",
                "summary": "Still missing required support.",
                "confidence": 0.6,
                "unresolved_issues": ["Required criteria remain open."],
                "evidence_summary": ["The current evidence is insufficient to satisfy all criteria."],
                "source_artifact_ids": [],
            },
        ]

        vote = engine._select_final_answer(
            state=state,
            stage_name="judge",
            artifacts=artifacts,
        )

        self.assertEqual(vote["source"], "llm_judge_non_direct_fallback")
        self.assertNotEqual(vote["answer"], "Lingnan University")
        self.assertIn(vote["selected_artifact_id"], {"blocked_1", "blocked_2"})

    def test_orchestrator_finalize_replaces_stale_merge_with_supported_latest_artifact(self) -> None:
        engine = LangGraphMASEngine(_JudgeLLM(text="unused"))
        state = {
            "topology": "orchestrator_with_discussion",
            "benchmark_name": "browsecomp",
            "final_vote_mode": "deterministic",
            "termination_decision": {
                "consensus_is_substantive": False,
                "progress_status": "stalled",
                "expected_improvement": "low",
            },
            "llm_client": engine.llm_client,
            "task_id": "769",
            "run_index": 0,
            "task_prompt": "Which institution matches all criteria?",
            "round_index": 1,
            "discussion_index": 1,
            "dispatch_id": 4,
            "phase": "descriptor_monitor",
            "artifacts": [
                {
                    "artifact_id": "merge_old",
                    "node_name": "orchestrator_merge",
                    "agent_id": "agent_0",
                    "round_index": 1,
                    "discussion_index": 1,
                    "dispatch_id": 1,
                    "answer": "Queen Arwa University",
                    "summary": "Queen Arwa University",
                    "confidence": 0.95,
                    "unresolved_issues": ["The 2022 criteria remain unverified."],
                    "evidence_summary": ["Document 82002 confirms the graduation ceremony date."],
                    "source_artifact_ids": [],
                },
                {
                    "artifact_id": "blocked_1",
                    "node_name": "specialists_revision_round",
                    "agent_id": "agent_1",
                    "round_index": 1,
                    "discussion_index": 1,
                    "dispatch_id": 4,
                    "answer": "The institution cannot be determined from the currently retrieved evidence.",
                    "summary": "Criteria remain unresolved.",
                    "confidence": 0.7,
                    "unresolved_issues": ["The available evidence is incomplete."],
                    "evidence_summary": ["No evidence retrieved so far resolves all criteria."],
                    "source_artifact_ids": [],
                },
                {
                    "artifact_id": "blocked_2",
                    "node_name": "specialists_revision_round",
                    "agent_id": "agent_2",
                    "round_index": 1,
                    "discussion_index": 1,
                    "dispatch_id": 4,
                    "answer": "The answer cannot be determined from the currently retrieved evidence.",
                    "summary": "Still missing required support.",
                    "confidence": 0.65,
                    "unresolved_issues": ["Required criteria remain open."],
                    "evidence_summary": ["The current evidence is insufficient to satisfy all criteria."],
                    "source_artifact_ids": [],
                },
            ],
        }

        result = engine._finalize_node(state)

        self.assertNotEqual(result["final_answer"], "Queen Arwa University")
        self.assertEqual(result["final_vote_source"], "deterministic_non_direct_fallback")
        self.assertIn(result["selected_artifact_id"], {"blocked_1", "blocked_2"})

    def test_deterministic_vote_prefers_evidence_backed_direct_answer(self) -> None:
        engine = LangGraphMASEngine(_JudgeLLM(text="unused"))
        state = {
            "final_vote_mode": "deterministic",
            "llm_client": engine.llm_client,
            "task_id": "task",
            "run_index": 0,
            "task_prompt": "Which institution is correct?",
        }
        artifacts = [
            {
                "agent_id": "agent_0",
                "answer": "University of the Philippines Diliman",
                "summary": "UP Diliman",
                "confidence": 0.95,
                "evidence_summary": ["No evidence gathered yet."],
            },
            {
                "agent_id": "agent_1",
                "answer": "Queen Arwa University",
                "summary": "Queen Arwa University",
                "confidence": 0.6,
                "evidence_summary": ["Document 82002 confirms the graduation ceremony date."],
            },
        ]

        vote = engine._select_final_answer(
            state=state,
            stage_name="judge",
            artifacts=artifacts,
        )

        self.assertEqual(vote["answer"], "Queen Arwa University")

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

    def test_fallback_answer_returns_explicit_failure_when_artifacts_are_empty(self) -> None:
        engine = LangGraphMASEngine(_JudgeLLM(text="unused"))

        self.assertEqual(
            engine._fallback_answer_from_artifacts([]),
            "Unable to determine a supported final answer from the available agent outputs.",
        )

    def test_execute_agent_stage_does_not_fabricate_tool_calls(self) -> None:
        llm = _SequencedLLM(
            [
                LLMResult(
                    text=(
                        '{"answer_artifact":"Queen Arwa University","summary":"Queen Arwa University",'
                        '"critique":"","revision_request":"","confidence":0.7,'
                        '"unresolved_issues":[],"evidence_summary":["Visible evidence supports Queen Arwa University."]}'
                    ),
                    token_in=12,
                    token_out=8,
                    cost_usd=0.0,
                    model="judge-model",
                    mock_used=False,
                    metadata={},
                    tool_calls=[],
                )
            ]
        )
        engine = LangGraphMASEngine(llm)
        state = {
            "layout": build_layout(topology="sas", num_agents=1),
            "dispatch_id": 0,
            "round_index": 0,
            "discussion_index": 0,
            "artifacts": [],
            "domain_personas": {},
            "task_prompt": "Which institution is correct?",
            "benchmark_name": "browsecomp",
            "llm_client": llm,
            "task_id": "task",
            "run_index": 0,
            "tools": [
                {
                    "name": "search",
                    "description": "Search documents.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "handler": lambda args: [],
                },
                {
                    "name": "get_document",
                    "description": "Fetch a document.",
                    "parameters": {"type": "object", "properties": {"docid": {"type": "string"}}},
                    "handler": lambda args: {"docid": args.get("docid", "")},
                },
            ],
            "max_tool_iterations": 4,
            "agent_type_by_agent": {"agent_0": "general"},
        }

        result = engine._execute_agent_stage(
            state,
            agent_id="agent_0",
            node_name="single_agent",
            stage_role="worker",
            directive="Solve the task.",
            visible_messages=[],
        )

        artifact = result["artifacts"][0]
        self.assertEqual(artifact.get("tool_records", []), [])
        self.assertEqual(result["tool_records_log"], [])
        self.assertFalse(result["interaction_logs"][0]["tool_use_retry"])

    def test_execute_agent_stage_retries_once_for_missing_tool_use(self) -> None:
        llm = _SequencedLLM(
            [
                LLMResult(
                    text=(
                        '{"answer_artifact":"The task is currently blocked because no evidence has been retrieved.",'
                        '"summary":"Blocked","critique":"","revision_request":"","confidence":0.0,'
                        '"unresolved_issues":[],"evidence_summary":["No evidence has been retrieved."]}'
                    ),
                    token_in=11,
                    token_out=7,
                    cost_usd=0.0,
                    model="judge-model",
                    mock_used=False,
                    metadata={},
                    tool_calls=[],
                ),
                LLMResult(
                    text=(
                        '{"answer_artifact":"Queen Arwa University","summary":"Queen Arwa University",'
                        '"critique":"","revision_request":"","confidence":0.8,'
                        '"unresolved_issues":[],"evidence_summary":["Document 82002 confirms the graduation ceremony date."]}'
                    ),
                    token_in=14,
                    token_out=9,
                    cost_usd=0.0,
                    model="judge-model",
                    mock_used=False,
                    metadata={},
                    tool_calls=[
                        {
                            "tool_name": "search",
                            "arguments": {"query": "Queen Arwa University graduation June 22 2003"},
                            "status": "completed",
                            "error": None,
                            "output": [{"docid": "82002", "snippet": "Graduation ceremony..."}],
                        }
                    ],
                ),
            ]
        )
        engine = LangGraphMASEngine(llm)
        state = {
            "layout": build_layout(topology="sas", num_agents=1),
            "dispatch_id": 0,
            "round_index": 0,
            "discussion_index": 0,
            "artifacts": [],
            "domain_personas": {},
            "task_prompt": "Which institution is correct?",
            "benchmark_name": "browsecomp",
            "llm_client": llm,
            "task_id": "task",
            "run_index": 0,
            "tools": [
                {
                    "name": "search",
                    "description": "Search documents.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "handler": lambda args: [],
                }
            ],
            "max_tool_iterations": 4,
            "agent_type_by_agent": {"agent_0": "general"},
        }

        result = engine._execute_agent_stage(
            state,
            agent_id="agent_0",
            node_name="single_agent",
            stage_role="worker",
            directive="Solve the task.",
            visible_messages=[],
        )

        artifact = result["artifacts"][0]
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(result["interaction_logs"][0]["tool_use_retry"])
        self.assertEqual(len(artifact.get("tool_records", [])), 1)
        self.assertEqual(artifact["tool_records"][0]["tool_name"], "search")

    def test_execute_agent_stage_retries_for_progress_status_without_tools(self) -> None:
        llm = _SequencedLLM(
            [
                LLMResult(
                    text=(
                        "I am currently investigating the identity of the learning institution. "
                        "I have initiated a search but have not yet identified the institution."
                    ),
                    token_in=10,
                    token_out=7,
                    cost_usd=0.0,
                    model="judge-model",
                    mock_used=False,
                    metadata={},
                    tool_calls=[],
                ),
                LLMResult(
                    text=(
                        '{"answer_artifact":"Queen Arwa University","summary":"Queen Arwa University",'
                        '"critique":"","revision_request":"","confidence":0.8,'
                        '"unresolved_issues":[],"evidence_summary":["Document 82002 confirms the graduation ceremony date."]}'
                    ),
                    token_in=14,
                    token_out=9,
                    cost_usd=0.0,
                    model="judge-model",
                    mock_used=False,
                    metadata={},
                    tool_calls=[
                        {
                            "tool_name": "search",
                            "arguments": {"query": "Queen Arwa University graduation June 22 2003"},
                            "status": "completed",
                            "error": None,
                            "output": [{"docid": "82002", "snippet": "Graduation ceremony..."}],
                        }
                    ],
                ),
            ]
        )
        engine = LangGraphMASEngine(llm)
        state = {
            "layout": build_layout(topology="sas", num_agents=1),
            "dispatch_id": 0,
            "round_index": 0,
            "discussion_index": 0,
            "artifacts": [],
            "domain_personas": {},
            "task_prompt": "Which institution is correct?",
            "benchmark_name": "browsecomp",
            "llm_client": llm,
            "task_id": "task",
            "run_index": 0,
            "tools": [
                {
                    "name": "search",
                    "description": "Search documents.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "handler": lambda args: [],
                }
            ],
            "max_tool_iterations": 4,
            "agent_type_by_agent": {"agent_0": "general"},
        }

        result = engine._execute_agent_stage(
            state,
            agent_id="agent_0",
            node_name="single_agent",
            stage_role="worker",
            directive="Solve the task.",
            visible_messages=[],
        )

        artifact = result["artifacts"][0]
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(result["interaction_logs"][0]["tool_use_retry"])
        self.assertEqual(len(artifact.get("tool_records", [])), 1)

    def test_opening_debate_round_retries_after_blocked_tool_attempt(self) -> None:
        llm = _SequencedLLM(
            [
                LLMResult(
                    text=(
                        '{"answer_artifact":"","summary":"The initial search did not yield a direct match.",'
                        '"critique":"","revision_request":"","confidence":0.0,'
                        '"unresolved_issues":["Need different query angle."],"evidence_summary":["No evidence was retrieved."]}'
                    ),
                    token_in=12,
                    token_out=8,
                    cost_usd=0.0,
                    model="judge-model",
                    mock_used=False,
                    metadata={},
                    tool_calls=[
                        {
                            "tool_name": "search",
                            "arguments": {"query": "2022 plant trip academic department"},
                            "status": "completed",
                            "error": None,
                            "output": [{"docid": "59188", "snippet": "Irrelevant result"}],
                        }
                    ],
                ),
                LLMResult(
                    text=(
                        '{"answer_artifact":"Queen Arwa University","summary":"Queen Arwa University",'
                        '"critique":"","revision_request":"","confidence":0.8,'
                        '"unresolved_issues":[],"evidence_summary":["Refined search found the institution."]}'
                    ),
                    token_in=16,
                    token_out=10,
                    cost_usd=0.0,
                    model="judge-model",
                    mock_used=False,
                    metadata={},
                    tool_calls=[
                        {
                            "tool_name": "search",
                            "arguments": {"query": "\"Queen Arwa University\" 2002 2022"},
                            "status": "completed",
                            "error": None,
                            "output": [{"docid": "82002", "snippet": "Queen Arwa University..."}],
                        }
                    ],
                ),
            ]
        )
        engine = LangGraphMASEngine(llm)
        state = {
            "layout": build_layout(topology="fully_linked_debate", num_agents=4),
            "dispatch_id": 0,
            "round_index": 0,
            "discussion_index": 0,
            "artifacts": [],
            "domain_personas": {},
            "task_prompt": "Which institution is correct?",
            "benchmark_name": "browsecomp",
            "llm_client": llm,
            "task_id": "task",
            "run_index": 0,
            "tools": [
                {
                    "name": "search",
                    "description": "Search documents.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "handler": lambda args: [],
                }
            ],
            "max_tool_iterations": 4,
            "agent_type_by_agent": {"agent_0": "general"},
        }

        result = engine._execute_agent_stage(
            state,
            agent_id="agent_0",
            node_name="debate_round",
            stage_role="critic",
            directive="Debate using bounded peer summaries.",
            visible_messages=[],
        )

        artifact = result["artifacts"][0]
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(result["interaction_logs"][0]["tool_use_retry"])
        self.assertEqual(extract_substantive_answer(artifact["answer"]), "Queen Arwa University")

    def test_dynamic_role_assignment_is_persisted_and_prompt_injected(self) -> None:
        llm = _RoleAwareLLM()
        engine = LangGraphMASEngine(llm)
        task = _Task(
            task_id="role_task",
            prompt="Identify the institution from the retrieved evidence.",
            metadata={},
        )

        result = engine.run(
            task=task,
            run_index=0,
            seed=7,
            spec=ExperimentSpec(
                topology="sas",
                num_agents=1,
                rounds=1,
                benchmark_name="browsecomp",
                enable_dynamic_roles=True,
            ),
            agent_types=["general"],
            tools=[],
            max_tool_iterations=1,
        )

        self.assertIn("role_assigner", llm.calls)
        self.assertEqual(
            result.run_metadata["role_assignment"]["assignments"]["agent_0"]["role_name"],
            "Web Search Strategist",
        )
        self.assertEqual(
            result.run_metadata["domain_personas"]["agent_0"]["role_name"],
            "Web Search Strategist",
        )
        first_prompt = result.run_metadata["interaction_logs"][0]["prompt_messages"][0]["content"]
        self.assertIn("Domain Role: Web Search Strategist", first_prompt)

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
