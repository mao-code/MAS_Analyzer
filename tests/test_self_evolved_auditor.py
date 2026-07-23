import json
import unittest
from typing import Any

from MAS.config import SelfEvolvedConfig
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.relay import build_layout
from MAS.self_evolved.auditor import TraceAuditorAgent
from MAS.self_evolved.spec import spec_from_layout

BLOCKED_TEXT = "Insufficient evidence: the answer cannot be determined."


class _JudgeLLM(OpenRouterLLMClient):
    def __init__(self, text: str, *, mock_used: bool = False) -> None:
        self._text = text
        self._mock_used = mock_used
        self.prompts: list[Any] = []

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
        self.prompts.append(prompt)
        return LLMResult(
            text=self._text,
            token_in=5,
            token_out=5,
            cost_usd=0.0,
            model="judge",
            mock_used=self._mock_used,
            metadata={},
        )


def _artifact(
    agent_id: str,
    *,
    stage_role: str = "worker",
    answer: str = "The answer is 42.",
    confidence: float = 0.8,
    evidence: list[str] | None = None,
    unresolved: list[str] | None = None,
    round_index: int = 0,
) -> dict[str, Any]:
    return {
        "artifact_id": f"artifact:{agent_id}:r{round_index}",
        "agent_id": agent_id,
        "stage_role": stage_role,
        "round_index": round_index,
        "answer": answer,
        "confidence": confidence,
        "evidence_summary": list(evidence or []),
        "unresolved_issues": list(unresolved or []),
    }


def _spec(num_agents: int = 3):
    return spec_from_layout(
        build_layout(topology="orchestrator_no_discussion", num_agents=num_agents)
    )


def _auditor(**config_kwargs: Any) -> TraceAuditorAgent:
    return TraceAuditorAgent(_JudgeLLM("", mock_used=True), SelfEvolvedConfig(**config_kwargs))


def _base_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "task_id": "t",
        "run_index": 0,
        "artifacts": [],
        "tool_records_log": [],
        "messages": [],
    }
    state.update(overrides)
    return state


class TestTraceAuditorHeuristics(unittest.TestCase):
    def _modes(self, report: dict[str, Any]) -> set[str]:
        return {str(mode["mode"]) for mode in report["detected_modes"]}

    def test_failed_tool_records_flag_tool_error_cascade(self) -> None:
        state = _base_state(
            artifacts=[_artifact("agent_1", evidence=["found doc"])],
            tool_records_log=[
                {"agent_id": "agent_1", "round_index": 0, "tool_name": "search", "status": "error"},
                {"agent_id": "agent_1", "round_index": 0, "tool_name": "search", "status": "error"},
            ],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertIn("tool_error_cascade", self._modes(report))
        self.assertTrue(report["repair_recommended"])
        mode = next(m for m in report["detected_modes"] if m["mode"] == "tool_error_cascade")
        self.assertEqual(mode["severity"], "high")
        self.assertEqual(mode["agent_ids"], ["agent_1"])

    def test_blocked_contributions_flag_branch_collapse(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact("agent_1", answer=BLOCKED_TEXT, confidence=0.0),
                _artifact("agent_2", answer="The answer is 7.", evidence=["doc"]),
            ],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertIn("branch_collapse", self._modes(report))
        self.assertTrue(report["repair_recommended"])

    def test_evidence_lost_before_synthesis(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact("agent_1", evidence=["retrieved filing"]),
                _artifact("agent_2", evidence=["retrieved report"]),
                _artifact("agent_0", stage_role="aggregator", evidence=[], confidence=0.7),
            ],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertIn("evidence_lost_before_synthesis", self._modes(report))

    def test_impossibility_claim_that_names_constructive_path_is_challenged(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact(
                    "agent_1",
                    answer=(
                        "Impossible: the target cannot be crafted, although the existing "
                        "item must be transformed to become the target."
                    ),
                    confidence=1.0,
                )
            ]
        )

        report = _auditor().audit(state, _spec(), turn_index=0)

        self.assertIn("unsupported_impossibility_claim", self._modes(report))
        self.assertTrue(report["repair_recommended"])
        self.assertTrue(report["challenge_consensus"])

    def test_unanimous_impossibility_is_challenged_once_without_reference_labels(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact(
                    "agent_1",
                    answer="Impossible: only two blocks are available but three are required.",
                    confidence=1.0,
                ),
                _artifact(
                    "agent_2",
                    answer="The requested transformation is not possible with this input.",
                    confidence=1.0,
                ),
            ]
        )

        report = _auditor().audit(state, _spec(), turn_index=0)

        mode = next(
            mode
            for mode in report["detected_modes"]
            if mode["mode"] == "unverified_impossibility_consensus"
        )
        self.assertEqual(mode["severity"], "medium")
        self.assertEqual(mode["agent_ids"], ["agent_1", "agent_2"])
        self.assertTrue(report["repair_recommended"])
        self.assertTrue(report["challenge_consensus"])
        self.assertIn("counterexample", mode["detail"])

    def test_mixed_constructive_and_impossible_candidates_do_not_trigger_unanimous_gate(
        self,
    ) -> None:
        state = _base_state(
            artifacts=[
                _artifact("agent_1", answer="Impossible: one ingredient is missing."),
                _artifact("agent_2", answer="move: from [I2] to [B1] with quantity 1"),
            ]
        )

        report = _auditor().audit(state, _spec(), turn_index=0)

        self.assertNotIn("unverified_impossibility_consensus", self._modes(report))

    def test_premature_consensus_on_low_confidence_agreement(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact("agent_1", answer="The answer is 9.", confidence=0.3),
                _artifact("agent_2", answer="The answer is 9.", confidence=0.3),
            ],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertIn("premature_consensus", self._modes(report))

    def test_confident_evidence_action_consensus_can_execute_with_open_issue(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact(
                    "agent_1",
                    answer="search: prismarine_brick_stairs",
                    confidence=1.0,
                    unresolved=["The exact recipe has not been retrieved yet."],
                ),
                _artifact(
                    "agent_2",
                    answer="search: prismarine_brick_stairs",
                    confidence=1.0,
                    unresolved=["Recipe placement remains unknown until search executes."],
                ),
            ],
        )

        report = _auditor().audit(state, _spec(), turn_index=0)

        self.assertNotIn("premature_consensus", self._modes(report))
        self.assertFalse(report["challenge_consensus"])

    def test_low_confidence_evidence_action_consensus_is_still_challenged(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact("agent_1", answer="search: unknown_item", confidence=0.3),
                _artifact("agent_2", answer="search: unknown_item", confidence=0.3),
            ],
        )

        report = _auditor().audit(state, _spec(), turn_index=0)

        self.assertIn("premature_consensus", self._modes(report))

    def test_missing_validator_on_low_confidence(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact("agent_1", answer="Maybe 4.", confidence=0.4, evidence=["doc"]),
                _artifact("agent_2", answer="Maybe 5.", confidence=0.4, evidence=["doc"]),
            ],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertIn("missing_validator", self._modes(report))

    def test_duplicate_state_mutation_across_agents(self) -> None:
        # Two workers issue the identical side-effecting call -> double-apply risk.
        state = _base_state(
            artifacts=[_artifact("agent_1"), _artifact("agent_2")],
            tools=[{"name": "calendar.create_event"}],
            tool_records_log=[
                {
                    "agent_id": "agent_1",
                    "round_index": 0,
                    "tool_name": "calendar.create_event",
                    "arguments": {"name": "Standup", "date": "2026-06-25"},
                    "status": "ok",
                },
                {
                    "agent_id": "agent_2",
                    "round_index": 0,
                    "tool_name": "calendar.create_event",
                    "arguments": {"name": "Standup", "date": "2026-06-25"},
                    "status": "ok",
                },
            ],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertIn("duplicate_state_mutation", self._modes(report))
        self.assertTrue(report["repair_recommended"])
        mode = next(m for m in report["detected_modes"] if m["mode"] == "duplicate_state_mutation")
        self.assertEqual(mode["severity"], "high")
        self.assertEqual(mode["agent_ids"], ["agent_1", "agent_2"])

    def test_distinct_tool_args_do_not_flag_duplicate(self) -> None:
        # Different arguments = legitimately distinct calls; must not flag.
        state = _base_state(
            artifacts=[_artifact("agent_1"), _artifact("agent_2")],
            tools=[{"name": "search"}],
            tool_records_log=[
                {
                    "agent_id": "agent_1",
                    "round_index": 0,
                    "tool_name": "search",
                    "arguments": {"query": "a"},
                    "status": "ok",
                },
                {
                    "agent_id": "agent_2",
                    "round_index": 0,
                    "tool_name": "search",
                    "arguments": {"query": "b"},
                    "status": "ok",
                },
            ],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertNotIn("duplicate_state_mutation", self._modes(report))

    def test_insufficient_search_coverage_no_document_opened(self) -> None:
        # Retrieval run searched but never opened a document -> high severity.
        state = _base_state(
            artifacts=[_artifact("agent_1", evidence=["snippet"])],
            tools=[{"name": "search"}, {"name": "get_document"}],
            tool_records_log=[
                {
                    "agent_id": "agent_1",
                    "round_index": 0,
                    "tool_name": "search",
                    "arguments": {"query": "a"},
                    "status": "ok",
                },
            ],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertIn("insufficient_search_coverage", self._modes(report))
        mode = next(
            m for m in report["detected_modes"] if m["mode"] == "insufficient_search_coverage"
        )
        self.assertEqual(mode["severity"], "high")

    def test_insufficient_search_coverage_when_retrieval_tools_are_unused(self) -> None:
        state = _base_state(
            artifacts=[_artifact("agent_1", answer=BLOCKED_TEXT)],
            tools=[{"name": "search"}, {"name": "get_document"}],
        )

        report = _auditor().audit(state, _spec(), turn_index=0)

        mode = next(
            m for m in report["detected_modes"] if m["mode"] == "insufficient_search_coverage"
        )
        self.assertEqual(mode["severity"], "high")
        self.assertTrue(report["repair_recommended"])

    def test_constraint_search_counts_as_search_coverage(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact("agent_1", evidence=["doc"]),
                _artifact("agent_2", evidence=["doc"]),
            ],
            tools=[{"name": "search"}, {"name": "get_document"}],
            tool_records_log=[
                {
                    "agent_id": "agent_1",
                    "round_index": 0,
                    "tool_name": "constraint_search",
                    "status": "ok",
                },
                {
                    "agent_id": "agent_2",
                    "round_index": 0,
                    "tool_name": "constraint_search",
                    "status": "ok",
                },
                {
                    "agent_id": "agent_1",
                    "round_index": 0,
                    "tool_name": "get_document",
                    "status": "ok",
                },
            ],
        )

        report = _auditor().audit(state, _spec(), turn_index=0)

        self.assertNotIn("insufficient_search_coverage", self._modes(report))

    def test_search_coverage_ok_with_multiple_searchers_and_read(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact("agent_1", evidence=["doc"]),
                _artifact("agent_2", evidence=["doc"]),
            ],
            tools=[{"name": "search"}, {"name": "get_document"}],
            tool_records_log=[
                {
                    "agent_id": "agent_1",
                    "round_index": 0,
                    "tool_name": "search",
                    "arguments": {"query": "a"},
                    "status": "ok",
                },
                {
                    "agent_id": "agent_2",
                    "round_index": 0,
                    "tool_name": "search",
                    "arguments": {"query": "b"},
                    "status": "ok",
                },
                {
                    "agent_id": "agent_1",
                    "round_index": 0,
                    "tool_name": "get_document",
                    "arguments": {"docid": "d1"},
                    "status": "ok",
                },
            ],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertNotIn("insufficient_search_coverage", self._modes(report))

    def test_clean_turn_recommends_no_repair(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact("agent_1", answer="The answer is 4.", confidence=0.9, evidence=["doc"]),
                _artifact("agent_2", answer="The answer is 5.", confidence=0.9, evidence=["doc"]),
                _artifact(
                    "agent_0",
                    stage_role="aggregator",
                    answer="The answer is 4.",
                    confidence=0.9,
                    evidence=["doc"],
                ),
            ],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertFalse(report["repair_recommended"])
        self.assertEqual(report["source"], "heuristic")

    def test_other_round_artifacts_ignored(self) -> None:
        state = _base_state(
            artifacts=[_artifact("agent_1", answer=BLOCKED_TEXT, round_index=1)],
        )
        report = _auditor().audit(state, _spec(), turn_index=0)
        self.assertEqual(report["detected_modes"], [])


class TestTraceAuditorLLMJudge(unittest.TestCase):
    def test_llm_judge_overrides_heuristic_recommendation(self) -> None:
        judge_response = json.dumps(
            {"repair_recommended": False, "recommendation": "Blocked branch is benign."}
        )
        auditor = TraceAuditorAgent(
            _JudgeLLM(judge_response), SelfEvolvedConfig(audit_mode="llm_judge")
        )
        state = _base_state(
            artifacts=[_artifact("agent_1", answer=BLOCKED_TEXT, confidence=0.0)],
        )
        report = auditor.audit(state, _spec(), turn_index=0)
        self.assertEqual(report["source"], "llm_judge")
        self.assertFalse(report["repair_recommended"])
        self.assertIn("benign", report["recommendation"])
        # Heuristic findings stay recorded.
        self.assertTrue(report["detected_modes"])

    def test_mock_judge_keeps_heuristics(self) -> None:
        auditor = TraceAuditorAgent(
            _JudgeLLM("", mock_used=True), SelfEvolvedConfig(audit_mode="llm_judge")
        )
        state = _base_state(
            artifacts=[_artifact("agent_1", answer=BLOCKED_TEXT, confidence=0.0)],
        )
        report = auditor.audit(state, _spec(), turn_index=0)
        self.assertEqual(report["source"], "heuristic")
        self.assertTrue(report["repair_recommended"])


class TestTraceAuditorOpenSetHybrid(unittest.TestCase):
    def test_audit_snapshot_preserves_long_answer_conclusion(self) -> None:
        llm = _JudgeLLM(
            json.dumps(
                {
                    "repair_recommended": False,
                    "recommendation": "Complete answer.",
                    "new_failure_modes": [],
                }
            )
        )
        auditor = TraceAuditorAgent(llm, SelfEvolvedConfig(audit_mode="hybrid"))
        long_answer = "reasoning " * 400 + "Final conclusion: \\boxed{28}"
        state = _base_state(artifacts=[_artifact("agent_1", answer=long_answer)])

        auditor.audit(state, _spec(), turn_index=0)

        prompt_text = " ".join(str(message.get("content", "")) for message in llm.prompts[0])
        self.assertIn("middle omitted for audit budget", prompt_text)
        self.assertIn(r"\\boxed{28}", prompt_text)

    def test_structured_prompt_keeps_current_user_separate_from_old_examples(self) -> None:
        response = json.dumps(
            {
                "repair_recommended": True,
                "recommendation": "Repair the alleged mismatch.",
                "new_failure_modes": [
                    {
                        "mode": "hallucinated_task_context",
                        "severity": "high",
                        "confidence": 0.99,
                        "repairable": True,
                        "agent_ids": ["agent_1"],
                        "evidence": [
                            {"ref": "task", "quote": "iron_ingot"},
                            {
                                "ref": "artifact:agent_1:r0",
                                "quote": "smelt the salmon",
                            },
                        ],
                        "detail": "The current task asks for iron but the agent uses salmon.",
                    }
                ],
            }
        )
        llm = _JudgeLLM(response)
        auditor = TraceAuditorAgent(llm, SelfEvolvedConfig(audit_mode="hybrid"))
        state = _base_state(
            task_prompt=[
                {"role": "system", "content": "Return one environment action."},
                {"role": "user", "content": "Example: craft iron_ingot."},
                {"role": "assistant", "content": "smelt: iron ore"},
                {"role": "user", "content": "Current task: craft cooked_salmon."},
            ],
            artifacts=[_artifact("agent_1", answer="smelt the salmon", confidence=1.0)],
        )

        report = auditor.audit(state, _spec(), turn_index=0)

        self.assertEqual(report["novel_modes"], [])
        prompt_text = " ".join(str(message.get("content", "")) for message in llm.prompts[0])
        self.assertIn('"current_task": "Current task: craft cooked_salmon."', prompt_text)
        self.assertIn('"recent_context"', prompt_text)

    def test_grounded_novel_failure_can_trigger_repair_and_challenge_consensus(self) -> None:
        response = json.dumps(
            {
                "repair_recommended": True,
                "recommendation": "Add an independent verifier for the omitted requirement.",
                "new_failure_modes": [
                    {
                        "mode": "requirement_coverage_gap",
                        "severity": "high",
                        "confidence": 0.91,
                        "repairable": True,
                        "agent_ids": ["agent_1", "invented_agent"],
                        "evidence": [
                            {"ref": "task", "quote": "Return both the value"},
                            {"ref": "artifact:agent_1:r0", "quote": "The value is 42."},
                        ],
                        "detail": "The answer covers only one of the task's two requested outputs.",
                    }
                ],
            }
        )
        llm = _JudgeLLM(response)
        auditor = TraceAuditorAgent(llm, SelfEvolvedConfig(audit_mode="hybrid"))
        state = _base_state(
            task_prompt="Return both the value and its justification.",
            artifacts=[_artifact("agent_1", answer="The value is 42.", confidence=0.95)],
        )

        report = auditor.audit(state, _spec(), turn_index=0)

        self.assertEqual(report["source"], "hybrid")
        self.assertTrue(report["repair_recommended"])
        self.assertTrue(report["challenge_consensus"])
        self.assertGreaterEqual(report["risk_score"], 0.9)
        novel = report["novel_modes"][0]
        self.assertEqual(novel["mode"], "requirement_coverage_gap")
        self.assertEqual(novel["agent_ids"], ["agent_1"])
        self.assertEqual(novel["evidence_refs"], ["task", "artifact:agent_1:r0"])
        prompt_text = " ".join(str(message.get("content", "")) for message in llm.prompts[0])
        self.assertIn("Return both the value", prompt_text)
        self.assertIn("artifact:agent_1:r0", prompt_text)

    def test_ungrounded_or_low_confidence_novel_modes_are_observations_not_actions(self) -> None:
        response = json.dumps(
            {
                "repair_recommended": True,
                "recommendation": "Repair it.",
                "new_failure_modes": [
                    {
                        "mode": "imagined_failure",
                        "severity": "high",
                        "confidence": 0.99,
                        "repairable": True,
                        "agent_ids": ["agent_1"],
                        "evidence": [{"ref": "not-in-the-trace", "quote": "Unsupported"}],
                        "detail": "Unsupported claim.",
                    },
                    {
                        "mode": "weak_hunch",
                        "severity": "high",
                        "confidence": 0.4,
                        "repairable": True,
                        "agent_ids": ["agent_1"],
                        "evidence": [{"ref": "artifact:agent_1:r0", "quote": "The answer is 42."}],
                        "detail": "Below the confidence floor.",
                    },
                ],
            }
        )
        auditor = TraceAuditorAgent(_JudgeLLM(response), SelfEvolvedConfig(audit_mode="hybrid"))
        state = _base_state(artifacts=[_artifact("agent_1", confidence=0.9)])

        report = auditor.audit(state, _spec(), turn_index=0)

        self.assertEqual(report["novel_modes"], [])
        self.assertFalse(report["repair_recommended"])
        self.assertFalse(report["challenge_consensus"])

    def test_high_confidence_medium_mode_can_challenge_wrong_consensus(self) -> None:
        response = json.dumps(
            {
                "repair_recommended": True,
                "recommendation": "Re-check the omitted action.",
                "new_failure_modes": [
                    {
                        "mode": "unsupported_confident_synthesis",
                        "severity": "medium",
                        "confidence": 0.95,
                        "repairable": True,
                        "agent_ids": ["agent_1"],
                        "evidence": [
                            {"ref": "task", "quote": "explicit action"},
                            {"ref": "artifact:agent_1:r0", "quote": "The answer is 42."},
                        ],
                        "detail": "The answer ignores an action explicitly present in the task.",
                    }
                ],
            }
        )
        auditor = TraceAuditorAgent(_JudgeLLM(response), SelfEvolvedConfig(audit_mode="hybrid"))
        state = _base_state(
            task_prompt="Perform the explicit action and report the result.",
            artifacts=[_artifact("agent_1", confidence=1.0)],
        )

        report = auditor.audit(state, _spec(), turn_index=0)

        self.assertTrue(report["challenge_consensus"])

    def test_invented_task_quote_rejects_hallucinated_failure_mode(self) -> None:
        response = json.dumps(
            {
                "repair_recommended": True,
                "recommendation": "Repair the alleged task mismatch.",
                "new_failure_modes": [
                    {
                        "mode": "hallucinated_task_context",
                        "severity": "high",
                        "confidence": 0.99,
                        "repairable": True,
                        "agent_ids": ["agent_1"],
                        "evidence": [
                            {"ref": "task", "quote": "craft an iron_ingot"},
                            {
                                "ref": "artifact:agent_1:r0",
                                "quote": "smelt the salmon",
                            },
                        ],
                        "detail": "The task asks for iron_ingot but the agent smelts salmon.",
                    }
                ],
            }
        )
        auditor = TraceAuditorAgent(_JudgeLLM(response), SelfEvolvedConfig(audit_mode="hybrid"))
        state = _base_state(
            task_prompt="Craft cooked_salmon from the salmon in inventory.",
            artifacts=[_artifact("agent_1", answer="smelt the salmon", confidence=1.0)],
        )

        report = auditor.audit(state, _spec(), turn_index=0)

        self.assertEqual(report["novel_modes"], [])
        self.assertFalse(report["repair_recommended"])
        self.assertFalse(report["challenge_consensus"])

    def test_single_anchor_novel_mode_is_observed_but_cannot_spend_repair_budget(self) -> None:
        response = json.dumps(
            {
                "repair_recommended": True,
                "recommendation": "Add a verifier.",
                "new_failure_modes": [
                    {
                        "mode": "possible_reasoning_gap",
                        "severity": "high",
                        "confidence": 0.99,
                        "repairable": True,
                        "agent_ids": ["agent_1"],
                        "evidence": [{"ref": "artifact:agent_1:r0", "quote": "The answer is 42."}],
                        "detail": "The answer may omit a derivation.",
                    }
                ],
            }
        )
        auditor = TraceAuditorAgent(_JudgeLLM(response), SelfEvolvedConfig(audit_mode="hybrid"))
        state = _base_state(artifacts=[_artifact("agent_1", confidence=1.0)])

        report = auditor.audit(state, _spec(), turn_index=0)

        self.assertEqual(len(report["novel_modes"]), 1)
        self.assertFalse(report["novel_modes"][0]["repairable"])
        self.assertFalse(report["repair_recommended"])
        self.assertFalse(report["challenge_consensus"])

    def test_hybrid_model_cannot_veto_deterministic_failure(self) -> None:
        response = json.dumps(
            {
                "repair_recommended": False,
                "recommendation": "No model-discovered issue.",
                "new_failure_modes": [],
            }
        )
        auditor = TraceAuditorAgent(_JudgeLLM(response), SelfEvolvedConfig(audit_mode="hybrid"))
        state = _base_state(artifacts=[_artifact("agent_1", answer=BLOCKED_TEXT, confidence=0.0)])

        report = auditor.audit(state, _spec(), turn_index=0)

        self.assertIn("branch_collapse", {mode["mode"] for mode in report["detected_modes"]})
        self.assertTrue(report["repair_recommended"])


if __name__ == "__main__":
    unittest.main()
