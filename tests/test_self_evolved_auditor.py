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

    def test_premature_consensus_on_low_confidence_agreement(self) -> None:
        state = _base_state(
            artifacts=[
                _artifact("agent_1", answer="The answer is 9.", confidence=0.3),
                _artifact("agent_2", answer="The answer is 9.", confidence=0.3),
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


if __name__ == "__main__":
    unittest.main()
