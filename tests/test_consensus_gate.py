"""Consensus stopping-gate and broadened finalize-synthesis trigger.

The gate mirrors the Trace Auditor's premature_consensus predicate: high answer
agreement is only a stop signal when it is decision-grade (avg confidence >= 0.5
and no open issues) while another step remains. The consensus_ratio metric is
unchanged; only the stop decision gains the gate.
"""

import unittest
from typing import Any

from MAS.config import OpenRouterConfig, SelfEvolvedConfig
from MAS.langgraph_engine import LangGraphMASEngine
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.self_evolved.engine import SelfEvolvedEngine


class _MockClient(OpenRouterLLMClient):
    def __init__(self) -> None:  # noqa: D401 - mock, no real config needed
        pass

    def generate(self, **_kwargs: Any) -> LLMResult:
        return LLMResult(
            text="MOCK", token_in=0, token_out=0, cost_usd=0.0, model="m", mock_used=True, metadata={}
        )


def _agreeing(confidence: float, *, unresolved: list[str] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": agent_id,
            "answer": "The answer is 9.",
            "confidence": confidence,
            "unresolved_issues": list(unresolved or []),
        }
        for agent_id in ("a0", "a1")
    ]


def _decide(engine: LangGraphMASEngine, artifacts: list[dict[str, Any]], *, max_reached: bool):
    state = {
        "termination_consensus_mode": "lexical",
        "llm_client": engine.llm_client,
        "task_id": "t",
        "run_index": 0,
        "task_prompt": "q",
    }
    return engine._termination_decision(
        state,
        stage_name="meta",
        round_index=0,
        discussion_index=0,
        minimum_required_rounds=0,
        candidate_artifacts=artifacts,
        previous_candidate_artifacts=[],
        consensus_artifacts=artifacts,
        expected_count=2,
        max_reached=max_reached,
        continue_next_step="apply_mutation",
        stop_next_step="finalize",
    )


class TestConsensusGate(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = LangGraphMASEngine(_MockClient())

    def test_low_confidence_agreement_is_gated_when_repair_available(self) -> None:
        decision = _decide(self.engine, _agreeing(0.3), max_reached=False)
        self.assertNotEqual(decision["reason"], "consensus_reached")
        self.assertFalse(decision["should_stop"])
        self.assertTrue(decision["consensus_gate_blocked"])

    def test_unresolved_issue_agreement_is_gated(self) -> None:
        decision = _decide(self.engine, _agreeing(0.9, unresolved=["open q"]), max_reached=False)
        self.assertNotEqual(decision["reason"], "consensus_reached")
        self.assertTrue(decision["consensus_gate_blocked"])

    def test_confident_clean_agreement_reaches_consensus(self) -> None:
        decision = _decide(self.engine, _agreeing(0.9), max_reached=False)
        self.assertEqual(decision["reason"], "consensus_reached")
        self.assertFalse(decision["consensus_gate_blocked"])

    def test_no_repair_left_still_stops_on_agreement(self) -> None:
        decision = _decide(self.engine, _agreeing(0.3), max_reached=True)
        self.assertEqual(decision["reason"], "consensus_reached")
        self.assertFalse(decision["consensus_gate_blocked"])


def _se_engine() -> SelfEvolvedEngine:
    client = OpenRouterLLMClient(OpenRouterConfig(api_key=None), {"default": "test-model"})
    return SelfEvolvedEngine(client, SelfEvolvedConfig())


class TestNeedsFinalSynthesis(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _se_engine()

    def _state(self, *, confidence: float = 0.8, unresolved: list[str] | None = None):
        return {
            "artifacts": [
                {
                    "artifact_id": "win",
                    "agent_id": "a0",
                    "answer": "The answer is Paris.",
                    "confidence": confidence,
                    "unresolved_issues": list(unresolved or []),
                }
            ]
        }

    def test_strong_confident_pick_no_synthesis(self) -> None:
        vote = {"selected_artifact_id": "win", "tally": {"the answer is paris.": 1}}
        self.assertFalse(
            self.engine._needs_final_synthesis(
                "The answer is Paris.", vote_result=vote, state=self._state()
            )
        )

    def test_empty_answer_needs_synthesis(self) -> None:
        self.assertTrue(
            self.engine._needs_final_synthesis(
                "", vote_result={"tally": {}}, state={"artifacts": []}
            )
        )

    def test_vote_tie_needs_synthesis(self) -> None:
        vote = {"selected_artifact_id": "win", "tally": {"paris": 1, "london": 1}}
        self.assertTrue(
            self.engine._needs_final_synthesis(
                "The answer is Paris.", vote_result=vote, state=self._state()
            )
        )

    def test_low_confidence_winner_needs_synthesis(self) -> None:
        vote = {"selected_artifact_id": "win", "tally": {"the answer is paris.": 1}}
        self.assertTrue(
            self.engine._needs_final_synthesis(
                "The answer is Paris.", vote_result=vote, state=self._state(confidence=0.3)
            )
        )

    def test_unresolved_winner_needs_synthesis(self) -> None:
        vote = {"selected_artifact_id": "win", "tally": {"the answer is paris.": 1}}
        self.assertTrue(
            self.engine._needs_final_synthesis(
                "The answer is Paris.", vote_result=vote, state=self._state(unresolved=["check date"])
            )
        )


if __name__ == "__main__":
    unittest.main()
