import json
import unittest
from dataclasses import dataclass, field
from typing import Any

from MAS.config import SelfEvolvedConfig
from MAS.langgraph_engine import ExperimentSpec
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.relay import build_layout
from MAS.self_evolved.context import SharedContextController
from MAS.self_evolved.engine import SelfEvolvedEngine
from MAS.self_evolved.spec import MutationOp, TopologyMutation, spec_from_layout

_NESTED_PLAN = json.dumps(
    {
        "rationale": "Two branches need decomposition and verification.",
        "pattern": "star",
        "num_agents": 3,
        "expansions": [
            {"member_index": 0, "pattern": "star", "num_subagents": 2},
            {"member_index": 1, "pattern": "debate", "num_subagents": 2},
        ],
    }
)


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: Any
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _nested_spec():
    """Root star agent_0..2 + star subgroup on agent_1 (agent_3, agent_4) +
    debate subgroup on agent_2 (agent_5, agent_6)."""

    spec = spec_from_layout(build_layout(topology="orchestrator_no_discussion", num_agents=3))
    mutation = TopologyMutation(
        rationale="",
        target_failure_modes=(),
        ops=(
            MutationOp(
                op="expand_agent_to_group",
                args={"agent_id": "agent_1", "pattern": "star", "num_subagents": 2},
            ),
            MutationOp(
                op="expand_agent_to_group",
                args={"agent_id": "agent_2", "pattern": "debate", "num_subagents": 2},
            ),
        ),
    )
    return mutation.apply(spec, max_agents=10)


def _with_policy(spec, agent_id: str, **policy_args: Any):
    mutation = TopologyMutation(
        rationale="",
        target_failure_modes=(),
        ops=(MutationOp(op="set_context_policy", args={"agent_id": agent_id, **policy_args}),),
    )
    return mutation.apply(spec, max_agents=10)


def _packet(
    sender: str,
    recipients: list[str],
    *,
    kind: str = "specialist_report",
    message_id: str = "m_1",
    content: str = "claim",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "message_id": message_id,
        "dispatch_id": 1,
        "sender": sender,
        "recipients": recipients,
        "kind": kind,
        "phase": "test",
        "round": 0,
        "discussion_index": 0,
        "artifact_id": "a_1",
        "content": content,
        "payload": payload or {"artifact_id": "a_1", "summary": content},
    }


def _artifact(agent_id: str, *, answer: str, evidence: list[str] | None = None) -> dict[str, Any]:
    return {
        "artifact_id": f"art_{agent_id}",
        "agent_id": agent_id,
        "answer": answer,
        "summary": answer,
        "confidence": 0.8,
        "evidence_summary": list(evidence or []),
    }


class TestShareScope(unittest.TestCase):
    def test_misaddressed_packet_hidden_from_sibling_branch(self) -> None:
        controller = SharedContextController(_nested_spec())
        # agent_3 lives in g_agent_1; default share_scope="group" means a packet
        # mis-addressed to agent_5 (sibling branch) must stay invisible.
        state = {"messages": [_packet("agent_3", ["agent_5", "agent_1"])]}
        self.assertEqual(controller.visible_packets(state, agent_id="agent_5"), [])
        # The attachment parent still sees it.
        visible = controller.visible_packets(state, agent_id="agent_1")
        self.assertEqual(len(visible), 1)

    def test_parent_only_scope(self) -> None:
        spec = _with_policy(_nested_spec(), "agent_3", share_scope="parent_only")
        controller = SharedContextController(spec)
        state = {"messages": [_packet("agent_3", ["agent_1", "agent_4"])]}
        # Group peer is blocked, parent allowed.
        self.assertEqual(controller.visible_packets(state, agent_id="agent_4"), [])
        self.assertEqual(len(controller.visible_packets(state, agent_id="agent_1")), 1)

    def test_global_scope_crosses_branches(self) -> None:
        spec = _with_policy(_nested_spec(), "agent_3", share_scope="global")
        controller = SharedContextController(spec)
        state = {"messages": [_packet("agent_3", ["agent_5"])]}
        self.assertEqual(len(controller.visible_packets(state, agent_id="agent_5")), 1)

    def test_group_scope_allows_own_subgroup_children(self) -> None:
        controller = SharedContextController(_nested_spec())
        # agent_1 (default group scope) plans task packets for its children.
        state = {"messages": [_packet("agent_1", ["agent_3"], kind="task_package")]}
        self.assertEqual(len(controller.visible_packets(state, agent_id="agent_3")), 1)


class TestReaderBounds(unittest.TestCase):
    def test_summary_only_compacts_payload_and_content(self) -> None:
        spec = _with_policy(_nested_spec(), "agent_1", summary_only=True, max_packet_chars=64)
        controller = SharedContextController(spec)
        long_text = "evidence " * 50
        original = _packet(
            "agent_3",
            ["agent_1"],
            content=long_text,
            payload={
                "artifact_id": "a_1",
                "summary": long_text,
                "answer_artifact": long_text,
                "critique": long_text,
                "confidence": 0.7,
            },
        )
        state = {"messages": [original]}
        visible = controller.visible_packets(state, agent_id="agent_1")
        self.assertEqual(len(visible), 1)
        compacted = visible[0]
        self.assertLessEqual(len(compacted["content"]), 64)
        self.assertTrue(compacted["content"].endswith("..."))
        self.assertNotIn("answer_artifact", compacted["payload"])
        self.assertNotIn("critique", compacted["payload"])
        self.assertEqual(compacted["payload"]["confidence"], 0.7)
        # The stored packet is untouched; only the reader's copy is compacted.
        self.assertEqual(state["messages"][0]["content"], long_text)

    def test_max_packet_chars_bounds_payload_fields(self) -> None:
        spec = _with_policy(_nested_spec(), "agent_1", max_packet_chars=48)
        controller = SharedContextController(spec)
        long_text = "x" * 500
        state = {
            "messages": [
                _packet(
                    "agent_3",
                    ["agent_1"],
                    content=long_text,
                    payload={"summary": long_text, "answer_artifact": long_text},
                )
            ]
        }
        visible = controller.visible_packets(state, agent_id="agent_1")
        payload = visible[0]["payload"]
        self.assertLessEqual(len(visible[0]["content"]), 48)
        self.assertLessEqual(len(payload["summary"]), 48)
        self.assertLessEqual(len(payload["answer_artifact"]), 48)
        self.assertTrue(payload["summary"].endswith("..."))

    def test_default_policy_leaves_packets_untouched(self) -> None:
        controller = SharedContextController(_nested_spec())
        long_text = "y" * 500
        state = {"messages": [_packet("agent_3", ["agent_1"], content=long_text)]}
        visible = controller.visible_packets(state, agent_id="agent_1")
        self.assertEqual(visible[0]["content"], long_text)


class TestEvidenceLedger(unittest.TestCase):
    def _controller_with_ledger(self):
        controller = SharedContextController(_nested_spec())
        state: dict[str, Any] = {"round_index": 0, "dispatch_id": 3}
        controller.record_evidence(
            state,
            _artifact("agent_3", answer="Branch one says 42.", evidence=["doc A"]),
            agent_id="agent_3",
        )
        controller.record_evidence(
            state,
            _artifact("agent_5", answer="Branch two says 41.", evidence=["doc B"]),
            agent_id="agent_5",
        )
        controller.record_evidence(
            state, _artifact("agent_0", answer="Root merge pending."), agent_id="agent_0"
        )
        return controller, state

    def test_branch_agent_ids(self) -> None:
        controller = SharedContextController(_nested_spec())
        self.assertEqual(controller.branch_agent_ids("agent_3"), {"agent_1", "agent_3", "agent_4"})
        self.assertEqual(controller.branch_agent_ids("agent_5"), {"agent_2", "agent_5", "agent_6"})
        self.assertEqual(controller.branch_agent_ids("agent_0"), {"agent_0"})

    def test_digest_scopes(self) -> None:
        controller, state = self._controller_with_ledger()

        self.assertIsNone(controller.evidence_digest_packet(state, agent_id="agent_3"))

        branch = controller.evidence_digest_packet(state, agent_id="agent_3", scope="branch")
        self.assertIsNotNone(branch)
        self.assertIn("agent_3[r0]", branch["content"])
        self.assertIn("doc A", branch["content"])
        self.assertNotIn("agent_5[r0]", branch["content"])

        global_digest = controller.evidence_digest_packet(state, agent_id="agent_0", scope="global")
        self.assertIn("agent_3[r0]", global_digest["content"])
        self.assertIn("agent_5[r0]", global_digest["content"])
        self.assertEqual(global_digest["kind"], "evidence_digest")
        self.assertEqual(global_digest["recipients"], ["agent_0"])

    def test_policy_evidence_access_enables_digest(self) -> None:
        spec = _with_policy(_nested_spec(), "agent_3", evidence_access="global")
        controller = SharedContextController(spec)
        state: dict[str, Any] = {"round_index": 0}
        controller.record_evidence(
            state, _artifact("agent_5", answer="Across branches."), agent_id="agent_5"
        )
        digest = controller.evidence_digest_packet(state, agent_id="agent_3")
        self.assertIsNotNone(digest)
        self.assertIn("agent_5[r0]", digest["content"])

    def test_digest_is_bounded(self) -> None:
        controller = SharedContextController(_nested_spec())
        state: dict[str, Any] = {"round_index": 0}
        for index in range(15):
            controller.record_evidence(
                state,
                _artifact("agent_3", answer=f"claim {index} " + "z" * 1000),
                agent_id="agent_3",
            )
        digest = controller.evidence_digest_packet(state, agent_id="agent_0", scope="global")
        entries = digest["payload"]["entries"]
        self.assertEqual(len(entries), 12)
        for entry in entries:
            self.assertLessEqual(len(entry["claim"]), 160)
            self.assertTrue(entry["claim"].endswith("..."))

    def test_empty_or_unsubstantive_artifacts_not_recorded(self) -> None:
        controller = SharedContextController(_nested_spec())
        state: dict[str, Any] = {"round_index": 0}
        controller.record_evidence(state, None, agent_id="agent_3")
        controller.record_evidence(
            state, {"artifact_id": "a", "answer": "", "summary": ""}, agent_id="agent_3"
        )
        self.assertNotIn("evidence_ledger", state)


class _NestedLLM(OpenRouterLLMClient):
    """Planner proposes the nested topology; every agent answers directly.
    Captures each agent's prompts so digest injection can be asserted."""

    def __init__(self) -> None:
        self.prompts_by_agent: dict[str, list[str]] = {}

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
        prompt_text = (
            prompt
            if isinstance(prompt, str)
            else " ".join(str(m.get("content", "")) for m in prompt)
        )
        self.prompts_by_agent.setdefault(agent_id, []).append(prompt_text)
        text = _NESTED_PLAN if agent_id == "topology_planner" else "The answer is 42."
        return LLMResult(
            text=text,
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="nested",
            mock_used=False,
            metadata={},
        )


class TestExecutorDigestIntegration(unittest.TestCase):
    def _run(self) -> tuple[Any, _NestedLLM]:
        client = _NestedLLM()
        engine = SelfEvolvedEngine(
            client,
            SelfEvolvedConfig(max_initial_agents=7, max_total_agents=10, playbook_read=False),
        )
        spec = ExperimentSpec(
            topology="self_evolved",
            num_agents=7,
            rounds=2,
            communication_budget_per_agent=2,
            termination_consensus_mode="lexical",
            final_vote_mode="deterministic",
            benchmark_name="finance_agent",
            enable_dynamic_roles=False,
        )
        result = engine.run(
            task=_Task(task_id="ctx-task", prompt="Question with two branches"),
            run_index=0,
            seed=11,
            spec=spec,
            agent_types=["general"],
            tools=[],
        )
        return result, client

    def test_root_aggregator_prompt_contains_global_evidence_digest(self) -> None:
        _, client = self._run()
        merge_prompt = client.prompts_by_agent["agent_0"][-1]
        self.assertIn("Evidence ledger (scope=global", merge_prompt)
        # Both branches' entries reach the final synthesis stage.
        self.assertIn("agent_3[r0]", merge_prompt)
        self.assertIn("agent_5[r0]", merge_prompt)

    def test_branch_aggregator_sees_only_its_branch(self) -> None:
        _, client = self._run()
        merge_prompt = client.prompts_by_agent["agent_1"][-1]
        self.assertIn("Evidence ledger (scope=branch", merge_prompt)
        self.assertIn("agent_3[r0]", merge_prompt)
        self.assertNotIn("agent_5[r0]", merge_prompt)

    def test_workers_get_no_digest_by_default(self) -> None:
        _, client = self._run()
        for worker in ("agent_3", "agent_4"):
            for prompt in client.prompts_by_agent.get(worker, []):
                self.assertNotIn("Evidence ledger", prompt)

    def test_branch_packets_never_in_sibling_views(self) -> None:
        result, _ = self._run()
        branch_one = {"agent_3", "agent_4"}
        branch_two = {"agent_5", "agent_6"}
        for view in result.run_metadata["message_views"]:
            viewer = view["viewer"]
            senders = set(view["visible_senders"])
            if viewer in branch_one:
                self.assertFalse(senders & branch_two, view)
            elif viewer in branch_two:
                self.assertFalse(senders & branch_one, view)


if __name__ == "__main__":
    unittest.main()
