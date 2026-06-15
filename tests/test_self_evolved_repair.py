import json
import unittest
from dataclasses import dataclass, field
from typing import Any

from descriptor.schema import validate_trace_events
from MAS.config import SelfEvolvedConfig
from MAS.langgraph_engine import ExperimentSpec
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.relay import build_layout
from MAS.self_evolved.engine import SelfEvolvedEngine
from MAS.self_evolved.spec import MutationOp, TopologyMutation, spec_from_layout

BLOCKED_TEXT = "Insufficient evidence: the answer cannot be determined."

_INITIAL_PLAN = json.dumps(
    {
        "rationale": "Star with three branches for a decomposable task.",
        "pattern": "star",
        "num_agents": 3,
        "expansions": [],
    }
)

# The doc's example repair: one leaf becomes a star-orchestrator, the other a
# fully-linked debate group.
_MUTATION_PLAN = json.dumps(
    {
        "rationale": "Branch agent_1 lacked decomposition; branch agent_2 needs verification.",
        "ops": [
            {
                "op": "expand_agent_to_group",
                "agent_id": "agent_1",
                "pattern": "star",
                "num_subagents": 2,
            },
            {
                "op": "expand_agent_to_group",
                "agent_id": "agent_2",
                "pattern": "debate",
                "num_subagents": 2,
            },
        ],
    }
)


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: Any
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class _ScriptedLLM(OpenRouterLLMClient):
    """Planner gets scripted responses; workers are blocked in round 0 and
    answer directly in round 1; the coordinator always answers directly."""

    def __init__(self, planner_responses: list[str], *, blocked_round0: bool = True) -> None:
        self._planner_responses = list(planner_responses)
        self._blocked_round0 = blocked_round0
        self.calls: list[str] = []
        self.planner_prompts: list[str] = []

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
        prompt_text = (
            prompt
            if isinstance(prompt, str)
            else " ".join(str(m.get("content", "")) for m in prompt)
        )
        if agent_id == "topology_planner":
            self.planner_prompts.append(prompt_text)
            text = self._planner_responses.pop(0) if self._planner_responses else "{}"
        elif agent_id == "agent_0":
            text = "The final answer is 42."
        elif self._blocked_round0 and '"round_index": 0' in prompt_text:
            text = BLOCKED_TEXT
        else:
            text = "The answer is 42."
        return LLMResult(
            text=text,
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="scripted",
            mock_used=False,
            metadata={},
        )


def _experiment_spec() -> ExperimentSpec:
    return ExperimentSpec(
        topology="self_evolved",
        num_agents=3,
        rounds=2,
        communication_budget_per_agent=2,
        termination_consensus_mode="lexical",
        final_vote_mode="deterministic",
        benchmark_name="finance_agent",
        enable_dynamic_roles=False,
    )


def _run(engine: SelfEvolvedEngine) -> Any:
    return engine.run(
        task=_Task(task_id="repair-task", prompt="Question with two weak branches"),
        run_index=0,
        seed=3,
        spec=_experiment_spec(),
        agent_types=["general"],
    )


class TestSingleRepairLoop(unittest.TestCase):
    def test_repair_applied_exactly_once(self) -> None:
        client = _ScriptedLLM([_INITIAL_PLAN, _MUTATION_PLAN])
        engine = SelfEvolvedEngine(client, SelfEvolvedConfig())
        result = _run(engine)
        validate_trace_events(result.trace_events)

        meta = result.run_metadata["self_evolved"]
        self.assertEqual(len(meta["topology_spec_versions"]), 2)
        self.assertEqual(len(meta["topology_spec_versions"][0]["agents"]), 3)
        self.assertEqual(len(meta["topology_spec_versions"][1]["agents"]), 7)

        mutation = meta["mutation"]
        self.assertIsNotNone(mutation)
        self.assertEqual(len(mutation["mutation"]["ops"]), 2)
        self.assertEqual(len(mutation["added_agents"]), 4)
        self.assertIn("branch_collapse", mutation["mutation"]["target_failure_modes"])
        self.assertEqual(len(meta["short_term_playbook_entries"]), 2)
        self.assertEqual(meta["short_term_playbook_entries"][0]["scope"], "short_term")
        self.assertEqual(len(meta["context_state_versions"]), 2)
        self.assertEqual(meta["context_state_versions"][0]["reason"], "spawn")
        self.assertEqual(meta["context_state_versions"][1]["reason"], "mutation")
        self.assertIn("turn-level process memory", client.planner_prompts[1])
        self.assertIn("branch_collapse", client.planner_prompts[1])

        revise_events = [
            event
            for event in result.trace_events
            if event.actor == "orchestrator"
            and event.event_type == "revise"
            and event.payload.get("node") == "apply_mutation"
        ]
        self.assertEqual(len(revise_events), 1)

        self.assertEqual(result.run_metadata["turns_executed"], 2)
        history = result.run_metadata["termination_history"]
        self.assertFalse(history[0]["should_stop"])
        self.assertTrue(history[-1]["should_stop"])
        # The repaired turn converges, so the loop stops on consensus.
        self.assertEqual(history[-1]["reason"], "consensus_reached")

        # Agents minted by the mutation actually executed in turn 1.
        actors = {event.actor for event in result.trace_events}
        for new_agent in ("agent_3", "agent_4", "agent_5", "agent_6"):
            self.assertIn(new_agent, actors)

        self.assertTrue(str(result.final_answer).strip())
        self.assertIn("42", result.final_answer)

    def test_consensus_stops_without_mutation(self) -> None:
        engine = SelfEvolvedEngine(
            _ScriptedLLM([_INITIAL_PLAN, _MUTATION_PLAN], blocked_round0=False),
            SelfEvolvedConfig(),
        )
        result = _run(engine)

        meta = result.run_metadata["self_evolved"]
        self.assertEqual(len(meta["topology_spec_versions"]), 1)
        self.assertIsNone(meta["mutation"])
        self.assertEqual(result.run_metadata["turns_executed"], 1)
        history = result.run_metadata["termination_history"]
        self.assertEqual(history[-1]["reason"], "consensus_reached")

        revise_events = [
            event for event in result.trace_events if event.payload.get("node") == "apply_mutation"
        ]
        self.assertEqual(revise_events, [])

    def test_unusable_mutation_response_skips_repair(self) -> None:
        engine = SelfEvolvedEngine(
            _ScriptedLLM([_INITIAL_PLAN, "no mutation for you"]), SelfEvolvedConfig()
        )
        result = _run(engine)

        meta = result.run_metadata["self_evolved"]
        self.assertEqual(len(meta["topology_spec_versions"]), 1)
        self.assertIsNone(meta["mutation"])
        skip_events = [
            event
            for event in result.trace_events
            if event.payload.get("node") == "mutation_skipped"
        ]
        self.assertEqual(len(skip_events), 1)
        self.assertEqual(skip_events[0].payload.get("reason"), "invalid_or_unparseable_mutation")

    def test_audit_events_present(self) -> None:
        engine = SelfEvolvedEngine(
            _ScriptedLLM([_INITIAL_PLAN, _MUTATION_PLAN]), SelfEvolvedConfig()
        )
        result = _run(engine)
        audit_events = [event for event in result.trace_events if event.actor == "trace_auditor"]
        self.assertEqual(len(audit_events), 2)
        self.assertTrue(
            audit_events[0].payload["audit"]["repair_recommended"],
        )
        self.assertEqual(len(result.run_metadata["self_evolved"]["audit_reports"]), 2)


class TestMutationOps(unittest.TestCase):
    def _star_spec(self):
        return spec_from_layout(build_layout(topology="orchestrator_no_discussion", num_agents=3))

    def test_expand_agent_to_group(self) -> None:
        spec = self._star_spec()
        mutation = TopologyMutation(
            rationale="split branch",
            target_failure_modes=("branch_collapse",),
            ops=(
                MutationOp(
                    op="expand_agent_to_group",
                    args={"agent_id": "agent_1", "pattern": "star", "num_subagents": 3},
                ),
            ),
        )
        mutated = mutation.apply(spec, max_agents=10)
        self.assertEqual(mutated.version, 1)
        self.assertEqual(len(mutated.agents), 6)
        subgroup = mutated.subgroup_of("agent_1")
        self.assertEqual(subgroup.pattern, "star")
        layout = mutated.to_layout()
        self.assertEqual(layout.parent_by_agent["agent_3"], "agent_1")

    def test_expand_rejects_double_expansion(self) -> None:
        spec = self._star_spec()
        op = MutationOp(
            op="expand_agent_to_group",
            args={"agent_id": "agent_1", "pattern": "star", "num_subagents": 2},
        )
        mutation = TopologyMutation(rationale="", target_failure_modes=(), ops=(op,))
        mutated = mutation.apply(spec, max_agents=10)
        with self.assertRaises(ValueError):
            mutation.apply(mutated, max_agents=10)

    def test_expand_rejects_budget_overflow(self) -> None:
        spec = self._star_spec()
        mutation = TopologyMutation(
            rationale="",
            target_failure_modes=(),
            ops=(
                MutationOp(
                    op="expand_agent_to_group",
                    args={"agent_id": "agent_1", "pattern": "voting", "num_subagents": 5},
                ),
            ),
        )
        with self.assertRaises(ValueError):
            mutation.apply(spec, max_agents=5)

    def test_set_group_pattern_adjusts_leader(self) -> None:
        spec = self._star_spec()
        to_debate = TopologyMutation(
            rationale="",
            target_failure_modes=(),
            ops=(
                MutationOp(
                    op="set_group_pattern", args={"group_id": "g_root", "pattern": "debate"}
                ),
            ),
        )
        mutated = to_debate.apply(spec, max_agents=10)
        root = mutated.group("g_root")
        self.assertEqual(root.pattern, "debate")
        self.assertIsNone(root.leader_id)

    def test_add_agent_with_critic_stage(self) -> None:
        spec = self._star_spec()
        mutation = TopologyMutation(
            rationale="add validator",
            target_failure_modes=("missing_validator",),
            ops=(
                MutationOp(
                    op="add_agent",
                    args={
                        "group_id": "g_root",
                        "structural_role": "verifier",
                        "stage_role": "critic",
                    },
                ),
            ),
        )
        mutated = mutation.apply(spec, max_agents=10)
        self.assertEqual(len(mutated.agents), 4)
        new_agent = mutated.agent("agent_3")
        self.assertEqual(new_agent.stage_role, "critic")
        self.assertIn("agent_3", mutated.group("g_root").member_ids)

    def test_edge_ops_round_trip(self) -> None:
        spec = self._star_spec()
        add = TopologyMutation(
            rationale="",
            target_failure_modes=(),
            ops=(MutationOp(op="add_edge", args={"src": "agent_1", "dst": "agent_2"}),),
        )
        mutated = add.apply(spec, max_agents=10)
        self.assertIn(("agent_1", "agent_2"), mutated.extra_edges)
        self.assertIn("agent_2", mutated.to_layout().adjacency["agent_1"])

        remove = TopologyMutation(
            rationale="",
            target_failure_modes=(),
            ops=(MutationOp(op="remove_edge", args={"src": "agent_2", "dst": "agent_1"}),),
        )
        back = remove.apply(mutated, max_agents=10)
        self.assertEqual(back.extra_edges, ())

    def test_set_context_policy(self) -> None:
        spec = self._star_spec()
        mutation = TopologyMutation(
            rationale="widen evidence",
            target_failure_modes=("evidence_lost_before_synthesis",),
            ops=(
                MutationOp(
                    op="set_context_policy",
                    args={"agent_id": "agent_0", "evidence_access": "global"},
                ),
            ),
        )
        mutated = mutation.apply(spec, max_agents=10)
        self.assertEqual(mutated.agent("agent_0").context.evidence_access, "global")
        # Other agents untouched.
        self.assertEqual(mutated.agent("agent_1").context.evidence_access, "own")


if __name__ == "__main__":
    unittest.main()
