import json
import unittest
from dataclasses import dataclass, field
from typing import Any

from MAS.config import SelfEvolvedConfig
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.self_evolved.planner import TopologyPlannerAgent


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: Any
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class _PlanLLM(OpenRouterLLMClient):
    """Returns a canned plan for the topology planner and generic text otherwise."""

    def __init__(self, plan_text: str) -> None:
        self._plan_text = plan_text
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
        text = self._plan_text if agent_id == "topology_planner" else f"Answer from {agent_id}"
        return LLMResult(
            text=text,
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="plan-model",
            mock_used=False,
            metadata={},
        )


def _planner(plan_text: str, **config_kwargs: Any) -> TopologyPlannerAgent:
    return TopologyPlannerAgent(_PlanLLM(plan_text), SelfEvolvedConfig(**config_kwargs))


_VALID_PLAN = json.dumps(
    {
        "rationale": "Decomposable retrieval task with one contested branch.",
        "pattern": "star",
        "num_agents": 3,
        "verifier": True,
        "expansions": [{"member_index": 0, "pattern": "debate", "num_subagents": 2}],
    }
)


class TestTopologyPlanner(unittest.TestCase):
    def test_valid_plan_parsed_into_spec(self) -> None:
        planner = _planner(_VALID_PLAN)
        proposal = planner.propose_initial(
            task=_Task(task_id="t", prompt="Find the figure"),
            benchmark_name="finance_agent",
            num_agents=5,
        )
        self.assertFalse(proposal.used_fallback)
        self.assertIn("contested branch", proposal.rationale)

        spec = proposal.spec
        root = spec.group(spec.root_group_id)
        self.assertEqual(root.pattern, "star")
        self.assertEqual(root.leader_id, "agent_0")
        self.assertEqual(len(spec.agents), 5)

        # Expansion: root worker 0 (agent_1) owns a debate subgroup of 2.
        subgroup = spec.subgroup_of("agent_1")
        self.assertIsNotNone(subgroup)
        self.assertEqual(subgroup.pattern, "debate")
        self.assertEqual(len(subgroup.member_ids), 2)

        # Verifier: the last root worker contributes as a critic.
        self.assertEqual(spec.agent("agent_2").stage_role, "critic")
        self.assertEqual(spec.agent("agent_2").structural_role, "verifier")

    def test_task_analysis_folded_into_rationale(self) -> None:
        plan = json.dumps(
            {
                "task_analysis": {
                    "task_type": "state mutation",
                    "attributes": ["mutates external state"],
                    "failure_risks": ["duplicated writes"],
                },
                "rationale": "single executor avoids a double create",
                "pattern": "singleton",
                "num_agents": 1,
                "verifier": False,
                "expansions": [],
            }
        )
        proposal = _planner(plan).propose_initial(
            task=_Task(task_id="t", prompt="Create a calendar event"),
            benchmark_name="workbench",
            num_agents=3,
        )
        self.assertFalse(proposal.used_fallback)
        self.assertEqual(proposal.spec.group(proposal.spec.root_group_id).pattern, "singleton")
        # The planner's own analysis is surfaced in the rationale (and thus the trace).
        self.assertIn("type=state mutation", proposal.rationale)
        self.assertIn("risks=duplicated writes", proposal.rationale)

    def test_initial_prompt_carries_topology_guidance(self) -> None:
        planner = _planner(_VALID_PLAN)
        messages = planner._build_initial_prompt(
            task=_Task(task_id="t", prompt="q"),
            benchmark_name="workbench",
            max_agents=3,
            playbook_entries=[],
            principles=[],
        )
        blob = messages[0]["content"] + messages[1]["content"]
        self.assertIn("ANALYZE", blob)
        self.assertIn("External state mutation", blob)
        self.assertIn("task_analysis", blob)

    def test_malformed_response_falls_back(self) -> None:
        planner = _planner("I think a star topology would be nice.")
        proposal = planner.propose_initial(
            task=_Task(task_id="t", prompt="q"),
            benchmark_name="finance_agent",
            num_agents=4,
        )
        self.assertTrue(proposal.used_fallback)
        self.assertEqual(proposal.fallback_reason, "invalid_or_unparseable_plan")
        root = proposal.spec.group(proposal.spec.root_group_id)
        self.assertEqual(root.pattern, "star")
        self.assertEqual(len(proposal.spec.agents), 4)

    def test_unknown_pattern_falls_back(self) -> None:
        planner = _planner(json.dumps({"pattern": "mesh", "num_agents": 3}))
        proposal = planner.propose_initial(
            task=_Task(task_id="t", prompt="q"),
            benchmark_name="finance_agent",
            num_agents=3,
        )
        self.assertTrue(proposal.used_fallback)

    def test_agent_budget_clamped(self) -> None:
        planner = _planner(json.dumps({"pattern": "voting", "num_agents": 50}))
        proposal = planner.propose_initial(
            task=_Task(task_id="t", prompt="q"),
            benchmark_name="finance_agent",
            num_agents=4,
        )
        self.assertFalse(proposal.used_fallback)
        self.assertEqual(len(proposal.spec.agents), 4)

    def test_voting_uses_diversity_quorum_when_budget_allows(self) -> None:
        planner = _planner(json.dumps({"pattern": "voting", "num_agents": 2}))
        proposal = planner.propose_initial(
            task=_Task(task_id="t", prompt="q"),
            benchmark_name="math500",
            num_agents=5,
        )

        self.assertFalse(proposal.used_fallback)
        self.assertEqual(len(proposal.spec.agents), 4)

    def test_oversized_expansion_skipped(self) -> None:
        plan = json.dumps(
            {
                "pattern": "star",
                "num_agents": 4,
                "expansions": [{"member_index": 0, "pattern": "voting", "num_subagents": 9}],
            }
        )
        planner = _planner(plan)
        proposal = planner.propose_initial(
            task=_Task(task_id="t", prompt="q"),
            benchmark_name="finance_agent",
            num_agents=5,
        )
        self.assertFalse(proposal.used_fallback)
        self.assertIsNone(proposal.spec.subgroup_of("agent_1"))
        self.assertEqual(len(proposal.spec.agents), 4)

    def test_playbook_entries_appear_in_prompt(self) -> None:
        client = _PlanLLM(_VALID_PLAN)
        planner = TopologyPlannerAgent(client, SelfEvolvedConfig())
        proposal = planner.propose_initial(
            task=_Task(task_id="t", prompt="q"),
            benchmark_name="finance_agent",
            num_agents=5,
            playbook_entries=[
                {"key": "finance_agent::retrieval", "pattern": "star", "notes": "works well"}
            ],
        )
        self.assertEqual(proposal.playbook_keys, ["finance_agent::retrieval"])
        prompt_text = json.dumps(proposal.prompt_messages)
        self.assertIn("finance_agent::retrieval", prompt_text)
        self.assertIn("Historical experience", prompt_text)


class TestPlannerEngineIntegration(unittest.TestCase):
    def test_engine_runs_planner_proposed_topology(self) -> None:
        from MAS.langgraph_engine import ExperimentSpec
        from MAS.self_evolved.engine import SelfEvolvedEngine

        engine = SelfEvolvedEngine(_PlanLLM(_VALID_PLAN), SelfEvolvedConfig())
        spec = ExperimentSpec(
            topology="self_evolved",
            num_agents=5,
            rounds=2,
            communication_budget_per_agent=2,
            termination_consensus_mode="lexical",
            final_vote_mode="deterministic",
            benchmark_name="finance_agent",
            enable_dynamic_roles=False,
        )
        result = engine.run(
            task=_Task(task_id="t-e2e", prompt="Question needing decomposition"),
            run_index=0,
            seed=1,
            spec=spec,
            agent_types=["general"],
        )

        planner_meta = result.run_metadata["self_evolved"]["planner"]
        self.assertFalse(planner_meta["used_fallback"])

        spec_payload = result.run_metadata["self_evolved"]["topology_spec_versions"][0]
        self.assertEqual(len(spec_payload["groups"]), 2)
        patterns = {group["pattern"] for group in spec_payload["groups"]}
        self.assertEqual(patterns, {"star", "debate"})

        actors = {event.actor for event in result.trace_events}
        # Debate subgroup members executed.
        self.assertIn("agent_3", actors)
        self.assertIn("agent_4", actors)

        planner_events = [
            event for event in result.trace_events if event.actor == "topology_planner"
        ]
        self.assertEqual(len(planner_events), 1)
        self.assertFalse(planner_events[0].payload["used_fallback"])
        self.assertTrue(str(result.final_answer).strip())


if __name__ == "__main__":
    unittest.main()
