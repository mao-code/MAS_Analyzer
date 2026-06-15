import unittest
from dataclasses import dataclass, field
from typing import Any

from descriptor.schema import validate_trace_events
from MAS.config import OpenRouterConfig, SelfEvolvedConfig
from MAS.langgraph_engine import ExperimentSpec
from MAS.llm import OpenRouterLLMClient
from MAS.self_evolved.engine import SelfEvolvedEngine


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: Any
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _mock_client() -> OpenRouterLLMClient:
    return OpenRouterLLMClient(OpenRouterConfig(api_key=None), {"default": "test-model"})


def _run_engine(num_agents: int) -> Any:
    engine = SelfEvolvedEngine(_mock_client(), SelfEvolvedConfig())
    spec = ExperimentSpec(
        topology="self_evolved",
        num_agents=num_agents,
        rounds=2,
        discussion_rounds=1,
        communication_budget_per_agent=2,
        termination_consensus_mode="lexical",
        final_vote_mode="deterministic",
        benchmark_name="finance_agent",
        enable_dynamic_roles=True,
    )
    return engine.run(
        task=_Task(task_id="t0", prompt="What is the answer to the question?"),
        run_index=0,
        seed=42,
        spec=spec,
        agent_types=["general"],
        tools=[],
        max_tool_iterations=2,
    )


class TestSelfEvolvedEngineSmoke(unittest.TestCase):
    def test_multi_agent_run_produces_valid_trace(self) -> None:
        result = _run_engine(num_agents=3)

        self.assertTrue(str(result.final_answer).strip())
        validate_trace_events(result.trace_events)

        metadata = result.run_metadata
        self.assertEqual(metadata["topology"], "self_evolved")
        self.assertEqual(metadata["self_evolved"]["harness_backend"], "openrouter")
        self.assertEqual(len(metadata["self_evolved"]["topology_spec_versions"]), 1)
        self.assertTrue(metadata["self_evolved"]["planner"]["used_fallback"])
        self.assertEqual(metadata["topology_layout"]["topology"], "self_evolved")

        actors = {event.actor for event in result.trace_events}
        self.assertIn("topology_planner", actors)
        self.assertIn("orchestrator", actors)
        self.assertIn("agent_0", actors)

        planner_events = [
            event for event in result.trace_events if event.actor == "topology_planner"
        ]
        self.assertEqual(planner_events[0].event_type, "plan")
        self.assertTrue(planner_events[0].payload["used_fallback"])

        finalize_events = [event for event in result.trace_events if event.event_type == "finalize"]
        self.assertEqual(len(finalize_events), 1)
        self.assertEqual(finalize_events[0].payload["status"], "completed")

        # Code-level termination decision is recorded.
        history = metadata["termination_history"]
        self.assertTrue(history)
        self.assertTrue(history[-1]["should_stop"])

    def test_workers_only_see_leader_task_packets(self) -> None:
        result = _run_engine(num_agents=3)
        views = result.run_metadata["message_views"]
        worker_views = [view for view in views if view["viewer"] in {"agent_1", "agent_2"}]
        self.assertTrue(worker_views)
        for view in worker_views:
            self.assertTrue(set(view["visible_senders"]) <= {"agent_0"})

    def test_single_agent_run(self) -> None:
        result = _run_engine(num_agents=1)
        self.assertTrue(str(result.final_answer).strip())
        validate_trace_events(result.trace_events)
        spec_payloads = result.run_metadata["self_evolved"]["topology_spec_versions"]
        self.assertEqual(spec_payloads[0]["groups"][0]["pattern"], "singleton")

    def test_runner_dispatches_self_evolved_engine(self) -> None:
        from MAS.config import ExperimentConfig, ExperimentRuntimeConfig, MASConfig
        from MAS.runner import MASRunner

        config = ExperimentConfig(
            openrouter=OpenRouterConfig(api_key=None),
            mas=MASConfig(
                levels=1,
                intra_level_link_ratio=1.0,
                full_linked=True,
                topology="self_evolved",
                number_of_agents=3,
                agent_types=["general"],
                communication_count_internally=2,
                final_vote_mode="deterministic",
                termination_consensus_mode="lexical",
            ),
            experiment=ExperimentRuntimeConfig(runs_per_task=1),
            models={"default": "test-model"},
        )
        config.validate()
        runner = MASRunner(config, _mock_client())
        result = runner.run_task(
            _Task(task_id="t1", prompt="A runner-dispatched question"),
            run_index=0,
            seed=7,
            benchmark_name="finance_agent",
        )
        self.assertEqual(result.run_metadata["topology"], "self_evolved")
        self.assertTrue(str(result.final_answer).strip())


if __name__ == "__main__":
    unittest.main()
