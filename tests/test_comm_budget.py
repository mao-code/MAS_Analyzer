import unittest

from MAS.config import ExperimentConfig, ExperimentRuntimeConfig, MASConfig, OpenRouterConfig
from MAS.llm import OpenRouterLLMClient
from MAS.runner import MASRunner
from benchmark.base import BenchmarkTask


class TestCommunicationBudget(unittest.TestCase):
    def test_per_agent_total_budget_across_turns(self) -> None:
        config = ExperimentConfig(
            openrouter=OpenRouterConfig(api_key=None),
            mas=MASConfig(
                levels=2,
                intra_level_link_ratio=1.0,
                full_linked=True,
                number_of_agents=4,
                agents_per_level=[2, 2],
                agent_types=["general"],
                communication_count_internally=1,
                turn_mode="multi_turn",
                max_turns=5,
            ),
            experiment=ExperimentRuntimeConfig(output_dir="outputs", runs_per_task=1, seed=3),
            models={"default": "openai/gpt-4o-mini"},
        )
        config.validate()

        llm = OpenRouterLLMClient(config.openrouter, config.models)
        runner = MASRunner(config, llm)

        task = BenchmarkTask(task_id="t1", prompt="Compute result for [123]", reference_answer="x")
        run = runner.run_task(task=task, run_index=0, seed=99)

        by_agent = run.run_metadata["messages_sent_by_agent"]
        self.assertTrue(by_agent)
        self.assertTrue(all(count <= 1 for count in by_agent.values()))
        self.assertLessEqual(run.run_metadata["turns_executed"], config.mas.max_turns)


if __name__ == "__main__":
    unittest.main()
