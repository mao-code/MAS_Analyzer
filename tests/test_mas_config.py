import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from benchmark.base import BenchmarkTask
from MAS.config import load_experiment_config
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.runner import MASRunner


class _ArtifactLLM(OpenRouterLLMClient):
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
            text=(
                '{"answer_artifact":"4","summary":"4","critique":"","revision_request":"",'
                '"confidence":1.0,"unresolved_issues":[],"evidence_summary":[]}'
            ),
            token_in=3,
            token_out=5,
            cost_usd=0.0,
            model="mock-model",
            mock_used=False,
            metadata={},
        )


class TestMASConfig(unittest.TestCase):
    def _write(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        tmp.write(textwrap.dedent(content))
        tmp.flush()
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def test_load_valid_config(self) -> None:
        path = self._write(
            """
            [openrouter]
            api_key = "abc"

            [experiment]
            output_dir = "outputs"
            runs_per_task = 2
            seed = 7

            [mas]
            levels = 2
            intra_level_link_ratio = 0.5
            full_linked = false
            number_of_agents = 4
            agent_types = ["planner", "researcher"]
            communication_count_internally = 1
            turn_mode = "multi_turn"
            max_turns = 3

            [models]
            default = "openai/gpt-4o-mini"
            planner = "openai/gpt-4o-mini"
            """
        )

        cfg = load_experiment_config(path)
        self.assertEqual(cfg.mas.total_agents, 4)
        self.assertEqual(cfg.mas.turn_mode, "multi_turn")
        self.assertEqual(cfg.models["default"], "openai/gpt-4o-mini")
        self.assertEqual(cfg.mas.termination_consensus_mode, "llm_judge")
        self.assertEqual(cfg.mas.final_vote_mode, "llm_judge")
        self.assertEqual(cfg.self_evolved.max_turns, 5)
        self.assertEqual(cfg.self_evolved.repair_budget, 4)
        self.assertEqual(cfg.self_evolved.audit_mode, "hybrid")

    def test_env_override_api_key(self) -> None:
        path = self._write(
            """
            [openrouter]
            api_key = "file_key"

            [experiment]
            runs_per_task = 1
            seed = 1

            [mas]
            levels = 1
            intra_level_link_ratio = 1.0
            full_linked = true
            number_of_agents = 1
            agent_types = ["general"]
            communication_count_internally = 0
            turn_mode = "single_turn"
            max_turns = 1

            [models]
            default = "openai/gpt-4o-mini"
            """
        )

        old = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "env_key"
        try:
            cfg = load_experiment_config(path)
            self.assertEqual(cfg.openrouter.api_key, "env_key")
        finally:
            if old is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old

    def test_invalid_agents_per_level_length(self) -> None:
        path = self._write(
            """
            [openrouter]
            api_key = ""

            [experiment]
            runs_per_task = 1
            seed = 1

            [mas]
            levels = 2
            intra_level_link_ratio = 1.0
            full_linked = true
            agents_per_level = [2]
            agent_types = ["general"]
            communication_count_internally = 0
            turn_mode = "single_turn"
            max_turns = 1

            [models]
            default = "openai/gpt-4o-mini"
            """
        )

        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_invalid_termination_consensus_mode(self) -> None:
        path = self._write(
            """
            [openrouter]
            api_key = ""

            [experiment]
            runs_per_task = 1
            seed = 1

            [mas]
            levels = 1
            intra_level_link_ratio = 1.0
            full_linked = true
            number_of_agents = 1
            agent_types = ["general"]
            communication_count_internally = 0
            turn_mode = "single_turn"
            max_turns = 1
            termination_consensus_mode = "semantic_magic"

            [models]
            default = "openai/gpt-4o-mini"
            """
        )

        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_invalid_final_vote_mode(self) -> None:
        path = self._write(
            """
            [openrouter]
            api_key = ""

            [experiment]
            runs_per_task = 1
            seed = 1

            [mas]
            levels = 1
            intra_level_link_ratio = 1.0
            full_linked = true
            number_of_agents = 1
            agent_types = ["general"]
            communication_count_internally = 0
            turn_mode = "single_turn"
            max_turns = 1
            final_vote_mode = "semantic_magic"

            [models]
            default = "openai/gpt-4o-mini"
            """
        )

        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_default_max_turns_is_20_when_omitted(self) -> None:
        path = self._write(
            """
            [openrouter]
            api_key = ""

            [experiment]
            runs_per_task = 1
            seed = 1

            [mas]
            levels = 1
            intra_level_link_ratio = 1.0
            full_linked = true
            number_of_agents = 1
            agent_types = ["general"]
            communication_count_internally = 0
            turn_mode = "single_turn"

            [models]
            default = "openai/gpt-4o-mini"
            """
        )

        cfg = load_experiment_config(path)
        self.assertEqual(cfg.mas.max_turns, 20)

    def test_single_turn_runtime_uses_one_round_even_with_default_max_turns(self) -> None:
        path = self._write(
            """
            [openrouter]
            api_key = ""

            [experiment]
            runs_per_task = 1
            seed = 1

            [mas]
            levels = 1
            intra_level_link_ratio = 1.0
            full_linked = true
            number_of_agents = 1
            agent_types = ["general"]
            communication_count_internally = 0
            turn_mode = "single_turn"

            [models]
            default = "openai/gpt-4o-mini"
            """
        )

        cfg = load_experiment_config(path)
        runner = MASRunner(cfg, _ArtifactLLM())
        run = runner.run_task(
            task=BenchmarkTask(task_id="t1", prompt="What is 2 + 2?", reference_answer="4"),
            run_index=0,
            seed=1,
        )

        self.assertEqual(cfg.mas.max_turns, 20)
        self.assertEqual(run.run_metadata["rounds_configured"], 1)
        self.assertEqual(run.run_metadata["turns_executed"], 1)


if __name__ == "__main__":
    unittest.main()
