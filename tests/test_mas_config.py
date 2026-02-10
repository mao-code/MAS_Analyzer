import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from MAS.config import load_experiment_config


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


if __name__ == "__main__":
    unittest.main()
