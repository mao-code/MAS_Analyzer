import csv
import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import main as main_module


class TestSelfEvolvedMainSmoke(unittest.TestCase):
    def setUp(self) -> None:
        # load_experiment_config prefers OPENROUTER_API_KEY from the environment
        # (auto-loaded from .env) over the TOML's empty api_key; blank it so the
        # smoke test always runs in offline mock mode.
        patcher = patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_run_finance_agent_with_self_evolved_system(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            csv_path = base / "public.csv"
            cfg_path = base / "experiment.toml"
            out_dir = base / "outputs"

            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Question",
                        "Answer",
                        "Question Type",
                        "Expert time (mins)",
                        "Rubric",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Question": "What is X?",
                        "Answer": "X is 42",
                        "Question Type": "Quantitative Retrieval",
                        "Expert time (mins)": "2",
                        "Rubric": (
                            '[{"operator": "correctness", "criteria": "42"}, '
                            '{"operator": "contradiction", "criteria": "13"}]'
                        ),
                    }
                )

            cfg_path.write_text(
                textwrap.dedent(
                    f"""
                    [openrouter]
                    api_key = ""

                    [experiment]
                    output_dir = "{out_dir.as_posix()}"
                    runs_per_task = 1
                    seed = 42

                    [mas]
                    levels = 1
                    intra_level_link_ratio = 1.0
                    full_linked = true
                    topology = "self_evolved"
                    number_of_agents = 3
                    agent_types = ["general"]
                    communication_count_internally = 2
                    turn_mode = "single_turn"
                    max_turns = 1
                    termination_consensus_mode = "lexical"
                    final_vote_mode = "deterministic"

                    [self_evolved]
                    harness_backend = "openrouter"
                    max_initial_agents = 3
                    max_total_agents = 6
                    max_turns = 2

                    [models]
                    default = "openai/gpt-4o-mini"

                    [finance_agent]
                    local_csv_path = "{csv_path.as_posix()}"
                    success_threshold = 0.5
                    eval_mode = "substring"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            exit_code = main_module.main(
                [
                    "run",
                    "--config",
                    str(cfg_path),
                    "--benchmark",
                    "finance_agent",
                    "--task-limit",
                    "1",
                    "--runs-per-task",
                    "1",
                ]
            )
            self.assertEqual(exit_code, 0)

            run_dirs = [item for item in out_dir.iterdir() if item.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            root = run_dirs[0]

            task_dir = root / "finance_agent" / "0"
            self.assertTrue((task_dir / "run_0.trace.jsonl").exists())
            self.assertTrue((task_dir / "run_0.eval.json").exists())
            self.assertTrue((task_dir / "descriptor.json").exists())
            self.assertTrue((task_dir / "descriptor.csv").exists())
            self.assertTrue((task_dir / "analysis.json").exists())
            self.assertTrue((root / "summary.csv").exists())

            raw_payload = json.loads((task_dir / "run_0.raw.json").read_text(encoding="utf-8"))
            run_metadata = raw_payload.get("run_metadata", {})
            self.assertEqual(run_metadata.get("topology"), "self_evolved")
            self.assertIn("self_evolved", run_metadata)

    def test_hierarchical_layout_writes_dynamic_graph_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            csv_path = base / "public.csv"
            cfg_path = base / "experiment.toml"
            out_dir = base / "outputs"

            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Question",
                        "Answer",
                        "Question Type",
                        "Expert time (mins)",
                        "Rubric",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Question": "What is Y?",
                        "Answer": "Y is 7",
                        "Question Type": "Quantitative Retrieval",
                        "Expert time (mins)": "1",
                        "Rubric": '[{"operator": "correctness", "criteria": "7"}]',
                    }
                )

            cfg_path.write_text(
                textwrap.dedent(
                    f"""
                    [openrouter]
                    api_key = ""

                    [experiment]
                    output_dir = "{out_dir.as_posix()}"
                    runs_per_task = 1
                    seed = 42

                    [mas]
                    levels = 1
                    intra_level_link_ratio = 1.0
                    full_linked = true
                    topology = "self_evolved"
                    number_of_agents = 2
                    agent_types = ["general"]
                    communication_count_internally = 1
                    turn_mode = "single_turn"
                    max_turns = 1
                    termination_consensus_mode = "lexical"
                    final_vote_mode = "deterministic"

                    [models]
                    default = "openai/gpt-4o-mini"

                    [finance_agent]
                    local_csv_path = "{csv_path.as_posix()}"
                    success_threshold = 0.5
                    eval_mode = "substring"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            exit_code = main_module.main(
                [
                    "run",
                    "--config",
                    str(cfg_path),
                    "--benchmark",
                    "finance_agent",
                    "--task-limit",
                    "1",
                    "--runs-per-task",
                    "1",
                    "--output-layout",
                    "hierarchical",
                    "--experiment-id",
                    "se_smoke",
                ]
            )
            self.assertEqual(exit_code, 0)

            run_root = out_dir / "se_smoke" / "finance_agent" / "self_evolved"
            self.assertTrue(run_root.is_dir())
            graph_payload = json.loads((run_root / "mas_graph.json").read_text(encoding="utf-8"))
            self.assertTrue(graph_payload["dynamic"])
            self.assertEqual(graph_payload["topology"], "self_evolved")


if __name__ == "__main__":
    unittest.main()
