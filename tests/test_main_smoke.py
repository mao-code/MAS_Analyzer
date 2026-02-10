import csv
import tempfile
import textwrap
import unittest
from pathlib import Path

import main as main_module


class TestMainSmoke(unittest.TestCase):
    def test_run_finance_agent_one_task_one_run(self) -> None:
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
                    number_of_agents = 1
                    agent_types = ["general"]
                    communication_count_internally = 0
                    turn_mode = "single_turn"
                    max_turns = 1

                    [models]
                    default = "openai/gpt-4o-mini"

                    [finance_agent]
                    local_csv_path = "{csv_path.as_posix()}"
                    success_threshold = 0.5
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
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
