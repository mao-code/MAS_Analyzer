import math
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import main as main_module


class TestMainBrowseCompSmoke(unittest.TestCase):
    def test_run_browsecomp_one_task_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            decrypted_path = base / "browsecomp_decrypted.jsonl"
            qrel_evidence = base / "qrel_evidence.txt"
            qrel_golds = base / "qrel_golds.txt"
            cfg_path = base / "experiment.toml"
            out_dir = base / "outputs"

            row = {
                "query_id": "q1",
                "query": "Which city is called the Eternal City?",
                "answer": "Rome",
                "gold_docs": [{"docid": "100", "text": "Rome is known as Eternal City.", "url": "u"}],
                "evidence_docs": [{"docid": "200", "text": "Evidence text", "url": "u"}],
                "negative_docs": [],
            }
            decrypted_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            qrel_evidence.write_text("q1 Q0 200 1\n", encoding="utf-8")
            qrel_golds.write_text("q1 Q0 100 1\n", encoding="utf-8")

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

                    [browsecomp]
                    decrypted_path = "{decrypted_path.as_posix()}"
                    qrel_evidence_path = "{qrel_evidence.as_posix()}"
                    qrel_golds_path = "{qrel_golds.as_posix()}"
                    eval_mode = "substring"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "MAS_DISABLE_LIVE_LLM": "1",
                    "MAS_REQUIRE_LIVE_LLM": "0",
                    "OPENROUTER_API_KEY": "",
                },
                clear=False,
            ):
                exit_code = main_module.main(
                    [
                        "run",
                        "--config",
                        str(cfg_path),
                        "--benchmark",
                        "browsecomp",
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

            task_dir = root / "browsecomp" / "q1"
            self.assertTrue((task_dir / "run_0.trace.jsonl").exists())
            self.assertTrue((task_dir / "run_0.eval.json").exists())
            self.assertTrue((task_dir / "descriptor.json").exists())
            self.assertTrue((task_dir / "descriptor.csv").exists())
            self.assertTrue((task_dir / "analysis.json").exists())
            self.assertTrue((task_dir / "run_0.answer.txt").exists())

            analysis = json.loads((task_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis["descriptor"]["C4_tool_calls_total"], 0.0)
            self.assertEqual(
                analysis["descriptor"]["success_rate"],
                analysis["evaluation"]["success_rate"],
            )
            self.assertEqual(
                analysis["descriptor"]["eval_avg_score"],
                analysis["evaluation"]["avg_score"],
            )
            self.assertEqual(
                analysis["descriptor"]["pass_at_1"],
                analysis["descriptor"]["success_rate"],
            )
            self.assertTrue(math.isnan(analysis["descriptor"]["pass_at_3"]))
            self.assertTrue(math.isnan(analysis["descriptor"]["stability"]))
            self.assertTrue(math.isnan(analysis["descriptor"]["tokens_cv"]))

            eval_payload = json.loads((task_dir / "run_0.eval.json").read_text(encoding="utf-8"))
            run_metadata = eval_payload["details"]["run_metadata"]
            self.assertNotIn("search", run_metadata.get("tool_call_counts", {}))
            answer_text = (task_dir / "run_0.answer.txt").read_text(encoding="utf-8").strip()
            self.assertTrue(answer_text)

            settings_path = root / "experiment_settings.json"
            self.assertTrue(settings_path.exists())
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(settings["system"]["mode"], "SAS")
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
