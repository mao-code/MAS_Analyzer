import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import main as main_module


class TestMainHierarchicalOutput(unittest.TestCase):
    def test_hierarchical_layout_writes_graph_and_trajectory(self) -> None:
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

            def fake_graph_artifact(*, runner, config, run_root):
                graph_path = run_root / "mas_graph.png"
                mermaid_path = run_root / "mas_graph.mmd"
                metadata_path = run_root / "mas_graph.json"
                graph_path.write_bytes(b"fake-png")
                mermaid_path.write_text("graph TD;\n", encoding="utf-8")
                payload = {
                    "topology": config.mas.resolved_topology(),
                    "render_backend": "test",
                    "render_error": "",
                    "png_path": str(graph_path.resolve()),
                    "mermaid_path": str(mermaid_path.resolve()),
                    "layout": {},
                }
                metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                return payload

            with patch.object(
                main_module,
                "_write_system_graph_artifact",
                side_effect=fake_graph_artifact,
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
                        "--output-layout",
                        "hierarchical",
                        "--experiment-id",
                        "exp123",
                        "--system-label",
                        "sas",
                        "--topology",
                        "sas",
                        "--agents",
                        "1",
                        "--mas-rounds",
                        "1",
                        "--communication-budget",
                        "0",
                    ]
                )
            self.assertEqual(exit_code, 0)

            experiment_root = out_dir / "exp123"
            system_dir = experiment_root / "browsecomp" / "sas"
            task_dir = system_dir / "q1"

            self.assertTrue((system_dir / "mas_graph.png").exists())
            self.assertTrue((system_dir / "experiment_settings.json").exists())
            self.assertTrue((system_dir / "summary.json").exists())
            self.assertTrue((system_dir / "summary.csv").exists())

            self.assertTrue((task_dir / "task.json").exists())
            self.assertTrue((task_dir / "run_0.answer.txt").exists())
            self.assertTrue((task_dir / "run_0.metadata.json").exists())
            self.assertTrue((task_dir / "run_0.result.json").exists())
            self.assertTrue((task_dir / "run_0.eval.json").exists())
            self.assertTrue((task_dir / "run_0.trajectory.json").exists())
            self.assertTrue((task_dir / "run_0.trajectory.md").exists())
            self.assertTrue((task_dir / "task_summary.json").exists())

            trajectory = json.loads((task_dir / "run_0.trajectory.json").read_text(encoding="utf-8"))
            self.assertTrue(trajectory["tool_definitions"])
            self.assertTrue(trajectory["steps"])
            first_step = trajectory["steps"][0]
            prompt_roles = {item.get("role") for item in first_step["prompt_messages"]}
            self.assertIn("user", prompt_roles)
            self.assertEqual(first_step["assistant_message"]["role"], "assistant")

            eval_payload = json.loads((task_dir / "run_0.eval.json").read_text(encoding="utf-8"))
            self.assertIn("run_metadata_path", eval_payload["details"])
            self.assertIn("tool_call_counts", eval_payload["details"]["run_metadata"])

            summarize_exit_code = main_module.main(
                [
                    "summarize-experiment",
                    "--experiment-root",
                    str(experiment_root),
                ]
            )
            self.assertEqual(summarize_exit_code, 0)
            self.assertTrue((experiment_root / "experiment_summary.json").exists())
            self.assertTrue((experiment_root / "experiment_summary.csv").exists())
            self.assertTrue((experiment_root / "browsecomp" / "benchmark_summary.json").exists())
            self.assertTrue((experiment_root / "browsecomp" / "benchmark_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
