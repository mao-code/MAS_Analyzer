import math
import json
import tempfile
import textwrap
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main as main_module
from benchmark.base import BenchmarkEvaluation, BenchmarkTask
from cli import commands as commands_module  # run_command's own module: patch targets live here


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
                analysis["descriptor"]["accuracy"],
                analysis["evaluation"]["accuracy"],
            )
            self.assertEqual(
                analysis["descriptor"]["pass_at_1"],
                analysis["descriptor"]["success_rate"],
            )
            self.assertTrue(math.isnan(analysis["descriptor"]["pass_at_3"]))
            self.assertTrue(math.isnan(analysis["descriptor"]["stability"]))
            self.assertTrue(math.isnan(analysis["descriptor"]["tokens_cv"]))
            self.assertIn("latency_e2e", analysis["descriptor"])
            self.assertEqual(
                analysis["descriptor"]["token_total"],
                analysis["descriptor"]["tokens_total"],
            )

            eval_payload = json.loads((task_dir / "run_0.eval.json").read_text(encoding="utf-8"))
            run_metadata = eval_payload["details"]["run_metadata"]
            self.assertNotIn("search", run_metadata.get("tool_call_counts", {}))
            answer_text = (task_dir / "run_0.answer.txt").read_text(encoding="utf-8").strip()
            self.assertTrue(answer_text)

            trace_metrics = json.loads(
                (task_dir / "run_0.trace_metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("accuracy", trace_metrics["metrics"])
            self.assertIn("latency_e2e", trace_metrics["metrics"])
            self.assertEqual(
                trace_metrics["metrics"]["token_total"],
                trace_metrics["metrics"]["tokens_total"],
            )

            settings_path = root / "experiment_settings.json"
            self.assertTrue(settings_path.exists())
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(settings["system"]["mode"], "SAS")
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "summary.csv").exists())

    def test_run_resumes_from_next_completed_task_in_hierarchical_layout(self) -> None:
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
                    number_of_agents = 4
                    agent_types = ["general"]
                    communication_count_internally = 50
                    turn_mode = "multi_turn"
                    max_turns = 2
                    discussion_rounds = 2
                    topology = "orchestrator_with_discussion"

                    [models]
                    default = "openai/gpt-4o-mini"
                    judge = "openai/gpt-4o-mini"

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

            experiment_id = "resume_case"
            system_label = "orchestrator_with_discussion"
            run_root = out_dir / experiment_id / "browsecomp" / system_label
            completed_task_dir = run_root / "775"
            completed_task_dir.mkdir(parents=True, exist_ok=True)
            (completed_task_dir / "analysis.json").write_text("{}", encoding="utf-8")
            (completed_task_dir / "descriptor.json").write_text("{}", encoding="utf-8")
            (completed_task_dir / "descriptor.csv").write_text("", encoding="utf-8")
            (completed_task_dir / "task_summary.json").write_text(
                json.dumps(
                    {
                        "task_id": "775",
                        "benchmark": "browsecomp",
                        "system": {
                            "system_label": system_label,
                            "topology": "orchestrator_with_discussion",
                            "agents": 4,
                            "max_turns": 2,
                            "discussion_rounds": 2,
                            "termination_consensus_mode": "llm_judge",
                            "final_vote_mode": "llm_judge",
                            "peer_artifact_max_chars": 0,
                            "communication_budget": 50,
                        },
                        "task_dir": str(completed_task_dir.resolve()),
                        "prompt_preview": "resume task 775",
                        "reference_answer": "A",
                        "evaluation": {
                            "accuracy": 1.0,
                            "avg_score": 1.0,
                            "completion_rate": 1.0,
                            "count": 1,
                            "success_rate": 1.0,
                        },
                        "descriptor": {
                            "accuracy": 1.0,
                            "completion_rate": 1.0,
                            "eval_avg_score": 1.0,
                            "latency_e2e": 0.0,
                            "success_rate": 1.0,
                            "token_total": 0.0,
                            "tokens_total": 0.0,
                        },
                        "stage_bottleneck": {},
                        "runs": [{"run_index": 0}],
                        "artifacts": {
                            "analysis_path": str((completed_task_dir / "analysis.json").resolve()),
                            "descriptor_json_path": str((completed_task_dir / "descriptor.json").resolve()),
                            "descriptor_csv_path": str((completed_task_dir / "descriptor.csv").resolve()),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            tasks = [
                BenchmarkTask(task_id="775", prompt="resume task 775", reference_answer="A"),
                BenchmarkTask(task_id="776", prompt="resume task 776", reference_answer="B"),
            ]
            run_calls: list[str] = []

            class FakeBenchmark:
                def load_tasks(self, task_limit: int | None = None):
                    return tasks[:task_limit] if task_limit is not None else tasks

                def run(self, task, runner, run_index, seed):
                    run_calls.append(task.task_id)
                    return SimpleNamespace(
                        final_answer=f"answer-{task.task_id}",
                        trace_events=[],
                        run_metadata={},
                    )

                def evaluate(self, task, prediction, *, run_metadata=None):
                    return BenchmarkEvaluation(
                        task_id=task.task_id,
                        score=1.0 if task.task_id == "776" else 0.0,
                        success=(task.task_id == "776"),
                        details={},
                    )

                def requirements(self):
                    return {}

            args = Namespace(
                config=str(cfg_path),
                benchmark="browsecomp",
                task_limit=2,
                runs_per_task=1,
                seed=42,
                output_dir=str(out_dir),
                output_layout="hierarchical",
                experiment_id=experiment_id,
                system_label=system_label,
                topology="orchestrator_with_discussion",
                agents=4,
                mas_rounds=2,
                discussion_rounds=2,
                communication_budget=50,
                termination_consensus_mode=None,
                final_vote_mode=None,
                default_model=None,
                judge_model=None,
                benchmark_eval_judge_model=None,
                peer_artifact_max_chars=None,
                agents_per_level=None,
                group_sizes=None,
                agent_types=None,
                no_dynamic_roles=False,
            )

            with patch.object(commands_module, "get_benchmark", return_value=FakeBenchmark()):
                with patch.object(commands_module, "OpenRouterLLMClient", return_value=object()):
                    with patch.object(commands_module, "MASRunner", return_value=SimpleNamespace()):
                        with patch.object(
                            commands_module,
                            "_write_system_graph_artifact",
                            return_value={"png_path": "graph.png"},
                        ):
                            exit_code = main_module.run_command(args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(run_calls, ["776"])

            summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual([task["task_id"] for task in summary["tasks"]], ["775", "776"])
            summary_csv = (run_root / "summary.csv").read_text(encoding="utf-8")
            self.assertIn("775", summary_csv)
            self.assertIn("776", summary_csv)
            self.assertTrue((run_root / "776" / "task_summary.json").exists())

    def test_run_failure_is_written_as_rerunnable_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg_path = base / "experiment.toml"
            out_dir = base / "outputs"
            cfg_path.write_text(
                textwrap.dedent(
                    f"""
                    [openrouter]
                    api_key = ""

                    [experiment]
                    output_dir = "{out_dir.as_posix()}"
                    runs_per_task = 2
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
                    judge = "openai/gpt-4o-mini"

                    [browsecomp]
                    eval_mode = "substring"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            task = BenchmarkTask(task_id="task_fail", prompt="prompt", reference_answer="answer")

            class FakeBenchmark:
                def load_tasks(self, task_limit: int | None = None):
                    return [task]

                def run(self, task, runner, run_index, seed):
                    if run_index == 0:
                        raise RuntimeError("provider 429")
                    return SimpleNamespace(
                        final_answer="answer",
                        trace_events=[],
                        run_metadata={
                            "task_id": task.task_id,
                            "run_index": run_index,
                            "seed": seed,
                            "topology": "sas",
                            "run_status": "completed",
                        },
                    )

                def evaluate(self, task, prediction, *, run_metadata=None):
                    return BenchmarkEvaluation(
                        task_id=task.task_id,
                        score=1.0 if prediction == "answer" else 0.0,
                        success=prediction == "answer",
                        details={},
                    )

                def requirements(self):
                    return {}

            args = Namespace(
                config=str(cfg_path),
                benchmark="browsecomp",
                task_limit=1,
                runs_per_task=2,
                seed=42,
                output_dir=str(out_dir),
                output_layout="hierarchical",
                experiment_id="failure_case",
                system_label="sas",
                topology="sas",
                agents=1,
                mas_rounds=1,
                discussion_rounds=1,
                communication_budget=0,
                termination_consensus_mode=None,
                final_vote_mode=None,
                default_model=None,
                judge_model=None,
                benchmark_eval_judge_model=None,
                peer_artifact_max_chars=None,
                agents_per_level=None,
                group_sizes=None,
                agent_types=None,
                no_dynamic_roles=False,
            )

            with patch.object(commands_module, "get_benchmark", return_value=FakeBenchmark()):
                with patch.object(commands_module, "OpenRouterLLMClient", return_value=object()):
                    with patch.object(commands_module, "MASRunner", return_value=SimpleNamespace()):
                        with patch.object(
                            commands_module,
                            "_write_system_graph_artifact",
                            return_value={"png_path": "graph.png"},
                        ):
                            exit_code = main_module.run_command(args)

            self.assertEqual(exit_code, 0)
            task_dir = out_dir / "failure_case" / "browsecomp" / "sas" / "task_fail"
            metadata = json.loads((task_dir / "run_0.metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["run_status"], "failed")
            self.assertTrue(metadata["fallback"])
            task_summary = json.loads((task_dir / "task_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(task_summary["needs_rerun"])
            self.assertEqual(task_summary["run_failure_count"], 1)
            self.assertEqual(task_summary["fallback_count"], 1)

            rerun_calls: list[int] = []

            class SecondPassBenchmark(FakeBenchmark):
                def run(self, task, runner, run_index, seed):
                    rerun_calls.append(run_index)
                    return SimpleNamespace(
                        final_answer="answer",
                        trace_events=[],
                        run_metadata={
                            "task_id": task.task_id,
                            "run_index": run_index,
                            "seed": seed,
                            "topology": "sas",
                            "run_status": "completed",
                        },
                    )

            with patch.object(commands_module, "get_benchmark", return_value=SecondPassBenchmark()):
                with patch.object(commands_module, "OpenRouterLLMClient", return_value=object()):
                    with patch.object(commands_module, "MASRunner", return_value=SimpleNamespace()):
                        with patch.object(
                            commands_module,
                            "_write_system_graph_artifact",
                            return_value={"png_path": "graph.png"},
                        ):
                            exit_code = main_module.run_command(args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(rerun_calls, [0])
            task_summary = json.loads((task_dir / "task_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(task_summary["needs_rerun"])
            self.assertEqual(task_summary["run_failure_count"], 0)
            self.assertEqual(task_summary["fallback_count"], 0)


if __name__ == "__main__":
    unittest.main()
