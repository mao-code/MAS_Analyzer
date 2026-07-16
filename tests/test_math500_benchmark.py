import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import main as main_module
from benchmark.base import BenchmarkTask
from benchmark.math500 import Math500Benchmark, extract_answer, is_equiv

SAMPLE_ROWS = [
    {
        "problem": "What is $1 + 1$?",
        "solution": "Adding gives $\\boxed{2}$.",
        "answer": "2",
        "subject": "Prealgebra",
        "level": 1,
        "unique_id": "test/prealgebra/0001.json",
    },
    {
        "problem": "Simplify $\\frac{2}{4}$.",
        "solution": "$\\boxed{\\frac{1}{2}}$",
        "answer": "\\frac{1}{2}",
        "subject": "Algebra",
        "level": 2,
        "unique_id": "test/algebra/0002.json",
    },
]


def _write_dataset(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in SAMPLE_ROWS:
            handle.write(json.dumps(row) + "\n")


class TestMath500AnswerMatching(unittest.TestCase):
    def test_extract_boxed(self) -> None:
        answer, match_type = extract_answer("Step 1... so the result is $\\boxed{42}$.")
        self.assertEqual(answer, "42")
        self.assertEqual(match_type, "boxed")

    def test_extract_last_boxed(self) -> None:
        answer, _ = extract_answer("First \\boxed{1}, but actually \\boxed{\\frac{1}{2}}")
        self.assertEqual(answer, "\\frac{1}{2}")

    def test_extract_nested_braces(self) -> None:
        answer, _ = extract_answer("\\boxed{\\sqrt{2}}")
        self.assertEqual(answer, "\\sqrt{2}")

    def test_extract_phrase_fallback(self) -> None:
        answer, match_type = extract_answer("Reasoning...\nThe final answer is 7.")
        self.assertEqual(answer, "7")
        self.assertEqual(match_type, "phrase")

    def test_extract_empty(self) -> None:
        answer, match_type = extract_answer("")
        self.assertEqual(answer, "")
        self.assertEqual(match_type, "empty")

    def test_is_equiv_exact(self) -> None:
        self.assertTrue(is_equiv("42", "42"))
        self.assertFalse(is_equiv("41", "42"))

    def test_is_equiv_fraction_forms(self) -> None:
        self.assertTrue(is_equiv("1/2", "\\frac{1}{2}"))
        self.assertTrue(is_equiv("0.5", "\\frac{1}{2}"))
        self.assertTrue(is_equiv("\\frac12", "\\frac{1}{2}"))
        self.assertTrue(is_equiv("\\dfrac{1}{2}", "\\frac{1}{2}"))

    def test_is_equiv_latex_noise(self) -> None:
        self.assertTrue(is_equiv("\\left(3, \\frac{\\pi}{2}\\right)", "(3,\\frac{\\pi}{2})"))
        self.assertTrue(is_equiv("90^\\circ", "90"))
        self.assertTrue(is_equiv("\\$5", "5"))
        self.assertTrue(is_equiv("x = 4", "4"))

    def test_is_equiv_text_wrapper(self) -> None:
        # MATH-500 wraps word/categorical answers in \text{...}.
        self.assertTrue(is_equiv("Evelyn", "\\text{Evelyn}"))
        self.assertTrue(is_equiv("\\text{yes}", "yes"))
        self.assertTrue(is_equiv("\\textbf{no}", "no"))

    def test_is_equiv_units_still_stripped(self) -> None:
        # The unit form \text{ ...} must still be removed (not turned into content).
        self.assertTrue(is_equiv("50\\text{ square feet}", "50"))

    def test_is_equiv_numeric_tolerance(self) -> None:
        self.assertTrue(is_equiv("1,000", "1000"))
        self.assertFalse(is_equiv("1000.5", "1000"))


class TestMath500Benchmark(unittest.TestCase):
    def test_load_tasks_local_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "math500.jsonl"
            _write_dataset(dataset)
            bench = Math500Benchmark({"dataset_path": str(dataset)})
            tasks = bench.load_tasks()

            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[0].task_id, "test_prealgebra_0001")
            self.assertIn("What is $1 + 1$?", tasks[0].prompt)
            self.assertIn("\\boxed", tasks[0].prompt)
            self.assertEqual(tasks[0].reference_answer, "2")
            self.assertEqual(tasks[0].metadata["subject"], "Prealgebra")

            limited = bench.load_tasks(task_limit=1)
            self.assertEqual(len(limited), 1)

    def test_run_delegates_to_runner(self) -> None:
        bench = Math500Benchmark()
        task = BenchmarkTask(task_id="t1", prompt="p", reference_answer="2", metadata={})
        runner = MagicMock()
        result = bench.run(task, runner, run_index=0, seed=42)
        runner.run_task.assert_called_once_with(
            task=task, run_index=0, seed=42, benchmark_name="math500"
        )
        self.assertIs(result, runner.run_task.return_value)

    def test_evaluate_success(self) -> None:
        bench = Math500Benchmark()
        task = BenchmarkTask(
            task_id="t1",
            prompt="p",
            reference_answer="\\frac{1}{2}",
            metadata={"subject": "Algebra", "level": 2},
        )
        result = bench.evaluate(task, "So the answer is $\\boxed{1/2}$.")
        self.assertTrue(result.success)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.details["match_type"], "boxed")

    def test_evaluate_failure(self) -> None:
        bench = Math500Benchmark()
        task = BenchmarkTask(task_id="t1", prompt="p", reference_answer="2", metadata={})
        result = bench.evaluate(task, "\\boxed{3}")
        self.assertFalse(result.success)
        self.assertEqual(result.score, 0.0)

    def test_evaluate_empty_prediction(self) -> None:
        bench = Math500Benchmark()
        task = BenchmarkTask(task_id="t1", prompt="p", reference_answer="2", metadata={})
        result = bench.evaluate(task, "")
        self.assertFalse(result.success)

    def test_registry(self) -> None:
        from benchmark.registry import get_benchmark, list_benchmarks

        self.assertIn("math500", list_benchmarks())
        bench = get_benchmark("math500", config={"split": "test"})
        self.assertIsInstance(bench, Math500Benchmark)


class TestMath500MainSmoke(unittest.TestCase):
    """End-to-end offline (mock LLM) runs through main.py."""

    def setUp(self) -> None:
        # Blank the env API key so the run always uses the offline mock client.
        patcher = patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run_and_check(self, mas_section: str, self_evolved_section: str = "") -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset = base / "math500.jsonl"
            _write_dataset(dataset)
            cfg_path = base / "experiment.toml"
            out_dir = base / "outputs"

            cfg_path.write_text(
                textwrap.dedent(
                    f"""
                    [openrouter]
                    api_key = ""

                    [experiment]
                    output_dir = "{out_dir.as_posix()}"
                    runs_per_task = 1
                    seed = 42

                    {mas_section}

                    {self_evolved_section}

                    [models]
                    default = "openai/gpt-4o-mini"

                    [math500]
                    dataset_path = "{dataset.as_posix()}"
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
                    "math500",
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

            task_dir = root / "math500" / "test_prealgebra_0001"
            self.assertTrue((task_dir / "run_0.trace.jsonl").exists())
            self.assertTrue((task_dir / "run_0.eval.json").exists())
            self.assertTrue((task_dir / "descriptor.json").exists())
            self.assertTrue((task_dir / "analysis.json").exists())
            self.assertTrue((root / "summary.csv").exists())

            eval_payload = json.loads((task_dir / "run_0.eval.json").read_text(encoding="utf-8"))
            self.assertEqual(eval_payload["task_id"], "test_prealgebra_0001")
            self.assertIn("extracted_answer", eval_payload["details"])

            raw_payload = json.loads((task_dir / "run_0.raw.json").read_text(encoding="utf-8"))
            return dict(raw_payload.get("run_metadata", {}))

    def test_run_math500_static_mas(self) -> None:
        run_metadata = self._run_and_check(
            textwrap.dedent(
                """
                [mas]
                levels = 1
                intra_level_link_ratio = 1.0
                full_linked = true
                number_of_agents = 3
                agent_types = ["general"]
                communication_count_internally = 2
                turn_mode = "single_turn"
                max_turns = 1
                termination_consensus_mode = "lexical"
                final_vote_mode = "deterministic"
                """
            ).strip()
        )
        self.assertNotEqual(run_metadata.get("topology"), "self_evolved")

    def test_run_math500_self_evolved(self) -> None:
        with tempfile.TemporaryDirectory() as state_dir:
            skill_path = Path(state_dir) / "topology_skill.md"
            playbook_path = Path(state_dir) / "topology_playbook.json"
            run_metadata = self._run_and_check(
                textwrap.dedent(
                    """
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
                    """
                ).strip(),
                textwrap.dedent(
                    f"""
                    [self_evolved]
                    harness_backend = "openrouter"
                    max_initial_agents = 3
                    max_total_agents = 6
                    max_turns = 2
                    skill_update_batch_size = 0
                    skill_path = "{skill_path.as_posix()}"
                    playbook_path = "{playbook_path.as_posix()}"
                    """
                ).strip(),
            )
            self.assertEqual(run_metadata.get("topology"), "self_evolved")
            self.assertIn("self_evolved", run_metadata)


if __name__ == "__main__":
    unittest.main()
