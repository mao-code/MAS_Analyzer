import csv
import tempfile
import unittest
from pathlib import Path

from benchmark.finance_agent import FinanceAgentBenchmark


class TestFinanceBenchmark(unittest.TestCase):
    def test_csv_load_and_rubric_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "public.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
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

            benchmark = FinanceAgentBenchmark(
                {
                    "local_csv_path": str(path),
                    "success_threshold": 0.5,
                }
            )
            tasks = benchmark.load_tasks()
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].prompt, "What is X?")

            good_eval = benchmark.evaluate(tasks[0], "The answer is 42")
            self.assertGreaterEqual(good_eval.score, 0.5)
            self.assertTrue(good_eval.success)

            bad_eval = benchmark.evaluate(tasks[0], "The answer is 13")
            self.assertLess(bad_eval.score, 0.5)
            self.assertFalse(bad_eval.success)


if __name__ == "__main__":
    unittest.main()
