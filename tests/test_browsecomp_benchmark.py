import json
import tempfile
import unittest
from pathlib import Path

from benchmark.browsecomp import BrowseCompBenchmark


class TestBrowseCompBenchmark(unittest.TestCase):
    def test_load_and_recall_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            decrypted = base / "decrypted.jsonl"
            qrel_evidence = base / "qrel_evidence.txt"
            qrel_golds = base / "qrel_golds.txt"

            row = {
                "query_id": "q1",
                "query": "Which doc contains answer?",
                "answer": "42",
                "gold_docs": [{"docid": "100", "text": "gold", "url": "u"}],
                "evidence_docs": [{"docid": "200", "text": "ev", "url": "u"}],
            }
            decrypted.write_text(json.dumps(row) + "\n", encoding="utf-8")
            qrel_evidence.write_text("q1 Q0 200 1\n", encoding="utf-8")
            qrel_golds.write_text("q1 Q0 100 1\n", encoding="utf-8")

            bench = BrowseCompBenchmark(
                {
                    "decrypted_path": str(decrypted),
                    "qrel_evidence_path": str(qrel_evidence),
                    "qrel_golds_path": str(qrel_golds),
                }
            )
            tasks = bench.load_tasks()
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertEqual(task.task_id, "q1")

            eval_ok = bench.evaluate(
                task,
                "Final answer: 42",
                run_metadata={"retrieved_docids": ["100", "200"]},
            )
            self.assertTrue(eval_ok.success)
            self.assertEqual(eval_ok.details["recall_evidence"], 1.0)
            self.assertEqual(eval_ok.details["recall_gold"], 1.0)

            eval_bad = bench.evaluate(
                task,
                "Final answer: 0",
                run_metadata={"retrieved_docids": []},
            )
            self.assertFalse(eval_bad.success)
            self.assertEqual(eval_bad.score, 0.0)


if __name__ == "__main__":
    unittest.main()
