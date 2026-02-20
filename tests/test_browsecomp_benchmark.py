import json
import tempfile
import unittest
from pathlib import Path

from benchmark.browsecomp import (
    BrowseCompBenchmark,
    compute_citation_metrics,
    extract_citations_from_response,
    parse_judge_response,
)


class TestBrowseCompBenchmark(unittest.TestCase):
    def test_load_and_evaluate_substring(self) -> None:
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
                    "eval_mode": "substring",
                }
            )
            tasks = bench.load_tasks()
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertEqual(task.task_id, "q1")

            # Good prediction
            eval_ok = bench.evaluate(
                task,
                "Final answer: 42",
                run_metadata={"retrieved_docids": ["100", "200"]},
            )
            self.assertTrue(eval_ok.success)
            self.assertEqual(eval_ok.details["retrieval"]["recall_evidence"], 1.0)
            self.assertEqual(eval_ok.details["retrieval"]["recall_gold"], 1.0)

            # Bad prediction
            eval_bad = bench.evaluate(
                task,
                "Final answer: 0",
                run_metadata={"retrieved_docids": []},
            )
            self.assertFalse(eval_bad.success)
            self.assertEqual(eval_bad.score, 0.0)


class TestParseJudgeResponse(unittest.TestCase):
    def test_parse_standard_format(self) -> None:
        response = (
            "extracted_final_answer: Paris\n\n"
            "reasoning: The answer matches the correct answer.\n\n"
            "correct: yes\n\n"
            "confidence: 95%"
        )
        result = parse_judge_response(response)
        self.assertEqual(result["extracted_final_answer"], "Paris")
        self.assertTrue(result["correct"])
        self.assertEqual(result["confidence"], 95.0)
        self.assertFalse(result["parse_error"])

    def test_parse_bold_format(self) -> None:
        response = (
            "**extracted_final_answer:** Tokyo\n"
            "**reasoning:** Matches.\n"
            "**correct:** no\n"
            "**confidence:** 30"
        )
        result = parse_judge_response(response)
        self.assertEqual(result["extracted_final_answer"], "Tokyo")
        self.assertFalse(result["correct"])
        self.assertEqual(result["confidence"], 30.0)

    def test_parse_empty(self) -> None:
        result = parse_judge_response("")
        self.assertTrue(result["parse_error"])

    def test_parse_missing_correct(self) -> None:
        response = "extracted_final_answer: something\nreasoning: blah"
        result = parse_judge_response(response)
        self.assertTrue(result["parse_error"])
        self.assertIsNone(result["correct"])


class TestCitations(unittest.TestCase):
    def test_extract_half_width(self) -> None:
        text = "The answer is X [123] and Y [456, 789]."
        docids = extract_citations_from_response(text)
        self.assertIn("123", docids)
        self.assertIn("456", docids)
        self.assertIn("789", docids)

    def test_extract_fullwidth(self) -> None:
        text = "回答是 X【100】。"
        docids = extract_citations_from_response(text)
        self.assertIn("100", docids)

    def test_empty(self) -> None:
        self.assertEqual(extract_citations_from_response(""), [])

    def test_citation_metrics(self) -> None:
        metrics = compute_citation_metrics(["1", "2", "3"], ["2", "3", "4"])
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)

    def test_citation_metrics_empty(self) -> None:
        metrics = compute_citation_metrics([], ["1", "2"])
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
