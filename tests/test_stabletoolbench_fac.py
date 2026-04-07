import unittest
from unittest.mock import patch

from benchmark.base import BenchmarkTask
from benchmark.stabletoolbench import StableToolBenchBenchmark


class TestStableToolBenchFAC(unittest.TestCase):
    def test_judge_uses_openrouter_env_fallback(self) -> None:
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "", "OPENROUTER_API_KEY": "or-test-key"},
            clear=False,
        ):
            benchmark = StableToolBenchBenchmark({"eval_mode": "fac"})
        self.assertEqual(benchmark.judge_api_key, "or-test-key")
        self.assertEqual(benchmark.judge_api_base, "https://openrouter.ai/api/v1")

    def test_parse_fac_payload_from_text(self) -> None:
        payload = StableToolBenchBenchmark._parse_fac_payload(
            "Answer Status\nSolved\nReason\nThe answer addresses all requested parts."
        )
        self.assertEqual(payload["answer_status"], "Solved")
        self.assertIn("addresses", payload["reason"])

    def test_parse_fac_payload_from_json(self) -> None:
        payload = StableToolBenchBenchmark._parse_fac_payload(
            '{"answer_status":"Unsolved","reason":"Missing required part"}'
        )
        self.assertEqual(payload["answer_status"], "Unsolved")
        self.assertEqual(payload["reason"], "Missing required part")

    def test_evaluate_fac_mode_uses_fac_judge(self) -> None:
        benchmark = StableToolBenchBenchmark({"eval_mode": "fac"})
        task = BenchmarkTask(
            task_id="q1",
            prompt="",
            reference_answer="",
            metadata={"query": "test query", "task_set": "G1_instruction", "query_id": "q1"},
        )
        with patch.object(
            StableToolBenchBenchmark,
            "_judge_answer_fac",
            return_value={"answer_status": "Solved", "reason": "good"},
        ):
            evaluation = benchmark.evaluate(task, "some final answer", run_metadata={})
        self.assertEqual(evaluation.score, 1.0)
        self.assertTrue(evaluation.success)
        self.assertEqual(evaluation.details["eval_mode"], "fac")
        self.assertEqual(evaluation.details["answer_status"], "Solved")


if __name__ == "__main__":
    unittest.main()
