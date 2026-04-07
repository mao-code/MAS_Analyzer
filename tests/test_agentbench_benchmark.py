import unittest

from benchmark.agentbench import AgentBenchBenchmark


class TestAgentBenchFormatting(unittest.TestCase):
    def setUp(self) -> None:
        self.bench = AgentBenchBenchmark({"task_name": "os"})
        self.history = [
            {
                "role": "user",
                "content": (
                    'For each turn use Think/Act. Actions: "bash", "finish", "answer".\n'
                    "Think: ...\nAct: bash\n```bash\n...\n```\n"
                    "Act: answer(...)\n"
                ),
            }
        ]

    def test_wraps_free_form_into_answer_action(self) -> None:
        normalized = self.bench._normalize_agent_content(
            raw_content="The answer is 42.",
            history=self.history,
        )
        self.assertIn("Think:", normalized)
        self.assertIn("Act: answer(", normalized)
        self.assertIn("The answer is 42.", normalized)

    def test_canonicalizes_inline_bash(self) -> None:
        normalized = self.bench._normalize_agent_content(
            raw_content="Think: check file. Act: bash ```bash cat /tmp/x ```",
            history=self.history,
        )
        self.assertIn("Act: bash", normalized)
        self.assertIn("```bash", normalized)
        self.assertIn("cat /tmp/x", normalized)

    def test_keeps_non_think_act_tasks_unchanged(self) -> None:
        raw = "plain model output"
        normalized = self.bench._normalize_agent_content(
            raw_content=raw,
            history=[{"role": "user", "content": "some other task format"}],
        )
        self.assertEqual(normalized, raw)


if __name__ == "__main__":
    unittest.main()
