import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.scicode import (
    BACKGROUND_COMMENT_TEMPLATE,
    MULTISTEP_TEMPLATE,
    SciCodeBenchmark,
    _extract_python_script,
)


class TestSciCodeBenchmark(unittest.TestCase):
    @patch.object(SciCodeBenchmark, "_ensure_data_exists", autospec=True)
    def test_template_mapping_without_background(self, _mock_ensure: object) -> None:
        bench = SciCodeBenchmark({"with_background": False})
        self.assertEqual(bench.template, BACKGROUND_COMMENT_TEMPLATE)

    @patch.object(SciCodeBenchmark, "_ensure_data_exists", autospec=True)
    def test_template_mapping_with_background(self, _mock_ensure: object) -> None:
        bench = SciCodeBenchmark({"with_background": True})
        self.assertEqual(bench.template, MULTISTEP_TEMPLATE)

    def test_extract_python_script_removes_imports(self) -> None:
        response = """```python
import os
from math import sin
def solve():
    return 42
```"""
        code = _extract_python_script(response)
        self.assertIn("def solve():", code)
        self.assertNotIn("import os", code)
        self.assertNotIn("from math import sin", code)

    @patch.object(SciCodeBenchmark, "_ensure_data_exists", autospec=True)
    def test_llm_repair_is_enabled_by_default(self, _mock_ensure: object) -> None:
        bench = SciCodeBenchmark({})
        self.assertTrue(bench.llm_repair)

    @patch.object(SciCodeBenchmark, "_ensure_data_exists", autospec=True)
    def test_llm_repair_can_be_disabled_explicitly(self, _mock_ensure: object) -> None:
        bench = SciCodeBenchmark({"llm_repair": False})
        runner = SimpleNamespace(llm_client=SimpleNamespace(client=None))
        raw = "```python # Background: x def solve(): return 1```"
        self.assertEqual(
            bench._repair_response_if_needed(
                raw_response=raw,
                runner=runner,
                function_header="def solve():",
                step_task_id="1.1",
            ),
            raw,
        )

    def test_build_repair_prompt_mentions_format_only(self) -> None:
        prompt = SciCodeBenchmark._build_repair_prompt(
            raw_response="```python # Background: ... def solve(): pass```",
            function_header="def solve():",
        )
        self.assertEqual(prompt[0]["role"], "system")
        self.assertIn("formatting only", prompt[0]["content"])
        self.assertIn("Target function header", prompt[1]["content"])


if __name__ == "__main__":
    unittest.main()
