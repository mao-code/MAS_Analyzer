import unittest
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


if __name__ == "__main__":
    unittest.main()
