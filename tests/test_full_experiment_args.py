import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import scripts.full_experiment as batch_module


BENCHMARK_NAMES = [
    "agentbench",
    "browsecomp",
    "finance_agent",
    "plancraft",
    "scicode",
    "stabletoolbench",
    "webshop",
    "workbench",
]


class TestFullExperimentArgs(unittest.TestCase):
    def _write_configs(self, config_dir: Path) -> None:
        for benchmark_name in BENCHMARK_NAMES:
            (config_dir / f"{benchmark_name}_10.toml").write_text("", encoding="utf-8")

    def test_parse_batch_args_combines_benchmark_selectors(self) -> None:
        options = batch_module.parse_batch_args(
            [
                "--benchmarks",
                "browsecomp,workbench",
                "--benchmark",
                "scicode",
                "--benchmark",
                "browsecomp",
            ]
        )

        self.assertEqual(options.benchmarks, ["browsecomp", "workbench", "scicode"])

    def test_list_benchmarks_prints_discovered_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            self._write_configs(config_dir)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = batch_module.main(
                    ["--list-benchmarks", "--config-dir", str(config_dir)]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip().splitlines(), sorted(BENCHMARK_NAMES))

    def test_select_benchmarks_defaults_to_all_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            self._write_configs(config_dir)

            options = batch_module.parse_batch_args(["--config-dir", str(config_dir)])
            selected = batch_module.select_benchmarks(options)

        self.assertEqual(sorted(selected), sorted(BENCHMARK_NAMES))


if __name__ == "__main__":
    unittest.main()
