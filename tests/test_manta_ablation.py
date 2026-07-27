import argparse
import json
import tempfile
import unittest
from pathlib import Path

from MAS.config import load_experiment_config
from scripts import run_manta_ablation


class TestMantaAblation(unittest.TestCase):
    def _args(self, output_root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            experiment_id="test_ablation",
            output_root=str(output_root),
            manifest=str(run_manta_ablation.DEFAULT_MANIFEST),
            model="test-model",
            allow_mock=True,
        )

    def test_committed_manifest_has_thirty_unique_tasks_per_benchmark(self) -> None:
        manifest = run_manta_ablation._load_manifest(run_manta_ablation.DEFAULT_MANIFEST)

        self.assertEqual(len(manifest["execution_order"]), 90)
        for benchmark in ("browsecomp", "workbench", "plancraft"):
            task_ids = manifest["benchmarks"][benchmark]["task_ids"]
            self.assertEqual(len(task_ids), run_manta_ablation.TASKS_PER_BENCHMARK)
            self.assertEqual(len(set(task_ids)), run_manta_ablation.TASKS_PER_BENCHMARK)

    def test_prepare_writes_five_parseable_variant_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp))
            _, state_root = run_manta_ablation.prepare(args)

            for variant in run_manta_ablation.VARIANTS:
                config_path = run_manta_ablation._config_path(state_root, variant, "browsecomp")
                config = load_experiment_config(config_path)
                self.assertEqual(config.models["default"], "test-model")
                self.assertEqual(
                    config.self_evolved.initial_planner_mode,
                    variant.initial_planner_mode,
                )
                self.assertEqual(config.self_evolved.repair_budget, variant.repair_budget)
                self.assertEqual(config.self_evolved.max_turns, variant.max_turns)
                self.assertEqual(config.self_evolved.playbook_read, variant.playbook_read)
                self.assertTrue(Path(config.self_evolved.skill_path).exists())

            run_manifest = json.loads(
                (state_root / "experiment_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(run_manifest["variants"]), 5)
            self.assertEqual(run_manifest["runs_per_task"], run_manta_ablation.RUNS_PER_TASK)
            self.assertEqual(run_manta_ablation.RUNS_PER_TASK, 1)
            self.assertEqual(
                run_manifest["reflection_batch_size"],
                run_manta_ablation.REFLECTION_BATCH_SIZE,
            )
            self.assertEqual(run_manta_ablation.REFLECTION_BATCH_SIZE, 10)
            self.assertEqual(run_manifest["seed"], 42)


if __name__ == "__main__":
    unittest.main()
