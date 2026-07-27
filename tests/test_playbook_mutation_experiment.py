import argparse
import json
import tempfile
import unittest
from pathlib import Path

from MAS.config import load_experiment_config
from scripts import run_playbook_mutation_experiment as experiment


class TestPlaybookMutationExperiment(unittest.TestCase):
    def _args(self, output_root: Path, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "command": "prepare",
            "experiment_id": "test_playbook_mutation",
            "output_root": str(output_root),
            "manifest": str(experiment.DEFAULT_MANIFEST),
            "model": "test-model",
            "source_benchmarks": "plancraft",
            "target_benchmarks": "workbench",
            "mutation_budgets": "0,2",
            "training_mutation_budget": None,
            "allow_mock": True,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_default_design_is_two_benchmarks_and_four_mutation_budgets(self) -> None:
        args = experiment.build_parser().parse_args(["prepare"])
        sources, targets, budgets = experiment._design(args)

        self.assertEqual(sources, ("workbench", "plancraft"))
        self.assertEqual(targets, ("workbench", "plancraft"))
        self.assertEqual(budgets, (0, 1, 2, 3))
        self.assertEqual(len(experiment.evaluation_cells(sources, targets, budgets)), 24)
        self.assertEqual(len(sources) * experiment.TASKS_PER_BENCHMARK, 60)
        self.assertEqual(
            len(experiment.evaluation_cells(sources, targets, budgets))
            * experiment.TASKS_PER_BENCHMARK,
            720,
        )

    def test_prepare_builds_seed_control_and_learned_transfer_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp))
            _, state_root = experiment.prepare(args)

            cells = experiment.evaluation_cells(
                ("plancraft",),
                ("workbench",),
                (0, 2),
            )
            self.assertEqual(
                cells,
                [
                    ("seed", "workbench", 0),
                    ("seed", "workbench", 2),
                    ("plancraft", "workbench", 0),
                    ("plancraft", "workbench", 2),
                ],
            )

            learned_config = load_experiment_config(
                experiment._config_path(
                    state_root,
                    phase="evaluation",
                    source="plancraft",
                    target="workbench",
                    budget=2,
                )
            )
            self.assertEqual(learned_config.self_evolved.repair_budget, 2)
            self.assertEqual(learned_config.self_evolved.max_turns, 3)
            self.assertEqual(learned_config.mas.max_turns, 3)
            self.assertEqual(learned_config.self_evolved.skill_update_batch_size, 0)
            self.assertTrue(Path(learned_config.self_evolved.skill_path).exists())

            run_manifest = json.loads(
                (state_root / "experiment_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_manifest["reflection_batch_size"], 10)
            self.assertEqual(run_manifest["learning_runs"], 30)
            self.assertEqual(run_manifest["evaluation_cells"], 4)
            self.assertEqual(run_manifest["evaluation_runs"], 120)
            self.assertEqual(run_manifest["training_mutation_budget"], 2)

    def test_resume_rejects_a_changed_factorial_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            experiment.prepare(self._args(output_root))

            with self.assertRaisesRegex(RuntimeError, "settings differ"):
                experiment.prepare(self._args(output_root, mutation_budgets="0,1,2"))

    def test_reflection_recovery_commits_a_written_skill_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp))
            _, state_root = experiment.prepare(args)
            source = "plancraft"
            skill_path = experiment._training_skill(state_root, source)
            before_sha = experiment._sha256(skill_path)
            pending = [
                {"run_key": f"{source}/task-{index}/run_0", "summary": {}}
                for index in range(experiment.REFLECTION_BATCH_SIZE)
            ]
            state = experiment._load_reflection_state(state_root, source)
            state["pending"] = pending
            state["seen_run_keys"] = [row["run_key"] for row in pending]
            state["inflight"] = {
                "run_keys": [row["run_key"] for row in pending],
                "skill_before_sha256": before_sha,
            }
            experiment._save_reflection_state(state_root, source, state)
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8") + "\nRecovered lesson.\n",
                encoding="utf-8",
            )

            experiment._recover_inflight_reflection(state_root, source)
            recovered = experiment._load_reflection_state(state_root, source)
            self.assertEqual(recovered["pending"], [])
            self.assertIsNone(recovered["inflight"])
            self.assertEqual(len(recovered["updates"]), 1)
            self.assertEqual(
                recovered["updates"][0]["reason"],
                "recovered_after_skill_write",
            )

            experiment._recover_inflight_reflection(state_root, source)
            self.assertEqual(
                len(experiment._load_reflection_state(state_root, source)["updates"]),
                1,
            )

    def test_summary_reports_paired_seed_delta_and_actual_mutation_groups(self) -> None:
        rows = [
            {
                "playbook_source": source,
                "target_benchmark": "workbench",
                "transfer_type": ("seed_control" if source == "seed" else "cross_domain"),
                "configured_mutation_budget": budget,
                "actual_mutations": actual,
                "task_id": task_id,
                "success": success,
                "score": float(success),
                "turns_executed": actual + 1,
            }
            for source, budget, actual, task_id, success in (
                ("seed", 0, 0, "a", 0),
                ("seed", 0, 0, "b", 1),
                ("plancraft", 0, 0, "a", 1),
                ("plancraft", 0, 0, "b", 1),
                ("seed", 2, 0, "a", 0),
                ("seed", 2, 1, "b", 1),
                ("plancraft", 2, 2, "a", 1),
                ("plancraft", 2, 1, "b", 1),
            )
        ]

        configured, actual = experiment.summarize_results(rows)
        learned_budget_zero = next(
            row
            for row in configured
            if row["playbook_source"] == "plancraft" and row["configured_mutation_budget"] == 0
        )
        self.assertEqual(learned_budget_zero["n"], 2)
        self.assertEqual(learned_budget_zero["success_rate"], 1.0)
        self.assertEqual(learned_budget_zero["paired_success_delta_vs_seed"], 0.5)
        self.assertTrue(
            any(
                row["playbook_source"] == "plancraft"
                and row["actual_mutations"] == 2
                and row["n"] == 1
                for row in actual
            )
        )

    def test_static_curve_renderer_writes_a_nonempty_png(self) -> None:
        rows = [
            {
                "playbook_source": source,
                "target_benchmark": "workbench",
                "configured_mutation_budget": budget,
                "success_rate": rate,
                "success_ci95_low": max(0.0, rate - 0.1),
                "success_ci95_high": min(1.0, rate + 0.1),
            }
            for source, budget, rate in (
                ("seed", 0, 0.4),
                ("seed", 1, 0.5),
                ("plancraft", 0, 0.5),
                ("plancraft", 1, 0.7),
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "curve.png"
            experiment._render_curve(
                rows,
                x_field="configured_mutation_budget",
                output_path=output_path,
                title="Success rate by configured mutation budget",
                subtitle="Synthetic test data.",
            )
            self.assertGreater(output_path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
