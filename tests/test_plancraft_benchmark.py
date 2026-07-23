import unittest
from unittest.mock import MagicMock, patch

from benchmark.base import BenchmarkTask
from benchmark.plancraft import PlancraftBenchmark
from benchmark.plancraft.search import exhaustive_recipe_search


class TestPlancraftBenchmark(unittest.TestCase):
    @patch("benchmark.plancraft.get_plancraft_examples", create=True)
    def test_load_tasks(self, mock_get: MagicMock) -> None:
        """Placeholder: mock plancraft examples to test load_tasks."""
        # PlanCraft examples have .id, .target, .recipe_type, .complexity,
        # .impossible, .slotted_inventory
        ex = MagicMock()
        ex.id = "VAL0001"
        ex.target = "oak_planks"
        ex.recipe_type = "shaped"
        ex.complexity = 1
        ex.impossible = False
        ex.slotted_inventory = {1: {"type": "oak_log", "quantity": 1}}
        mock_get.return_value = [ex]

        bench = PlancraftBenchmark({"split": "val"})

        with patch("plancraft.simple.get_plancraft_examples", return_value=[ex]):
            tasks = bench.load_tasks(task_limit=1)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "VAL0001")
        self.assertIn("oak_planks", tasks[0].prompt)
        self.assertTrue(tasks[0].metadata["recipe_search"])
        self.assertIn("search: <recipe name>", bench._system_prompt_text)

    def test_recipe_search_can_be_disabled_for_historical_protocol_replay(self) -> None:
        bench = PlancraftBenchmark({"recipe_search": False})

        self.assertNotIn("search: <recipe name>", bench._system_prompt_text)
        self.assertNotIn("search", [handler.action_name for handler in bench._action_handlers])

    def test_recipe_search_lists_all_accepted_shaped_alternatives(self) -> None:
        result = exhaustive_recipe_search("red_sandstone_slab")

        self.assertIn("chiseled_red_sandstone|red_sandstone at [A1]", result)
        self.assertIn("chiseled_red_sandstone|red_sandstone at [A2]", result)
        self.assertIn("chiseled_red_sandstone|red_sandstone at [A3]", result)
        self.assertIn("produces 6 red_sandstone_slab", result)

    def test_evaluate_success(self) -> None:
        task = BenchmarkTask(
            task_id="VAL0001",
            prompt="Craft oak_planks",
            reference_answer="oak_planks",
            metadata={"impossible": False, "recipe_type": "shaped", "complexity": 1},
        )
        bench = PlancraftBenchmark()
        result = bench.evaluate(
            task,
            "move I1 C1 1",
            run_metadata={"reward": 1.0, "num_steps": 3, "terminated": True},
        )
        self.assertTrue(result.success)
        self.assertEqual(result.score, 1.0)

    def test_evaluate_failure(self) -> None:
        task = BenchmarkTask(
            task_id="VAL0002",
            prompt="Craft iron_ingot",
            reference_answer="iron_ingot",
            metadata={"impossible": False, "recipe_type": "smelting", "complexity": 2},
        )
        bench = PlancraftBenchmark()
        result = bench.evaluate(
            task,
            "move I1 C1 1",
            run_metadata={"reward": 0.0, "num_steps": 30, "truncated": True},
        )
        self.assertFalse(result.success)
        self.assertEqual(result.score, 0.0)

    def test_evaluate_impossible_correct(self) -> None:
        task = BenchmarkTask(
            task_id="VAL0003",
            prompt="Craft diamond_block",
            reference_answer="impossible",
            metadata={"impossible": True, "recipe_type": "shaped", "complexity": 1},
        )
        bench = PlancraftBenchmark()
        result = bench.evaluate(
            task,
            "impossible",
            run_metadata={"reward": 0.0, "num_steps": 1, "terminated": True},
        )
        self.assertTrue(result.success)

    def test_evaluate_impossible_wrong(self) -> None:
        task = BenchmarkTask(
            task_id="VAL0004",
            prompt="Craft diamond_block",
            reference_answer="impossible",
            metadata={"impossible": True, "recipe_type": "shaped", "complexity": 1},
        )
        bench = PlancraftBenchmark()
        result = bench.evaluate(
            task,
            "move I1 C1 1",
            run_metadata={"reward": 0.0, "num_steps": 30, "truncated": True},
        )
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
