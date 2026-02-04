import unittest

import pandas as pd

from descriptor.pareto import ideal_point_distance, pareto_frontier


class TestPareto(unittest.TestCase):
    def test_pareto_frontier(self) -> None:
        df = pd.DataFrame(
            {
                "quality": [1.0, 2.0, 3.0],
                "cost": [3.0, 2.0, 4.0],
            },
            index=["A", "B", "C"],
        )
        objectives = {"quality": "max", "cost": "min"}
        frontier = pareto_frontier(df, objectives)
        self.assertListEqual(sorted(frontier.index.tolist()), ["B", "C"])

    def test_ideal_point_distance(self) -> None:
        df = pd.DataFrame(
            {
                "quality": [1.0, 2.0, 3.0],
                "cost": [3.0, 2.0, 4.0],
            },
            index=["A", "B", "C"],
        )
        objectives = {"quality": "max", "cost": "min"}
        distances, ideal, norm_df = ideal_point_distance(df, objectives)
        self.assertEqual(distances.name, "d_ideal")
        self.assertEqual(len(ideal), 2)
        self.assertEqual(norm_df.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
