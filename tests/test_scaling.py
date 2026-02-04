import unittest

import numpy as np

from descriptor.scaling import robust_scale


class TestScaling(unittest.TestCase):
    def test_robust_scale(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaled, scaler = robust_scale(X, feature_names=["a", "b"])
        expected = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(scaled, expected)
        self.assertEqual(scaler.feature_names, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
