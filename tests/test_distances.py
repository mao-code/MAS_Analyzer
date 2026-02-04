import unittest

import numpy as np

from descriptor.distances import covariance_inverse, mahalanobis_distance


class TestDistances(unittest.TestCase):
    def test_mahalanobis_identity(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        cov, cov_inv = covariance_inverse(X, regularization=0.0)
        self.assertEqual(cov.shape, (2, 2))
        x = np.array([1.0, 2.0])
        mean = np.array([0.0, 0.0])
        dist = mahalanobis_distance(x, mean, np.eye(2))
        self.assertAlmostEqual(dist, np.sqrt(5.0))


if __name__ == "__main__":
    unittest.main()
