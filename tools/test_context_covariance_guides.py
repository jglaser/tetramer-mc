import unittest
import numpy as np
from prepare_context_covariance_guides import freeze_moments


class CovarianceGuideTests(unittest.TestCase):
    def test_full_diagonal_controls_preserve_mean_and_marginal_variances(self):
        scales = np.array([.1]*3+[.2]*3)
        lower = np.eye(6)
        lower[3, 0] = .8
        lower[4, 2] = -.4
        scaled = lower@lower.T
        mean = np.arange(6)*.01
        actual, full, diagonal, epsilon = freeze_moments(mean, scaled*np.outer(scales, scales), scales)
        np.testing.assert_array_equal(actual, mean)
        self.assertEqual(epsilon, 1e-6*np.trace(scaled)/6)
        np.testing.assert_allclose(full, (scaled+epsilon*np.eye(6))*np.outer(scales, scales), atol=1e-16)
        np.testing.assert_array_equal(np.diag(full), np.diag(diagonal))
        self.assertGreater(abs(full[3, 0]), 0)
        self.assertEqual(diagonal[3, 0], 0)

    def test_degenerate_training_has_declared_floor_and_invalid_data_fails(self):
        _, full, diagonal, epsilon = freeze_moments(np.zeros(6), np.zeros((6, 6)), np.ones(6))
        self.assertEqual(epsilon, 1e-10)
        np.testing.assert_array_equal(full, np.eye(6)*1e-10)
        np.testing.assert_array_equal(full, diagonal)
        for matrix in (np.eye(6)*-1, np.full((6, 6), np.nan), np.eye(6)+np.diag([.1]*5, 1)):
            with self.assertRaises(ValueError):freeze_moments(np.zeros(6), matrix, np.ones(6))


if __name__ == '__main__':
    unittest.main()
