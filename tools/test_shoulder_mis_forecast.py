#!/usr/bin/env python3
"""Controls for fixed-quota forecast variance, independent of protein geometry."""
import unittest
import numpy as np

from forecast_shoulder_mis import fixed_quota_cost_variance, zero_inclusive_variance


class ShoulderMisForecastTests(unittest.TestCase):
    def test_invalid_zeros_remain_in_stratum_variance(self):
        self.assertAlmostEqual(zero_inclusive_variance([1., 3.], 4), 2.)
        self.assertAlmostEqual(zero_inclusive_variance([1., 3.], 4), np.var([0., 1., 0., 3.], ddof=1))
        self.assertEqual(zero_inclusive_variance([], 10), 0.)

    def test_equal_laws_and_costs_have_no_allocation_gain(self):
        for ratio in (0, 1, 4, 16, 64):
            self.assertAlmostEqual(fixed_quota_cost_variance(7., 7., ratio, 3., 3.), 21.)

    def test_fixed_quota_variance_uses_squared_total_denominator(self):
        self.assertAlmostEqual(fixed_quota_cost_variance(4., 9., 4, 2., .5), 6.4)
        self.assertAlmostEqual(fixed_quota_cost_variance(4., 9., 0, 2., .5), 8.)


if __name__ == '__main__':
    unittest.main()
