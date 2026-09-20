#!/usr/bin/env python3
"""Independent population-error controls, including unobserved populations."""
import math
import unittest

from compare_shoulder_covers import population_error


class ShoulderCoverComparisonTests(unittest.TestCase):
    def test_population_errors_use_linear_normalizers(self):
        populations = [{'physical': {'logQ': 0.}}, {'physical': {'logQ': math.log(3.)}}]
        self.assertAlmostEqual(population_error(populations, 'physical'), .5)

    def test_unobserved_population_remains_in_denominator(self):
        populations = [{'hard': {'logQ': None}}, {'hard': {'logQ': math.log(2.)}}]
        self.assertAlmostEqual(population_error(populations, 'hard'), 1.)
        self.assertIsNone(population_error([{'hard': {'logQ': None}}]*4, 'hard'))

    def test_constant_population_estimates_have_zero_observed_error(self):
        self.assertEqual(population_error([{'physical': {'logQ': 40.}}]*4, 'physical'), 0.)


if __name__ == '__main__':
    unittest.main()
