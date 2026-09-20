"""Finite reference tests for fixed prefixes and their shared-row covariance."""
import math
import unittest

import numpy as np

from analyze_expanded_contact_atlas import PrefixMoments, nested_prefix_comparison


class ExpandedAtlasPrefixTests(unittest.TestCase):
    def test_fixed_prefixes_keep_every_zero_and_unchanged_regions(self):
        accumulator = PrefixMoments(8, [2, 4, 8])
        weights = [0., 2., 4., 0., 8., 3., 0., 1.]
        old = [None, .25, 3., None, 3., 1.5, None, 3.]
        new = [None, 3., .5, None, 3., 3., None, 2.]
        for i, w in enumerate(weights):
            if w: accumulator.add(i, old[i], new[i], 8., math.log(w), 0., [math.log(w)]*2)
            else: accumulator.add(i)
        accumulator.complete()
        for n, moments in accumulator.prefixes.items():
            report = moments.report(); expected = np.array(weights[:n])
            full = report['physical']['full']
            self.assertEqual(full['draws'], n)
            self.assertAlmostEqual(math.exp(full['logQ']), expected.mean())
            self.assertAlmostEqual(math.exp(full['log_variance_of_mean']), expected.var(ddof=1)/n)
            partition = report['disjoint_threeway']['physical']
            self.assertAlmostEqual(sum(math.exp(partition[key]['logQ']) if partition[key]['logQ'] is not None else 0.
                for key in ('old_ball', 'new_ball', 'outside_both')), expected.mean())

    def test_prefix_comparison_uses_covariance_not_independent_errors(self):
        accumulator = PrefixMoments(8, [2, 4, 8])
        for i, w in enumerate([1., 3., 2., 4., 8., 6., 7., 9.]):
            accumulator.add(i, .25, 3., 1., math.log(w), 0., [math.log(w)]*2)
        small = accumulator.prefixes[2].report()['physical']['full']
        full = accumulator.prefixes[8].report()['physical']['full']
        comparison = nested_prefix_comparison(small, full)
        self.assertAlmostEqual(comparison['theoretical_IID_correlation'], .5)
        self.assertAlmostEqual(math.exp(comparison['estimated_log_variance_of_difference']),
                               3*math.exp(full['log_variance_of_mean']))
        self.assertAlmostEqual(comparison['relative_prefix_minus_full'], 2/5-1)

    def test_missing_and_repeated_rows_fail(self):
        accumulator = PrefixMoments(4, [2, 4])
        with self.assertRaises(ValueError): accumulator.add(1)
        accumulator.add(0)
        with self.assertRaises(ValueError): accumulator.add(0)
        with self.assertRaises(ValueError): accumulator.complete()

    def test_zero_prefix_is_not_a_zero_mass_bound(self):
        accumulator = PrefixMoments(4, [2, 4])
        accumulator.add(0); accumulator.add(1)
        accumulator.add(2, 3., 3., 1., 0., 0., [0., 0.]); accumulator.add(3)
        prefix = accumulator.prefixes[2].report()['physical']['full']
        full = accumulator.prefixes[4].report()['physical']['full']
        self.assertIsNone(prefix['logQ'])
        comparison = nested_prefix_comparison(prefix, full)
        self.assertEqual(comparison['relative_prefix_minus_full'], -1.)
        self.assertIn('do not bound unseen tails', comparison['qualification'])


if __name__ == '__main__': unittest.main()
