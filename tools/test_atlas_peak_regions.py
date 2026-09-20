"""Check disjoint geometric-region diagnostics with exact finite reference data."""
import math
import unittest

import numpy as np

from compare_intermediate_local_reference import PartitionMoments
from diagnose_atlas_peak_regions import center_separation, threeway_add, threeway_report


class AtlasPeakRegionsTests(unittest.TestCase):
    def test_threeway_preserves_full_denominator_and_same_row_variance(self):
        rows = [(3., .5, 1.), (2., 3., 4.), (3., 2., 2.), (3., 3., 8.), (None, None, 0.)]
        moments = PartitionMoments(2.)
        for new, old, weight in rows:
            if weight:
                value = math.log(weight); threeway_add(moments, new, old, value, 0., [value]*2)
            else: threeway_add(moments)
        population = dict(id='r0', seed=1, samples=len(rows), **moments.report())
        report = threeway_report(moments, [population])
        expected = {'old_ball': np.array([1., 0., 2., 0., 0.]),
                    'new_ball': np.array([0., 4., 0., 0., 0.]),
                    'outside_both': np.array([0., 0., 0., 8., 0.])}
        for key, values in expected.items():
            measured = report['physical'][key]
            self.assertEqual(measured['draws'], len(rows))
            self.assertAlmostEqual(math.exp(measured['logQ']), values.mean())
            self.assertAlmostEqual(math.exp(measured['log_variance_of_mean']), values.var(ddof=1)/len(rows))
        self.assertAlmostEqual(math.exp(report['physical']['full']['logQ']), 3.)
        self.assertLess(report['partition_checks']['physical']['relative_variance_error'], 1e-12)

    def test_overlapping_ball_membership_is_rejected(self):
        with self.assertRaises(ValueError):
            threeway_add(PartitionMoments(2.), 2., 2., 0., 0., [0., 0.])

    def test_rms_triangle_margin_is_translation_distance(self):
        cfg = dict(metadata=dict(rigid_members=[dict(position=[1., 0., 0.]), dict(position=[-1., 0., 0.])]))
        old = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
        new = dict(position=[3., 4., 0.], orientation=[1., 0., 0., 0.])
        result = center_separation(cfg, new, old)
        self.assertEqual(result['member_RMS_distance_A'], 5.)
        self.assertEqual(result['positive_RMS_separation_lower_bound_A'], 1.)
        self.assertTrue(result['two_radius_2_balls_disjoint'])


if __name__ == '__main__': unittest.main()
