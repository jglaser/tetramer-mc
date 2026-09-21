"""Full-N radial partitions and overlapping-mask covariance."""
import math
import unittest

import numpy as np

from compare_far_local_history import (HistoryMoments, flat_visit_rate, independent_comparison,
    selected_keys, validate_runtime_metric)


class FarLocalHistoryTests(unittest.TestCase):
    def test_runtime_metric_ignores_only_nonmetric_metadata(self):
        metric = dict(native_poses=[1], rigid_members=[2], member_error_scale=2., angle_error_scale_deg=15.)
        validate_runtime_metric(metric, dict(metric, source_smc='report only', site=0))
        with self.assertRaises(ValueError): validate_runtime_metric(dict(metric, member_error_scale=4.), metric)

    def test_boundaries_infinite_seam_and_bad_radius(self):
        self.assertIn('ball0p5', selected_keys(.5))
        self.assertNotIn('ball0p5', selected_keys(np.nextafter(.5, 1.)))
        self.assertIn('ball1', selected_keys(1.))
        self.assertIn('ball2', selected_keys(2.))
        self.assertEqual(selected_keys(math.inf), ('full', 'outside2'))
        for value in (-.1, math.nan):
            with self.assertRaises(ValueError): selected_keys(value)

    def test_full_n_and_covariance_for_nested_and_disjoint_masks(self):
        radius = np.array([.2, .7, 1.7, 3., .3, 1.2])
        weights = np.array([2., 5., 3., 7., 0., 0.])
        moments = HistoryMoments()
        for r, w in zip(radius, weights):
            if w:
                value = math.log(w); moments.add(r, value, 0., [value, value])
            else: moments.add()
        result = moments.report(); balls = {name: weights*(radius <= bound) for name, bound in [('ball0p5', .5), ('ball1', 1.), ('ball2', 2.)]}
        balls['outside2'] = weights*(radius > 2.)
        for left, a in balls.items():
            observed = result['physical'][left]
            self.assertEqual(observed['draws'], len(weights))
            self.assertAlmostEqual(math.exp(observed['logQ']), a.mean())
            for right, b in balls.items():
                item = result['same_row_covariances']['physical'][left][right]
                actual = item['sign']*math.exp(item['log_absolute_covariance']) if item['sign'] else 0.
                self.assertAlmostEqual(actual, np.cov(a, b, ddof=1)[0, 1]/len(weights))
        self.assertGreater(result['same_row_covariances']['physical']['ball0p5']['ball1']['sign'], 0)
        self.assertLess(result['same_row_covariances']['physical']['ball2']['outside2']['sign'], 0)
        self.assertLess(result['partition_checks']['physical']['full']['scaled_variance_error'], 1e-12)

    def test_nonuniform_proposal_and_invalid_zeros(self):
        p = [.1, .2, .3, .4]; f = [.5, 2., 0., 1.]; radius = [.2, .7, 1.2, 3.]
        moments = HistoryMoments()
        for probability, mass, r in zip(p, f, radius):
            for _ in range(round(10*probability)):
                if mass:
                    value = math.log(mass/probability); moments.add(r, value, 0., [value, value])
                else: moments.add()
        result = moments.report()['physical']
        self.assertEqual(result['full']['draws'], 10)
        self.assertAlmostEqual(math.exp(result['full']['logQ']), sum(f))
        self.assertAlmostEqual(math.exp(result['ball1']['logQ']), sum(f[:2]))
        self.assertAlmostEqual(math.exp(result['outside2']['logQ']), f[-1])
        self.assertIsNone(result['radial_2']['logQ'])

    def test_observed_difference_uses_linear_mass_errors(self):
        a = dict(logQ=math.log(4), row_RSE=.25, independent_population_RSE=.5)
        b = dict(logQ=math.log(1), row_RSE=.2, independent_population_RSE=.4)
        result = independent_comparison(a, b)
        self.assertAlmostEqual(result['historical_to_fresh_ratio'], 4.)
        self.assertAlmostEqual(result['linear_difference_in_combined_row_SE'], 3/math.hypot(1., .2))
        self.assertAlmostEqual(result['linear_difference_in_combined_population_SE'], 3/math.hypot(2., .4))
        self.assertIn('exploratory', result['scope'])
        self.assertIsNone(independent_comparison(dict(a, logQ=None), b)['historical_minus_fresh_logQ'])

    def test_empty_supported_masks_preserve_zero_covariance(self):
        moments = HistoryMoments(); moments.add(); moments.add()
        result = moments.report()
        self.assertIsNone(result['physical']['full']['logQ'])
        self.assertEqual(result['physical']['ball2']['draws'], 2)
        self.assertEqual(result['same_row_covariances']['physical']['ball2']['outside2']['sign'], 0)

    def test_flat_hit_rate_and_jacobian_bound(self):
        n = 100; density = .00001; volume = .01; rse = .2
        result = flat_visit_rate(n, density, dict(logQ=math.log(volume), row_RSE=rse, independent_population_RSE=.3), np.diag([2., 3., 4.]))
        self.assertAlmostEqual(result['expected_hits'], n*density*volume)
        self.assertAlmostEqual(result['probability_zero_hits'], (1-density*volume)**n)
        self.assertAlmostEqual(result['mean_independent_draws_to_hit'], 1/(density*volume))
        self.assertAlmostEqual(result['expected_hits_observed_SE'], n*density*volume*rse)
        bound = result['geometric_upper_bound']; j0 = 1/(8*math.pi**2*math.sqrt(24))
        self.assertAlmostEqual(bound['maximum_physical_jacobian'], j0)
        self.assertAlmostEqual(bound['physical_volume_A3'], j0*math.pi**3/6*2**6)
        self.assertGreater(bound['physical_volume_A3'], volume)
        self.assertLess(bound['probability_zero_hits_lower_bound'], result['probability_zero_hits'])


if __name__ == '__main__': unittest.main()
