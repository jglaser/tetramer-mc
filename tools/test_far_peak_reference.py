"""Far-only masks, full-N positive partitions and independent shell sums."""
import math
import unittest

import numpy as np

from analyze_far_peak_reference import (MaskedMoments, OLD_KEYS, WINDOW,
    add_population_errors, combine_shells, selected_keys, sum_independent_estimates)
from analyze_latent_region import original_q_contains


class FarPeakReferenceTests(unittest.TestCase):
    def test_exact_far_and_radial_boundaries(self):
        self.assertFalse(original_q_contains(np.nextafter(5., 0.), WINDOW))
        self.assertTrue(original_q_contains(5., WINDOW))
        self.assertTrue(original_q_contains(np.nextafter(37., 0.), WINDOW))
        self.assertFalse(original_q_contains(37., WINDOW))
        self.assertIn('old_r_le_3', selected_keys(.5, 3., 2.))
        self.assertIn('radial_0', selected_keys(.5, 3., 2.))
        self.assertIn('old_r_gt_3', selected_keys(.5, np.nextafter(3., 4.), 2.))
        self.assertIn('radial_1', selected_keys(np.nextafter(.5, 1.), 2., 2.))
        self.assertIn('radial_2', selected_keys(2., 2., 2.))
        with self.assertRaises(ValueError): selected_keys(2.0001, 2., 2.)

    def test_original_denominator_and_negative_covariance(self):
        weights = np.array([2., 0., 5., 7., 0., 3.]); membership = np.array([0, 0, 1, 1, 0, 0])
        moments = MaskedMoments(1.)
        for w, group in zip(weights, membership):
            logw = math.log(w) if w else None
            moments.add(.25+.5*group, 2.+2*group, logw, 0. if w else None, [logw, logw] if w else None)
        result = moments.report(); full = result['physical']['full']; left, right = [result['physical'][k] for k in OLD_KEYS]
        self.assertEqual(left['draws'], 6); self.assertEqual(left['nonzero'], 2)
        self.assertAlmostEqual(math.exp(full['logQ']), weights.mean())
        self.assertAlmostEqual(math.exp(left['logQ']), (weights*(membership == 0)).mean())
        cov = result['partition_checks']['physical']['old_chart']['covariances'][0]
        expected_cov = np.cov(weights*(membership == 0), weights*(membership == 1), ddof=1)[0, 1]/6
        self.assertAlmostEqual(-math.exp(cov['log_absolute_covariance']), expected_cov)
        self.assertAlmostEqual(math.exp(full['log_variance_of_mean']), weights.var(ddof=1)/6)
        self.assertAlmostEqual(sum(math.exp(r['log_variance_of_mean']) for r in (left, right))+2*expected_cov, weights.var(ddof=1)/6)

    def test_nonuniform_finite_proposal_recovers_masked_mass(self):
        probability = np.array([.1, .2, .3, .4]); target = np.array([.5, 2., 7., 1.])
        moments = MaskedMoments(2.)
        for i, n in enumerate(np.rint(10*probability).astype(int)):
            value = math.log(target[i]/probability[i])
            for _ in range(n): moments.add([.2, .7, 1.2, 1.7][i], [2., 4., 2., 4.][i], value, 0., [value, value])
        result = moments.report()
        self.assertAlmostEqual(math.exp(result['physical']['full']['logQ']), target.sum())
        self.assertAlmostEqual(math.exp(result['physical']['radial_0']['logQ']), target[0])
        self.assertAlmostEqual(math.exp(result['physical']['old_r_le_3']['logQ']), target[[0, 2]].sum())

    def test_empty_masks_remain_unresolved(self):
        moments = MaskedMoments(.5)
        moments.add(.1, 4.); moments.add(.2, 4.)
        result = moments.report()
        self.assertIsNone(result['physical']['full']['logQ'])
        self.assertEqual(result['physical']['old_r_gt_3']['draws'], 2)
        self.assertIn('not a mass upper bound', result['physical']['full']['coverage'])
        with self.assertRaises(ValueError): moments.merge(MaskedMoments(1.))
        empty = result['physical']['full']; empty['independent_population_RSE'] = None
        combined = sum_independent_estimates([empty, empty])
        self.assertIsNone(combined['logQ']); self.assertEqual(combined['unresolved_zero_pieces'], 2)

    def test_disjoint_shell_sum_not_nested_ball_sum(self):
        campaigns = []; arrays = []
        for index, radius in enumerate((.5, 1., 2.)):
            aggregate = MaskedMoments(radius); populations = []
            # Earlier bins deliberately have huge contributions. The selected
            # shell estimator must not add these nested whole-ball masses.
            sample = np.array([2., 5., 0., 3.])*(index+1)
            arrays.append(sample)
            shell_radius = (.25, .75, 1.5)[index]
            for pop in range(2):
                local = MaskedMoments(radius)
                for j, w in enumerate(sample[2*pop:2*pop+2]):
                    value = math.log(w) if w else None
                    local.add(shell_radius, 2. if j == 0 else 4., value, 0. if w else None, [value, value] if w else None)
                if index:
                    local.add(.1, 2., math.log(1000.), 0., [math.log(1000.)]*2)
                populations.append(dict(samples=local.values['physical']['full'].count, **local.report()))
                aggregate.merge(local)
            report = add_population_errors(aggregate.report(), populations, aggregate.keys)
            campaigns.append(dict(root=f'run{index}', radius_A=radius, **report))
        result = combine_shells(campaigns)
        # Denominators include the deliberate earlier-bin observations.
        full_rows = [np.r_[a, [0., 0.]] if i else a for i, a in enumerate(arrays)]
        expected = sum(a.mean() for a in full_rows)
        variance = sum(a.var(ddof=1)/len(a) for a in full_rows)
        self.assertAlmostEqual(math.exp(result['physical']['full']['logQ']), expected)
        self.assertAlmostEqual(math.exp(result['physical']['full']['log_variance_of_mean']), variance)
        self.assertAlmostEqual(result['physical']['full']['row_RSE'], math.sqrt(variance)/expected)
        self.assertLess(result['physical']['old_partition_audit']['scaled_variance_error'], 1e-12)
        self.assertNotAlmostEqual(expected, sum(math.exp(c['physical']['full']['logQ']) for c in campaigns))

    def test_sum_variance_not_variance_of_average(self):
        def estimate(q, variance):
            return dict(logQ=math.log(q), log_variance_of_mean=math.log(variance),
                independent_population_RSE=math.sqrt(variance)/q, draws=100, nonzero=100)
        result = sum_independent_estimates([estimate(2., .3), estimate(5., .7)])
        self.assertAlmostEqual(math.exp(result['logQ']), 7.)
        self.assertAlmostEqual(math.exp(result['log_variance_of_mean']), 1.)
        self.assertAlmostEqual(result['independent_population_RSE'], 1/7)


if __name__ == '__main__': unittest.main()
