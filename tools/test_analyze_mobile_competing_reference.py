"""Synthetic paired-statistic and disjoint-stratum controls only."""
import math
import unittest
import numpy as np
from analyze_mobile_competing_reference import paired_moments, sum_independent_regions, contrast, check_estimate


def logs(values):
    return np.log(np.asarray(values, dtype=float), where=np.asarray(values)>0,
                  out=np.full(len(values), -math.inf))


def region(name, z, h):
    value = paired_moments(logs(z), logs(h))
    return dict(name=name, row_uncertainty=value, population_uncertainty=value)


class MobileCompetingStatisticsTests(unittest.TestCase):
    def test_paired_ratio_matches_linear_delta_residual_with_zeros(self):
        z, h = np.array([0., 2., 11., 4., 0., 6.]), np.array([0., 1., 2., 3., 0., 2.])
        value = paired_moments(logs(z), logs(h))
        expected = np.var(z/z.mean()-h/h.mean(), ddof=1)/len(z)
        self.assertAlmostEqual(value['log_enhancement_SE']**2, expected)
        self.assertAlmostEqual(value['log_Qz'], math.log(z.mean()))
        self.assertAlmostEqual(value['log_Q0'], math.log(h.mean()))
        covariance = np.cov(np.stack([z/z.mean(), h/h.mean()]), ddof=1)/len(z)
        np.testing.assert_allclose(value['covariance_relative'], covariance, atol=1e-15)

    def test_perfect_pairing_removes_ratio_error_without_erasing_marginal_error(self):
        h = [0., 1., 2., 8., 0., 3.]
        value = paired_moments(logs(np.array(h)*math.exp(20)), logs(h))
        self.assertGreater(value['Qz_relative_SE'], .2)
        self.assertLess(value['log_enhancement_SE'], 1e-14)
        self.assertAlmostEqual(value['log_enhancement'], 20.)

    def test_missing_zero_denominator_and_invalid_support_are_rejected(self):
        value = paired_moments(logs([0, 1, 2, 0]), logs([0, 1, 1, 0]))
        with self.assertRaisesRegex(ValueError, 'denominator'):
            check_estimate(value, dict(logQ=value['log_Qz'], draws=2), dict(logQ=value['log_Q0'], draws=2))
        with self.assertRaisesRegex(ValueError, 'invalid zeros'):
            paired_moments(logs([0, 1, 2]), logs([1, 1, 2]))
        self.assertIsNone(paired_moments(logs([0, 0]), logs([0, 0]))['log_enhancement'])

    def test_disjoint_sum_adds_integrals_and_independent_absolute_covariance(self):
        a = region('core', [1., 2., 5., 0.], [1., 1., 2., 0.])
        b = region('shell', [8., 0., 4., 2.], [2., 0., 3., 1.])
        result = sum_independent_regions([a, b], 'union')['row_uncertainty']
        means = [np.array([math.exp(x['row_uncertainty'][f]) for f in ('log_Qz', 'log_Q0')]) for x in (a, b)]
        absolute = sum(np.array(x['row_uncertainty']['covariance_relative'])*np.outer(m, m) for x, m in zip((a, b), means))
        total = sum(means)
        expected_cov = absolute/np.outer(total, total)
        np.testing.assert_allclose(result['covariance_relative'], expected_cov, atol=1e-15)
        self.assertAlmostEqual(math.exp(result['log_Qz']), total[0])
        self.assertAlmostEqual(result['log_enhancement_SE']**2, expected_cov[0, 0]+expected_cov[1, 1]-2*expected_cov[0, 1])
        # Averaging strata together would incorrectly halve the union integral.
        pooled = paired_moments(logs([1, 2, 5, 0, 8, 0, 4, 2]), logs([1, 1, 2, 0, 2, 0, 3, 1]))
        self.assertAlmostEqual(result['log_Qz']-pooled['log_Qz'], math.log(2))

    def test_independent_region_contrast_keeps_correct_total_error(self):
        a = region('native', [2., 6., 4., 0.], [1., 2., 1., 0.])
        b = region('competitor', [1., 3., 8., 0.], [2., 1., 3., 0.])
        result = contrast(a, b)['row_uncertainty']
        self.assertAlmostEqual(result['log_Qz_ratio'], result['log_Q0_ratio']+result['log_enhancement_difference'])
        self.assertAlmostEqual(result['log_Qz_ratio_SE']**2, a['row_uncertainty']['Qz_relative_SE']**2+b['row_uncertainty']['Qz_relative_SE']**2)


if __name__ == '__main__':
    unittest.main()
