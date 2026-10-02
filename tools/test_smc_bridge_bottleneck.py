"""Poisson reuse, endpoint and population-ratio checks; no physical sampling."""
import math
import unittest
import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson
import analyze_smc_bridge_bottleneck as bridge


class BridgeReuseTests(unittest.TestCase):
    def test_endpoints_zero_activity_and_invalid_counts(self):
        h, g, z, intensity = -3., 2., .035, 4.48
        lower, counts = [2., 2.], [11, 23]
        result = bridge.bridge_log_weights(h, g, lower, counts, z, intensity)
        self.assertAlmostEqual(result[0], h+g, places=14)
        expected = h+logsumexp([z*l+k*math.log1p(z/intensity) for l, k in zip(lower, counts)])-math.log(2)
        self.assertAlmostEqual(result[-1], expected, places=14)
        np.testing.assert_allclose(bridge.bridge_log_weights(h, g, [0., 0.], [0, 0], 0., 1.),
                                   h+(1-np.array(bridge.BETAS))*g, atol=1e-14)
        for bad_counts in ([1.5, 2.], [-1, 2]):
            with self.assertRaisesRegex(ValueError, 'cloud pair'):
                bridge.bridge_log_weights(h, g, lower, bad_counts, z, intensity)

    def test_reused_poisson_pgf_is_unbiased_and_not_power_of_old_weight(self):
        overlap, lower, z, intensity = 1.7, .2, .8, 3.
        counts = np.arange(100)
        probability = poisson.pmf(counts, intensity*(overlap-lower))
        for beta in bridge.BETAS:
            values = np.array([math.exp(bridge.bridge_log_weights(0., 0., [lower]*2,
                np.array([k, k]), z, intensity, [beta])[0]) for k in counts])
            self.assertAlmostEqual(float(probability@values), math.exp(beta*z*overlap), places=12)
        beta = .5
        powered = np.exp(beta*(z*lower+counts*math.log1p(z/intensity)))
        self.assertGreater(abs(float(probability@powered)-math.exp(beta*z*overlap)), .01)

    def test_pooled_row_concentration_keeps_zero_attempts_and_populations(self):
        arrays = [np.full(16384, -np.inf) for _ in range(4)]
        arrays[0][0] = math.log(2)
        arrays[3][0] = math.log(4)
        result = bridge.aggregate_moments([bridge.moments(a) for a in arrays])
        self.assertAlmostEqual(result['log_mass'], math.log(6/65536), places=12)
        self.assertAlmostEqual(result['importance_ESS'], 36/20, places=12)
        self.assertAlmostEqual(result['largest_draw_fraction'], 2/3, places=12)
        self.assertEqual(result['population_log_masses'][1:3], [None, None])
        self.assertFalse(result['passed'])
        empty = bridge.aggregate_moments([bridge.moments(np.full(16384, -np.inf)) for _ in range(4)])
        self.assertFalse(empty['observed'])
        self.assertIsNone(empty['log_mass'])

    def test_fraction_is_ratio_of_linear_means_and_retains_paired_covariance(self):
        rows = []
        for total, remainder in [(1., .9), (10., 1.), (1., .9), (10., 1.)]:
            rows.append(dict(total=math.log(total), registered_native_entry=math.log(total),
                old_R5_intersection_native=math.log(total-remainder), remaining_R4_native=math.log(remainder),
                contact_no_native_entry=None, unbound_no_native_entry=None))
        result = bridge.fraction(bridge.population_statistics(rows))
        self.assertAlmostEqual(result['fraction'], 1.9/11)
        self.assertNotAlmostEqual(result['fraction'], .5)
        expected_relative = np.array([.9, 1., .9, 1.])/.95-np.array([1., 10., 1., 10.])/5.5
        self.assertAlmostEqual(result['paired_population_log_ratio_SE'], expected_relative.std(ddof=1)/2)

    def test_smc_comparison_is_linear_and_missing_mass_unresolved(self):
        left = list(np.log([1.5, 1.5, 4.5, 4.5])); right = [0.]*4
        result = bridge.compare_logs(left, right)
        self.assertTrue(result['SE_passed'])
        self.assertFalse(result['absolute_passed'])
        self.assertAlmostEqual(result['ratio_left_over_right'], 3.)
        result = bridge.compare_logs([None]*4, right)
        self.assertFalse(result['observed'])
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
