"""Exact endpoint preservation and paired uncertainty for changed SMC bridges."""
import unittest
import math
import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson
import analyze_endpoint_proposal_bridge as alternate
from analyze_smc_bridge_bottleneck import bridge_log_weights, moments


class EndpointBridgeTests(unittest.TestCase):
    def test_replacement_endpoints_invalid_zeros_and_unconditional_denominator(self):
        valid = np.array([True, False, True])
        h, g = np.array([3., -np.inf, -2.]), np.array([-4., -np.inf, 1.])
        old = np.full((3, 9), -np.inf)
        for i in [0, 2]:
            old[i] = bridge_log_weights(h[i], g[i], [1., 2.], [4, 9], .035, 4.48)
        with np.errstate(invalid='raise'):
            new, error = alternate.change_bridge(old, h, g, valid)
        np.testing.assert_array_equal(new[:, -1], old[:, -1])
        np.testing.assert_array_equal(new[valid, 0], 0.)
        self.assertTrue(np.isneginf(new[1]).all())
        self.assertLess(error, 1e-12)
        self.assertAlmostEqual(moments(new[:, 0])['logQ'], math.log(2/3))
        beta = np.array(alternate.BETAS)
        cloud = logsumexp(beta[:, None]*.035*np.array([1., 2.])
                         + np.log1p(beta[:, None]/128)*[4, 9], axis=1)-math.log(2)
        np.testing.assert_allclose(new[valid], beta[None, :]*h[valid, None]+cloud, atol=1e-14)
        bad = old.copy(); bad[0, 0] += .1
        with self.assertRaisesRegex(ValueError, 'cancellation'):
            alternate.change_bridge(bad, h, g, valid)

    def test_discrete_target_after_changing_guide(self):
        # Two physical cells of unequal size, normalized q, and a rejected cell.
        # Explicit PGF expectation checks the intermediate physical measure.
        q, J = np.array([.2, .5, .3]), np.array([2., 3., 5.])
        volume = [1.1, 2.2, 0.]
        z, intensity = .6, 2.
        for beta in alternate.BETAS:
            estimate = 0.
            for i in [0, 1]:
                k = np.arange(120)
                pgf = float(poisson.pmf(k, intensity*volume[i]) @ (1+beta*z/intensity)**k)
                estimate += q[i]*(J[i]/q[i])**beta*pgf
            exact = sum(J[i]*(q[i]/J[i])**(1-beta)*math.exp(beta*z*volume[i]) for i in [0, 1])
            self.assertAlmostEqual(estimate, exact, places=11)

    def test_paired_fraction_change_cancels_identical_and_common_scaling(self):
        rows = []
        for total, remaining in [(1., .2), (3., 1.), (2., .8), (8., 2.)]:
            rows.append(dict(total=math.log(total), registered_native_entry=math.log(total),
                old_R5_intersection_native=math.log(total-remaining), remaining_R4_native=math.log(remaining),
                contact_no_native_entry=None, unbound_no_native_entry=None))
        same = alternate.paired_fraction_change(rows, rows)
        self.assertEqual(same['log_new_over_old'], 0.)
        self.assertLess(same['paired_population_log_ratio_SE'], 1e-15)
        changed = [dict(r, remaining_R4_native=r['remaining_R4_native']+math.log(1.5)) for r in rows]
        result = alternate.paired_fraction_change(rows, changed)
        self.assertAlmostEqual(result['fraction_ratio'], 1.5)
        self.assertLess(result['paired_population_log_ratio_SE'], 1e-15)


if __name__ == '__main__':
    unittest.main()
