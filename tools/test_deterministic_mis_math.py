#!/usr/bin/env python3
"""Exact finite-state MIS fixtures; no random draws or production kernels.

The positive two-point auxiliary factor replaces a Poisson estimator solely
to make every pose/cloud realization enumerable. Its conditional mean is one,
so it tests the same importance and variance identities independently.
"""
from fractions import Fraction as F
from itertools import product
import math
import unittest


def mixture_laws(counts=(2, 2), proposals=((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))),
                 target=(F(2), F(1)), cloud_factors=(F(1, 2), F(3, 2))):
    """Return exact unconditional one-draw laws for two independent clouds."""
    total = sum(counts)
    assert len(counts) == len(proposals) and all(n > 0 for n in counts)
    assert all(sum(g) == 1 and all(p >= 0 for p in g) for g in proposals)
    assert sum(cloud_factors)/len(cloud_factors) == 1
    density = tuple(sum(F(n, total)*g[x] for n, g in zip(counts, proposals)) for x in range(len(target)))
    if any(f > 0 and density[x] == 0 for x, f in enumerate(target)):
        raise ValueError('Mixture misses positive target support')
    laws = []
    for proposal in proposals:
        outcomes = []
        for x, probability in enumerate(proposal):
            if not probability:
                continue
            for a, b in product(cloud_factors, repeat=2):
                z1, z2 = target[x]*a/density[x], target[x]*b/density[x]
                outcomes.append((probability/len(cloud_factors)**2, (z1+z2)/2, z1, z2))
        assert sum(row[0] for row in outcomes) == 1
        laws.append(outcomes)
    return density, laws


def sample_variance(values):
    mean = sum(values)/len(values)
    return sum((x-mean)**2 for x in values)/(len(values)-1)


def enumerate_equal_quota_fixture():
    """All 8^4 pose/cloud outcomes: exact expected uncertainty estimators."""
    counts = (2, 2); n = sum(counts)
    _, laws = mixture_laws(counts)
    totals = dict(probability=F(0), mean=F(0), variance=F(0), estimated_fixed_variance=F(0),
                  estimated_iid_variance=F(0), estimated_cloud_variance=F(0))
    for rows in product(laws[0], laws[0], laws[1], laws[1]):
        probability = math.prod(row[0] for row in rows)
        values = [row[1] for row in rows]; mean = sum(values)/n
        fixed = sum(F(size, n*n)*sample_variance(values[start:start+size])
                    for start, size in ((0, 2), (2, 2)))
        iid = sample_variance(values)/n
        cloud = sum((row[2]-row[3])**2/4 for row in rows)/(n*n)
        totals['probability'] += probability
        totals['mean'] += probability*mean
        totals['variance'] += probability*(mean-3)**2
        totals['estimated_fixed_variance'] += probability*fixed
        totals['estimated_iid_variance'] += probability*iid
        totals['estimated_cloud_variance'] += probability*cloud
    return totals


class DeterministicMixtureMathTests(unittest.TestCase):
    def test_complete_pose_cloud_enumeration_and_fixed_quota_variance(self):
        observed = enumerate_equal_quota_fixture()
        self.assertEqual(observed['probability'], 1)
        self.assertEqual(observed['mean'], 3)
        self.assertEqual(observed['variance'], F(1, 2))
        self.assertEqual(observed['estimated_fixed_variance'], F(1, 2))
        self.assertEqual(observed['estimated_iid_variance'], F(7, 12))
        self.assertEqual(observed['estimated_cloud_variance'], F(5, 16))
        self.assertEqual(observed['variance']-observed['estimated_cloud_variance'], F(3, 16))

    def test_unequal_quota_denominator_uses_actual_allocation(self):
        density, laws = mixture_laws(counts=(2, 3))
        self.assertEqual(density, (F(9, 20), F(11, 20)))
        means = [sum(p*z for p, z, _, _ in law) for law in laws]
        self.assertEqual(means, [F(125, 33), F(245, 99)])
        self.assertEqual(F(2, 5)*means[0]+F(3, 5)*means[1], 3)
        # Replacing the true allocation by an equal mixture is biased here.
        wrong_equal_density_mean = F(9, 20)*4+F(11, 20)*2
        self.assertEqual(wrong_equal_density_mean, F(29, 10))
        self.assertNotEqual(wrong_equal_density_mean, 3)

    def test_disjoint_support_can_be_exact_despite_nonzero_pooled_iid_error(self):
        _, laws = mixture_laws(proposals=((F(1), F(0)), (F(0), F(1))), cloud_factors=(F(1),))
        self.assertEqual(laws[0][0][1], 4)
        self.assertEqual(laws[1][0][1], 2)
        values = [F(4), F(4), F(2), F(2)]
        self.assertEqual(sum(values)/4, 3)
        self.assertEqual(sample_variance(values[:2])+sample_variance(values[2:]), 0)
        self.assertEqual(sample_variance(values)/4, F(1, 3))

    def test_target_zeros_remain_unconditional_and_missing_support_is_rejected(self):
        _, laws = mixture_laws(target=(F(2), F(0)))
        means = [sum(p*z for p, z, _, _ in law) for law in laws]
        self.assertEqual(sum(means)/2, 2)
        self.assertTrue(any(z == 0 and p > 0 for law in laws for p, z, _, _ in law))
        with self.assertRaises(ValueError):
            mixture_laws(proposals=((F(1), F(0)), (F(1), F(0))))

    def test_mis_and_own_component_control_have_paired_covariance(self):
        density = (F(1, 2), F(1, 2)); target = (F(2), F(1))
        guide = (F(3, 4), F(1, 4)); observations = []
        for state, probability in enumerate(guide):
            for a, b in product((F(1, 2), F(3, 2)), repeat=2):
                mean_factor = (a+b)/2
                z = target[state]*mean_factor/density[state]
                own = target[state]*mean_factor/guide[state]
                observations.append((probability/4, z, own))
        z_mean = sum(p*z for p, z, u in observations)
        own_mean = sum(p*u for p, z, u in observations)
        within_covariance = sum(p*(z-z_mean)*(u-own_mean) for p, z, u in observations)
        own_draw_variance = sum(p*(u-own_mean)**2 for p, z, u in observations)
        self.assertEqual(within_covariance, F(3, 4))
        self.assertEqual(own_draw_variance, F(3, 2))
        # N=4, n_G=2. MIS and own-guide estimates reuse the same two draws.
        covariance_of_estimators = within_covariance/4
        variance_of_own = own_draw_variance/2
        difference_variance = F(1, 2)+variance_of_own-2*covariance_of_estimators
        self.assertEqual(covariance_of_estimators, F(3, 16))
        self.assertEqual(difference_variance, F(7, 8))
        self.assertNotEqual(difference_variance, F(1, 2)+variance_of_own)


if __name__ == '__main__':
    unittest.main()
