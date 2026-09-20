#!/usr/bin/env python3
"""Exact finite-state controls for the actual fixed-quota MIS helpers.

These deterministic fixtures exercise the production moment implementation;
the independent Fraction derivation lives in test_deterministic_mis_math.py.
No protein poses or Poisson clouds are generated.
"""
from fractions import Fraction as F
from itertools import product
import math
import unittest

import numpy as np

from shoulder_mis import (LogMoments, log_difference_square,
                          mixture_log_density, quota_summary,
                          signed_quota_difference)
from test_deterministic_mis_math import mixture_laws, sample_variance


def log_value(value):
    return math.log(float(value)) if value else -math.inf


def logged_moments(values, cloud_pairs=None):
    moments = LogMoments()
    if cloud_pairs is None:
        cloud_pairs = [None]*len(values)
    for value, pair in zip(values, cloud_pairs):
        moments.add(log_value(value), None if pair is None else list(map(log_value, pair)))
    return moments


def reported_variance(summary, field='log_variance_of_mean'):
    return math.exp(summary[field]) if summary[field] is not None else 0.


class ShoulderMisMomentsTests(unittest.TestCase):
    def test_actual_helpers_against_every_equal_quota_pose_cloud_outcome(self):
        """All 4096 outcomes, including within-source constant realizations."""
        _, laws = mixture_laws()
        target = (F(2), F(1))
        guide = (F(3, 4), F(1, 4))
        totals = dict(probability=0., mean=0., variance=0., estimated_variance=0.,
                      cloud_variance=0., difference_mean=0., difference_variance=0.,
                      estimated_difference_variance=0.)
        for indexed in product(enumerate(laws[0]), enumerate(laws[0]),
                               enumerate(laws[1]), enumerate(laws[1])):
            rows = [row for _, row in indexed]
            probability = float(math.prod(row[0] for row in rows))
            values = [row[1] for row in rows]
            pairs = [(row[2], row[3]) for row in rows]
            strata = [logged_moments(values[:2], pairs[:2]),
                      logged_moments(values[2:], pairs[2:])]
            result = quota_summary(strata)
            expected_mean = sum(values)/4
            expected_variance = (sample_variance(values[:2])+sample_variance(values[2:]))/8
            expected_cloud = sum((a-b)**2/4 for a, b in pairs)/16
            actual_mean = math.exp(result['logQ'])
            actual_variance = reported_variance(result)
            actual_cloud = sum(math.exp(s.noise) for s in strata)/16
            self.assertAlmostEqual(actual_mean, float(expected_mean), delta=3e-13)
            self.assertAlmostEqual(actual_variance, float(expected_variance), delta=3e-13)
            self.assertAlmostEqual(actual_cloud, float(expected_cloud), delta=3e-13)
            if expected_variance > 0 and expected_cloud > 0:
                self.assertAlmostEqual(result['paired_cloud_variance_fraction'],
                                       float(expected_cloud/expected_variance), delta=2e-11)

            # The source control uses its own denominator and the same clouds.
            # For the equal mixture Z=f*cloudmean/(1/2); state index is exact.
            own_guide = []
            for (index, _), z in zip(indexed[:2], values[:2]):
                state = index//4
                cloud_mean = z/(2*target[state])
                own_guide.append(target[state]*cloud_mean/guide[state])
            h_negative = [2*u-z for u, z in zip(own_guide, values[:2])]
            self.assertTrue(all(h >= 0 for h in h_negative))
            expected_difference = expected_mean-sum(own_guide)/2
            expected_difference_variance = (sample_variance(h_negative)+sample_variance(values[2:]))/8
            paired = signed_quota_difference(logged_moments(h_negative), logged_moments(values[2:]),
                                             log_value(sum(own_guide)/2))
            actual_difference = (paired['sign']*math.exp(paired['log_absolute_difference'])
                                 if paired['sign'] else 0.)
            actual_difference_variance = reported_variance(paired, 'log_variance_of_difference')
            self.assertAlmostEqual(actual_difference, float(expected_difference), delta=3e-13)
            self.assertAlmostEqual(actual_difference_variance,
                                   float(expected_difference_variance), delta=3e-13)
            self.assertAlmostEqual(paired['relative_difference_to_component'],
                                   float(expected_difference/(sum(own_guide)/2)), delta=3e-13)

            totals['probability'] += probability
            totals['mean'] += probability*actual_mean
            totals['variance'] += probability*(actual_mean-3)**2
            totals['estimated_variance'] += probability*actual_variance
            totals['cloud_variance'] += probability*actual_cloud
            totals['difference_mean'] += probability*actual_difference
            totals['difference_variance'] += probability*actual_difference**2
            totals['estimated_difference_variance'] += probability*actual_difference_variance
        for key, expected in dict(probability=1., mean=3., variance=.5,
                                  estimated_variance=.5, cloud_variance=5/16,
                                  difference_mean=0., difference_variance=7/8,
                                  estimated_difference_variance=7/8).items():
            self.assertAlmostEqual(totals[key], expected, delta=2e-12, msg=key)

    def test_actual_unequal_quota_density_and_known_source_means(self):
        density, laws = mixture_laws(counts=(2, 3))
        result = mixture_log_density(np.log([.75, .25]), np.log([.25, .75]), 2, 3)
        np.testing.assert_allclose(np.exp(result), [float(d) for d in density], rtol=2e-15)
        means = [sum(p*z for p, z, _, _ in law) for law in laws]
        self.assertEqual(means, [F(125, 33), F(245, 99)])
        self.assertAlmostEqual(.4*float(means[0])+.6*float(means[1]), 3.)
        result = quota_summary([logged_moments([1, 4]), logged_moments([0, 2, 8])])
        self.assertEqual(result['draws'], 5)
        self.assertAlmostEqual(math.exp(result['logQ']), 3.)
        expected_variance = (2*sample_variance([F(1), F(4)])
                             +3*sample_variance([F(0), F(2), F(8)]))/25
        self.assertAlmostEqual(reported_variance(result), float(expected_variance))

    def test_disjoint_support_fixed_quotas_have_zero_variance(self):
        summary = quota_summary([logged_moments([4, 4]), logged_moments([2, 2])])
        self.assertAlmostEqual(math.exp(summary['logQ']), 3.)
        self.assertLess(reported_variance(summary), 2e-14)
        self.assertLess(summary['stratified_RSE'], 1e-7)
        self.assertAlmostEqual(summary['weight_ESS'], 3.6)
        # Pooled iid s^2/N=1/3 is inappropriate for these enforced quotas.
        self.assertEqual(sample_variance([F(4), F(4), F(2), F(2)])/4, F(1, 3))
        density = mixture_log_density([0., -math.inf], [-math.inf, 0.], 2, 2)
        np.testing.assert_allclose(density, [math.log(.5), math.log(.5)])

    def test_identical_proposals_reduce_to_ordinary_importance_weights(self):
        log_density = np.log([.5, .5])
        np.testing.assert_allclose(mixture_log_density(log_density, log_density, 2, 3),
                                   log_density, rtol=2e-15)
        rows = ([F(4), F(2)], [F(2), F(4), F(4)])
        result = quota_summary([logged_moments(values) for values in rows])
        self.assertAlmostEqual(math.exp(result['logQ']), 16/5)
        expected_variance = sum(len(v)*sample_variance(v) for v in rows)/25
        self.assertAlmostEqual(reported_variance(result), float(expected_variance))
        # Even identical laws do not force the two realized variance estimators
        # (fixed quotas and pooled iid) to coincide sample by sample.
        self.assertNotEqual(expected_variance, sample_variance([*rows[0], *rows[1]])/5)

    def test_implicit_zero_padding_matches_every_explicit_unconditional_zero(self):
        explicit = [logged_moments([0, 3, 0, 0, 7]), logged_moments([0, 0, 0])]
        padded = [logged_moments([3, 7]), LogMoments()]
        padded[0].count = 5
        padded[1].count = 3
        self.assertEqual(quota_summary(explicit), quota_summary(padded))
        summary = quota_summary(padded)
        self.assertEqual(summary['draws'], 8)
        self.assertEqual(summary['nonzero'], 2)
        self.assertAlmostEqual(math.exp(summary['logQ']), 10/8)
        self.assertNotAlmostEqual(math.exp(summary['logQ']), 10/2)
        all_zero = quota_summary([logged_moments([0, 0]), logged_moments([0, 0, 0])])
        self.assertEqual(all_zero['draws'], 5)
        self.assertIsNone(all_zero['logQ'])
        self.assertIsNone(all_zero['stratified_RSE'])
        self.assertIn('not a mass upper bound', all_zero['coverage'])

    def test_log_scale_invariance_and_merging_source_specific_moments(self):
        source_values = ([0, 1, 4, 0, 2, 7], [3, 0, 0, 5, 1, 0])
        baseline = quota_summary([logged_moments(v) for v in source_values])
        merged = [logged_moments(v[:3]).merge(logged_moments(v[3:])) for v in source_values]
        for key in ('logQ', 'log_variance_of_mean', 'weight_ESS', 'stratified_RSE'):
            self.assertAlmostEqual(quota_summary(merged)[key], baseline[key], delta=2e-13)
        for shift in (-700., 700.):
            strata = []
            for values in source_values:
                moment = LogMoments()
                for value in values:
                    moment.add(log_value(value)+shift)
                strata.append(moment)
            result = quota_summary(strata)
            self.assertAlmostEqual(result['logQ'], baseline['logQ']+shift, delta=3e-12)
            self.assertAlmostEqual(result['log_variance_of_mean'],
                                   baseline['log_variance_of_mean']+2*shift, delta=3e-12)
            self.assertAlmostEqual(result['stratified_RSE'], baseline['stratified_RSE'], delta=3e-12)
            self.assertAlmostEqual(result['weight_ESS'], baseline['weight_ESS'], delta=3e-11)

    def test_log_difference_and_invalid_quota_guards(self):
        self.assertEqual(log_difference_square(-math.inf, -math.inf), -math.inf)
        self.assertEqual(log_difference_square(700., -math.inf), 1400.)
        self.assertAlmostEqual(log_difference_square(math.log(3), math.log(2)), 0.)
        # 1-exp(-delta)=delta+O(delta^2), checked after a huge common scale.
        delta = 700.-(700.-1e-10)
        self.assertAlmostEqual(log_difference_square(700., 700.-1e-10),
                               1400.+2*math.log(delta), delta=2e-9)
        for counts in ((0, 2), (2, 0), (True, 2), (2., 2)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                mixture_log_density([0.], [0.], *counts)
        for strata in ([], [LogMoments()], [logged_moments([1])]):
            with self.assertRaises(ValueError):
                quota_summary(strata)


if __name__ == '__main__':
    unittest.main()
