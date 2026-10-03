"""Pure arithmetic fixtures; no saved science is read or recomputed."""
import copy
import hashlib
import json
import math
import unittest

import numpy as np
from scipy.stats import t

from contact_population_statistics import compare_population_masses, summarize_populations


def fixture(native, competing=None, *, seed_start=100, arm='example', draws=4096, logged=False):
    if competing is None:
        competing = native
    assert len(native) == len(competing)
    p = len(native)
    slots = [dict(id=f'p{i}', seed=seed_start+i) for i in range(p)]
    declaration = dict(arm_id=arm, population_count=p, draws_per_population=draws,
                       total_unconditional_draws=p*draws, regions=['native', 'competing'],
                       target_and_regions_sha256=hashlib.sha256(b'declared synthetic target and partition').hexdigest(),
                       populations=copy.deepcopy(slots))
    def log(v):
        return v if logged else None if v == 0 else math.log(v)
    records = [dict(**slot, draws=draws, unconditional_denominator=draws,
                    log_masses=dict(native=log(n), competing=log(c)))
               for slot, n, c in zip(slots, native, competing)]
    return declaration, records


class PopulationStatisticsTest(unittest.TestCase):
    def test_four_and_eight_linear_means_and_se(self):
        for p in (4, 8):
            with self.subTest(p=p):
                x = np.arange(1., p+1)
                declaration, records = fixture(x)
                result = summarize_populations(declaration, list(reversed(records)))
                estimate = result['estimates']['native']
                self.assertAlmostEqual(estimate['linear_mean']['value'], x.mean())
                self.assertAlmostEqual(estimate['linear_population_SE']['value'], x.std(ddof=1)/math.sqrt(p))
                self.assertAlmostEqual(estimate['log_linear_mean'], math.log(x.mean()))
                self.assertNotAlmostEqual(estimate['log_linear_mean'], np.log(x).mean())
                self.assertEqual(result['population_count'], p)
                self.assertEqual(result['total_unconditional_draws'], p*4096)
                self.assertNotIn('ESS', json.dumps(result))

    def test_correlated_ratio_covariance_and_t_quantile(self):
        for p in (4, 8):
            with self.subTest(p=p):
                x = np.arange(1., p+1)
                y = x + np.array([.8, -.4] * (p//2))
                result = summarize_populations(*fixture(x, y))
                fe = result['free_energy_contrast']
                cov = np.cov(np.stack([x, y]), ddof=1) / p
                variance = cov[0, 0]/x.mean()**2 + cov[1, 1]/y.mean()**2 - 2*cov[0, 1]/(x.mean()*y.mean())
                expected_se = math.sqrt(variance)
                expected_fe = math.log(y.mean()/x.mean())
                self.assertAlmostEqual(fe['population_SE'], expected_se)
                self.assertAlmostEqual(fe['beta_F_native_minus_competing'], expected_fe)
                self.assertEqual(fe['degrees_of_freedom'], p-1)
                self.assertAlmostEqual(fe['Student_t_quantile'], t.ppf(.975, p-1))
                self.assertAlmostEqual(fe['halfwidth_95'], t.ppf(.975, p-1)*expected_se)
                self.assertAlmostEqual(fe['interval_95'][0], expected_fe-fe['halfwidth_95'])
                # Keeping covariance matters: treating the regional estimates
                # as independent grossly overstates uncertainty here.
                self.assertLess(fe['population_SE'], math.sqrt(cov[0, 0]/x.mean()**2 + cov[1, 1]/y.mean()**2)/2)
                scales = result['covariance_of_population_means']['per_region_log_scales']
                recovered = np.asarray(result['covariance_of_population_means']['scaled_matrix']) * np.exp(np.add.outer(scales, scales))
                np.testing.assert_allclose(recovered, cov, atol=1e-13)

    def test_proportional_contacts_cancel_noisy_population_scale(self):
        x = [1., 2., 4., 16., 2., 8., 32., 4.]
        result = summarize_populations(*fixture(x, [4*v for v in x]))
        fe = result['free_energy_contrast']
        self.assertAlmostEqual(fe['beta_F_native_minus_competing'], math.log(4))
        self.assertLess(fe['population_SE'], 1e-15)
        self.assertGreater(result['estimates']['native']['population_relative_SE'], .3)

    def test_partial_zero_populations_keep_unconditional_denominators(self):
        result = summarize_populations(*fixture([0, 0, 2, 6], [1, 1, 2, 4]))
        e = result['estimates']['native']
        self.assertAlmostEqual(e['linear_mean']['value'], 2.)
        self.assertEqual(e['nonzero_populations'], 2)
        self.assertEqual(e['observed_zero_populations'], 2)
        self.assertTrue(e['observed_positive'])
        self.assertTrue(result['free_energy_contrast']['observed'])
        self.assertEqual(result['total_unconditional_draws'], 16384)

    def test_all_zero_mass_is_unresolved_not_precise_or_a_bound(self):
        result = summarize_populations(*fixture([0]*4, [1, 2, 3, 4]))
        e = result['estimates']['native']
        self.assertFalse(e['observed_positive'])
        self.assertEqual(e['linear_mean']['value'], 0.)
        self.assertIsNone(e['log_linear_mean'])
        self.assertIsNone(e['population_relative_SE'])
        self.assertIsNone(e['scaled_population_SE'])
        self.assertIsNone(e['linear_population_SE']['value'])
        self.assertEqual(e['observed_zero_populations'], 4)
        self.assertFalse(result['free_energy_contrast']['observed'])
        self.assertIsNone(result['free_energy_contrast']['interval_95'])
        a, ar = fixture([None, -math.inf, None, -math.inf], logged=True)
        b, br = fixture([None]*8, logged=True, seed_start=200)
        comparison = compare_population_masses(a, ar, b, br, 'native')
        self.assertFalse(comparison['passed'])
        self.assertFalse(comparison['observed'])
        self.assertIsNone(comparison['SE_passed'])
        json.dumps(result, allow_nan=False)
        json.dumps(comparison, allow_nan=False)

    def test_extreme_absolute_log_scales_preserve_finite_log_ratio(self):
        for offset in (-1000., 1000.):
            a = [offset+math.log(v) for v in (1, 2, 3, 4)]
            b = [v+math.log(2) for v in a]
            result = summarize_populations(*fixture(a, b, logged=True))
            self.assertAlmostEqual(result['estimates']['native']['log_linear_mean'], offset+math.log(2.5))
            self.assertIsNone(result['estimates']['native']['linear_mean']['value'])
            self.assertAlmostEqual(result['free_energy_contrast']['beta_F_native_minus_competing'], math.log(2))
            self.assertLess(result['free_energy_contrast']['population_SE'], 1e-13)
            json.dumps(result, allow_nan=False)
        # Different column scales must not erase the tiny-but-observed class.
        result = summarize_populations(*fixture([1000.]*4, [-1000.]*4, logged=True))
        self.assertTrue(result['free_energy_contrast']['observed'])
        self.assertEqual(result['free_energy_contrast']['beta_F_native_minus_competing'], -2000.)

    def test_independent_cross_arm_unequal_population_counts(self):
        x, y = np.array([1., 2., 3., 4.]), np.array([1., 2., 3., 4., 2., 3., 4., 2.])
        a, ar = fixture(x)
        b, br = fixture(y, seed_start=200, draws=8192)
        out = compare_population_masses(a, ar, b, br, 'native')
        scale = math.exp(out['log_scale'])
        expected = math.hypot(x.std(ddof=1)/math.sqrt(4), y.std(ddof=1)/math.sqrt(8))
        self.assertAlmostEqual(out['scaled_independent_difference_SE']*scale, expected)
        self.assertAlmostEqual(out['scaled_linear_difference']*scale, x.mean()-y.mean())
        self.assertAlmostEqual(out['difference_in_combined_SE'], (x.mean()-y.mean())/expected)
        self.assertEqual(out['left_population_count'], 4)
        self.assertEqual(out['right_population_count'], 8)
        self.assertTrue(out['passed'])

    def test_independent_comparison_uses_linear_not_log_se_gate(self):
        # Fixed synthetic values where the delta log-SE and linear tests differ.
        a, ar = fixture([1.]*4)
        b, br = fixture([1.041, 1.041, 1.159, 1.159], seed_start=200)
        out = compare_population_masses(a, ar, b, br, 'native')
        self.assertEqual(out['SE_passed'], abs(out['scaled_linear_difference']) <= 3*out['scaled_independent_difference_SE'] + out['scaled_comparison_roundoff_tolerance'])
        # General regression of the formula rather than rounded stored verdicts.
        self.assertEqual(out['passed'], out['SE_passed'] and out['absolute_passed'])
        self.assertTrue(out['SE_passed'])
        self.assertFalse(abs(out['log_left_minus_right']) <= 3*out['combined_population_log_delta_SE'])

    def test_constants_agree_or_fail_despite_zero_observed_se(self):
        a, ar = fixture([1]*4)
        b, br = fixture([1]*8, seed_start=200)
        out = compare_population_masses(a, ar, b, br, 'native')
        self.assertTrue(out['passed'])
        self.assertEqual(out['difference_in_combined_SE'], 0.)
        b, br = fixture([1.01]*8, seed_start=200)
        out = compare_population_masses(a, ar, b, br, 'native')
        self.assertFalse(out['passed'])
        self.assertFalse(out['SE_passed'])
        self.assertTrue(out['absolute_passed'])
        self.assertTrue(out['zero_variance_disagreement'])
        self.assertIsNone(out['difference_in_combined_SE'])
        json.dumps(out, allow_nan=False)

    def test_absolute_gate_is_required_even_for_high_variance(self):
        a, ar = fixture([0, 0, 0, 4])
        b, br = fixture([0, 0, 0, 8], seed_start=200)
        out = compare_population_masses(a, ar, b, br, 'native')
        self.assertTrue(out['SE_passed'])
        self.assertFalse(out['absolute_passed'])
        self.assertFalse(out['passed'])

    def test_shared_scale_underflow_does_not_convert_positive_to_unobserved(self):
        a, ar = fixture([1000.]*4, logged=True)
        b, br = fixture([-1000.+math.log(v) for v in (1, 2, 3, 4)], logged=True, seed_start=200)
        out = compare_population_masses(a, ar, b, br, 'native')
        self.assertTrue(out['observed'])
        self.assertTrue(out['shared_display_scale_underflow']['right'])
        self.assertFalse(out['passed'])
        self.assertAlmostEqual(out['log_left_minus_right'], 2000.-math.log(2.5))
        self.assertEqual(out['right']['nonzero_populations'], 4)
        json.dumps(out, allow_nan=False)
        a, ar = fixture([0.]*4, logged=True)
        b, br = fixture([-743., -744., -744., -745.], logged=True, seed_start=200)
        out = compare_population_masses(a, ar, b, br, 'native')
        self.assertTrue(out['standard_error_ratio_overflow'])
        json.dumps(out, allow_nan=False)

    def test_declaration_identity_denominator_and_missing_population_checks(self):
        for mutation in ('count', 'missing', 'duplicate', 'wrongid', 'seed', 'repeatseed',
                         'draws', 'denominator', 'total', 'missingregion', 'zerodraws'):
            with self.subTest(mutation=mutation):
                a, rows = fixture([1, 2, 3, 4])
                if mutation == 'count': a['population_count'] = 8
                if mutation == 'missing': rows.pop()
                if mutation == 'duplicate': rows[1]['id'] = rows[0]['id']
                if mutation == 'wrongid': rows[1]['id'] = 'undeclared'
                if mutation == 'seed': rows[1]['seed'] += 1000
                if mutation == 'repeatseed': a['populations'][1]['seed'] = a['populations'][0]['seed']
                if mutation == 'draws': rows[1]['draws'] -= 1
                if mutation == 'denominator': rows[1]['unconditional_denominator'] -= 1
                if mutation == 'total': a['total_unconditional_draws'] -= 1
                if mutation == 'missingregion': del rows[1]['log_masses']['native']
                if mutation == 'zerodraws': a['draws_per_population'] = 0
                with self.assertRaises(ValueError): summarize_populations(a, rows)

    def test_reject_nan_inf_and_boolean_counts_or_logs(self):
        for value in (math.nan, math.inf, True, '0'):
            with self.subTest(log=value):
                a, rows = fixture([1, 2, 3, 4])
                rows[0]['log_masses']['native'] = value
                with self.assertRaises(ValueError): summarize_populations(a, rows)
        for name in ('seed', 'draws', 'unconditional_denominator'):
            with self.subTest(field=name):
                a, rows = fixture([1, 2, 3, 4])
                rows[0][name] = True
                with self.assertRaises(ValueError): summarize_populations(a, rows)
        for name in ('population_count', 'draws_per_population', 'total_unconditional_draws'):
            a, rows = fixture([1, 2, 3, 4])
            a[name] = True
            with self.assertRaises(ValueError): summarize_populations(a, rows)

    def test_cross_arm_seed_reuse_rejected_even_for_zero_mass(self):
        for values in ([1, 2, 3, 4], [0]*4):
            a, ar = fixture(values)
            b, br = fixture(values, arm='other')
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                compare_population_masses(a, ar, b, br, 'native')

    def test_scope_identity_must_match_and_be_a_sha256(self):
        a, ar = fixture([1, 2, 3, 4])
        b, br = fixture([1, 2, 3, 4], seed_start=200)
        out = compare_population_masses(a, ar, b, br, 'native')
        self.assertEqual(out['target_and_regions_sha256'], a['target_and_regions_sha256'])
        self.assertEqual(summarize_populations(a, ar)['target_and_regions_sha256'], a['target_and_regions_sha256'])
        b['target_and_regions_sha256'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'target or complete region definitions differ'):
            compare_population_masses(a, ar, b, br, 'native')
        for value in ('not a hash', 'A'*64, 'a'*63, None, True):
            a['target_and_regions_sha256'] = value
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                summarize_populations(a, ar)

    def test_partition_means_and_covariance_sum_under_arbitrary_ordering(self):
        # Explicitly declared additive columns are a fixture property, not
        # region semantics imposed by the general statistics API.
        leaves = dict(
            native_old=np.array([1., 0., 2., 4., 3., 0., 5., 1.]),
            native_remainder=np.array([0., 1., 1., 0., 2., 3., 0., 4.]),
            competing=np.array([6., 5., 2., 1., 4., 2., 1., 3.]),
            unbound=np.array([20., 18., 25., 22., 19., 23., 21., 17.]))
        values = dict(leaves, native=leaves['native_old'] + leaves['native_remainder'],
                      total=sum(leaves.values()))
        order = ['total', 'unbound', 'native_remainder', 'competing', 'native', 'native_old']
        for p, region_order in ((4, order), (8, list(reversed(order)))):
            with self.subTest(populations=p, order=region_order):
                declaration, records = fixture(values['native'][:p], values['competing'][:p])
                declaration['regions'] = region_order
                declaration['target_and_regions_sha256'] = hashlib.sha256(b'synthetic six-region partition').hexdigest()
                for i, row in enumerate(records):
                    row['log_masses'] = {name: None if vector[i] == 0 else math.log(vector[i])
                                         for name, vector in values.items()}
                result = summarize_populations(declaration, list(reversed(records)))
                self.assertEqual(result['region_order'], region_order)
                means = {name: result['estimates'][name]['linear_mean']['value'] for name in region_order}
                for name in region_order:
                    self.assertAlmostEqual(means[name], values[name][:p].mean())
                self.assertAlmostEqual(means['native'], means['native_old'] + means['native_remainder'])
                self.assertAlmostEqual(means['total'], means['native'] + means['competing'] + means['unbound'])
                cov_report = result['covariance_of_population_means']
                scales = cov_report['per_region_log_scales']
                covariance = np.asarray(cov_report['scaled_matrix']) * np.exp(np.add.outer(scales, scales))
                expected = np.cov(np.stack([values[name][:p] for name in region_order]), ddof=1) / p
                np.testing.assert_allclose(covariance, expected, rtol=1e-12, atol=1e-13)
                for aggregate, parts in [('native', ['native_old', 'native_remainder']),
                                         ('total', ['native', 'competing', 'unbound']),
                                         ('total', list(leaves))]:
                    ai = region_order.index(aggregate)
                    indices = [region_order.index(name) for name in parts]
                    np.testing.assert_allclose(covariance[ai], covariance[indices].sum(axis=0), atol=1e-13)
                    self.assertAlmostEqual(covariance[ai, ai], covariance[np.ix_(indices, indices)].sum())
                    reported_se = result['estimates'][aggregate]['linear_population_SE']['value']
                    self.assertAlmostEqual(reported_se**2, covariance[np.ix_(indices, indices)].sum())


if __name__ == '__main__': unittest.main()
