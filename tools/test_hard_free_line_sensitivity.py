"""Independent-population sensitivity diagnostics; no physical draws."""
import copy
import math
import unittest
import analyze_hard_free_line_sensitivity as analysis
from test_hard_free_line_population_size import estimate


def mass(values, seedbase):
    result = estimate(values, 16384, seedbase)
    result['population_count'] = 4
    # Synthetic independent row diagnostics, not new physical estimates.
    result['row_uncertainty'].update(Qz_ESS=600., largest_Qz_fraction=.01)
    return result


def arm(allocation, seedbase):
    sample = mass([.85, 1.15, .85, 1.15], seedbase)
    estimates = {r: copy.deepcopy(sample) for r in ['total', *analysis.DECISION_CLASSES]}
    strata = {}
    for family in analysis.PLAN['stratum_families']:
        count = 64 if family == 'orthant' else 3
        strata[family] = {region: [dict(copy.deepcopy(sample), bin=i,
            observed_class_fraction=dict(Qz=1/count, Q0=1/count)) for i in range(count)]
            for region in analysis.DECISION_CLASSES}
    ratio_key = '/'.join(analysis.DECISION_CLASSES[:2])
    return dict(allocation=copy.deepcopy(allocation), estimates=estimates, strata=strata,
        populations=[dict(id=f'r{i:02}', arm=allocation['id'], seed=seedbase+i, samples=16384) for i in range(4)],
        primary_ratios={ratio_key: dict(population=dict(log_ratio=0., log_ratio_SE=.1))},
        native_entry_unbound_anomaly_count=0, native_partition_sum_verified=True)


def stages():
    binding = dict(definition='/old/definition.json', definition_sha256='definition', runtime_sha256='runtime',
        input_sha256={'shape.json': 'shape'}, criteria={'entry': 2.}, scope='unchanged classifier')
    def stage(schema, allocations, seedbase):
        return dict(schema=schema, complete=True, total_unconditional_draws=131072,
            arms={a['id']: arm(a, seedbase+4*i) for i, a in enumerate(allocations)},
            diagnostics=dict(full_vessel_gate_open=False, assembly_gate_open=False),
            native_definition=copy.deepcopy(binding), unbound_R4_bound={'log_Qz_upper': -9.})
    pilot = stage('hard-free-line-physical-comparison-v1',
        [dict(analysis.REFERENCE, id='baseline', beta=0.), analysis.REFERENCE], 100)
    current = stage(analysis.SCHEMA, analysis.CONTROLS, 200)
    current['native_definition']['definition'] = '/new/definition.json'
    return pilot, current


def shift(entry, amount, kind='Qz'):
    for p in entry['populations']:
        if p['log_'+kind] is not None:
            p['log_'+kind] += amount
    for key in ('row_uncertainty', 'population_uncertainty'):
        if entry[key]['log_'+kind] is not None:
            entry[key]['log_'+kind] += amount


class LinearComparisonTests(unittest.TestCase):
    def test_statistical_gate_uses_linear_means(self):
        a = mass([1., 1., 1., 1.], 10)
        b = mass([1.5, 1.5, 4.5, 4.5], 20)
        result = analysis.compare_linear(a, b)
        self.assertTrue(result['SE_passed'])
        self.assertFalse(result['absolute_passed'])
        self.assertGreater(abs(result['log_control_minus_reference']), 3*result['combined_population_log_delta_SE'])
        self.assertAlmostEqual(result['scaled_independent_difference_SE']*math.exp(result['log_scale']), math.sqrt(.75))

    def test_zeros_preserve_denominators_and_unobserved_is_not_zero(self):
        a, b = mass([0., 0., 0., 4.], 10), mass([0., 0., 0., 4.], 20)
        result = analysis.compare_linear(a, b)
        self.assertEqual(result['log_reference_linear_mean'], 0.)
        self.assertEqual(result['reference_population_log_masses'][:3], [None]*3)
        self.assertTrue(result['passed'])
        result = analysis.compare_linear(a, mass([0.]*4, 30))
        self.assertFalse(result['observed'])
        self.assertFalse(result['passed'])
        b['row_uncertainty']['draws'] -= 1
        with self.assertRaisesRegex(ValueError, 'denominator'):
            analysis.compare_linear(a, b)

    def test_extreme_finite_mass_remains_observed(self):
        a, b = mass([1.]*4, 10), mass([1.]*4, 20)
        shift(a, -1000.)
        result = analysis.compare_linear(a, b)
        self.assertTrue(result['observed'])
        self.assertTrue(result['shared_display_scale_underflow']['reference'])
        self.assertEqual(result['log_control_minus_reference'], 1000.)
        self.assertFalse(result['passed'])


class SensitivityTests(unittest.TestCase):
    def test_identical_scientific_bindings_can_live_at_different_paths(self):
        pilot, current = stages()
        before = copy.deepcopy((pilot, current))
        result = analysis.compare_sensitivity(pilot, current, analysis.PLAN)
        self.assertTrue(result['sensitivity_checks_passed'])
        self.assertEqual(len(result['all_stratum_comparisons']), 2*4*70)
        self.assertFalse(result['stages_pooled'])
        self.assertFalse(result['full_vessel_gate_open'])
        self.assertFalse(result['assembly_gate_open'])
        self.assertEqual(result['old_native_classifier_calls'], 0)
        self.assertEqual((pilot, current), before)
        current['native_definition']['input_sha256']['shape.json'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Classifier'):
            analysis.compare_sensitivity(pilot, current, analysis.PLAN)

    def test_direct_contrast_can_fail_when_individual_masses_pass(self):
        pilot, current = stages()
        changed = current['arms']['alpha02']
        native, competing = analysis.DECISION_CLASSES[:2]
        shift(changed['estimates'][native], .15)
        shift(changed['estimates'][competing], -.15)
        changed['primary_ratios'][native+'/'+competing]['population']['log_ratio'] = .3
        result = analysis.compare_sensitivity(pilot, current, analysis.PLAN)
        self.assertTrue(result['checks']['regional_mass_agreement'])
        self.assertFalse(result['checks']['free_energy_contrast_agreement'])
        self.assertFalse(result['sensitivity_checks_passed'])
        self.assertAlmostEqual(result['free_energy_contrast_comparisons']['alpha02']['beta_deltaF_left_minus_right'], -.3)

    def test_failed_material_stratum_retained_and_hard_checks_visible(self):
        pilot, current = stages()
        entry = current['arms']['lambda64']['strata']['orthant'][analysis.DECISION_CLASSES[1]][39]
        shift(entry, math.log(2))
        entry['observed_class_fraction']['Qz'] = .015
        shift(current['arms']['alpha02']['estimates']['total'], .5, 'Q0')
        result = analysis.compare_sensitivity(pilot, current, analysis.PLAN)
        self.assertEqual(len(result['failed_material_strata']), 1)
        failure = result['failed_material_strata'][0]
        self.assertEqual((failure['control'], failure['bin']), ('lambda64', 39))
        self.assertIn('Q0', failure['comparisons'])
        self.assertFalse(result['checks']['total_hard_mass_agreement'])
        self.assertFalse(result['checks']['material_physical_strata_agreement'])

    def test_plan_target_allocation_population_and_stream_mutations_rejected(self):
        for mutate in (
            lambda p, c: c.update(complete=False),
            lambda p, c: c['arms']['alpha02']['allocation'].update(alpha=.5),
            lambda p, c: c['arms']['alpha02']['populations'][0].update(seed=104),
            lambda p, c: c['arms']['alpha02']['estimates']['total']['populations'][0].update(seed=300),
            lambda p, c: c['arms']['lambda64']['strata']['orthant'][analysis.DECISION_CLASSES[0]].pop(),
            lambda p, c: c['diagnostics'].update(full_vessel_gate_open=True),
            lambda p, c: c.update(total_unconditional_draws=100),
            lambda p, c: c['unbound_R4_bound'].update(log_Qz_upper=3.),
            lambda p, c: c['arms']['alpha02'].update(native_partition_sum_verified=False),
        ):
            pilot, current = stages()
            mutate(pilot, current)
            with self.assertRaises(ValueError):
                analysis.compare_sensitivity(pilot, current, analysis.PLAN)
        pilot, current = stages()
        plan = copy.deepcopy(analysis.PLAN)
        plan['convergence']['log_agreement_absolute_max'] = .3
        with self.assertRaisesRegex(ValueError, 'plan changed'):
            analysis.compare_sensitivity(pilot, current, plan)

    def test_reference_or_control_quality_and_precision_failures_remain_visible(self):
        for stage, name in [('pilot', 'conditioned'), ('current', 'alpha02'), ('current', 'lambda64')]:
            pilot, current = stages()
            selected = (pilot if stage == 'pilot' else current)['arms'][name]
            selected['estimates'][analysis.DECISION_CLASSES[0]]['row_uncertainty']['Qz_ESS'] = 199.
            result = analysis.compare_sensitivity(pilot, current, analysis.PLAN)
            self.assertFalse(result['checks']['observed_regional_quality'])
            self.assertFalse(result['sensitivity_checks_passed'])
            pilot, current = stages()
            selected = (pilot if stage == 'pilot' else current)['arms'][name]
            selected['primary_ratios']['/'.join(analysis.DECISION_CLASSES[:2])]['population']['log_ratio_SE'] = .2
            result = analysis.compare_sensitivity(pilot, current, analysis.PLAN)
            self.assertFalse(result['checks']['free_energy_precision'])
            self.assertFalse(result['sensitivity_checks_passed'])


if __name__ == '__main__':
    unittest.main()
