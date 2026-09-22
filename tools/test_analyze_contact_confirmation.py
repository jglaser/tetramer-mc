"""Protect one-pass classification, exact native partition, and all control gates."""
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import analyze_contact_confirmation as analyzer


def protocol():
    return {'decision_arms': list(analyzer.MAIN_ARMS),
        'comparisons': [['bank', name] for name in analyzer.ARM_NAMES if name != 'bank'],
        'convergence': dict(population_relative_SE_max=.1, importance_ESS_min=200., largest_draw_max=.02,
            log_agreement_absolute_max=.2, log_agreement_combined_SE_max=3., deltaF_95_halfwidth_max=.5,
            significant_stratum_mass_fraction=.01, stratum_log_agreement_absolute_max=.2)}


def estimate(logz=60., logh=-12.):
    row = dict(log_Qz=logz, log_Q0=logh, Qz_relative_SE=.05, Q0_relative_SE=.05,
        Qz_ESS=1000., largest_Qz_fraction=.005)
    return dict(row_uncertainty=copy.deepcopy(row), population_uncertainty=copy.deepcopy(row), population_count=4)


def passing_arms():
    result = {}
    for arm in analyzer.ARM_NAMES:
        estimates = {name: estimate() for name in analyzer.CLASSES}
        estimates[analyzer.PRIMARY[1]] = estimate(42.)
        for name in analyzer.PARTS:
            estimates[name] = estimate(60. - math.log(2))
        strata = {family: {name: [dict(estimate(estimates[name]['row_uncertainty']['log_Qz']),
            bin=0, observed_class_fraction={'Qz': .1, 'Q0': .1})] for name in analyzer.DECISION_CLASSES}
            for family in ('radial', 'angular', 'orthant')}
        result[arm] = dict(estimates=estimates, strata=strata,
            primary_ratios={analyzer.PRIMARY[0] + '/' + analyzer.PRIMARY[1]: {'population': {'log_ratio': 18., 'log_ratio_SE': .1}}},
            native_entry_unbound_anomaly_count=0, native_partition_sum_verified=True)
    return result


def classified_fixture():
    n = 5
    arrays = dict(z=np.array([math.log(2), math.log(3), math.log(7), -np.inf, -np.inf]),
        h=np.array([0., 0., 0., -np.inf, -np.inf]), support=np.array([True, True, True, True, False]),
        branch=np.array([0, 1, 1, 0, 1]), component=np.array([-1, 0, 1, -1, 2]))
    arrays['pairs'] = np.column_stack((arrays['z'], arrays['z']))
    rows = [dict(draw=i, q=1. if i == 1 else 2., capture_valid=True,
        pose={'id': i, 'position': [0., 0., 0.], 'orientation': [1., 0., 0., 0.]}) for i in range(n)]
    coordinates = np.zeros((n, 6)); coordinates[:, 0] = [5., 5., 6., 1., 7.]
    class Native:
        def __init__(self): self.calls = []
        def classify(self, pose):
            self.calls.append(pose['id']); native = pose['id'] < 2
            return dict(native_any=native, native_anchor_count=int(native), registry_consistent_triangle=False)
    class Contact:
        def __init__(self): self.calls = []
        def classify(self, pose):
            self.calls.append(pose['id'])
            return dict(exclusion_contact=True, near_zero_negative_gap=False)
    native, contact, stream = Native(), Contact(), io.BytesIO()
    with patch.object(analyzer, 'chart_coordinates', return_value=coordinates):
        classified, diagnostics = analyzer.classify_rows_once(rows, arrays, native, contact, {}, stream)
    arrays.update(classified)
    arrays.update(u=np.zeros((n, 6)), log_q=np.zeros(n), draw=np.arange(n), source_n=np.full(n, n),
        bin_radial=np.array([0, 1, 2, 0, -1]), bin_angular=np.array([0, 1, 2, 0, -1]),
        bin_orthant=np.array([0, 1, 2, 0, -1]))
    return arrays, diagnostics, native, contact, stream


class ContactConfirmationTests(unittest.TestCase):
    def test_five_controls_and_four_decision_regions_pass_only_declared_gates(self):
        result = analyzer.evaluate(passing_arms(), protocol())
        self.assertTrue(result['confirmation_passed'])
        self.assertFalse(result['passed'])
        self.assertEqual(set(result['comparisons']), {'wide', 'small', 'alpha02', 'intensity256'})
        self.assertEqual(set(result['quality']['bank']), set(analyzer.DECISION_CLASSES))
        self.assertFalse(result['full_wall_coverage_established'])
        self.assertFalse(result['assembly_stability_established'])

    def test_direct_contrast_catches_compensating_subthreshold_mass_shifts(self):
        arms = passing_arms()
        for level in ('row_uncertainty', 'population_uncertainty'):
            arms['wide']['estimates'][analyzer.NATIVE][level]['log_Qz'] += .15
            arms['wide']['estimates'][analyzer.PRIMARY[1]][level]['log_Qz'] -= .15
        arms['wide']['primary_ratios'][analyzer.PRIMARY[0] + '/' + analyzer.PRIMARY[1]]['population']['log_ratio'] += .30
        result = analyzer.evaluate(arms, protocol())
        self.assertTrue(result['comparisons']['wide']['region_agreement'])
        self.assertFalse(result['comparisons']['wide']['free_energy_contrast']['passed'])
        self.assertFalse(result['confirmation_passed'])

    def test_native_complement_disagreement_cannot_hide_in_full_native(self):
        arms = passing_arms()
        for level in ('row_uncertainty', 'population_uncertainty'):
            arms['alpha02']['estimates'][analyzer.PARTS[1]][level]['log_Qz'] += .3
        result = analyzer.evaluate(arms, protocol())
        self.assertTrue(result['comparisons']['alpha02']['masses'][analyzer.NATIVE]['Qz']['passed'])
        self.assertFalse(result['comparisons']['alpha02']['masses'][analyzer.PARTS[1]]['Qz']['passed'])
        self.assertFalse(result['confirmation_passed'])

    def test_strict_Q0_and_combined_population_SE_limits_both_apply(self):
        arms = passing_arms()
        arms['intensity256']['estimates'][analyzer.PARTS[0]]['population_uncertainty']['log_Q0'] += .21
        result = analyzer.evaluate(arms, protocol())
        self.assertFalse(result['comparisons']['intensity256']['masses'][analyzer.PARTS[0]]['Q0']['absolute_passed'])
        arms = passing_arms()
        for arm in ('bank', 'wide'):
            arms[arm]['estimates'][analyzer.PARTS[1]]['population_uncertainty']['Qz_relative_SE'] = .001
        arms['wide']['estimates'][analyzer.PARTS[1]]['population_uncertainty']['log_Qz'] += .05
        check = analyzer.evaluate(arms, protocol())['comparisons']['wide']['masses'][analyzer.PARTS[1]]['Qz']
        self.assertTrue(check['absolute_passed'])
        self.assertFalse(check['SE_passed'])

    def test_small_standalone_precision_diagnostic_but_agreement_still_mandatory(self):
        arms = passing_arms()
        for name in analyzer.DECISION_CLASSES:
            arms['small']['estimates'][name]['row_uncertainty'].update(Qz_ESS=1., largest_Qz_fraction=1.)
            arms['small']['estimates'][name]['population_uncertainty']['Qz_relative_SE'] = 1.
        arms['small']['primary_ratios'][analyzer.PRIMARY[0] + '/' + analyzer.PRIMARY[1]]['population']['log_ratio_SE'] = 1.
        result = analyzer.evaluate(arms, protocol())
        self.assertFalse(result['quality']['small'][analyzer.NATIVE]['passed'])
        self.assertFalse(result['free_energy_intervals']['small']['passed'])
        self.assertTrue(result['confirmation_passed'])
        arms['small']['estimates'][analyzer.PARTS[1]]['population_uncertainty']['log_Qz'] += .3
        self.assertFalse(analyzer.evaluate(arms, protocol())['confirmation_passed'])

    def test_all_main_arms_require_precision_for_both_native_subsets(self):
        for arm in analyzer.MAIN_ARMS:
            arms = passing_arms()
            arms[arm]['estimates'][analyzer.PARTS[1]]['row_uncertainty']['Qz_ESS'] = 199.
            self.assertFalse(analyzer.evaluate(arms, protocol())['confirmation_passed'])

    def test_original_significant_orthant_bins_are_checked_for_complement(self):
        arms = passing_arms()
        entry = arms['wide']['strata']['orthant'][analyzer.PARTS[1]][0]
        entry['population_uncertainty']['log_Qz'] += .5
        result = analyzer.evaluate(arms, protocol())
        self.assertFalse(result['checks']['significant_original_strata_agreement'])
        self.assertEqual(result['significant_stratum_disagreements'][0]['region'], analyzer.PARTS[1])

    def test_missing_mass_is_unresolved_and_missing_control_arm_rejected(self):
        arms = passing_arms()
        for level in ('row_uncertainty', 'population_uncertainty'):
            arms['bank']['estimates'][analyzer.PARTS[1]][level]['log_Qz'] = None
        result = analyzer.evaluate(arms, protocol())
        self.assertFalse(result['confirmation_passed'])
        self.assertFalse(result['quality']['bank'][analyzer.PARTS[1]]['observed'])
        arms.pop('small')
        with self.assertRaisesRegex(ValueError, 'Five frozen'):
            analyzer.evaluate(arms, protocol())

    def test_single_native_and_contact_call_per_valid_pose_with_same_pass_partition(self):
        arrays, diagnostics, native, contact, stream = classified_fixture()
        self.assertEqual(native.calls, [0, 1, 2]); self.assertEqual(contact.calls, [0, 1, 2])
        self.assertEqual(diagnostics['full_native_classifier_calls'], 3)
        np.testing.assert_array_equal(arrays[analyzer.PARTS[0]], [True, False, False, False, False])
        np.testing.assert_array_equal(arrays[analyzer.PARTS[1]], [False, True, False, False, False])
        np.testing.assert_array_equal(arrays['class_id'], [0, 1, 2, -2, -2])
        labels = [json.loads(line) for line in stream.getvalue().decode().splitlines()]
        self.assertEqual(len(labels), 5)
        self.assertTrue(labels[0]['native_partition'][analyzer.PARTS[0]])
        self.assertTrue(labels[1]['native_partition'][analyzer.PARTS[1]])
        self.assertIsNone(labels[4]['native_partition'])
        self.assertIsNone(labels[4]['classification'])

    def test_native_partition_all_N_and_unchanged_physical_weights_are_preserved(self):
        arrays, diagnostics, *_ = classified_fixture()
        specification = dict(radial_edges=[0., 2., 3., 4.], angular_projection_squared_edges=[0., 4., 9., 16.])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); records = []
            for index in range(4):
                path = root / f'p{index}.npz'; np.savez_compressed(path, **arrays)
                records.append(dict(id=f'r{index:02d}', seed=10 + index, samples=5, records=path.name,
                    records_sha256=analyzer.sha(path), sampler_cpu_seconds=1., **diagnostics))
            with patch.object(analyzer, 'check_estimate'):
                result = analyzer.summarize_arm(root, {'samples': 5, 'component_count': 80}, records, specification,
                    {'estimate': {}, 'hard_region': {}, 'importance_sampling': {}})
        for name in analyzer.DECISION_CLASSES:
            estimate = result['estimates'][name]
            self.assertEqual(estimate['row_uncertainty']['draws'], 20)
            self.assertEqual([p['draws'] for p in estimate['populations']], [5] * 4)
        self.assertAlmostEqual(result['estimates'][analyzer.NATIVE]['row_uncertainty']['log_Qz'], 0.)
        self.assertAlmostEqual(result['estimates'][analyzer.PARTS[0]]['row_uncertainty']['log_Qz'], math.log(.4))
        self.assertAlmostEqual(result['estimates'][analyzer.PARTS[1]]['row_uncertainty']['log_Qz'], math.log(.6))
        self.assertTrue(result['native_partition_sum_verified'])
        self.assertEqual(len(result['strata']['orthant'][analyzer.PARTS[1]]), 64)

    def test_modified_masks_or_dropped_source_denominators_are_rejected(self):
        arrays, *_ = classified_fixture()
        specification = dict(radial_edges=[0., 2., 3., 4.], angular_projection_squared_edges=[0., 4., 9., 16.])
        changed = copy.deepcopy(arrays); changed[analyzer.PARTS[1]][0] = True
        with self.assertRaisesRegex(ValueError, 'supplemental mask'):
            analyzer.load_population_masks(changed, specification)
        changed = copy.deepcopy(arrays); changed['source_n'][:] = 3
        with self.assertRaisesRegex(ValueError, 'attempted-draw'):
            analyzer.load_population_masks(changed, specification)

    def test_worker_limit_and_no_overwrite_precede_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'At most four'):
                analyzer.analyze('unused', Path(directory) / 'new', workers=5)
            with self.assertRaisesRegex(ValueError, 'Fresh analysis output'):
                analyzer.analyze('unused', directory, workers=1)


if __name__ == '__main__':
    unittest.main()
