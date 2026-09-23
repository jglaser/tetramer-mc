"""Bounded two-arm validation cannot weaken thresholds or promote assembly gates."""
import copy
import math
import json
import os
import signal
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_smc_guide_pilot as analyzer
import analyze_contact_confirmation as reused
from test_analyze_contact_confirmation import passing_arms as old_passing_arms, classified_fixture


def protocol():
    arms = [dict(id=name, samples=analyzer.SAMPLES, component_count=count, alpha=.5, lambda_ratio=128.)
        for name, count in zip(analyzer.ARM_NAMES, (80, 84))]
    jobs = [dict(arm=arm, id=f'r{i:02d}', samples=analyzer.SAMPLES, seed=1000 * j + i)
        for j, arm in enumerate(analyzer.ARM_NAMES) for i in range(4)]
    return dict(arms=arms, jobs=jobs, total_unconditional_draws=analyzer.TOTAL_DRAWS,
        convergence=copy.deepcopy(analyzer.CONVERGENCE), strata=copy.deepcopy(analyzer.STRATA),
        comparisons=[['bank', 'smc']], decision_arms=['bank', 'smc'])


def passing_arms():
    source = old_passing_arms()
    arms = {'bank': source['bank'], 'smc': source['wide']}
    for a in arms.values():
        a['observed_importance_ESS_per_cpu_second'] = {name: 10. for name in analyzer.CLASSES}
    return arms


def synthetic_classification_worker(task):
    # Real process worker: enough latency to leave both running and queued work.
    time.sleep(.025)
    marker = Path(task['out']) / (task['arm']['id'] + '-' + task['job']['id'] + '.done')
    marker.write_text(json.dumps(dict(sigterm_ignored=signal.getsignal(signal.SIGTERM) == signal.SIG_IGN,
        sigint_ignored=signal.getsignal(signal.SIGINT) == signal.SIG_IGN)))
    return dict(arm=task['arm']['id'], id=task['job']['id'])


def terminate_before_results(futures):
    os.kill(os.getpid(), signal.SIGTERM)
    raise AssertionError('SIGTERM handler did not interrupt result collection')


class SmcGuidePilotAnalysisTests(unittest.TestCase):
    def test_predeclared_allocation_and_thresholds(self):
        analyzer.validate_design(protocol())
        for key, value in [('importance_ESS_min', 199.), ('largest_draw_max', .021),
                ('log_agreement_absolute_max', .201), ('population_relative_SE_max', .11)]:
            p = protocol(); p['convergence'][key] = value
            with self.assertRaisesRegex(ValueError, 'thresholds changed'):
                analyzer.validate_design(p)

    def test_missing_population_duplicate_seed_and_larger_allocation_rejected(self):
        p = protocol(); p['jobs'].pop()
        with self.assertRaisesRegex(ValueError, 'Eight independent'):
            analyzer.validate_design(p)
        p = protocol(); p['jobs'][1]['seed'] = p['jobs'][0]['seed']
        with self.assertRaisesRegex(ValueError, 'Eight independent'):
            analyzer.validate_design(p)
        p = protocol(); p['jobs'][0]['samples'] *= 2
        with self.assertRaisesRegex(ValueError, 'Four complete'):
            analyzer.validate_design(p)

    def test_intensity_defensive_mass_and_number_of_components_frozen(self):
        for key, value in [('lambda_ratio', 64.), ('alpha', .2), ('component_count', 85), ('samples', 16384)]:
            p = protocol(); p['arms'][1][key] = value
            with self.assertRaisesRegex(ValueError, 'proposal allocation'):
                analyzer.validate_design(p)

    def test_original_strata_and_both_arm_policies_frozen(self):
        p = protocol(); p['strata']['radial_edges'] = [0., 4.]
        with self.assertRaisesRegex(ValueError, 'strata changed'):
            analyzer.validate_design(p)
        p = protocol(); p['decision_arms'] = ['smc']
        with self.assertRaisesRegex(ValueError, 'Both proposal arms'):
            analyzer.validate_design(p)

    def test_passing_pilot_never_promotes_confirmation_or_assembly_gate(self):
        result = analyzer.evaluate(passing_arms(), protocol())
        self.assertTrue(result['pilot_diagnostics_passed'])
        for key in ('passed', 'original_confirmation_superseded', 'full_vessel_gate_open',
                'full_wall_coverage_established', 'assembly_stability_established'):
            self.assertFalse(result[key])
        self.assertEqual(set(result['quality']['smc']), set(analyzer.DECISION_CLASSES))

    def test_noentry_efficiency_improvement_is_not_required(self):
        arms = passing_arms()
        arms['smc']['observed_importance_ESS_per_cpu_second'][analyzer.PRIMARY[1]] = 1.
        result = analyzer.evaluate(arms, protocol())
        self.assertTrue(result['pilot_diagnostics_passed'])
        self.assertEqual(result['observed_importance_efficiency'][analyzer.PRIMARY[1]]['smc_over_bank'], .1)
        arms['smc']['estimates'][analyzer.PRIMARY[1]]['row_uncertainty']['Qz_ESS'] = 199.
        self.assertFalse(analyzer.evaluate(arms, protocol())['pilot_diagnostics_passed'])

    def test_each_arm_and_native_partition_retains_quality_checks(self):
        for arm in analyzer.ARM_NAMES:
            for region in analyzer.DECISION_CLASSES:
                arms = passing_arms()
                arms[arm]['estimates'][region]['row_uncertainty']['largest_Qz_fraction'] = .021
                self.assertFalse(analyzer.evaluate(arms, protocol())['pilot_diagnostics_passed'])

    def test_total_native_agreement_does_not_hide_complement_difference(self):
        arms = passing_arms()
        arms['smc']['estimates'][analyzer.PARTS[1]]['population_uncertainty']['log_Qz'] += .3
        result = analyzer.evaluate(arms, protocol())
        self.assertTrue(result['comparisons']['smc']['masses'][analyzer.PRIMARY[0]]['Qz']['passed'])
        self.assertFalse(result['comparisons']['smc']['masses'][analyzer.PARTS[1]]['Qz']['passed'])
        self.assertFalse(result['pilot_diagnostics_passed'])

    def test_direct_free_energy_and_three_se_conditions_remain(self):
        arms = passing_arms()
        ratio = arms['smc']['primary_ratios'][analyzer.PRIMARY[0] + '/' + analyzer.PRIMARY[1]]['population']
        ratio['log_ratio'] += .3
        self.assertFalse(analyzer.evaluate(arms, protocol())['checks']['direct_free_energy_agreement'])
        arms = passing_arms()
        for arm in analyzer.ARM_NAMES:
            arms[arm]['estimates'][analyzer.PARTS[1]]['population_uncertainty']['Qz_relative_SE'] = .001
        arms['smc']['estimates'][analyzer.PARTS[1]]['population_uncertainty']['log_Qz'] += .05
        check = analyzer.evaluate(arms, protocol())['comparisons']['smc']['masses'][analyzer.PARTS[1]]['Qz']
        self.assertTrue(check['absolute_passed']); self.assertFalse(check['SE_passed'])

    def test_population_student_t3_interval_and_precision_gate(self):
        result = analyzer.evaluate(passing_arms(), protocol())
        interval = result['free_energy_intervals']['smc']
        self.assertEqual(interval['degrees_of_freedom'], 3)
        self.assertAlmostEqual(interval['halfwidth_95'], .318244630528, places=10)
        arms = passing_arms()
        arms['smc']['primary_ratios'][analyzer.PRIMARY[0] + '/' + analyzer.PRIMARY[1]]['population']['log_ratio_SE'] = .2
        self.assertFalse(analyzer.evaluate(arms, protocol())['checks']['both_paired_free_energy_precision'])

    def test_unobserved_weight_is_not_zero_and_is_reported(self):
        arms = passing_arms()
        for level in ('row_uncertainty', 'population_uncertainty'):
            arms['smc']['estimates'][analyzer.PARTS[1]][level]['log_Qz'] = None
        result = analyzer.evaluate(arms, protocol())
        self.assertFalse(result['quality']['smc'][analyzer.PARTS[1]]['observed'])
        self.assertFalse(result['pilot_diagnostics_passed'])
        result = dict(arms=arms, convergence=result)
        self.assertIn('unobserved', analyzer.report(result))

    def test_significant_orthant_failure_and_missing_mass_retained(self):
        arms = passing_arms()
        entry = arms['smc']['strata']['orthant'][analyzer.PARTS[1]][0]
        entry['population_uncertainty']['log_Qz'] += .4
        result = analyzer.evaluate(arms, protocol())
        self.assertFalse(result['checks']['significant_original_strata_agreement'])
        self.assertEqual(result['significant_stratum_disagreements'][0]['region'], analyzer.PARTS[1])
        entry['population_uncertainty']['log_Qz'] = None
        self.assertFalse(analyzer.evaluate(arms, protocol())['checks']['significant_original_strata_agreement'])

    def test_reuses_original_single_pass_classifier_and_denominator_summarizer(self):
        self.assertIs(analyzer.classify_population, reused.classify_population)
        self.assertIs(analyzer.summarize_arm, reused.summarize_arm)
        arrays, diagnostics, native, contact, _ = classified_fixture()
        self.assertEqual(native.calls, [0, 1, 2]); self.assertEqual(contact.calls, [0, 1, 2])
        masks, counts = reused.load_population_masks(arrays, analyzer.STRATA)
        self.assertEqual(counts, dict(radial=3, angular=3, orthant=64))
        self.assertEqual(len(arrays['z']), 5)
        self.assertEqual(int(masks[analyzer.PARTS[0]].sum()), 1)
        self.assertEqual(int(masks[analyzer.PARTS[1]].sum()), 1)
        self.assertEqual(diagnostics['full_native_classifier_calls'], 3)

    def test_zero_efficiency_has_no_spurious_infinite_ratio(self):
        arms = passing_arms()
        arms['bank']['observed_importance_ESS_per_cpu_second'][analyzer.PRIMARY[2]] = 0.
        self.assertIsNone(analyzer.evaluate(arms, protocol())['observed_importance_efficiency'][analyzer.PRIMARY[2]]['smc_over_bank'])

    def test_sigterm_drains_real_synthetic_workers_records_failure_and_restores_handler(self):
        p = protocol()
        p.update(reference_region_sha256='reference', supplemental_definition_sha256='supplemental')
        status = dict(jobs=[dict(arm=j['arm'], id=j['id'], output={}) for j in p['jobs']])
        assessments = {name: dict(populations=[dict(id=f'r{i:02d}') for i in range(4)])
            for name in analyzer.ARM_NAMES}
        terminal = (p, status, assessments, Path('definition'), {}, Path('reference'), {})
        previous = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'analysis'
            with patch.object(analyzer, 'validate_terminal', return_value=terminal), \
                    patch.object(analyzer, 'classify_population', synthetic_classification_worker), \
                    patch.object(analyzer, 'as_completed', terminate_before_results):
                with self.assertRaisesRegex(InterruptedError, 'SIGTERM requested'):
                    analyzer.analyze('unused', out, workers=2)
            self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
            # Executor shutdown finishes all eight submitted tasks before failure status.
            markers = list(out.glob('*.done'))
            self.assertEqual(len(markers), 8)
            for path in markers:
                self.assertEqual(json.loads(path.read_text()), dict(sigterm_ignored=True, sigint_ignored=True))
            saved = json.loads((out / 'status.json').read_text())
            self.assertFalse(saved['complete'])
            self.assertEqual(saved['phase'], 'first_classification_pass_failed')
            self.assertIn('InterruptedError', saved['exception'])
            self.assertGreaterEqual(saved['finished'], max(path.stat().st_mtime for path in markers))
            self.assertFalse((out / 'status.json.tmp').exists())
            self.assertFalse((out / 'analysis.json').exists())

    def test_atomic_status_failure_preserves_previous_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            analyzer.write_status(path, dict(phase='before'))
            with patch.object(analyzer.os, 'replace', side_effect=OSError('replace failed')):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    analyzer.write_status(path, dict(phase='after'))
            self.assertEqual(json.loads(path.read_text()), dict(phase='before'))
            self.assertFalse(path.with_name(path.name + '.tmp').exists())

    def test_worker_limit_and_no_overwrite_precede_terminal_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(analyzer, 'validate_terminal') as validation:
                with self.assertRaisesRegex(ValueError, 'At most four'):
                    analyzer.analyze('unused', Path(directory) / 'new', workers=5)
                with self.assertRaisesRegex(ValueError, 'Fresh analysis output'):
                    analyzer.analyze('unused', directory, workers=1)
                validation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
