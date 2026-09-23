"""Frozen allocation, physical gates and first-pass failure draining; no physics."""
import copy
import json
import signal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_protected_guide_validation as analyzer
import analyze_contact_confirmation as reused
from test_analyze_contact_confirmation import passing_arms as previous_arms
from test_analyze_smc_guide_pilot import synthetic_classification_worker, terminate_before_results


def protocol():
    arms = [dict(id=name, samples=n, component_count=k, lambda_ratio=l, alpha=.5)
            for name, (n, k, l) in zip(analyzer.ARM_NAMES, analyzer.ALLOCATION)]
    jobs = [dict(arm=a['id'], id=f'r{i:02d}', samples=a['samples'], seed=1000*j+i)
            for j, a in enumerate(arms) for i in range(4)]
    return dict(arms=arms, jobs=jobs, total_unconditional_draws=analyzer.TOTAL_DRAWS,
        convergence=copy.deepcopy(analyzer.CONVERGENCE), strata=copy.deepcopy(analyzer.STRATA),
        comparisons=copy.deepcopy(analyzer.COMPARISONS), decision_arms=list(analyzer.DECISION_ARMS))


def passing_arms():
    source = previous_arms()['bank']
    arms = {a: copy.deepcopy(source) for a in analyzer.ARM_NAMES}
    for a in arms.values():
        a['observed_importance_ESS_per_cpu_second'] = {r: 10. for r in analyzer.CLASSES}
    return arms


class ProtectedAnalysisTests(unittest.TestCase):
    def test_fixed_allocation_and_matched_intensity_control(self):
        p = protocol(); analyzer.validate_design(p)
        self.assertEqual(sum(j['samples'] for j in p['jobs']), 10_485_760)
        r = analyzer.evaluate(passing_arms(), p)
        self.assertTrue(r['regional_diagnostics_passed'])
        self.assertEqual(len(r['comparisons']), 4)
        self.assertTrue(r['comparisons']['small/intensity256']['matched_population_size'])
        self.assertFalse(r['comparisons']['protected/intensity256']['matched_population_size'])

    def test_thresholds_allocation_and_arm_policy_cannot_relax(self):
        for change in [lambda p: p['convergence'].update(largest_draw_max=.03),
                       lambda p: p['arms'][3].update(lambda_ratio=128.),
                       lambda p: p['arms'][0].update(samples=262144),
                       lambda p: p['strata'].update(latent_orthants='21 fitted groups'),
                       lambda p: p['decision_arms'].remove('bank'),
                       lambda p: p['comparisons'].pop()]:
            p = protocol(); change(p)
            with self.assertRaises(ValueError): analyzer.validate_design(p)

    def test_missing_duplicate_or_partial_populations_rejected(self):
        for change in [lambda p: p['jobs'].pop(),
                       lambda p: p['jobs'][1].update(seed=p['jobs'][0]['seed']),
                       lambda p: p['jobs'][1].update(id='r00'),
                       lambda p: p['jobs'][0].update(samples=1)]:
            p = protocol(); change(p)
            with self.assertRaises(ValueError): analyzer.validate_design(p)

    def test_passing_regional_diagnostics_never_promote_physical_gates(self):
        r = analyzer.evaluate(passing_arms(), protocol())
        self.assertTrue(r['regional_diagnostics_passed'])
        for key in ('passed', 'original_confirmation_superseded', 'full_vessel_gate_open',
                    'full_wall_coverage_established', 'assembly_stability_established'):
            self.assertFalse(r[key])

    def test_small_standalone_quality_diagnostic_but_size_agreement_required(self):
        arms = passing_arms(); region = analyzer.PRIMARY[1]
        arms['small']['estimates'][region]['row_uncertainty']['Qz_ESS'] = 5.
        self.assertTrue(analyzer.evaluate(arms, protocol())['regional_diagnostics_passed'])
        arms['small']['estimates'][region]['population_uncertainty']['log_Qz'] += .3
        r = analyzer.evaluate(arms, protocol())
        self.assertFalse(r['regional_diagnostics_passed'])
        self.assertFalse(r['comparisons']['protected/small']['masses'][region]['Qz']['passed'])

    def test_all_decision_regions_retain_quality_gate(self):
        for name in analyzer.DECISION_ARMS:
            for region in analyzer.DECISION_CLASSES:
                arms = passing_arms()
                arms[name]['estimates'][region]['row_uncertainty']['largest_Qz_fraction'] = .021
                self.assertFalse(analyzer.evaluate(arms, protocol())['regional_diagnostics_passed'])

    def test_efficiency_is_not_a_gate(self):
        arms = passing_arms()
        for name in ('protected', 'small', 'intensity256'):
            arms[name]['observed_importance_ESS_per_cpu_second'][analyzer.PRIMARY[1]] = .001
        self.assertTrue(analyzer.evaluate(arms, protocol())['regional_diagnostics_passed'])

    def test_three_se_and_hard_only_checks_remain(self):
        arms = passing_arms(); region = analyzer.PARTS[1]
        for a in arms.values():
            a['estimates'][region]['population_uncertainty']['Q0_relative_SE'] = .001
        arms['protected']['estimates'][region]['population_uncertainty']['log_Q0'] += .05
        r = analyzer.evaluate(arms, protocol())
        c = r['comparisons']['bank/protected']['masses'][region]['Q0']
        self.assertTrue(c['absolute_passed']); self.assertFalse(c['SE_passed'])
        self.assertFalse(r['regional_diagnostics_passed'])

    def test_original_strata_and_unobserved_contributions_are_retained(self):
        arms = passing_arms(); region = analyzer.PARTS[1]
        arms['intensity256']['strata']['orthant'][region][0]['population_uncertainty']['log_Qz'] = None
        r = analyzer.evaluate(arms, protocol())
        self.assertFalse(r['checks']['significant_original_strata_agreement'])
        self.assertFalse(r['regional_diagnostics_passed'])
        for level in ('row_uncertainty', 'population_uncertainty'):
            arms['protected']['estimates'][region][level]['log_Qz'] = None
        r = analyzer.evaluate(arms, protocol())
        self.assertFalse(r['quality']['protected'][region]['observed'])
        self.assertIn('unobserved', analyzer.report(dict(arms=arms, convergence=r)))

    def test_paired_interval_and_original_classifier_reused(self):
        r = analyzer.evaluate(passing_arms(), protocol())
        self.assertEqual(r['free_energy_intervals']['protected']['degrees_of_freedom'], 3)
        self.assertAlmostEqual(r['free_energy_intervals']['protected']['halfwidth_95'], .318244630528, places=10)
        self.assertIs(analyzer.classify_population, reused.classify_population)
        self.assertIs(analyzer.summarize_arm, reused.summarize_arm)

    def test_sigterm_drains_all_sixteen_submitted_jobs_without_reclassification(self):
        p = protocol(); p.update(reference_region_sha256='reference', supplemental_definition_sha256='partition', repository='unused')
        status = dict(jobs=[dict(arm=j['arm'], id=j['id'], output={}) for j in p['jobs']])
        assessments = {n: dict(populations=[dict(id=f'r{i:02d}') for i in range(4)]) for n in analyzer.ARM_NAMES}
        terminal = (p, status, assessments, Path('definition'), {}, Path('reference'), {})
        previous = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'analysis'
            with patch.object(analyzer, 'validate_terminal', return_value=terminal), \
                    patch.object(analyzer, 'wait_for_classification_capacity'), \
                    patch.object(analyzer, 'classify_population', synthetic_classification_worker), \
                    patch.object(analyzer, 'as_completed', terminate_before_results):
                with self.assertRaises(InterruptedError): analyzer.analyze('unused', out, workers=2)
            self.assertEqual(len(list(out.glob('*.done'))), 16)
            saved = json.loads((out/'status.json').read_text())
            self.assertFalse(saved['complete']); self.assertIn('failed', saved['phase'])
            self.assertFalse((out/'analysis.json').exists())
            self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_global_capacity_includes_pool_and_management_threads(self):
        counts = iter([dict(workers=n, physical_pids=[]) for n in (20, 15, 14)])
        snapshots, waits, state = [], [], {}
        analyzer.wait_for_classification_capacity('workspace', 16, state,
            lambda: snapshots.append(copy.deepcopy(state)), capacity=lambda _: next(counts),
            pause=lambda delay: waits.append(delay))
        self.assertEqual(waits, [.5, .5])
        self.assertEqual(state['phase'], 'first_classification_pass')
        self.assertEqual(state['classification_capacity']['observed_workers'], 14)
        self.assertEqual(state['classification_capacity']['reserved_management_threads'], 2)
        self.assertEqual(len(snapshots), 4)

    def test_worker_cap_and_no_overwrite_checked_before_loading_physical_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(analyzer, 'validate_terminal') as validate:
                with self.assertRaises(ValueError): analyzer.analyze('unused', Path(directory)/'new', workers=17)
                with self.assertRaises(ValueError): analyzer.analyze('unused', directory, workers=16)
                validate.assert_not_called()


if __name__ == '__main__': unittest.main()
