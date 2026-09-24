"""Independent-stream comparison controls using synthetic retained histories.

No physical sampler, protein classifier, or historical audit is run. Every
matrix member gets a legitimate summary from the frozen pure graph observer.
"""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import compare_finite_assembly as compare
import contact_benchmark_contract as benchmark
from finite_assembly_observer import REGIONS, graph_summary, summarize_series
from prepare_finite_assembly_observer import REGION_RULES


PREPARATIONS = ('dispersed', 'competing-aggregate', 'native-seeded')
ARMS = ('local', 'redraw', 'transport')
MOTIFS = [dict(id=i, relative_position=[float(i+1), 0., 0.],
               relative_orientation=[1., 0., 0., 0.]) for i in range(2)]


def observation(label):
    native = []
    exclusion = []
    if label == 'contact_no_entry':
        exclusion = [(0, 1)]
    elif label == 'registered_eight':
        native = [(i, i+1, 0) for i in range(7)]
    elif label == 'remaining':
        native = [(0, 1, 0), (1, 2, 0), (0, 2, 0)]
    exclusion += sorted({(i, j) for i, j, _ in native})
    result = graph_summary(12, exclusion, native, MOTIFS, [[float(i), 0., 0.] for i in range(12)])
    assert result['environment'] == label
    return dict(tokens=[(i, j, 'surface', 'surface') for i, j in exclusion], **result)


def synthetic_history(stream=0, mode='mixed'):
    pattern = ['dispersed', 'contact_no_entry', 'contact_no_entry', 'remaining',
               'registered_eight', 'registered_eight', 'contact_no_entry', 'dispersed']*2
    for index in range(stream):
        pattern[index] = 'registered_eight'
    if mode == 'no-native':
        pattern = ['dispersed', 'contact_no_entry']*8
    elif mode == 'constant':
        pattern = ['contact_no_entry']*16
    observations = [observation('dispersed')]+[observation(label) for label in pattern]
    frames = [dict(sweep=i, sampler_cpu_seconds=.2+.5*i) for i in range(17)]
    window = dict(indices=list(range(17)), sample_indices=list(range(1, 17)),
        cpu_seconds=8., cadence_sweeps=1, burn_sweep=0, end_sweep=16)
    return observations, frames, window


def matrix(mode='mixed'):
    identity = dict(shape_sha256='a'*64, bodies=12,
        boundary=dict(kind='spherical', radius=354.47871325728795, center=[0., 0., 0.]),
        box_lengths=[571.4168336408595]*3, depletant_radius=1.5, reservoir_density=.035,
        fixed_body_indices=[], measure='d^3t times normalized Haar for each labelled rigid body; ideal bath wall-permeable')
    common = dict(local_translation_std_A=.1, local_small_angle_std_degrees=1.,
        gca_probability=.1, center_shift_probability=.1, poisson_lambda_ratio=64.,
        endpoint_gate=dict(max_cells=64, max_depth=4, min_width=.01))
    arms = dict(local=dict(role='local', global_probability=0., method='local-uniform',
                    model_sha256=None, frozen_posterior=None, learned_uniform_weight=.5),
                redraw=dict(role='independent-redraw', global_probability=.5, method='learned',
                    model_sha256='b'*64, frozen_posterior=None, learned_uniform_weight=.5),
                transport=dict(role='correlated-transport', global_probability=.5, method='learned',
                    model_sha256='b'*64, frozen_posterior=dict(probability=.9, correlation=.9), learned_uniform_weight=.5))
    contract = dict(schema=benchmark.SCHEMA, physical_identity_sha256=benchmark.digest(identity),
        observer_definition_sha256='c'*64, region_definition_sha256=benchmark.digest(REGION_RULES),
        common_schedule=common, window=dict(burn_sweep=0, end_sweep=16, cadence_sweeps=1),
        preparations=list(PREPARATIONS), streams_per_preparation=4, arms=arms,
        single_body_attempt_budget=benchmark.ATTEMPTS)
    result = []
    for arm in ARMS:
        for preparation in PREPARATIONS:
            for stream in range(4):
                report = summarize_series(*synthetic_history(stream, mode))
                law = arms[arm]
                report.update(schema='finite-assembly-observation-v1', complete=True,
                    run=f'/synthetic/{arm}/{preparation}/{stream}', physical_identity=identity,
                    definition_sha256='d'*64, observer_definition_sha256=contract['observer_definition_sha256'],
                    region_definition_sha256=contract['region_definition_sha256'],
                    implementation_sha256={'finite_assembly_observer.py': 'e'*64},
                    initialization=dict(master_seed=1000+len(result), preparation_id=preparation,
                        proposal_arm=arm, initial_poses_sha256=benchmark.digest([preparation, stream]),
                        invocation_initial_sweep=0, resume=None),
                    schedule=dict(common, global_probability=law['global_probability']),
                    proposal={key: law[key] for key in benchmark.PROPOSAL},
                    window_attempts=dict(local=192 if arm == 'local' else 96,
                                         **{'global': 0 if arm == 'local' else 96}),
                    benchmark_contract=dict(path='/synthetic/contract.json', sha256='f'*64,
                        content=contract, content_sha256=benchmark.digest(contract)),
                    dependency_sha256={'/synthetic/contract.json': 'f'*64},
                    source_sha256={'trajectory.jsonl': benchmark.digest([arm, preparation, stream])},
                    physical_reference=None, frozen_bias_supported=False,
                    observer_cpu_seconds=.01)
                result.append(copy.deepcopy(report))
    return result


class PopulationStatisticsTests(unittest.TestCase):
    def test_uncertainty_uses_four_independent_population_values(self):
        values = [0., .25, .5, .75]
        result = compare.mean_statistics(values)
        self.assertEqual(result['values'], values)
        self.assertEqual(result['populations'], 4)
        self.assertAlmostEqual(result['mean'], .375)
        self.assertAlmostEqual(result['population_standard_error'], np.std(values, ddof=1)/2.)
        self.assertFalse(result['zero_observed_variance'])
        width = 3.182446305284263*np.std(values, ddof=1)/2.
        np.testing.assert_allclose(result['descriptive_t95'], [.375-width, .375+width], rtol=1e-12)

    def test_zero_observed_variance_has_no_fake_confidence_interval(self):
        for value in (0., 1., 3.):
            result = compare.mean_statistics([value]*4)
            self.assertEqual(result['population_standard_error'], 0.)
            self.assertTrue(result['zero_observed_variance'])
            self.assertIsNone(result['descriptive_t95'])

    def test_nonfinite_population_values_fail(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare.mean_statistics([0., 1., 2., value])


class CompleteMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid = matrix()

    def test_complete_36_population_matrix_is_accepted_without_mutation(self):
        reports = copy.deepcopy(self.valid)
        original = copy.deepcopy(reports)
        result = compare.compare_reports(reports)
        self.assertEqual(len(result['groups']), 9)
        self.assertTrue(result['contrasts'])
        self.assertIn('speedup_diagnostics', result)
        self.assertEqual(reports, original)

    def test_missing_extra_or_duplicate_populations_fail(self):
        for reports in (self.valid[:-1], self.valid+[self.valid[0]], self.valid[:-1]+[self.valid[0]]):
            with self.subTest(count=len(reports)), self.assertRaises(ValueError):
                compare.compare_reports(copy.deepcopy(reports))

    def test_independence_rejects_duplicate_invalid_seeds_and_resumes(self):
        for key, value in [('master_seed', self.valid[1]['initialization']['master_seed']),
                           ('master_seed', True), ('master_seed', -1), ('master_seed', 2**64),
                           ('invocation_initial_sweep', 1), ('resume', {'checkpoint': 'old'})]:
            reports = copy.deepcopy(self.valid); reports[0]['initialization'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                compare.compare_reports(reports)

    def test_target_measurement_window_and_schedule_mismatches_fail(self):
        cases = [('schema', None, 'passive-contact-efficiency-v1'), ('complete', None, False),
                 ('physical_identity', 'bodies', 24), ('physical_identity', 'boundary', {'kind': 'periodic'}),
                 ('definition_sha256', None, '0'*64), ('implementation_sha256', 'finite_assembly_observer.py', '0'*64),
                 ('observer_definition_sha256', None, '0'*64), ('region_definition_sha256', None, '0'*64),
                 ('window', 'cadence_sweeps', 2), ('window', 'burn_sweep', 1), ('window', 'end_sweep', 15),
                 ('schedule', 'local_translation_std_A', .2), ('schedule', 'gca_probability', 0.)]
        for key, subkey, value in cases:
            reports = copy.deepcopy(self.valid)
            if subkey is None: reports[0][key] = value
            else: reports[0][key][subkey] = value
            with self.subTest(key=key, subkey=subkey), self.assertRaises(ValueError):
                compare.compare_reports(reports)

    def test_matched_initial_pose_multisets_are_required_but_stream_order_is_not(self):
        reports = copy.deepcopy(self.valid)
        reports[0]['initialization']['initial_poses_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            compare.compare_reports(reports)
        reports = copy.deepcopy(self.valid)
        reports[0]['initialization']['initial_poses_sha256'], reports[1]['initialization']['initial_poses_sha256'] = (
            reports[1]['initialization']['initial_poses_sha256'], reports[0]['initialization']['initial_poses_sha256'])
        self.assertEqual(len(compare.compare_reports(reports)['groups']), 9)

    def test_contract_and_independent_redraw_law_are_mandatory(self):
        for mode in ('missing-contract', 'posterior-redraw', 'bad-attempts', 'renamed-preparation'):
            reports = copy.deepcopy(self.valid)
            if mode == 'missing-contract': reports[0]['benchmark_contract'] = None
            elif mode == 'posterior-redraw': reports[12]['proposal']['frozen_posterior'] = dict(probability=.9, correlation=0.)
            elif mode == 'bad-attempts': reports[0]['window_attempts']['local'] -= 1
            else: reports[0]['initialization']['preparation_id'] = 'selected-start'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                compare.compare_reports(reports)

    def test_histograms_cannot_lose_raw_body_mass_or_create_certified_mass(self):
        for name, value in [('exclusion', 0.), ('native_raw', 0.), ('native_certified', 13.)]:
            reports = copy.deepcopy(self.valid)
            reports[0]['component_number_statistics'][name]['mean_component_counts'] = [value]+[0.]*11
            with self.subTest(name=name), self.assertRaises(ValueError):
                compare.compare_reports(reports)

    def test_each_group_retains_population_uncertainty_and_unresolved_body_mass(self):
        result = compare.compare_reports(copy.deepcopy(self.valid))
        group = next(g for g in result['groups'] if g['proposal_arm'] == 'local'
                     and g['preparation_id'] == 'dispersed')
        expected = [r['environment_occupancies']['registered_eight']['fraction'] for r in self.valid[:4]]
        stats = group['region_occupancies']['registered_eight']
        self.assertEqual(stats['values'], expected)
        self.assertEqual(stats['populations'], 4)
        self.assertAlmostEqual(stats['population_standard_error'], np.std(expected, ddof=1)/2.)
        self.assertEqual(len(stats['half_window_changes']), 4)
        for name in ('exclusion', 'native_raw'):
            self.assertEqual(group['component_histograms'][name]['represented_body_fraction']['values'], [1.]*4)
        certified = group['component_histograms']['native_certified']
        self.assertEqual(certified['sizes'], list(range(1, 13)))
        self.assertEqual(len(certified['mean_count_by_size']), 12)
        self.assertTrue(all(value < 1. for value in certified['represented_body_fraction']['values']))
        self.assertFalse(result['convergence_established'])

    def test_one_null_native_population_keeps_the_group_and_speedup_unresolved(self):
        reports = copy.deepcopy(self.valid)
        reports[0].update(summarize_series(*synthetic_history(mode='no-native')))
        result = compare.compare_reports(reports)
        group = next(g for g in result['groups'] if g['proposal_arm'] == 'local'
                     and g['preparation_id'] == 'dispersed')
        native = group['fingerprint_rates']['native_fingerprint']
        self.assertEqual(native['null_populations'], 1)
        self.assertEqual(len(native['values']), 4)
        self.assertIsNone(native['population_statistics'])
        self.assertIsNotNone(group['fingerprint_rates']['fingerprint']['population_statistics'])
        affected = [x for x in result['speedup_diagnostics'] if x['fingerprint'] == 'native_fingerprint'
                    and ['local', 'dispersed'] in (x['left'], x['right'])]
        self.assertTrue(affected)
        for row in affected:
            self.assertIsNone(row['ratio_of_population_mean_rates'])
            self.assertFalse(row['equilibrium_speedup_established'])

    def test_zero_native_observations_do_not_create_confidence_or_agreement(self):
        result = compare.compare_reports(matrix(mode='no-native'))
        for group in result['groups']:
            stats = group['region_occupancies']['registered_eight']
            self.assertEqual(stats['observed_counts'], [0]*4)
            self.assertTrue(stats['all_zero_observations'])
            self.assertIsNone(stats['descriptive_t95'])
            self.assertIsNone(group['fingerprint_rates']['native_fingerprint']['population_statistics'])
        contrasts = [c for c in result['contrasts'] if c['observable'] == 'region:registered_eight']
        self.assertTrue(contrasts)
        for row in contrasts:
            self.assertIsNone(row['diagnostic_agreement_within_three_observed_se'])
            self.assertFalse(row['equilibrium_agreement_established'])

    def test_exchanges_and_remaining_episodes_are_aggregated_without_joining_streams(self):
        result = compare.compare_reports(copy.deepcopy(self.valid))
        group = next(g for g in result['groups'] if g['proposal_arm'] == 'local'
                     and g['preparation_id'] == 'dispersed')
        for pair in group['exchanges']:
            expected = [next(p for p in r['environment_exchanges']['pairs'] if p['regions'] == pair['regions'])
                        for r in self.valid[:4]]
            self.assertEqual(pair['completed_roundtrips'], sum(p['completed_nonoverlapping_roundtrips'] for p in expected))
            for got, old in zip(pair['per_stream'], expected):
                self.assertEqual({k: v for k, v in got.items() if k != 'seed'}, old)
        histories = group['per_stream_episode_histories']
        self.assertEqual(len(histories), 4)
        for index, row in enumerate(histories):
            self.assertEqual(row['sampled_episodes'], self.valid[index]['environment_exchanges']['sampled_episodes'])
            self.assertTrue(any(episode['environment'] == 'remaining' for episode in row['sampled_episodes']))

    def test_full_sampler_cpu_controls_the_apparent_efficiency_ratio(self):
        reports = copy.deepcopy(self.valid)
        for report in reports:
            if report['initialization']['proposal_arm'] == 'transport':
                stream = (report['initialization']['master_seed']-1000) % 4
                observations, frames, window = synthetic_history(stream)
                window['cpu_seconds'] = 4.
                report.update(summarize_series(observations, frames, window))
        result = compare.compare_reports(reports)
        rows = [r for r in result['speedup_diagnostics'] if r['left'] == ['local', 'dispersed']
                and r['right'] == ['transport', 'dispersed']]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(row['ratio_of_population_mean_rates'], .5)
            self.assertFalse(row['equilibrium_speedup_established'])


class FrozenReportFileTests(unittest.TestCase):
    """One adapter-produced report isolates authentication from allocation.

    compare_reports is mocked only here; the complete matrix is exercised above.
    The fixture supplies synthetic native geometry solely during initial report
    construction, never during the comparator being tested.
    """
    def make_report(self, data):
        import analyze_finite_assembly as adapter
        return adapter.observe(data.run, data.definition, data.plan, data.out)

    def reseal(self, data):
        import analyze_finite_assembly as adapter
        path = data.out/'freeze.json'
        freeze = json.loads(path.read_text())
        freeze['files'] = {name: adapter.sha(data.out/name) for name in freeze['files']}
        path.write_text(json.dumps(freeze))

    def test_authentic_report_files_use_cached_states_without_geometry_or_fft(self):
        import finite_assembly_observer as observer
        from test_analyze_finite_assembly import synthetic_run
        with synthetic_run() as data:
            report = self.make_report(data)
            calls_before = len(data.calls)
            with patch.object(compare, 'compare_reports', return_value={'groups': [], 'contrasts': []}) as aggregate, \
                 patch.object(observer.FiniteAssemblyObserver, 'classify', side_effect=AssertionError('geometry replay')), \
                 patch.object(np.fft, 'rfft', side_effect=AssertionError('FFT replay')):
                compare.compare_report_files([data.out/'analysis.json'])
            aggregate.assert_called_once()
            self.assertEqual(aggregate.call_args.args[0][0]['initialization'], report['initialization'])
            self.assertEqual(len(data.calls), calls_before)

    def test_frozen_observation_mutation_is_rejected(self):
        from test_analyze_finite_assembly import synthetic_run
        with synthetic_run() as data:
            self.make_report(data)
            with (data.out/'observations.jsonl').open('a') as stream:
                stream.write('\n')
            with patch.object(compare, 'compare_reports') as aggregate, self.assertRaises(ValueError):
                compare.compare_report_files([data.out/'analysis.json'])
            aggregate.assert_not_called()

    def test_resealed_summary_cannot_disagree_with_cached_labels(self):
        from test_analyze_finite_assembly import synthetic_run
        with synthetic_run() as data:
            report = self.make_report(data)
            occupied = next(k for k, v in report['environment_occupancies'].items() if v['observations'] > 0)
            other = next(k for k in REGIONS if k != occupied)
            for label, delta in [(occupied, -1), (other, 1)]:
                entry = report['environment_occupancies'][label]
                entry['observations'] += delta
                entry['fraction'] = entry['observations']/report['fingerprint']['samples']
            (data.out/'analysis.json').write_text(json.dumps(report))
            self.reseal(data)
            with patch.object(compare, 'compare_reports') as aggregate, self.assertRaises(ValueError):
                compare.compare_report_files([data.out/'analysis.json'])
            aggregate.assert_not_called()

    def test_resealed_cached_environment_cannot_disagree_with_its_graph(self):
        from test_analyze_finite_assembly import synthetic_run
        with synthetic_run() as data:
            self.make_report(data)
            path = data.out/'observations.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]['environment'] = 'remaining' if rows[1]['environment'] != 'remaining' else 'dispersed'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.reseal(data)
            with patch.object(compare, 'compare_reports') as aggregate, self.assertRaises(ValueError):
                compare.compare_report_files([data.out/'analysis.json'])
            aggregate.assert_not_called()

    def test_original_run_inputs_and_observer_dependencies_remain_bound(self):
        from test_analyze_finite_assembly import synthetic_run
        for location in ('run', 'source', 'definition'):
            with self.subTest(location=location), synthetic_run() as data:
                self.make_report(data)
                path = (data.run/'config.json' if location == 'run' else data.definition if location == 'definition'
                        else data.definition.parent/'source/finite_assembly_observer.py')
                with path.open('a') as stream:
                    stream.write('\n')
                with patch.object(compare, 'compare_reports') as aggregate, self.assertRaises(ValueError):
                    compare.compare_report_files([data.out/'analysis.json'])
                aggregate.assert_not_called()

    def test_resealed_cpu_and_all_rates_still_must_match_original_sampler_work(self):
        from test_analyze_finite_assembly import synthetic_run
        with synthetic_run() as data:
            report = self.make_report(data)
            def rescale(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in ('cpu_seconds', 'sampling_CPU_seconds'):
                            value[key] *= 2.
                        elif key in ('apparent_ess_per_sampling_CPU_second', 'completed_passages_per_sampler_cpu_second'):
                            value[key] = None if item is None else item/2.
                        else:
                            rescale(item)
                elif isinstance(value, list):
                    for item in value:
                        rescale(item)
            rescale(report)
            (data.out/'analysis.json').write_text(json.dumps(report))
            self.reseal(data)
            with patch.object(compare, 'compare_reports') as aggregate, self.assertRaises(ValueError):
                compare.compare_report_files([data.out/'analysis.json'])
            aggregate.assert_not_called()

    def test_resealed_initial_pose_schedule_and_proposal_must_match_original_run(self):
        from test_analyze_finite_assembly import synthetic_run
        for field in ('initial_poses_sha256', 'schedule', 'proposal'):
            with self.subTest(field=field), synthetic_run() as data:
                report = self.make_report(data)
                if field == 'initial_poses_sha256':
                    report['initialization'][field] = '0'*64
                elif field == 'schedule':
                    report['schedule']['local_translation_std_A'] *= 2.
                else:
                    report['proposal']['learned_uniform_weight'] *= .5
                (data.out/'analysis.json').write_text(json.dumps(report))
                self.reseal(data)
                with patch.object(compare, 'compare_reports') as aggregate, self.assertRaises(ValueError):
                    compare.compare_report_files([data.out/'analysis.json'])
                aggregate.assert_not_called()

    def test_original_trajectory_binding_cannot_be_removed_from_resealed_report(self):
        from test_analyze_finite_assembly import synthetic_run
        with synthetic_run() as data:
            report = self.make_report(data)
            del report['source_sha256']['trajectory.jsonl']
            (data.out/'analysis.json').write_text(json.dumps(report))
            self.reseal(data)
            with patch.object(compare, 'compare_reports') as aggregate, self.assertRaises(ValueError):
                compare.compare_report_files([data.out/'analysis.json'])
            aggregate.assert_not_called()

    def test_resealed_cache_cannot_drop_an_original_frame_outside_selected_window(self):
        from test_analyze_finite_assembly import synthetic_run
        with synthetic_run() as data:
            plan = json.loads(data.plan.read_text())
            plan.update(burn_sweep=2, end_sweep=6)
            data.plan.write_text(json.dumps(plan))
            report = self.make_report(data)
            rows_path = data.out/'observations.jsonl'
            rows = rows_path.read_text().splitlines()
            rows_path.write_text('\n'.join(rows[:-1])+'\n')
            report['labels_by_frame'].pop()
            (data.out/'analysis.json').write_text(json.dumps(report))
            self.reseal(data)
            with patch.object(compare, 'compare_reports') as aggregate, self.assertRaises(ValueError):
                compare.compare_report_files([data.out/'analysis.json'])
            aggregate.assert_not_called()

    def test_resealed_attempt_split_must_match_original_counters(self):
        from test_analyze_finite_assembly import synthetic_run
        with synthetic_run() as data:
            report = self.make_report(data)
            report['window_attempts']['local'] -= 1
            report['window_attempts']['global'] += 1
            (data.out/'analysis.json').write_text(json.dumps(report))
            self.reseal(data)
            with patch.object(compare, 'compare_reports') as aggregate, self.assertRaises(ValueError):
                compare.compare_report_files([data.out/'analysis.json'])
            aggregate.assert_not_called()

    def test_resealed_preparation_or_arm_cannot_depart_from_bound_analysis_plan(self):
        from test_analyze_finite_assembly import synthetic_run
        for field in ('preparation_id', 'proposal_arm'):
            with self.subTest(field=field), synthetic_run() as data:
                report = self.make_report(data)
                report['initialization'][field] = 'a-different-declared-group'
                (data.out/'analysis.json').write_text(json.dumps(report))
                self.reseal(data)
                with patch.object(compare, 'compare_reports') as aggregate, self.assertRaises(ValueError):
                    compare.compare_report_files([data.out/'analysis.json'])
                aggregate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
