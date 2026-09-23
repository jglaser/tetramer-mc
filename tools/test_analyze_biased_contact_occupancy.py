"""Synthetic saved histories and finite enumerated ensembles; no MC runs."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import analyze_biased_contact_occupancy as bias
from analyze_contact_efficiency import ContactObserver, read, save, sha, validate_frames
from test_analyze_contact_efficiency import file_fixture, pose

NATIVE = (0, 1, 'surface', 'surface')
COMPETING = (0, 2, 'surface', 'surface')
SECOND = (1, 2, 'surface', 'surface')
DEFINITIONS = [dict(id='native', required_tokens=[NATIVE, SECOND]),
               dict(id='competing', required_tokens=[COMPETING], forbidden_tokens=[NATIVE, SECOND])]


def biased_fixture(root, seed=811, values=None):
    run, plan_path = file_fixture(root, seed)
    c, m, s, cp = [read(run/(name+'.json')) for name in ('config', 'manifest', 'summary', 'checkpoint')]
    values = [0., math.log(1.5), math.log(3.)] if values is None else values
    table = dict(values=values)
    for obj in (c, m, s, cp): obj['assembly_bias'] = copy.deepcopy(table)
    m.update(assembly_bias_protocol=bias.PROTOCOL, physical_target=bias.TARGET,
             assembly_bias_reweighting=bias.REWEIGHTING)
    frames = [json.loads(line) for line in (run/'trajectory.jsonl').read_text().splitlines()]
    for frame in frames:
        step = frame['sweep']
        if step % 4 in (1, 2): frame['poses'] = [pose(0), pose(.7), pose(1.4)]
        elif step % 4 == 3: frame['poses'] = [pose(0), pose(10), pose(.7)]
        else: frame['poses'] = [pose(0), pose(5), pose(10)]
        size = 3 if step % 4 in (1, 2) else 2 if step % 4 == 3 else 1
        frame['assembly_bias'] = dict(largest_component_size=size, bias=values[size-1], log_reweight=values[size-1])
        gate = {kind: dict(physical_accepted=0, accepted=0, rejected=0) for kind in ('local', 'global', 'gca', 'center_shift')}
        reject = 0 if min(values) == max(values) else step
        gate['local'] = dict(physical_accepted=step+reject, accepted=step, rejected=reject)
        frame['counts']['assembly_bias'] = gate
    c['initial_poses'] = copy.deepcopy(frames[0]['poses'])
    cp['poses'] = copy.deepcopy(frames[-1]['poses'])
    cp['counts'] = copy.deepcopy(frames[-1]['counts'])
    s['counts'] = copy.deepcopy(cp['counts'])
    s['initial_counts'] = copy.deepcopy(frames[0]['counts'])
    s['segment_counts'] = copy.deepcopy(cp['counts'])
    cp['assembly_bias_state'] = copy.deepcopy(frames[-1]['assembly_bias'])
    s['assembly_bias_state'] = copy.deepcopy(cp['assembly_bias_state'])
    declared = read(run/'provenance/input-config.json')
    declared['assembly_bias'] = copy.deepcopy(table); declared['initial_poses'] = c['initial_poses']
    save(run/'provenance/input-config.json', declared)
    for obj in (m, s, cp): obj['config_sha256'] = sha(run/'provenance/input-config.json')
    for name, value in [('config', c), ('manifest', m), ('summary', s), ('checkpoint', cp)]: save(run/(name+'.json'), value)
    (run/'trajectory.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
    plan = read(plan_path); plan.update(schema=bias.PLAN_SCHEMA, environments=DEFINITIONS)
    save(plan_path, plan)
    return run, plan_path


def load_fixture(run):
    c, m, s, cp = [read(run/(name+'.json')) for name in ('config', 'manifest', 'summary', 'checkpoint')]
    frames = [json.loads(line) for line in (run/'trajectory.jsonl').read_text().splitlines()]
    observer = ContactObserver(read(run/'provenance/shape.json'), ['surface'], c)
    observations = [observer.classify(f['poses']) for f in frames]
    return frames, observations, c, m, s, cp


class BiasedContactOccupancyTests(unittest.TestCase):
    def test_exact_finite_native_competing_unbound_ensemble_and_wrong_sign(self):
        # Physical mass (.6,.3,.1), exp B=(3,1.5,1): biased mass (.4,.4,.2).
        labels = ['native']*4+['competing']*4+['remaining']*2
        logs = [math.log(3.)]*4+[math.log(1.5)]*4+[0.]*2
        ids = ['native', 'competing', 'remaining']
        correct = bias.weight_statistics(logs, labels, ids)
        np.testing.assert_allclose(list(correct['fractions'].values()), [.6, .3, .1], atol=1e-15)
        wrong = bias.weight_statistics([-x for x in logs], labels, ids)
        self.assertGreater(abs(wrong['fractions']['native']-.6), .3)
        self.assertAlmostEqual(sum(correct['fractions'].values()), 1.)
        self.assertLess(correct['weight_concentration']['effective_count'], len(labels))
        # The identity is exact for the fully enumerated measure. A finite random ratio is biased.
        target, weights = np.asarray([.6, .3, .1]), np.asarray([3., 1.5, 1.])
        sampled = target/weights; sampled /= sampled.sum()
        expected_n1_ratio = sampled[0]
        self.assertNotAlmostEqual(expected_n1_ratio, target[0])

    def test_bias_gauge_and_constant_limit_and_extreme_log_weights(self):
        labels = ['native', 'remaining', 'native', 'competing']; ids = ['native', 'competing', 'remaining']
        base = bias.weight_statistics([1., 0., 1., -.5], labels, ids)
        shifted = bias.weight_statistics([1001., 1000., 1001., 999.5], labels, ids)
        self.assertEqual(base['fractions'], shifted['fractions'])
        self.assertEqual(base['weight_concentration'], shifted['weight_concentration'])
        for shift in (0., 1000., -1000.):
            result = bias.weight_statistics([shift]*4, labels, ids)
            self.assertEqual(result['fractions'], dict(native=.5, competing=.25, remaining=.25))
            self.assertEqual(result['weight_concentration']['effective_count'], 4.)
        extreme = bias.weight_statistics([-1000., 1000.], ['native', 'remaining'], ids)
        self.assertEqual(extreme['fractions']['remaining'], 1.)
        self.assertEqual(extreme['weight_concentration']['numerically_underflowed_weights'], 1)
        json.dumps(extreme, allow_nan=False)

    def test_geometric_score_strict_tangency_periodic_and_no_history(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = biased_fixture(Path(temp)); _, _, config, *_ = load_fixture(run)
            shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.25)])
            observer = ContactObserver(shape, ['surface'], config)
            for xs, size in [([0., 1., 4.], 1), ([0., .999, 4.], 2), ([0., .999, 1.998], 3), ([0., 2., 4.], 1)]:
                obs = observer.classify([pose(x) for x in xs])
                self.assertEqual(bias.largest_component_size(obs['tokens'], 3), size)
            config['boundary'] = dict(kind='periodic'); config['box_lengths'] = [10.]*3
            observer = ContactObserver(shape, ['surface'], config)
            obs = observer.classify([pose(.1), pose(9.2), pose(5)])
            self.assertEqual(bias.largest_component_size(obs['tokens'], 3), 2)

    def test_complete_file_analysis_preserves_repeats_remaining_and_both_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, plan = biased_fixture(root)
            report = bias.analyze(run, plan, root/'out')
            self.assertEqual(report['weight_concentration']['samples'], 8)
            self.assertEqual(report['physical_occupancies']['native']['biased_observations'], 4)
            self.assertEqual(report['physical_occupancies']['remaining']['biased_observations'], 2)
            np.testing.assert_allclose([report['physical_occupancies'][k]['physical_fraction'] for k in
                                       ('native', 'competing', 'remaining')], [12/17, 3/17, 2/17])
            self.assertEqual(len((root/'out/observations.jsonl').read_text().splitlines()), 9)
            self.assertEqual(set(report['implementation_sha256']),
                {'analyze_biased_contact_occupancy.py', 'analyze_contact_efficiency.py', 'mobile_posterior_metrics.py'})
            self.assertIn('not finite-sample unbiased', report['scope'])
            self.assertEqual(report['biased_chain_diagnostics']['fingerprint']['samples'], 8)
            json.dumps(report, allow_nan=False)

    def test_bias_schema_table_state_protocol_and_counters_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = biased_fixture(Path(temp)); pristine = load_fixture(run)
            bias.validate_bias(*pristine)
            for mode in ('table', 'schema', 'nonfinite', 'difference', 'checkpoint', 'size', 'wrong_sign',
                         'missing_state', 'unknown_state', 'missing_counts', 'count_sum', 'final_accepts',
                         'physical_accepts', 'count_bool', 'checkpoint_bool', 'table_bool', 'protocol'):
                f, obs, c, m, s, cp = copy.deepcopy(pristine)
                if mode == 'table': cp['assembly_bias']['values'][0] += 1
                elif mode == 'schema': c['assembly_bias']['adaptive'] = True
                elif mode == 'nonfinite': c['assembly_bias']['values'][0] = float('nan')
                elif mode == 'difference': c['assembly_bias']['values'] = [-1e308, 0., 1e308]
                elif mode == 'checkpoint': cp['assembly_bias_state']['bias'] += 1
                elif mode == 'size': f[1]['assembly_bias']['largest_component_size'] = 2
                elif mode == 'wrong_sign': f[1]['assembly_bias']['log_reweight'] *= -1
                elif mode == 'missing_state': del f[1]['assembly_bias']
                elif mode == 'unknown_state': f[1]['assembly_bias']['ancestral_label'] = 3
                elif mode == 'missing_counts': del f[1]['counts']['assembly_bias']
                elif mode == 'count_sum': f[1]['counts']['assembly_bias']['local']['rejected'] += 1
                elif mode == 'final_accepts': f[1]['counts']['local']['accepted'] = 0
                elif mode == 'physical_accepts': f[1]['counts']['local']['hard_valid'] = 0
                elif mode == 'count_bool': f[1]['counts']['assembly_bias']['local']['accepted'] = True
                elif mode == 'checkpoint_bool': cp['assembly_bias_state']['largest_component_size'] = True
                elif mode == 'table_bool': cp['assembly_bias']['values'][0] = False
                else: m['assembly_bias_protocol'] = 'post-sweep-gate'
                with self.subTest(mode=mode), self.assertRaises(ValueError): bias.validate_bias(f, obs, c, m, s, cp)

    def test_collective_bias_rejections_complete_physical_kernel_but_retain_no_transform(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = biased_fixture(Path(temp)); pristine = load_fixture(run)
            for kind in ('gca', 'center_shift'):
                f, obs, c, m, s, cp = copy.deepcopy(pristine)
                for frame in f:
                    attempts = frame['sweep']; accepted = max(0, attempts-1)
                    frame['counts'][kind] = dict(attempted=attempts, completed=attempts,
                        transformed_bodies=(2 if kind == 'gca' else 3)*accepted)
                    frame['counts']['assembly_bias'][kind] = dict(physical_accepted=attempts,
                        accepted=accepted, rejected=attempts-accepted)
                cp['counts'] = copy.deepcopy(f[-1]['counts'])
                s['counts'] = copy.deepcopy(cp['counts'])
                s['initial_counts'] = copy.deepcopy(f[0]['counts'])
                s['segment_counts'] = copy.deepcopy(cp['counts'])
                validate_frames(f, c, s, m, cp, 0, 8, allow_frozen_assembly_bias=True)
                bias.validate_bias(f, obs, c, m, s, cp)
                # First completed kernel is rejected by the additional gate.
                self.assertEqual(f[1]['counts'][kind]['completed'], 1)
                self.assertEqual(f[1]['counts']['assembly_bias'][kind]['accepted'], 0)
                damaged = copy.deepcopy(f[1]['counts'])
                damaged[kind]['transformed_bodies'] = 1
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'transformed-body'):
                    bias.validate_bias_counts(damaged)
                damaged = copy.deepcopy(f[1]['counts'])
                damaged[kind]['completed'] = 0
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'physical completion'):
                    bias.validate_bias_counts(damaged)
            # Accepted zero-flip GCA is valid; center shifts count every body.
            counts = copy.deepcopy(pristine[0][0]['counts'])
            counts['gca'] = dict(attempted=1, completed=1, transformed_bodies=0)
            counts['assembly_bias']['gca'] = dict(physical_accepted=1, accepted=1, rejected=0)
            bias.validate_bias_counts(counts)

    def test_missing_frames_accepted_only_and_adaptive_mode_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, plan = biased_fixture(root)
            trajectory = run/'trajectory.jsonl'; original = trajectory.read_text()
            trajectory.write_text('\n'.join(original.splitlines()[1:])+'\n')
            with self.assertRaisesRegex(ValueError, 'saved endpoints'): bias.analyze(run, plan, root/'out')
            trajectory.write_text(original)
            c = read(run/'config.json'); c['contact_memory'] = dict(enabled=True); save(run/'config.json', c)
            with self.assertRaises(ValueError): bias.analyze(run, plan, root/'out')
            self.assertFalse((root/'out').exists())

    def test_constant_table_file_limit_and_unbiased_observer_stays_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, plan = biased_fixture(root, values=[900.]*3)
            report = bias.analyze(run, plan, root/'out')
            self.assertEqual(report['weight_concentration']['effective_count'], 8.)
            for occupancy in report['physical_occupancies'].values():
                self.assertEqual(occupancy['physical_fraction'], occupancy['biased_fraction'])
            f, obs, c, m, s, cp = load_fixture(run)
            with self.assertRaisesRegex(ValueError, 'unbiased'): validate_frames(f, c, s, m, cp, 0, 8)
            f[1]['counts']['assembly_bias']['local'].update(physical_accepted=2, rejected=1)
            with self.assertRaisesRegex(ValueError, 'Constant'): bias.validate_bias(f, obs, c, m, s, cp)

    def test_native_registry_is_independent_of_cluster_score(self):
        class Native:
            motifs = [dict(id=0, relative_position=[.7, 0, 0], relative_orientation=[1., 0, 0, 0])]
            def classify_pair(self, a, b):
                return [dict(motif_id=0)] if abs(b['position'][0]-a['position'][0]-.7) < 1e-9 else []
        with tempfile.TemporaryDirectory() as temp:
            run, _ = biased_fixture(Path(temp)); _, _, c, *_ = load_fixture(run)
            observer = ContactObserver(read(run/'provenance/shape.json'), ['surface'], c, Native())
            native = observer.classify([pose(0), pose(.7), pose(1.4)])
            competing = observer.classify([pose(0), pose(.8), pose(1.6)])
            self.assertEqual(bias.largest_component_size(native['tokens'], 3), 3)
            self.assertEqual(bias.largest_component_size(competing['tokens'], 3), 3)
            self.assertTrue(native['native_cycle']['consistent_connected'])
            self.assertFalse(competing['native_cycle']['consistent_connected'])
            self.assertEqual(observer.classify([pose(0), pose(4), pose(8)])['instantaneous_native_keys'], [])

    def test_weight_concentration_is_not_autocorrelation_or_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            run, _ = biased_fixture(Path(temp), values=[0.]*3)
            f, obs, c, m, s, cp = load_fixture(run)
            window = validate_frames(f, c, s, m, cp, 0, 8, allow_frozen_assembly_bias=True)
            obs = [dict(tokens=[], native_cycle=None) for _ in f]
            report = bias.summarize_occupancies(obs, f, window, DEFINITIONS)
            self.assertEqual(report['weight_concentration']['effective_count'], 8.)
            for value in report['physical_occupancies'].values():
                self.assertIsNone(value['ratio_residual_autocorrelation']['apparent_ess'])
            self.assertFalse(report['physical_occupancies']['native']['observed_support'])
            self.assertEqual(report['physical_occupancies']['remaining']['physical_fraction'], 1.)

    def test_independent_ratio_covariance_and_log_gauge(self):
        # Ratio numerator and denominator covary; a numerator-only SE would be incorrect.
        moments = [dict(log_scale=0., mean_scaled_weight=d, mean_scaled_weighted_indicators=dict(native=n))
                   for n, d in [(1., 2.), (2., 3.), (3., 4.), (2., 5.)]]
        result = bias.independent_ratio_diagnostic(moments, 'native')
        self.assertAlmostEqual(result['ratio_of_stream_moments'], 8/14)
        p = 8/14; residual = np.array([1, 2, 3, 2])-p*np.array([2, 3, 4, 5])
        self.assertAlmostEqual(result['delta_method_standard_error'], residual.std(ddof=1)/2/3.5)
        shifted = copy.deepcopy(moments)
        for row in shifted: row['log_scale'] += 1000
        self.assertEqual(bias.independent_ratio_diagnostic(shifted, 'native')['delta_method_standard_error'],
                         result['delta_method_standard_error'])
        self.assertIsNone(bias.independent_ratio_diagnostic(moments[:1], 'native')['delta_method_standard_error'])

    def test_independent_files_reject_reused_streams_and_tampered_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths = []
            for index in range(2):
                run, plan = biased_fixture(root/str(index), seed=811+index)
                bias.analyze(run, plan, root/str(index)/'out')
                paths.append(root/str(index)/'out/analysis.json')
            report = bias.compare_report_files(paths)
            self.assertEqual(report['groups'][0]['populations'], 2)
            self.assertIn('equal_stream_ratio_mean', report['groups'][0]['physical_occupancies']['native'])
            with self.assertRaisesRegex(ValueError, 'reused/paired'): bias.compare_report_files([paths[0], paths[0]])
            paths[0].write_text(paths[0].read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'Frozen occupancy report changed'): bias.compare_report_files(paths)

    def test_dependency_recheck_and_conditional_reference_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, plan = biased_fixture(root)
            original = bias.ContactObserver.classify; changed = False
            def mutate(observer, poses):
                nonlocal changed
                result = original(observer, poses)
                if not changed:
                    (root/'patch.json').write_text((root/'patch.json').read_text()+' '); changed = True
                return result
            with mock.patch.object(bias.ContactObserver, 'classify', mutate):
                with self.assertRaisesRegex(ValueError, 'dependency changed'): bias.analyze(run, plan, root/'out')
            self.assertFalse((root/'out').exists())
            settings = read(plan); settings['physical_reference'] = dict(path='conditional.json')
            with self.assertRaisesRegex(ValueError, 'conditional references'): bias.validate_definitions(settings)


if __name__ == '__main__':
    unittest.main()
