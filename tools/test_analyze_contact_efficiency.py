"""Synthetic retained histories only. No MC, depletant cloud or protein audit."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from analyze_contact_efficiency import (ContactObserver, PLAN_SCHEMA, SCHEMA, CONFIG_DEFAULTS,
    FRAME_CONVENTION, GATE_DEFAULTS, analyze, compare_report_files, compare_reports, digest,
    environment_label, native_input_bindings, physical_identity, reference_assessment,
    sampled_exchanges, save, sha, summarize_observations, validate_effective_config,
    validate_frames, validate_probability_bounds)


TOKEN_A = (0, 1, 'surface', 'surface')
TOKEN_B = (0, 2, 'surface', 'surface')
DEFINITIONS = [dict(id='A', required_tokens=[TOKEN_A], forbidden_tokens=[TOKEN_B]),
               dict(id='B', required_tokens=[TOKEN_B], forbidden_tokens=[TOKEN_A])]


def pose(x):
    return dict(position=[x, 0., 0.], orientation=[1., 0., 0., 0.])


def counts(sweep):
    zero = dict(attempted=0, hard_valid=0, accepted=0, hard_rejected=0, proposal_nulls=0)
    return dict(local=dict(attempted=3*sweep, hard_valid=2*sweep, accepted=sweep,
                           hard_rejected=sweep, proposal_nulls=0), **{'global': zero},
                gca=dict(attempted=0, completed=0, transformed_bodies=0),
                center_shift=dict(attempted=0, completed=0, transformed_bodies=0),
                selected_body_updates=3*sweep, selected_body_updates_by_body=[sweep]*3)


def fixture(total=8, stride=1):
    config = copy.deepcopy(CONFIG_DEFAULTS) | dict(shape='/synthetic/shape.json', initial_poses=[pose(0), pose(.7), pose(10)], seed=811,
                  boundary=dict(kind='spherical', radius=20.), box_lengths=[40.]*3,
                  depletant_radius=.25, reservoir_density=.035, sample_every=stride, sweeps=total,
                  method='local-uniform', local_translation_std_A=.2, local_small_angle_std_degrees=1.,
                  global_probability=.5, gca_probability=0., center_shift_probability=0.,
                  uniform_proposal_cube_lengths=[40.4]*3, coordinate_origin_sweep=0,
                  resolved_contact_memory=None, coordinate_frame_convention=FRAME_CONVENTION)
    frames = []
    for s in [0]+[k for k in range(1, total+1) if k % stride == 0 or k == total]:
        ps = [pose(0), pose(.7), pose(10)] if (s//2) % 2 == 0 else [pose(0), pose(10), pose(.7)]
        frames.append(dict(sweep=s, poses=ps, counts=counts(s), sampler_cpu_seconds=.01+.5*s,
                           boundary='spherical', spherical_wall_radius=20.))
    summary = dict(complete=True, all_bodies_mobile=True, bodies=3, initial_sweep=0,
                   completed_sweeps=total, requested_sweeps=total, counts=counts(total), initial_counts=counts(0),
                   segment_counts=counts(total), sampler_cpu_seconds=.02+.5*total, method='local-uniform', initial_metadata=None)
    manifest = dict(initial_sweep=0, model_sha256=None, boundary=copy.deepcopy(config['boundary']), resume=None)
    checkpoint = dict(poses=frames[-1]['poses'], counts=counts(total), completed_sweeps=total,
                      master_seed=811, method='local-uniform')
    return config, frames, summary, manifest, checkpoint


def file_fixture(root, seed=811):
    run = root/'run'; (run/'provenance').mkdir(parents=True)
    c, frames, s, m, cp = fixture()
    c['seed'] = cp['master_seed'] = seed
    shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.2)])
    save(run/'provenance/shape.json', shape)
    declared = {k: v for k, v in c.items() if k not in ('method', 'sweeps', 'sample_every',
        'uniform_proposal_cube_lengths', 'coordinate_origin_sweep', 'resolved_contact_memory',
        'coordinate_frame_convention')}
    declared['shape'] = 'shape.json'
    save(run/'provenance/input-config.json', declared)
    save(run/'provenance/source-bundle.json', dict(schema=1, files={}))
    for x in (s, m, cp):
        x.update(shape_sha256=sha(run/'provenance/shape.json'), config_sha256=sha(run/'provenance/input-config.json'), model_sha256=None)
    m['source_bundle_sha256'] = sha(run/'provenance/source-bundle.json')
    for name, value in [('config', c), ('manifest', m), ('summary', s), ('checkpoint', cp)]: save(run/(name+'.json'), value)
    (run/'trajectory.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
    save(root/'patch.json', dict(schema='body-frame-atom-patch-map-v1', shape_sha256=m['shape_sha256'], atom_patch_ids=['surface']))
    plan = dict(schema=PLAN_SCHEMA, physical_identity_sha256=digest(physical_identity(c, m['shape_sha256'])),
                patch_map=dict(path='patch.json', sha256=sha(root/'patch.json')), environments=DEFINITIONS,
                burn_sweep=0, end_sweep=8, preparation_id='synthetic', proposal_arm='local+uniform-global')
    save(root/'plan.json', plan)
    return run, root/'plan.json'


class ContactEfficiencyTests(unittest.TestCase):
    def window(self, data, burn=0, end=None):
        c, f, s, m, cp = data
        return validate_frames(f, c, s, m, cp, burn, c['sweeps'] if end is None else end)

    def test_retained_repeats_counts_and_cpu_are_preserved(self):
        data = fixture(); window = self.window(data)
        c, frames, *_ = data
        observer = ContactObserver(dict(atoms=[dict(center=[0., 0., 0.], radius=.2)]), ['surface'], c)
        obs = [observer.classify(f['poses']) for f in frames]
        report = summarize_observations(obs, frames, window, DEFINITIONS)
        self.assertEqual(window['sample_indices'], list(range(1, 9)))
        self.assertEqual(window['cpu_seconds'], 4.)
        self.assertEqual(report['fingerprint']['samples'], 8)
        self.assertEqual(report['environment_occupancies']['A']['fraction'], .5)
        self.assertEqual(report['environment_occupancies']['B']['fraction'], .5)
        self.assertEqual(report['environment_exchanges']['pairs'][0]['forward'], 2)
        self.assertEqual(report['environment_exchanges']['pairs'][0]['reverse'], 2)
        self.assertEqual(report['environment_exchanges']['pairs'][0]['completed_nonoverlapping_roundtrips'], 2)
        self.assertEqual(report['environment_exchanges']['pairs'][0]['equilibrium_interpretation']['status'], 'unassessed_equilibrium_weights')
        json.dumps(report, allow_nan=False)

    def test_missing_accepted_only_or_duplicate_frames_fail_closed(self):
        for mode in ('missing', 'duplicate'):
            d = fixture()
            if mode == 'missing': del d[1][1]
            else: d[1].insert(1, copy.deepcopy(d[1][0]))
            with self.assertRaisesRegex(ValueError, 'saved endpoints'): self.window(d)

    def test_missing_cpu_or_attempt_dispositions_fail_closed(self):
        for kind in ('cpu', 'counts', 'body'):
            d = fixture()
            if kind == 'cpu': d[1][2]['sampler_cpu_seconds'] = d[1][1]['sampler_cpu_seconds']
            elif kind == 'counts': d[1][2]['counts']['local']['attempted'] += 1
            else: d[1][2]['counts']['selected_body_updates_by_body'][0] += 1
            with self.assertRaises(ValueError): self.window(d)

    def test_irregular_terminal_requires_declared_regular_window(self):
        d = fixture(total=7, stride=2)
        with self.assertRaisesRegex(ValueError, 'Irregular'): self.window(d)
        w = self.window(d, end=6)
        self.assertEqual(w['sample_indices'], [1, 2, 3])
        self.assertEqual(w['cpu_seconds'], 3.)
        self.assertEqual(w['cadence_sweeps'], 2)

    def test_burn_is_endpoint_not_an_extra_sample(self):
        d = fixture(); w = self.window(d, burn=4)
        self.assertEqual(w['indices'], [4, 5, 6, 7, 8])
        self.assertEqual(w['sample_indices'], [5, 6, 7, 8])
        self.assertEqual(w['cpu_seconds'], 2.)

    def test_bias_auxiliary_and_missing_checkpoint_fail_closed(self):
        for kind in ('assembly_bias', 'atlas_transport', 'checkpoint'):
            d = fixture()
            if kind == 'checkpoint': d[4]['poses'] = [pose(9)]*3
            else: d[0][kind] = dict(enabled=True)
            with self.assertRaises(ValueError): self.window(d)

    def test_constant_or_unseen_contacts_have_no_invented_ess(self):
        d = fixture(); w = self.window(d)
        obs = [dict(tokens=[], native_cycle=None) for _ in d[1]]
        r = summarize_observations(obs, d[1], w, DEFINITIONS)
        self.assertIsNone(r['fingerprint']['apparent_ess'])
        self.assertIsNone(r['environment_occupancies']['A']['apparent_ess']['apparent_ess'])
        self.assertEqual(r['environment_occupancies']['remaining']['fraction'], 1.)

    def test_exchanges_allow_remaining_and_preserve_censoring(self):
        x = sampled_exchanges(['A', 'A', 'remaining', 'B', 'B', 'remaining', 'A'], list(range(7)), {'A', 'B'})
        self.assertEqual(len(x['events']), 2)
        self.assertEqual(x['events'][0]['intervening_saved_labels'], ['remaining'])
        self.assertEqual(x['events'][0]['source_last_observed_sweep'], 1)
        self.assertEqual(x['pairs'][0]['completed_nonoverlapping_roundtrips'], 1)
        self.assertTrue(x['sampled_episodes'][0]['left_censored'])
        self.assertTrue(x['sampled_episodes'][-1]['right_censored'])
        x = sampled_exchanges(['A', 'B', 'C', 'A'], list(range(4)), {'A', 'B', 'C'})
        ab = next(p for p in x['pairs'] if p['regions'] == ['A', 'B'])
        self.assertEqual(ab['completed_nonoverlapping_roundtrips'], 0)

    def test_unequal_weights_do_not_require_equal_transition_rates(self):
        pair = dict(regions=['A', 'B'], forward=3, reverse=1)
        ref = dict(probability_bounds={'A': [.85, .95], 'B': [.02, .08]})
        self.assertTrue(reference_assessment(pair, ref, .01)['completed_both_directions'])
        pair['reverse'] = 0
        self.assertTrue(reference_assessment(pair, ref, .01)['mixing_limitation_flag'])
        ref['probability_bounds']['B'] = [0., .00001]
        r = reference_assessment(pair, ref, .01)
        self.assertFalse(r['frequent_returns_required'])
        self.assertFalse(r['missing_transition_is_failure'])
        self.assertEqual(reference_assessment(pair, None, .01)['status'], 'unassessed_equilibrium_weights')

    def test_stateless_environment_partition_is_exhaustive_and_overlap_fails(self):
        obs = dict(tokens=[TOKEN_A], native_cycle=None)
        self.assertEqual(environment_label(obs, DEFINITIONS), 'A')
        self.assertEqual(environment_label(dict(tokens=[], native_cycle=None), DEFINITIONS), 'remaining')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            environment_label(obs, [dict(id='a'), dict(id='b')])
        with self.assertRaisesRegex(ValueError, 'frozen native observer'):
            environment_label(obs, [dict(id='a', native_consistent_connected=True)])

    def test_periodic_contact_and_atomic_wall_predicates(self):
        c = fixture()[0]; c['boundary'] = dict(kind='periodic'); c['box_lengths'] = [10.]*3
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.2)])
        obs = ContactObserver(shape, ['surface'], c)
        self.assertEqual(obs.classify([pose(.1), pose(9.4), pose(5)])['tokens'], [TOKEN_A])
        with self.assertRaisesRegex(ValueError, 'hard overlap'):
            obs.classify([pose(.1), pose(9.9), pose(5)])
        c['box_lengths'] = [1.]*3
        with self.assertRaisesRegex(ValueError, 'minimum-image'): ContactObserver(shape, ['surface'], c)
        c = fixture()[0]; obs = ContactObserver(shape, ['surface'], c)
        with self.assertRaisesRegex(ValueError, 'atomic wall'): obs.classify([pose(20.), pose(0), pose(10)])

    def test_patch_ids_track_surface_change_at_fixed_partner(self):
        c = fixture()[0]
        shape = dict(atoms=[dict(center=[-.3, 0, 0], radius=.1), dict(center=[.3, 0, 0], radius=.1)])
        observer = ContactObserver(shape, ['left', 'right'], c)
        first = observer.classify([pose(0), pose(.85), pose(10)])['tokens']
        second = observer.classify([pose(0), pose(-.85), pose(10)])['tokens']
        self.assertIn((0, 1, 'right', 'left'), first)
        self.assertIn((0, 1, 'left', 'right'), second)
        self.assertNotEqual(first, second)

    def test_native_pair_entry_and_cycle_adapter_are_instantaneous(self):
        class Native:
            motifs = [dict(id=0, relative_position=[.7, 0, 0], relative_orientation=[1., 0, 0, 0])]
            def classify_pair(self, anchor, partner):
                return [dict(motif_id=0)] if abs(partner['position'][0]-anchor['position'][0]-.7) < 1e-9 else []
        c = fixture()[0]
        observer = ContactObserver(dict(atoms=[dict(center=[0, 0, 0], radius=.2)]), ['surface'], c, Native())
        a = observer.classify([pose(0), pose(.7), pose(1.4)])
        self.assertEqual(a['instantaneous_native_keys'], [(0, 1, 0), (1, 2, 0)])
        self.assertTrue(a['native_cycle']['consistent_connected'])
        b = observer.classify([pose(0), pose(5), pose(10)])
        self.assertEqual(b['instantaneous_native_keys'], [])
        self.assertFalse(b['native_cycle']['connected'])

    def test_periodic_native_lift_rejects_winding_cycles(self):
        class Native:
            motifs = [dict(id=0, relative_position=[.7, 0, 0], relative_orientation=[1., 0, 0, 0])]
            def classify_pair(self, anchor, partner):
                return [dict(motif_id=0)]
        c = fixture()[0]; c['boundary'] = dict(kind='periodic'); c['box_lengths'] = [10.]*3
        observer = ContactObserver(dict(atoms=[dict(center=[0, 0, 0], radius=.2)]), ['surface'], c, Native())
        with self.assertRaisesRegex(ValueError, 'nonzero winding'):
            observer.classify([pose(0), pose(3.3), pose(6.6)])
        observer.classify([pose(9.5), pose(.5), pose(1.5)])  # Boundary-crossing finite lift exists.

    def test_independent_population_comparison_never_concatenates_chains(self):
        base = dict(schema=SCHEMA, complete=True, physical_identity={}, region_definition_sha256='x',
                    measurement_identity={}, observer_definition_sha256='obs',
                    implementation_sha256={'analyze_contact_efficiency.py': 'observer', 'mobile_posterior_metrics.py': 'ess'},
                    schedule={}, proposal={}, window=dict(burn_sweep=0, end_sweep=10),
                    initialization=dict(master_seed=1, proposal_arm='local', preparation_id='dispersed', initial_poses_sha256='a'),
                    fingerprint=dict(apparent_ess_per_sampling_CPU_second=None),
                    environment_occupancies=dict(A=dict(fraction=.1), B=dict(fraction=.9)))
        reports = []
        for i, (start, value) in enumerate([('dispersed', .1), ('dispersed', .3), ('seeded', .6), ('seeded', .8)]):
            r = copy.deepcopy(base); r['initialization'].update(master_seed=i+1, preparation_id=start)
            r['environment_occupancies']['A']['fraction'] = value
            r['environment_occupancies']['B']['fraction'] = 1-value
            reports.append(r)
        out = compare_reports(reports)
        self.assertAlmostEqual(out['groups'][0]['region_occupancies']['A']['mean'], .2)
        self.assertAlmostEqual(out['groups'][0]['region_occupancies']['A']['population_standard_error'], .1)
        self.assertIsNone(out['groups'][0]['fingerprint_ess_per_cpu'])
        self.assertEqual(len(out['contrasts']), 2)
        reports[-1]['initialization']['master_seed'] = 1
        with self.assertRaisesRegex(ValueError, 'reused/paired'): compare_reports(reports)

    def test_complete_file_path_freezes_inputs_and_never_runs_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, _ = file_fixture(root)
            result = analyze(run, root/'plan.json', root/'out')
            self.assertEqual(result['fingerprint']['samples'], 8)
            self.assertEqual(result['initialization']['preparation_id'], 'synthetic')
            self.assertEqual(len((root/'out/observations.jsonl').read_text().splitlines()), 9)
            with self.assertRaisesRegex(ValueError, 'fresh output'): analyze(run, root/'plan.json', root/'out')


    def test_effective_settings_cannot_depart_from_archived_input(self):
        changes = dict(reservoir_density=.04, depletant_radius=.3, seed=9,
                       global_probability=.2, poisson_lambda_ratio=32.,
                       endpoint_gate=dict(max_cells=7, max_depth=14, min_width=.5))
        for field, value in changes.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); run, plan = file_fixture(root)
                config = json.loads((run/'config.json').read_text()); config[field] = value
                save(run/'config.json', config)
                with self.assertRaisesRegex(ValueError, 'Effective config differs'):
                    analyze(run, plan, root/'out')

    def test_rust_defaults_cli_overrides_and_canonical_paths_are_reconciled(self):
        c, _, summary, manifest, _ = fixture()
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.2)])
        declared = {k: c[k] for k in ('shape', 'box_lengths', 'boundary', 'initial_poses',
                                     'seed', 'depletant_radius', 'reservoir_density')}
        declared.update(shape='shape.json', method='ignored-by-serde', sweeps=900,
                        sample_every=100, endpoint_gate={'max_depth': 14},
                        metadata={'native_pair_motifs': 'motifs.json', 'preparation': 'synthetic'})
        c['metadata'] = {'native_pair_motifs': '/synthetic/motifs.json', 'preparation': 'synthetic'}
        summary['initial_metadata'] = copy.deepcopy(c['metadata'])
        validate_effective_config(c, declared, manifest, summary, shape)
        c['metadata']['preparation'] = 'altered'
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            validate_effective_config(c, declared, manifest, summary, shape)

    def test_initial_pose_override_requires_periodic_wrapping_or_declared_resume(self):
        c, _, summary, manifest, _ = fixture()
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.2)])
        declared = copy.deepcopy(c); c['initial_poses'][1] = pose(8.)
        with self.assertRaisesRegex(ValueError, 'without a resumed'):
            validate_effective_config(c, declared, manifest, summary, shape)
        manifest.update(resume='checkpoint.json', initial_sweep=4)
        validate_effective_config(c, declared, manifest, summary, shape)
        manifest.update(resume=None, initial_sweep=0, boundary={'kind': 'periodic'})
        declared['boundary'] = c['boundary'] = {'kind': 'periodic'}
        declared['initial_poses'][0] = pose(-.5)
        c['initial_poses'] = copy.deepcopy(declared['initial_poses']); c['initial_poses'][0] = pose(39.5)
        c['uniform_proposal_cube_lengths'] = c['box_lengths']
        validate_effective_config(c, declared, manifest, summary, shape)

    def test_reference_intervals_must_cover_unit_total_probability(self):
        validate_probability_bounds({'A': [.2, .6], 'B': [.1, .4], 'remaining': [0., .7]}, ['A', 'B'])
        validate_probability_bounds({'A': [.25, .25], 'B': [.5, .5], 'remaining': [.25, .25]}, ['A', 'B'])
        for bounds in ({'A': [.6, .7], 'B': [.5, .6], 'remaining': [0., .2]},
                       {'A': [.1, .2], 'B': [.1, .2], 'remaining': [0., .2]}):
            with self.assertRaisesRegex(ValueError, 'exhaustive partition'):
                validate_probability_bounds(bounds, ['A', 'B'])

    def test_native_dependency_closure_binds_all_inputs_and_rechecks_after_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, plan_path = file_fixture(root)
            native_root = root/'native'; (native_root/'inputs').mkdir(parents=True)
            names = ['extra'+str(i)+'.json' for i in range(15)]
            for name in names: save(native_root/'inputs'/name, {'synthetic': name})
            definition = dict(input_sha256={name: sha(native_root/'inputs'/name) for name in names})
            native_path = native_root/'definition.json'; save(native_path, definition)
            self.assertEqual(len(native_input_bindings(native_path)), 16)
            plan = json.loads(plan_path.read_text())
            plan['native_definition'] = dict(path=str(native_path), sha256=sha(native_path)); save(plan_path, plan)
            native = SimpleNamespace(definition_sha256=sha(native_path),
                shape_sha256=sha(run/'provenance/shape.json'),
                motifs=[dict(id=0, relative_position=[.7, 0, 0], relative_orientation=[1., 0, 0, 0])],
                classify_pair=lambda *_: [])
            with mock.patch('native_contact_regions.NativeContactRegions', return_value=native):
                result = analyze(run, plan_path, root/'out')
                self.assertTrue(all(str(native_root/'inputs'/name) in result['dependency_sha256'] for name in names))
                original = ContactObserver.classify
                def mutate_after_initial_binding(observer, poses):
                    save(native_root/'inputs'/names[-1], {'changed': True})
                    return original(observer, poses)
                with mock.patch.object(ContactObserver, 'classify', mutate_after_initial_binding):
                    with self.assertRaisesRegex(ValueError, 'dependency changed during observation'):
                        analyze(run, plan_path, root/'out-mutated')

    def test_native_dependency_paths_cannot_escape_through_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'inputs').mkdir()
            save(root/'outside.json', {'outside': True})
            (root/'inputs/linked.json').symlink_to(root/'outside.json')
            save(root/'definition.json', dict(input_sha256={'linked.json': sha(root/'outside.json')}))
            with self.assertRaisesRegex(ValueError, 'escapes'):
                native_input_bindings(root/'definition.json')

    def test_comparison_checks_freezes_and_matches_cost_noise_and_implementations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths = []; reports = []
            for i in range(2):
                child = root/str(i); run, plan = file_fixture(child, seed=811+i)
                reports.append(analyze(run, plan, child/'out'))
                paths.append(child/'out/analysis.json')
            self.assertEqual(compare_report_files(paths)['groups'][0]['populations'], 2)
            for field, value in [('poisson_lambda_ratio', 32.), ('endpoint_gate', dict(max_cells=7)),
                                 ('global_probability', 0.)]:
                changed = copy.deepcopy(reports); changed[1]['schedule'][field] = value
                with self.assertRaisesRegex(ValueError, 'match schedule'): compare_reports(changed)
            for source in ('analyze_contact_efficiency.py', 'mobile_posterior_metrics.py'):
                changed = copy.deepcopy(reports); changed[1]['implementation_sha256'][source] = 'changed'
                with self.assertRaisesRegex(ValueError, 'match implementation'): compare_reports(changed)
            original = paths[1].read_bytes()
            changed = copy.deepcopy(reports[1]); changed['environment_occupancies']['A']['fraction'] = .99
            save(paths[1], changed)
            with self.assertRaisesRegex(ValueError, 'Frozen contact report changed'): compare_report_files(paths)
            paths[1].write_bytes(original)
            obs_path = paths[1].with_name('observations.jsonl'); obs_original = obs_path.read_bytes()
            obs_path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'Frozen contact report changed'): compare_report_files(paths)
            obs_path.write_bytes(obs_original)
            freeze_path = paths[1].with_name('freeze.json'); freeze = json.loads(freeze_path.read_text())
            freeze['ess_implementation_sha256'] = 'changed'; save(freeze_path, freeze)
            with self.assertRaisesRegex(ValueError, 'implementation freeze differs'): compare_report_files(paths)


if __name__ == '__main__':
    unittest.main()
