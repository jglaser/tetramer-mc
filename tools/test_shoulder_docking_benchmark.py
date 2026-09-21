#!/usr/bin/env python3
"""Small deterministic checks; no molecular simulation or binary build."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_smc_normalizer_atlas import read, write, sha
from prepare_shoulder_docking_benchmark import prepare, SOURCE, WINDOW, validate_design, validate_plan
from run_shoulder_docking_benchmark import check as check_run
from analyze_shoulder_docking_benchmark import (
    LABELS, assign_labels, contains, contact_incidence, descriptor_ess, one, transitions,
    validate_gate, validate_source_bundle, validate_terminal_status,
)


def pose(x):
    return dict(position=[x, 0., 0.], orientation=[1., 0., 0., 0.])


def source_fixture(archive, bundle_path):
    mapping = {name: name for name in ('Cargo.toml', 'Cargo.lock', 'build.rs')}
    mapping.update({'vendor/README.md': 'source/vendor/README.md', 'src/lib.rs': 'source/src/lib.rs'})
    files, hashes = {}, {}
    for name, stored in mapping.items():
        path = archive/stored
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'Synthetic source fixture for {name}\n')
        files[name] = dict(sha256=sha(path), text=path.read_text())
        hashes[stored] = sha(path)
    write(bundle_path, dict(schema=1, files=files))
    return dict(source_bundle_files=mapping, input_sha256=hashes)


class ShoulderPreparationTests(unittest.TestCase):
    def test_malformed_gate_counts_and_log_acceptance_are_rejected(self):
        gate = dict(gained=2, lost=3, raw_points=8, log_weight=-.25)
        validate_gate(gate, -.5)
        for key, value in [('gained', -1), ('lost', True), ('raw_points', 8.),
                           ('raw_points', 4), ('log_weight', float('inf'))]:
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                validate_gate(dict(gate, **{key: value}), -.5)
        for value in (True, .1, float('inf'), float('nan'), '-1'):
            with self.subTest(log_acceptance=value), self.assertRaises(AssertionError):
                validate_gate(gate, value)

    def test_source_bundle_is_tied_to_archived_bytes(self):
        with tempfile.TemporaryDirectory(prefix='shoulder-source-test-') as temporary:
            campaign = Path(temporary)
            archive = campaign/'provenance'
            archive.mkdir()
            path = campaign/'source-bundle.json'
            manifest = source_fixture(archive, path)
            validate_source_bundle(campaign, manifest, path)
            original = read(path)
            altered = copy.deepcopy(original)
            altered['files']['src/lib.rs']['text'] += '// changed embedded source\n'
            write(path, altered)
            with self.assertRaises(AssertionError):
                validate_source_bundle(campaign, manifest, path)
            write(path, original)
            (archive/'source/src/lib.rs').write_text('Changed archived source\n')
            with self.assertRaises(AssertionError):
                validate_source_bundle(campaign, manifest, path)

    def test_terminal_status_requires_complete_exact_job_set(self):
        with tempfile.TemporaryDirectory(prefix='shoulder-status-test-') as temporary:
            campaign = Path(temporary)
            manifest = dict(jobs=[dict(id='a'), dict(id='b')])
            status = dict(running=False, complete=True, jobs=[dict(id=name, exit_code=0, status='complete',
                started_at='optional extra field') for name in ('a', 'b')])
            write(campaign/'status.json', status)
            validate_terminal_status(campaign, manifest)
            bad_statuses = [dict(status, running=True), dict(status, complete=False),
                dict(status, jobs=status['jobs']+[status['jobs'][0]]),
                dict(status, jobs=[dict(id='extra', exit_code=0, status='complete'), status['jobs'][1]]),
                dict(status, jobs=[dict(id='a', exit_code=True, status='complete'), status['jobs'][1]]),
                dict(status, jobs=[dict(id='a', exit_code=1, status='complete'), status['jobs'][1]])]
            for bad in bad_statuses:
                write(campaign/'status.json', bad)
                with self.assertRaises(AssertionError):
                    validate_terminal_status(campaign, manifest)

    def test_preparation_is_inert_and_protocol_is_frozen(self):
        with tempfile.TemporaryDirectory(prefix='shoulder-preparation-test-') as temporary:
            out = Path(temporary)/'plan'
            result = prepare(out, SOURCE)
            self.assertFalse(result['launched'])
            self.assertFalse(result['binary_supplied'])
            self.assertFalse((out/'provenance/docking-mc').exists())
            self.assertFalse(any((out/'runs').iterdir()))
            self.assertEqual(result['seeds'], [115101010+1009*i for i in range(12)])
            self.assertEqual(len(result['jobs']), 12)
            self.assertEqual((result['cycles'], result['burn_cycles'], result['workers']), (5000, 1000, 4))
            self.assertAlmostEqual(sum(r['probability'] for r in result['physical_reference']['regions'].values()), 1.)
            self.assertIn('source/tests/docking_region.rs', result['input_sha256'])
            self.assertIn('build.rs', result['input_sha256'])
            self.assertIn('source/vendor/README.md', result['input_sha256'])
            self.assertEqual(result['total_attempts'], 180000)
            self.assertIn('exit 2', (out/'commands.sh').read_text())
            self.assertIn('SHOULDER_TERMINAL_STATUS_JSON', (out/'commands.sh').read_text())
            for job in result['jobs']:
                cfg = read(job['config'])
                self.assertEqual(cfg['target_region']['window'], WINDOW)
                self.assertEqual(cfg['proposal_anchor_index'], 0)
                self.assertEqual(len(cfg['fixed_poses']), 2)
                self.assertEqual(cfg['uniform_probability'], .05)
                self.assertEqual(cfg['local_attempts_per_cycle'], 2)
                self.assertEqual(sha(job['config']), job['config_sha256'])
                self.assertIn('--method', job['command'])
            with self.assertRaises(ValueError):
                prepare(out, SOURCE)

    def test_long_fixed_plan_and_runner_preflight_are_inert(self):
        with tempfile.TemporaryDirectory(prefix='shoulder-long-plan-test-') as temporary:
            root = Path(temporary)
            binary = root/'nonexecutable-binary-fixture'
            binary.write_text('Synthetic fixture: never execute\n')
            out, journal, assessment = root/'plan', root/'journal', root/'assessment'
            result = prepare(out, SOURCE, binary=binary, workers=12, cycles=40000,
                             burn_cycles=8000, seed_base=115201010)
            self.assertFalse(result['launched'])
            self.assertEqual(result['total_attempts'], 1440000)
            self.assertEqual(result['cycles']-result['burn_cycles'], 32000)
            self.assertEqual(result['workers'], 12)
            self.assertEqual(result['seed_base'], 115201010)
            self.assertEqual(result['seeds'], [115201010+1009*i for i in range(12)])
            self.assertIn('run_shoulder_docking_benchmark.py', result['input_sha256'])
            self.assertEqual(check_run(out, journal, assessment), result)
            for job in result['jobs']:
                argv = job['command']
                self.assertEqual(argv[argv.index('--cycles')+1], '40000')
                self.assertEqual(read(job['config'])['seed'], job['seed'])
            self.assertFalse(any((out/'runs').iterdir()))
            self.assertFalse(journal.exists())
            self.assertFalse(assessment.exists())
            self.assertFalse((out/'status.json').exists())
            # The old pilot schema without seed_base remains readable.
            historical = copy.deepcopy(result)
            del historical['seed_base']
            validate_plan(historical, out)
            for mutation in ('total', 'cycles_cli', 'duplicate_cycles_cli', 'seed', 'burn'):
                bad = copy.deepcopy(result)
                if mutation == 'total':
                    bad['total_attempts'] += 1
                elif mutation == 'cycles_cli':
                    argv = bad['jobs'][0]['command']
                    argv[argv.index('--cycles')+1] = '39999'
                elif mutation == 'duplicate_cycles_cli':
                    bad['jobs'][0]['command'] += ['--cycles', '40000']
                elif mutation == 'seed':
                    bad['jobs'][0]['seed'] += 1
                else:
                    bad['burn_cycles'] = 40000
                write(out/'manifest.json', bad)
                with self.subTest(mutation=mutation), self.assertRaises((AssertionError, ValueError)):
                    check_run(out, journal, assessment)
            write(out/'manifest.json', result)

    def test_invalid_allocations_fail_before_inputs_or_output(self):
        with tempfile.TemporaryDirectory(prefix='shoulder-invalid-design-test-') as temporary:
            root = Path(temporary)
            for params in (dict(cycles=0), dict(cycles=-1), dict(cycles=True), dict(cycles=5.),
                           dict(burn_cycles=-1), dict(burn_cycles=5000), dict(burn_cycles=5001),
                           dict(burn_cycles=4999), dict(burn_cycles=True), dict(burn_cycles=1.),
                           dict(cycles=1, burn_cycles=0), dict(cycles=2**64),
                           dict(seed_base=-1), dict(seed_base=True), dict(seed_base=2**64-1),
                           dict(workers=0), dict(workers=33), dict(workers=True), dict(workers=1.)):
                out = root/'must-not-exist'
                with self.subTest(params=params), self.assertRaises(ValueError):
                    prepare(out, root/'missing-input', **params)
                self.assertFalse(out.exists())
            for workers in (1, 12, 32):
                validate_design(2, 0, 0, workers)
            validate_design(40000, 8000, 2**64-1-11*1009, 12)

    def test_physical_mask_boundaries_and_priority(self):
        self.assertEqual(contains([1., 1.+1e-12, 2.-1e-12, 2., np.nan]).tolist(),
                         [False, True, True, False, False])
        q = np.array([1.05, 1.05, 1.05, 1.05, 1.1])
        radii = np.array([[.5, .1, .1], [.51, .5, .1], [.6, .6, .5], [.6, .6, .6], [.1, .1, .1]])
        self.assertEqual(assign_labels(q, radii).tolist(), [0, 1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            assign_labels(np.array([1.]), np.zeros((1, 3)))

    def test_contact_blocks_remain_separate_and_keep_repeats(self):
        cfg = dict(fixed_poses=[pose(0.), pose(4.)], depletant_radius=.75)
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.5)])
        frames = [dict(cycle=i+1, pose=pose(x), depletion_contact=True)
                  for i, x in enumerate([1.1, 1.1, 2.9, 2.9])]
        report = contact_incidence(cfg, shape, frames, 1.)
        self.assertEqual(report['samples'], 4)
        self.assertEqual([r['A']['mobile_contact_atoms'] for r in report['rows']], [1, 1, 0, 0])
        self.assertEqual([r['B']['mobile_contact_atoms'] for r in report['rows']], [0, 0, 1, 1])
        self.assertEqual(report['neighbors']['A']['mean_incidence'], [.5, .5])
        self.assertEqual(report['neighbors']['B']['mean_incidence'], [.5, .5])
        self.assertIsNone(descriptor_ess(np.ones(10), 1.)['apparent_ess'])
        self.assertIsNone(descriptor_ess(np.ones(10), 1.)['apparent_ess_per_sampler_cpu_second'])

    def test_all_attempt_flux_keeps_self_loops(self):
        result = transitions([0, 0, 1, 1, 0])
        self.assertEqual(result['counts'][0][:2], [1, 1])
        self.assertEqual(result['counts'][1][:2], [1, 1])
        self.assertEqual(result['self_loops'], 2)
        self.assertEqual(result['off_diagonal_events'], 2)

    def test_synthetic_observer_replays_rejections_and_fixed_burn(self):
        # Synthetic records exercise only analysis, not a simulation. Every
        # proposal exits the strict window and retains the same valid endpoint.
        with tempfile.TemporaryDirectory(prefix='shoulder-observer-test-') as temporary:
            campaign, out = Path(temporary)/'campaign', Path(temporary)/'report'
            archive, run = campaign/'provenance', campaign/'run'
            archive.mkdir(parents=True)
            run.mkdir()
            (run/'provenance').mkdir()
            (archive/'docking-mc').write_text('Synthetic nonexecutable provenance fixture\n')
            source_manifest = source_fixture(archive, run/'provenance/source-bundle.json')
            model = dict(angular_length=1., means=[[0.]*6], covariances=[np.eye(6).tolist()],
                weights=[1.], anchors=[dict(position=[0., 0., 0.], rotation=np.eye(3).tolist())])
            for name in ['model']+[f'geometric-{s}' for s in LABELS[:3]]:
                write(archive/f'{name}.json', model)
            shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.5)])
            write(archive/'shape.json', shape)
            metric = dict(native_poses=[pose(0.)], rigid_members=[pose(0.)], member_error_scale=2., angle_error_scale_deg=15.)
            cfg = dict(target_region=dict(metric=metric, window=WINDOW), proposal_anchor_index=0,
                fixed_poses=[pose(1.5), pose(8.)], initial_pose=pose(3.), capture_center=[0., 0., 0.],
                capture_radius=18., local_attempts_per_cycle=2, depletant_radius=1.5,
                reservoir_density=.035, poisson_lambda_ratio=64., seed=1)
            cfgpath = campaign/'config.json'
            write(cfgpath, cfg)
            job = dict(config=str(cfgpath), config_sha256=sha(cfgpath), directory=str(run),
                       id='synthetic', mode='local', correlation=0., start='direct', replicate=0, seed=1)
            manifest = dict(cycles=4, burn_cycles=1, target_region=cfg['target_region'],
                binary_supplied=True, binary_sha256=sha(archive/'docking-mc'),
                model_sha256=sha(archive/'model.json'), shape_sha256=sha(archive/'shape.json'),
                inference_scope='synthetic fixture', **source_manifest)
            write(campaign/'manifest.json', manifest)
            counts = [dict(attempted=12, accepted=0, region_rejected=12), {}]
            summary = dict(complete=True, completed_cycles=4, config_sha256=sha(cfgpath),
                model_sha256=manifest['model_sha256'], shape_sha256=manifest['shape_sha256'],
                method='local', correlation=0., counts=counts, sampler_cpu_seconds=4.)
            write(run/'summary.json', summary)
            hashes = dict(config_sha256=sha(cfgpath), model_sha256=manifest['model_sha256'], shape_sha256=manifest['shape_sha256'])
            write(run/'checkpoint.json', dict(pose=pose(3.), counts=counts, completed_cycles=4,
                                             method='local', correlation=0., **hashes))
            write(run/'manifest.json', dict(executable_sha256=manifest['binary_sha256'],
                source_bundle_sha256=sha(run/'provenance/source-bundle.json'), target_region=cfg['target_region'],
                proposal_anchor_index=0, cycles=4, sample_every=1, method='local', correlation=0., **hashes))
            import shutil
            for source, name in ((cfgpath, 'input-config.json'), (archive/'model.json', 'model.json'), (archive/'shape.json', 'shape.json')):
                shutil.copy2(source, run/'provenance'/name)
            frames = [dict(cycle=i, pose=pose(3.), depletion_contact=True, sampler_cpu_seconds=float(i)) for i in range(5)]
            moves = [dict(cycle=serial//3+1, attempt=serial%3, kind='local', branch='local',
                old_pose=pose(3.), proposed_pose=pose(0.), retained_pose=pose(3.),
                old_q=1.5, proposed_q=0., retained_q=1.5, accepted=False,
                capture_valid=True, region_valid=False, hard_valid=False, gate=None, log_acceptance=None,
                proposal=dict(branch='local', log_reverse_forward=0.), depletion_contact_before=True,
                depletion_contact=True, sampler_cpu_seconds=(serial+1)/3.) for serial in range(12)]
            import json
            (run/'trajectory.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
            (run/'moves.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in moves))
            report = one((campaign, out, job))
            self.assertEqual(report['retained_cycles'], 3)
            self.assertEqual(report['sampler_cpu_seconds_after_burn'], 3.)
            self.assertEqual(report['all_attempt_flux']['self_loops'], 9)
            self.assertEqual(report['attempt_flux_by_slot']['0']['self_loops'], 3)
            self.assertEqual(report['attempt_flux_by_kernel']['local']['self_loops'], 9)
            cost = report['attempt_cpu_allocation']
            self.assertAlmostEqual(cost['by_kernel']['local']['allocated_sampler_cpu_seconds'], 3.)
            self.assertAlmostEqual(cost['total_allocated_attempt_cpu_seconds'], 3.)
            self.assertAlmostEqual(cost['final_frame_cpu_tail_seconds'], 0.)
            for slot in range(3):
                self.assertAlmostEqual(cost['by_slot'][str(slot)]['allocated_sampler_cpu_seconds'], 1.)
            self.assertEqual(report['contact_incidence']['samples'], 3)
            self.assertIsNone(report['joint_label_ess']['apparent_ess'])
            self.assertEqual(report['occupancy']['outer_shoulder'], 1.)
            bad = copy.deepcopy(moves)
            bad[0]['retained_pose'] = pose(0.)
            (run/'moves.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in bad))
            with self.assertRaises(AssertionError):
                one((campaign, Path(temporary)/'bad-report', job))


if __name__ == '__main__':
    unittest.main()
