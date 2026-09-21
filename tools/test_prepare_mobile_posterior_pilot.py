#!/usr/bin/env python3
"""Inert model/start/design tests; no executable build or physical trajectory."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_mobile_posterior_pilot import (
    BURN, FORBIDDEN, MODES, RADIUS, RADIUS_12, ROOT, SEED_BASE, SWEEPS,
    atomic_start_check, combine_models, prepare, prepare_starts, read, sha,
    validate_configs, validate_model, verify_density_identity, write,
)


def pose(x=0., y=0., z=0.):
    return dict(position=[x, y, z], orientation=[1., 0., 0., 0.])


def model(ell=2., offset=0.):
    lower = np.diag([.8, .9, 1., .3, .4, .5])
    lower[3, 0], lower[5, 1] = .12, -.2
    covariance = lower@lower.T
    return dict(schema='weighted-pose-mixture-v1', coordinate_convention='anchor-body-relative',
        shape_sha256='a'*64, angular_length=ell,
        anchors=[dict(position=[offset, 0., 0.], rotation=np.eye(3).tolist()) for _ in range(2)],
        means=[[.1, -.2, .3, .4, -.5, .6], [-.4, .3, -.2, .1, .5, -.6]],
        covariances=[covariance.tolist(), (covariance*2).tolist()], weights=[.3, .7])


class MobilePreparationTests(unittest.TestCase):
    def test_merge_preserves_every_component_and_full_physical_density(self):
        sources = [model(2., 0.), model(7., 3.), model(.8, -2.)]
        before = copy.deepcopy(sources)
        combined, groups = combine_models(sources, [.5, .4, .1], 'a'*64, angular_length=4.)
        self.assertEqual(sources, before)
        self.assertEqual(len(combined['weights']), 6)
        self.assertEqual(combined['anchors'], sum([m['anchors'] for m in sources], []))
        self.assertTrue(np.allclose(combined['weights'], [.15, .35, .12, .28, .03, .07]))
        scale = np.diag([1.]*3+[2.]*3)
        self.assertTrue(np.allclose(combined['means'][0], scale@np.asarray(sources[0]['means'][0])))
        self.assertTrue(np.allclose(combined['covariances'][0], scale@np.asarray(sources[0]['covariances'][0])@scale.T))
        self.assertEqual([g['components'] for g in groups], [2, 2, 2])
        report = verify_density_identity(combined, sources, [.5, .4, .1])
        self.assertTrue(report['passed'])
        self.assertLess(report['maximum_log_density_difference'], 1e-8)
        changed = copy.deepcopy(combined)
        changed['means'][0][0] += .25
        with self.assertRaisesRegex(ValueError, 'mixture identity failed'):
            verify_density_identity(changed, sources, [.5, .4, .1])

    def test_bad_shape_weights_covariance_or_rotation_are_not_repaired(self):
        original = model()
        for change in ('shape', 'weights', 'covariance', 'rotation', 'missing'):
            bad = copy.deepcopy(original)
            if change == 'shape': bad['shape_sha256'] = 'b'*64
            elif change == 'weights': bad['weights'] = [0., 1.]
            elif change == 'covariance': bad['covariances'][0][0][0] = -1.
            elif change == 'rotation': bad['anchors'][0]['rotation'][0][0] = -1.
            else: bad['means'].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_model(bad, 'a'*64)

    def test_atomic_check_separates_wall_hard_and_exclusion_contacts(self):
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=1.)])
        valid = atomic_start_check([pose(-5), pose(0), pose(5)], shape, 10., rd=1., require_no_contacts=True)
        self.assertTrue(valid['passed'])
        contact = atomic_start_check([pose(-3), pose(0), pose(3)], shape, 10., rd=1., require_no_contacts=True)
        self.assertFalse(contact['passed'])
        self.assertTrue(contact['hard_valid'])
        self.assertFalse(contact['no_initial_exclusion_contacts'])
        self.assertFalse(atomic_start_check([pose(0), pose(1)], shape, 10.)['hard_valid'])
        self.assertFalse(atomic_start_check([pose(0), pose(10)], shape, 10.)['wall_valid'])

    def test_fallback_is_deterministic_and_preassociated_is_only_recentered(self):
        source = dict(initial_poses=[pose() for _ in range(12)], boundary=dict(kind='spherical', radius=RADIUS_12))
        ab = dict(fixed_poses=[pose(3., 0., 0.), pose(0., 4., 0.)], metadata=dict(native_poses=[pose()]))
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.5)])
        starts, report = prepare_starts(source, ab, shape)
        again, again_report = prepare_starts(source, ab, shape)
        self.assertEqual(starts, again)
        self.assertEqual(report, again_report)
        self.assertTrue(report['dispersed_generation']['fallback_used'])
        self.assertTrue(report['checks']['dispersed']['no_initial_exclusion_contacts'])
        self.assertTrue(np.allclose(np.mean([p['position'] for p in starts['preassociated']], axis=0), 0.))
        originals = ab['fixed_poses']+ab['metadata']['native_poses']
        for i in range(3):
            self.assertEqual(starts['preassociated'][i]['orientation'], originals[i]['orientation'])
            for j in range(3):
                self.assertTrue(np.allclose(np.asarray(starts['preassociated'][i]['position'])-starts['preassociated'][j]['position'],
                                            np.asarray(originals[i]['position'])-originals[j]['position']))
        ab['fixed_poses'][0] = pose(.25)
        with self.assertRaisesRegex(ValueError, 'no repair permitted'):
            prepare_starts(source, ab, shape)

    def test_real_preparation_is_inert_and_frozen_without_touching_example(self):
        user_example = ROOT/'examples/spherical-seeded-transport.json'
        before = sha(user_example)
        with tempfile.TemporaryDirectory(prefix='mobile-posterior-preparation-test-') as temporary:
            out = Path(temporary)/'prepared'
            result = prepare(out)
            self.assertTrue(result['preparation_only'])
            self.assertFalse(result['launched'])
            self.assertFalse(result['binary_supplied'])
            self.assertIsNone(result['binary_sha256'])
            self.assertEqual(read(out/'plan.json'), result)
            self.assertEqual((result['sweeps'], result['burn_sweeps'], result['sample_every'], result['workers']), (2000, 400, 1, 12))
            self.assertEqual(len(result['jobs']), 12)
            self.assertEqual(result['seeds'], [SEED_BASE+1009*i for i in range(12)])
            self.assertEqual(result['total_single_body_attempts'], 72000)
            self.assertEqual(result['total_collective_attempts'], 48000)
            self.assertEqual(len(read(out/'model.json')['weights']), 150)
            self.assertAlmostEqual((RADIUS/RADIUS_12)**3, .25)
            self.assertFalse(any((out/'runs').iterdir()))
            self.assertFalse(any((out/'logs').iterdir()))
            self.assertIn('exit 2', (out/'commands.sh').read_text())
            for job in result['jobs']:
                cfg = read(job['config'])
                self.assertFalse(FORBIDDEN.intersection(cfg))
                self.assertEqual(len(cfg['initial_poses']), 3)
                self.assertEqual(cfg['fixed_body_indices'], [])
                if job['mode'] == 'capture_only': self.assertNotIn('frozen_posterior', cfg)
                else: self.assertEqual(cfg['frozen_posterior']['probability'], .5)
            for name, digest in read(out/'freeze.json')['files'].items():
                self.assertEqual(sha(out/name), digest)
            validate_configs(result['jobs'])
            path = Path(result['jobs'][0]['config'])
            cfg = read(path)
            cfg['reservoir_density'] = .04
            write(path, cfg)
            altered = copy.deepcopy(result['jobs'])
            altered[0]['config_sha256'] = sha(path)
            with self.assertRaisesRegex(ValueError, 'Physical or matched proposal law'):
                validate_configs(altered)
            with self.assertRaisesRegex(ValueError, 'Fresh preparation'):
                prepare(out)
        self.assertEqual(sha(user_example), before)


if __name__ == '__main__':
    unittest.main()
