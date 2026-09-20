#!/usr/bin/env python3
"""Independent geometry and freezing controls for the selected-pose reference."""
import copy
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import prepare_peak_neighborhood as target
from prepare_cayley_rms_cover import uniform_draws
from prepare_smc_normalizer_atlas import Density, arrays, relative_poses


def pose(position, rotvec=(0., 0., 0.)):
    return {'position': list(position),
            'orientation': Rotation.from_rotvec(rotvec).as_quat()[[3, 0, 1, 2]].tolist()}


POINTS = np.array([[5., 0., 0.], [-5., 0., 0.], [0., 3., 0.],
                   [0., -3., 0.], [0., 0., 2.], [0., 0., -2.]])
METRIC = {'rigid_members': [pose(p) for p in POINTS],
          'native_poses': [pose([.1, -.2, .3], [.07, -.11, .03])],
          'member_error_scale': 2., 'angle_error_scale_deg': 15.,
          'fixture_metadata_must_survive': {'unchanged': True}}
FIXED = pose([11., -17., 7.], [.63, -.29, .18])
PEAK = pose([-.7, 1.3, 2.1], [-.14, .38, -.21])


def transformed(p, shift, rotation):
    t, _, r = arrays([p])
    return {'position': (rotation@t[0]+shift).tolist(),
            'orientation': Rotation.from_matrix(rotation@r[0]).as_quat()[[3, 0, 1, 2]].tolist()}


class PreparePeakNeighborhoodTests(unittest.TestCase):
    def test_finite_rotation_member_displacement_and_Haar_Jacobian(self):
        before = copy.deepcopy(METRIC)
        model, _ = target.geometry_model(METRIC, FIXED, PEAK, 'fixture', 17.)
        # Large radius makes the nonlinear Cayley denominator material.
        poses, latent, x, log_j = uniform_draws(model, FIXED, 5., 2048, 7031010)
        t, _, r = arrays(poses); pt, _, pr = arrays([PEAK]); _, _, fr = arrays([FIXED])
        moved = np.einsum('nij,kj->nki', r, POINTS)+t[:, None, :]
        original = POINTS@pr[0].T+pt[0]
        measured = np.mean(np.sum((moved-original)**2, axis=2), axis=1)
        # Cross-product actions give the rotational moment independently of
        # geometry_model's trace/covariance construction.
        moment = sum(np.dot(p, p)*np.eye(3)-np.outer(p, p) for p in POINTS)/len(POINTS)
        transform = fr[0].T@pr[0]
        c = x[:, 3:]/17.
        rotational = 4*np.einsum('ni,ij,nj->n', c, transform@moment@transform.T, c)
        predicted = np.sum(x[:, :3]**2, axis=1)+rotational/(1+np.sum(c*c, axis=1))
        np.testing.assert_allclose(measured, predicted, atol=5e-13, rtol=4e-14)
        self.assertLessEqual(float(np.max(measured-np.sum(latent*latent, axis=1))), 1e-12)
        self.assertGreater(float(np.max(np.sum(latent*latent, axis=1)-measured)), 1.)
        expected = -math.log(8*math.pi**2)-.5*np.linalg.slogdet(moment)[1]-2*np.log1p(np.sum(c*c, axis=1))
        np.testing.assert_allclose(log_j, expected, atol=2e-14, rtol=0.)
        log_density, radii, _ = Density(model).evaluate(relative_poses(poses, FIXED))
        np.testing.assert_allclose(radii[:, 0], np.linalg.norm(latent, axis=1), atol=8e-14, rtol=0.)
        np.testing.assert_allclose(log_density+log_j,
                                   -3*math.log(2*math.pi)-.5*np.sum(latent*latent, axis=1), atol=3e-13, rtol=0.)
        self.assertEqual(METRIC, before)

    def test_noncommuting_global_frame_change_preserves_physical_law(self):
        global_rotation = Rotation.from_rotvec([.7, -.5, 1.1]).as_matrix()
        global_shift = np.array([19., -4., 13.])
        moved_fixed = transformed(FIXED, global_shift, global_rotation)
        moved_peak = transformed(PEAK, global_shift, global_rotation)
        model, _ = target.geometry_model(METRIC, FIXED, PEAK, 'fixture', 23.)
        moved_model, _ = target.geometry_model(METRIC, moved_fixed, moved_peak, 'fixture', 23.)
        p, u, _, j = uniform_draws(model, FIXED, 2., 128, 7032019)
        p2, u2, _, j2 = uniform_draws(moved_model, moved_fixed, 2., 128, 7032019)
        t, _, r = arrays(p); t2, _, r2 = arrays(p2)
        np.testing.assert_array_equal(u, u2)
        np.testing.assert_allclose(t2, t@global_rotation.T+global_shift, atol=2e-14, rtol=0.)
        np.testing.assert_allclose(r2, global_rotation@r, atol=2e-15, rtol=0.)
        np.testing.assert_allclose(j, j2, atol=3e-14, rtol=0.)
        anchor = model['anchors'][0]; ft, _, fr = arrays([FIXED]); pt, _, pr = arrays([PEAK])
        np.testing.assert_allclose(fr[0]@anchor['position']+ft[0], pt[0], atol=2e-14, rtol=0.)
        np.testing.assert_allclose(fr[0]@anchor['rotation'], pr[0], atol=2e-15, rtol=0.)

    def test_angular_length_is_only_a_coordinate_convention(self):
        reference = None
        for ell in (.7, 17., 550.):
            model, _ = target.geometry_model(METRIC, FIXED, PEAK, 'fixture', ell)
            p, u, _, j = uniform_draws(model, FIXED, 3., 256, 7033028)
            t, _, r = arrays(p)
            if reference is None:
                reference = (t, r, u, j)
            else:
                np.testing.assert_allclose(t, reference[0], atol=1e-14, rtol=0.)
                np.testing.assert_allclose(r, reference[1], atol=3e-15, rtol=0.)
                np.testing.assert_array_equal(u, reference[2])
                np.testing.assert_allclose(j, reference[3], atol=2e-14, rtol=0.)

    def test_preparation_freezes_original_metric_masks_and_all_neighbors(self):
        # Synthetic fixture only: no physical production kernel is invoked.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); guides = root/'guides'; provenance = guides/'provenance'
            provenance.mkdir(parents=True)
            shape = root/'shape.json'
            target.write(shape, {'atoms': [{'center': [0., 0., 0.], 'radius': .2}],
                                 'volume': 4*math.pi*.2**3/3})
            cfg = {'shape': str(shape), 'fixed_poses': [FIXED, pose([-31., 4., -2.], [.1, .2, -.3])],
                   'capture_center': [0., 0., 0.], 'capture_radius': 18.,
                   'reservoir_density': .035, 'depletant_radius': 1.5, 'metadata': METRIC}
            source = {'physical_fixed_neighbors': cfg['fixed_poses'], 'fixed_neighbor': FIXED,
                      'capture_center': cfg['capture_center'], 'capture_radius': 18.,
                      'activity': .035, 'depletant_radius': 1.5, 'shape_sha256': target.sha(shape),
                      'physical_metric': METRIC, 'minimum_original_q': 2., 'maximum_original_q': 5.,
                      'minimum_original_q_inclusive': True, 'maximum_original_q_inclusive': False}
            target.write(guides/'config.json', cfg); target.write(provenance/'source-region.json', source)
            target.write(guides/'protocol.json', {'fixture': True})
            target.write(guides/'model-weighted.json', {'angular_length': 23.})
            target.write(guides/'freeze.json', {
                'config_sha256': target.sha(guides/'config.json'),
                'protocol_sha256': target.sha(guides/'protocol.json'),
                'model_sha256': {'weighted': target.sha(guides/'model-weighted.json')},
                'archived_sha256': {'source-region.json': target.sha(provenance/'source-region.json')}})
            extremes = root/'extremes.json'
            target.write(extremes, {'pieces': [{'name': 'remainder', 'top_poses': [
                {'draw': 151795, 'seed': 100601010, 'pose': PEAK}]}]})
            binary = root/'unused-kernel'; binary.write_bytes(b'fixture never executed')
            out = root/'prepared'
            original_config = (guides/'config.json').read_bytes()
            original_source = (provenance/'source-region.json').read_bytes()
            frozen_at_first_draw = []

            def checked_probe(*args, **kwargs):
                # Confirm the geometry and budgets were frozen before sampling.
                seal = target.read(out/'freeze.json')
                for name, digest in seal.items():
                    self.assertEqual(target.sha(out/name), digest)
                if frozen_at_first_draw:
                    self.assertEqual(seal, frozen_at_first_draw[0])
                else:
                    frozen_at_first_draw.append(seal)
                return uniform_draws(*args, **kwargs)

            with patch.multiple(target, GUIDES=guides, EXTREMES=extremes, BINARY=binary,
                                BINARY_SHA=target.sha(binary), PROBE_COUNT=8), \
                    patch.object(target, 'uniform_draws', side_effect=checked_probe), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = target.prepare(out, campaign_root=root/'campaigns', seed_base=777000)
            self.assertEqual(len(frozen_at_first_draw), 1)
            self.assertEqual((guides/'config.json').read_bytes(), original_config)
            self.assertEqual((provenance/'source-region.json').read_bytes(), original_source)
            frozen = target.read(out/'config.json')
            for key, value in cfg.items():
                if key != 'shape': self.assertEqual(frozen[key], value)
            self.assertEqual(target.sha(Path(frozen['shape'])), source['shape_sha256'])
            self.assertEqual(len(result['campaigns']), len(target.RADII))
            for index, campaign in enumerate(result['campaigns']):
                region = target.read(Path(campaign['region']))
                for key, value in source.items(): self.assertEqual(region[key], value)
                self.assertNotEqual(region['fixed_neighbor'], PEAK)
                self.assertEqual(region['mahalanobis_radius'], campaign['radius'])
                self.assertEqual(region['gaussian_chart']['means'], [[0.]*6])
                self.assertEqual(target.sha(Path(campaign['region'])), campaign['region_sha256'])
                label = f"{campaign['radius']:g}".replace('.', 'p')
                self.assertEqual(Path(campaign['output']), root/'campaigns'/f'r{label}')
                self.assertEqual(campaign['seed_base'], 777000+100000*index)
                self.assertEqual(campaign['seeds'], [777000+100000*index+1009*i for i in range(4)])
            for name, digest in target.read(out/'freeze.json').items():
                self.assertEqual(target.sha(out/name), digest)


if __name__ == '__main__':
    unittest.main()
