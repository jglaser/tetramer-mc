"""Reference controls for uniform shell preparation and physical pose measure."""
import copy
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_native_tail_regions import (
    decode_latent, encode_matrix, radial_quantile, sample_shell, shell_volume,
    validate_physics,
)


def pose(position, rotation):
    return {'position': list(position), 'orientation': rotation.as_quat()[[3, 0, 1, 2]].tolist()}


def fixture():
    lower = np.diag([.4, .7, .5, .3, .2, .6])
    lower[3, 0] = .2
    lower[4, 1] = -.3
    lower[5, 2] = .1
    return {'schema': 'weighted-pose-mixture-v1', 'coordinate_convention': 'anchor-body-relative',
            'shape_sha256': 'fixture', 'angular_length': 2.7,
            'anchors': [{'position': [.5, -1., 2.], 'rotation': Rotation.from_rotvec([.3, -.7, .2]).as_matrix().tolist()}],
            'means': [[.2, -.1, .3, .4, -.2, .1]],
            'covariances': [(lower@lower.T).tolist()], 'weights': [1.]}


class NativeTailPreparationTests(unittest.TestCase):
    def test_inverse_radial_cdf_and_shell_volume(self):
        for lo, hi in [(0., 3.), (4., 5.), (5., 8.), (8., 12.)]:
            u = np.array([0., .01, .2, .5, .9, 1.])
            r = radial_quantile(u, lo, hi)
            np.testing.assert_allclose((r**6-lo**6)/(hi**6-lo**6), u, atol=3e-15)
            np.testing.assert_allclose(shell_volume(lo, hi), math.pi**3*(hi**6-lo**6)/6, rtol=3e-15)
        for lo, hi in [(2., 2.), (-1., 3.), (0., float('inf'))]:
            with self.assertRaises(ValueError):
                shell_volume(lo, hi)
        with self.assertRaises(ValueError):
            radial_quantile([-.1], 4., 5.)

    def test_sample_is_uniform_in_volume_and_angularly_isotropic(self):
        sample = sample_shell(np.random.default_rng(6721), 65536, 4., 5.)
        radius = np.linalg.norm(sample, axis=1)
        self.assertTrue(np.all((radius >= 4.) & (radius <= 5.)))
        u = (radius**6-4**6)/(5**6-4**6)
        self.assertLess(abs(u.mean()-.5), 5/math.sqrt(12*len(u)))
        directions = sample/radius[:, None]
        np.testing.assert_allclose((directions.T@directions)/len(sample), np.eye(6)/6, atol=.005)
        self.assertLess(float(np.abs(directions.mean(axis=0)).max()), .006)

    def test_full_pose_map_roundtrip_and_reference_frame_covariance(self):
        model = fixture()
        latent = np.random.default_rng(7266).normal(size=(32, 6))
        fixed = pose([3., -5., 2.], Rotation.from_rotvec([-.4, .6, .8]))
        poses, jac = decode_latent(latent, model, fixed)
        np.testing.assert_allclose(encode_matrix(poses, model, fixed), latent, atol=2e-14)
        identity = pose([0., 0., 0.], Rotation.identity())
        relative, relative_jac = decode_latent(latent, model, identity)
        np.testing.assert_allclose(jac, relative_jac, atol=0)
        rotation = Rotation.from_quat(np.array(fixed['orientation'])[[1, 2, 3, 0]])
        np.testing.assert_allclose([p['position'] for p in poses], rotation.apply([p['position'] for p in relative])+fixed['position'], atol=1e-14)

    def test_haar_jacobian_matches_independent_six_dimensional_differential(self):
        model = fixture()
        fixed = pose([3., -5., 2.], Rotation.from_rotvec([-.4, .6, .8]))
        latent = np.array([.4, -.8, .5, 1.1, -.7, .3])
        base, logj = decode_latent(latent[None], model, fixed)
        base_r = Rotation.from_quat(np.array(base[0]['orientation'])[[1, 2, 3, 0]])
        eps = 2e-5
        differential = np.zeros((6, 6))
        for axis in range(6):
            shift = np.eye(6)[axis]*eps
            poses, _ = decode_latent(np.stack([latent+shift, latent-shift]), model, fixed)
            differential[:3, axis] = (np.array(poses[0]['position'])-poses[1]['position'])/(2*eps)
            rotations = Rotation.from_quat(np.array([p['orientation'] for p in poses])[:, [1, 2, 3, 0]])
            angular = (rotations*base_r.inv()).as_rotvec()
            differential[3:, axis] = (angular[0]-angular[1])/(2*eps)
        # Local rotation-vector volume is Haar volume times 8*pi^2.
        independent = abs(np.linalg.det(differential))/(8*math.pi**2)
        self.assertAlmostEqual(independent/math.exp(logj[0]), 1., delta=2e-9)

    def test_full_neighbor_and_metric_identity_are_required(self):
        cfg = {'fixed_poses': [pose([2, 0, 0], Rotation.identity()), pose([-2, 0, 0], Rotation.identity())],
               'capture_center': [0, 0, 0], 'capture_radius': 18., 'depletant_radius': 1.5,
               'reservoir_density': .035, 'metadata': {'member_error_scale': 2., 'tag': 'unchanged'}}
        region = {'fixed_neighbor': cfg['fixed_poses'][0], 'physical_fixed_neighbors': cfg['fixed_poses'],
                  'capture_center': cfg['capture_center'], 'capture_radius': cfg['capture_radius'],
                  'depletant_radius': cfg['depletant_radius'], 'activity': cfg['reservoir_density'],
                  'physical_metric': cfg['metadata'], 'shape_sha256': 'fixture', 'gaussian_chart': fixture()}
        validate_physics(region, cfg, 'fixture')
        bad = copy.deepcopy(region)
        bad['physical_fixed_neighbors'].pop()
        with self.assertRaises(AssertionError):
            validate_physics(bad, cfg, 'fixture')
        bad = copy.deepcopy(region)
        bad['physical_metric']['member_error_scale'] = 3.
        with self.assertRaises(AssertionError):
            validate_physics(bad, cfg, 'fixture')


if __name__ == '__main__':
    unittest.main()
