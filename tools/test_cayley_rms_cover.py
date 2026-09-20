#!/usr/bin/env python3
"""Physical-measure and frame controls for the complete geometric cover."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

from analyze_native_region_reference import native_q
from prepare_cayley_rms_cover import derive_model, uniform_draws, check_coordinates, prepare
from prepare_smc_normalizer_atlas import Density, arrays, relative_poses


def pose(position, rotvec=(0., 0., 0.)):
    return {'position': list(position),
            'orientation': Rotation.from_rotvec(rotvec).as_quat()[[3, 0, 1, 2]].tolist()}


def metric(points, delta=.8, alpha=35.):
    return {'native_poses': [pose([2., -3., 4.], [.23, -.37, .19])],
            'rigid_members': [pose(p) for p in points], 'member_error_scale': delta,
            'angle_error_scale_deg': alpha}


POINTS = [[10., 0., 0.], [-10., 0., 0.], [0., 4., 0.], [0., -4., 0.], [0., 0., 2.], [0., 0., -2.]]


class CayleyRmsCoverTests(unittest.TestCase):
    def test_noncommuting_frames_and_original_member_error_coverage(self):
        m = metric(POINTS)
        anchor = pose([11., -7., 9.], [-.42, .18, .71])
        model, proof = derive_model(m, anchor, 'sphere-fixture', 2.)
        poses, latent, x, log_j = uniform_draws(model, anchor, proof['mahalanobis_radius'], 4096, 593010)
        check_coordinates(model, anchor, poses, latent, x, log_j)
        # Direct physical member displacements independently check the right/left
        # chart conversion, rather than comparing two implementations of density.
        t, _, r = arrays(poses); rt, _, rr = arrays(m['native_poses'])
        members = np.asarray(POINTS)
        displaced = np.einsum('nij,kj->nki', r, members)+t[:, None, :]-(members@rr[0].T+rt[0])
        measured = np.mean(np.sum(displaced**2, axis=2), axis=1)
        cf = x[:, 3:]/model['angular_length']
        _, _, fr = arrays([anchor])
        transform = fr[0].T@rr[0]
        full_matrix = transform@np.asarray(proof['A_A2'])@transform.T
        predicted = np.sum(x[:, :3]**2, axis=1)+4*np.einsum('ni,ij,nj->n', cf, full_matrix, cf)/(1+np.sum(cf**2, axis=1))
        self.assertLess(float(np.max(np.abs(measured-predicted))), 2e-11)
        # Draw from an independent broader product cover, and test that every
        # actual max-member/angle target pose lies inside the smaller ellipsoid.
        rng = np.random.default_rng(594019)
        n = 8192; axes = rng.normal(size=(n, 3)); axes /= np.linalg.norm(axes, axis=1)[:, None]
        angles = rng.uniform(0, proof['full_product_cover_angle_cap'], n)
        rotations = rr[0]@Rotation.from_rotvec(axes*angles[:, None]).as_matrix()
        w = rng.uniform(-proof['mahalanobis_radius'], proof['mahalanobis_radius'], (n, 3))
        quaternions = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
        candidates = [{'position': p.tolist(), 'orientation': q.tolist()} for p, q in zip(w+rt[0], quaternions)]
        selected = [p for p in candidates if native_q(m, p) <= 2]
        self.assertGreater(len(selected), 50)
        norms = Density(model).evaluate(relative_poses(selected, anchor))[1][:, 0]
        self.assertLessEqual(float(norms.max()), proof['mahalanobis_radius']*(1+1e-12))

    def test_isotropic_image_volume_matches_independent_radial_integral(self):
        points = (3*np.r_[np.eye(3), -np.eye(3)]).tolist()
        m = metric(points, delta=.8, alpha=70.)
        anchor = pose([0., 0., 0.])
        model, proof = derive_model(m, anchor, 'sphere-fixture', 2.)
        radius = proof['mahalanobis_radius']; lam = proof['guarded_A_eigenvalues_A2'][0]
        k = proof['cosine_square_factor']; coefficient = 4*k*lam
        maximum = radius/math.sqrt(coefficient)
        # Integrate orientation's normalized Haar radial measure times the
        # available translation-ball volume. This bypasses the chart Jacobian.
        integrand = lambda c: (16/3)*c*c*max(0., radius*radius-coefficient*c*c)**1.5/(1+c*c)**2
        exact, error = quad(integrand, 0, maximum, epsabs=1e-12, epsrel=1e-12)
        self.assertLess(error, 1e-10)
        _, latent, _, log_j = uniform_draws(model, anchor, radius, 65536, 594819)
        values = math.pi**3*radius**6/6*np.exp(log_j)
        se = values.std(ddof=1)/math.sqrt(len(values))
        self.assertLess(abs(values.mean()-exact), 6*se)
        # Uniform sixth-power radial coordinate tests the measure of the draw.
        radial = (np.linalg.norm(latent, axis=1)/radius)**6
        self.assertLess(abs(radial.mean()-.5), 6/math.sqrt(12*len(radial)))

    def test_unsupported_geometry_is_rejected_without_ridge(self):
        original = metric(POINTS); anchor = pose([0., 0., 0.])
        cases = []
        shifted = copy.deepcopy(original); shifted['rigid_members'][0]['position'][0] += .1; cases.append(shifted)
        cases.append(metric([[1., 0., 0.], [-1., 0., 0.]]))
        cases.append(metric([[0., 0., 0.]]))
        cases.append(metric(POINTS, delta=100., alpha=180.))
        multiple = copy.deepcopy(original); multiple['native_poses'] *= 2; cases.append(multiple)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                derive_model(candidate, anchor, 'sphere-fixture', 2.)
        for bound in [0., -1., float('nan'), float('inf')]:
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                derive_model(original, anchor, 'sphere-fixture', bound)

    def test_narrow_window_recomputes_cap_and_has_constant_density_gain(self):
        m = metric(POINTS); anchor = pose([11., -7., 9.], [-.42, .18, .71])
        wide, wp = derive_model(m, anchor, 'sphere-fixture', 2.)
        narrow, npf = derive_model(m, anchor, 'sphere-fixture', 1.1)
        self.assertEqual(npf['mahalanobis_radius'], 1.1*m['member_error_scale'])
        self.assertLess(npf['full_product_cover_angle_cap'], wp['full_product_cover_angle_cap'])
        expected = (2/1.1)**6*(npf['cosine_square_factor']/wp['cosine_square_factor'])**1.5
        ref_t, _, ref_r = arrays(m['native_poses'])
        candidates = []
        for rv, translation in [([0., 0., 0.], [.83, 0., 0.]),
                                ([.004, -.003, .002], [.3, -.2, .1]),
                                ([-.008, .004, -.003], [-.2, .1, .15])]:
            rotation = ref_r[0]@Rotation.from_rotvec(rv).as_matrix()
            candidates.append({'position': (ref_t[0]+translation).tolist(),
                               'orientation': Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].tolist()})
        self.assertTrue(all(native_q(m, p) <= 1.1 for p in candidates))
        relative = relative_poses(candidates, anchor)
        log_densities = []
        for model, proof in [(wide, wp), (narrow, npf)]:
            gaussian_log, norms, _ = Density(model).evaluate(relative)
            self.assertLess(float(norms.max()), proof['mahalanobis_radius'])
            log_j = -3*math.log(2*math.pi)-.5*norms[:, 0]**2-gaussian_log
            log_densities.append(-math.log(math.pi**3*proof['mahalanobis_radius']**6/6)-log_j)
        self.assertLess(float(np.max(np.abs(log_densities[1]-log_densities[0]-math.log(expected)))), 2e-12)

    def test_preparation_preserves_all_neighbors_and_open_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = {'atoms': [{'center': [0., 0., 0.], 'radius': .2}], 'volume': 4*math.pi*.2**3/3}
            (root/'shape.json').write_text(json.dumps(shape))
            cfg = {'shape': 'shape.json', 'fixed_poses': [pose([30., 0., 0.]), pose([-30., 0., 0.])],
                   'capture_center': [2., -3., 4.], 'capture_radius': 18., 'depletant_radius': 1.5,
                   'reservoir_density': .035, 'metadata': metric(POINTS)}
            config = root/'config.json'; config.write_text(json.dumps(cfg))
            out = root/'prepared'
            result = prepare(config, out, 32, 595017)
            region = json.loads((out/'region.json').read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(region['physical_metric'], cfg['metadata'])
            self.assertEqual(region['physical_fixed_neighbors'], cfg['fixed_poses'])
            self.assertEqual(region['fixed_neighbor'], cfg['fixed_poses'][0])
            self.assertFalse(region['minimum_original_q_inclusive'])
            self.assertFalse(region['maximum_original_q_inclusive'])
            self.assertEqual((region['minimum_original_q'], region['maximum_original_q']), (1., 2.))
            self.assertEqual(len((out/'geometry-probes.jsonl').read_text().splitlines()), 32)
            with self.assertRaises(ValueError):
                prepare(config, out, 32, 595017)


if __name__ == '__main__':
    unittest.main()
