#!/usr/bin/env python3
"""Independent sphere/Haar references for the production MIS density evaluator.

Generators below use direct Gaussian, quaternion and six-ball constructions;
they never call the implementation's Density draw/evaluation helpers.
"""
import copy
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

from shoulder_mis import ProposalDensities, mixture_log_density


ELL = 2.3
CORE = .2
EXCLUSION = .55
DELTA = .8
ANGLE = math.radians(35.)
CAPTURE = 1.4
ACTIVITY = 8.


def pose(position, rotation):
    return {'position': np.asarray(position).tolist(),
            'orientation': rotation.as_quat()[[3, 0, 1, 2]].tolist()}


def haar_cap(angle):
    return (angle-math.sin(angle))/math.pi


def fixture(center=(.6, -1.3, .9)):
    center = np.asarray(center)
    frame = Rotation.from_rotvec([-.4, .2, .6])
    native = Rotation.from_rotvec([.3, -.5, .7])
    fixed_pose = pose(center, frame); native_pose = pose(center, native)
    anchor = {'position': [0., 0., 0.], 'rotation': (frame.inv()*native).as_matrix().tolist()}
    cfg = {'fixed_poses': [fixed_pose], 'capture_center': center.tolist(), 'capture_radius': CAPTURE,
           'depletant_radius': EXCLUSION-CORE, 'reservoir_density': ACTIVITY,
           'metadata': {'native_poses': [native_pose], 'rigid_members': [pose([0., 0., 0.], Rotation.identity())],
                        'member_error_scale': DELTA, 'angle_error_scale_deg': 35.}}
    lower = np.diag([.5, .65, .45, .25*ELL, .3*ELL, .18*ELL])
    lower[3, 0] = .18; lower[4, 1] = -.23; lower[5, 2] = .15
    lower[1, 0] = .1; lower[4, 2] = .08; lower[5, 0] = .07
    model = {'schema': 'weighted-pose-mixture-v1', 'coordinate_convention': 'anchor-body-relative',
             'angular_length': ELL, 'shape_sha256': 'analytic-sphere-fixture',
             'anchors': [copy.deepcopy(anchor), copy.deepcopy(anchor)],
             'means': [[.1, -.08, .05, .05*ELL, -.03*ELL, .02*ELL],
                       [-.2, .15, -.1, -.12*ELL, .08*ELL, -.04*ELL]],
             'covariances': [(lower@lower.T).tolist(), (1.7*lower@lower.T).tolist()],
             'weights': [.65, .35]}
    description = {'weight': .75, 'uniform_probability': .05, 'anchor_index': 0,
                   'anchor_pose': fixed_pose, 'capture_center': center.tolist(), 'cube_lengths': [2*CAPTURE]*3}
    h = 2*ANGLE; radius = 2*DELTA
    cover = {'centroid': [0., 0., 0.], 'reference': native_pose,
             'ball_radius': radius, 'angle_cap': h, 'volume': 4*math.pi*radius**3/3*haar_cap(h)}
    product = {'covers': [cover], 'weights': [1.], 'scales': [1.]}
    # Tests-only outer ellipsoid: each product-cover pose has translation
    # squared norm/radius^2 <= 1 and Cayley squared norm/tan(ANGLE)^2 <= 1.
    # Their sum is <= 2, so the sqrt(2) latent ball is a complete cover.
    # This does not use the production moment-cover construction, which needs
    # a nonsingular member metric and cannot use this singleton fixture.
    scales = np.array([radius]*3+[ELL*math.tan(ANGLE)]*3)
    chart = {'schema': model['schema'], 'coordinate_convention': model['coordinate_convention'],
             'angular_length': ELL, 'shape_sha256': model['shape_sha256'], 'anchors': [anchor],
             'means': [[0.]*6], 'covariances': [np.diag(scales**2).tolist()], 'weights': [1.]}
    region = {'fixed_neighbor': fixed_pose, 'physical_fixed_neighbors': [fixed_pose],
              'gaussian_chart': chart, 'mahalanobis_radius': math.sqrt(2),
              'minimum_original_q': 1., 'maximum_original_q': 2.,
              'minimum_original_q_inclusive': False, 'maximum_original_q_inclusive': False}
    return cfg, model, description, product, region, frame, native, scales


def directions(rng, count, dimension):
    value = rng.normal(size=(count, dimension))
    return value/np.linalg.norm(value, axis=1)[:, None]


def cap_rotation(rng, count, h):
    # Invert the normalized-Haar cap CDF directly, independently of the
    # product-cover generator or density implementation.
    wanted = rng.random(count)*(h-math.sin(h))
    lo = np.zeros(count); hi = np.full(count, h)
    for _ in range(54):
        mid = (lo+hi)/2
        low = mid-np.sin(mid) < wanted
        lo = np.where(low, mid, lo); hi = np.where(low, hi, mid)
    return Rotation.from_rotvec(directions(rng, count, 3)*((lo+hi)/2)[:, None])


def pack(positions, rotations):
    quaternion = rotations.as_quat()[:, [3, 0, 1, 2]]
    return [{'position': p.tolist(), 'orientation': q.tolist()} for p, q in zip(positions, quaternion)]


def chart_decode(values, cfg, anchor, frame):
    cayley = values[:, 3:]/ELL
    delta = Rotation.from_quat(np.column_stack((cayley, np.ones(len(values)))))
    rotations = frame*delta*Rotation.from_matrix(anchor['rotation'])
    positions = frame.apply(values[:, :3]+anchor['position'])+cfg['fixed_poses'][0]['position']
    return positions, rotations


def draw_guide(rng, count, f):
    cfg, model, description, product, _, frame, native, _ = f
    center = np.asarray(cfg['capture_center'])
    family = rng.choice(4, size=count, p=[.25, .75*.05, .75*.95*.65, .75*.95*.35])
    positions = np.empty((count, 3)); quaternions = np.empty((count, 4))
    for key in range(4):
        indices = np.flatnonzero(family == key); n = len(indices)
        if key == 0:
            positions[indices] = center+directions(rng, n, 3)*(2*DELTA*rng.random(n)**(1/3))[:, None]
            rotations = native*cap_rotation(rng, n, 2*ANGLE)
        elif key == 1:
            positions[indices] = center+rng.uniform(-CAPTURE, CAPTURE, size=(n, 3))
            rotations = Rotation.from_quat(directions(rng, n, 4))
        else:
            k = key-2
            lower = np.linalg.cholesky(model['covariances'][k])
            values = rng.normal(size=(n, 6))@lower.T+model['means'][k]
            positions[indices], rotations = chart_decode(values, cfg, model['anchors'][k], frame)
        quaternions[indices] = rotations.as_quat()
    return positions, Rotation.from_quat(quaternions), family


def draw_cover(rng, count, f):
    cfg, _, _, _, region, frame, _, scales = f
    latent = directions(rng, count, 6)*(math.sqrt(2)*rng.random(count)**(1/6))[:, None]
    positions, rotations = chart_decode(latent*scales, cfg, region['gaussian_chart']['anchors'][0], frame)
    return positions, rotations, latent


def overlap(distance):
    d = np.asarray(distance)
    return np.where(d < 2*EXCLUSION, math.pi*(4*EXCLUSION+d)*(2*EXCLUSION-d)**2/12, 0.)


def physical_reference():
    def integrand(radius):
        angular = haar_cap(2*ANGLE)-(haar_cap(ANGLE) if radius <= DELTA else 0.)
        return 4*math.pi*radius**2*angular*math.exp(ACTIVITY*float(overlap(radius)))
    physical = sum(quad(integrand, a, b, epsabs=2e-12, epsrel=2e-12)[0]
                   for a, b in zip((2*CORE, DELTA, 2*EXCLUSION), (DELTA, 2*EXCLUSION, CAPTURE)))
    hard = 4*math.pi/3*((CAPTURE**3-(2*CORE)**3)*haar_cap(2*ANGLE)
                         -(DELTA**3-(2*CORE)**3)*haar_cap(ANGLE))
    return physical, hard


class ShoulderMisDensityTests(unittest.TestCase):
    def test_positive_poisson_sphere_mis_matches_haar_and_lens_quadrature(self):
        f = fixture(); cfg, model, description, product, region, _, native, _ = f
        density = ProposalDensities(cfg, model, description, product, region)
        ng, nc = 16384, 65536
        gp, gr, family = draw_guide(np.random.default_rng(1273101), ng, f)
        cp, cr, latent = draw_cover(np.random.default_rng(1273102), nc, f)
        physical_arrays = []; hard_arrays = []; invalid_counts = []
        for index, (positions, rotations) in enumerate(((gp, gr), (cp, cr))):
            n = len(positions)
            lg, lc, radii = density.evaluate(pack(positions, rotations))
            # Both strata are evaluated under the same fixed-quota full density.
            lm = mixture_log_density(lg, lc, ng, nc)
            d = np.linalg.norm(positions-cfg['capture_center'], axis=1)
            angle = (native.inv()*rotations).magnitude()
            q = np.maximum(d/DELTA, angle/ANGLE)
            valid = (q > 1) & (q < 2) & (d >= 2*CORE) & (d <= CAPTURE)
            invalid_counts.append(int((~valid).sum()))
            self.assertTrue(np.isfinite(lg[valid]).all())
            self.assertTrue(np.isfinite(lc[valid]).all(), 'The tests-only enclosing ellipsoid must cover the full target')
            hard = np.zeros(n); hard[valid] = np.exp(-lm[valid])
            # Positive Poisson estimators from the analytical overlap volume,
            # with two independent clouds at each valid pose.
            rng = np.random.default_rng(1273201+index)
            intensity = ACTIVITY*64
            points = rng.poisson(intensity*overlap(d[valid])[:, None], size=(int(valid.sum()), 2))
            weight = np.exp(points*math.log1p(ACTIVITY/intensity)).mean(axis=1)
            physical = np.zeros(n); physical[valid] = hard[valid]*weight
            hard_arrays.append(hard); physical_arrays.append(physical)
        expected_physical, expected_hard = physical_reference()
        for arrays, expected in ((physical_arrays, expected_physical), (hard_arrays, expected_hard)):
            estimate = sum(a.sum() for a in arrays)/(ng+nc)
            variance = sum(len(a)*np.var(a, ddof=1) for a in arrays)/(ng+nc)**2
            self.assertLess(math.sqrt(variance)/expected, .03)
            self.assertLess(abs(estimate-expected), 6*math.sqrt(variance))
            # Omitting invalid zeros would change the denominator appreciably.
            self.assertGreater(sum(invalid_counts)/(ng+nc), .5)
        self.assertEqual(len(set(family)), 4)

    def test_uniform_latent_density_has_analytic_jacobian_in_noncommuting_frames(self):
        f = fixture(); cfg, model, description, product, region, frame, native, scales = f
        density = ProposalDensities(cfg, model, description, product, region)
        positions, rotations, latent = draw_cover(np.random.default_rng(71961), 1024, f)
        lg, lc, radii = density.evaluate(pack(positions, rotations))
        np.testing.assert_allclose(radii, np.linalg.norm(latent, axis=1), atol=2e-14)
        # Known Jacobian from the independently generated chart values.
        c = latent[:, 3:]*scales[3:]/ELL
        logj = np.log(scales).sum()-3*math.log(ELL)-2*math.log(math.pi)-2*np.log1p(np.sum(c*c, axis=1))
        logv = 3*math.log(math.pi)+6*math.log(math.sqrt(2))-math.log(6)
        np.testing.assert_allclose(lc, -logv-logj, atol=2e-13)
        self.assertGreater(np.max(np.abs((frame*native).as_matrix()-(native*frame).as_matrix())), .1)

    def test_density_is_unconditional_and_seam_keeps_only_cube_support(self):
        f = fixture(center=(0., 0., 0.)); cfg, model, description, product, region, frame, native, _ = f
        density = ProposalDensities(cfg, model, description, product, region)
        lg, lc, _ = density.evaluate([pose([0., 0., 0.], native)])
        self.assertTrue(np.isfinite(lg[0]) and np.isfinite(lc[0]), 'q=0 and hard-invalid poses still have proposal densities')
        anchor = Rotation.from_matrix(model['anchors'][0]['rotation'])
        seam = frame*Rotation.from_quat([1., 0., 0., 0.])*anchor
        points = [pose([0., 0., 0.], seam), pose([-CAPTURE, 0., 0.], seam),
                  pose([CAPTURE, 0., 0.], seam)]
        lg, lc, _ = density.evaluate(points)
        expected = math.log(.75*.05/(2*CAPTURE)**3)
        np.testing.assert_allclose(lg[:2], expected, atol=2e-12)
        self.assertTrue(np.isneginf(lc).all())
        # At the upper half-open cube edge, Gaussian density may be a tiny
        # finite floating-point seam residual; it must have vanished physically.
        self.assertLess(lg[2], -1000.)


if __name__ == '__main__':
    unittest.main()
