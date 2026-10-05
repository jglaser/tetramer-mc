"""Deterministic independent-density controls; no protein or random queries."""
import math
import copy
import unittest

import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation

from source_guide_reference import (SourceDensity, compose, mixture_log_density,
                                    pose, pose_error, expected_covariance, atlas_decode)
from audit_context_candidate_bank import MapDensityReference
from test_context_candidate_bank_audit import model


def spec():
    st, theta, ell = .2, 1., 30.
    return dict(translation_sigma=st, rotation_scale_deg=theta, angular_length=ell,
                covariance=np.diag([st*st]*3+[(ell*math.tan(theta*math.pi/360))**2]*3).tolist())


class SourceGuideReferenceTests(unittest.TestCase):
    def test_decode_density_has_normalized_haar_jacobian(self):
        anchor = pose([1., 2., -.3], Rotation.from_rotvec([.3, -.2, .4]).as_matrix())
        source = pose([-1., .2, 2.], Rotation.from_rotvec([-.5, .3, .1]).as_matrix())
        reference = SourceDensity(spec(), source, anchor)
        pose_error(reference.decode(np.zeros(6)), source)
        for z in (np.zeros(6), np.array([.4, -.7, .2, .6, -.1, .5])):
            x = reference.lower@z
            u = x[3:]/reference.ell
            log_j = reference.logdet-3*math.log(reference.ell)-2*math.log(math.pi)-2*math.log1p(u@u)
            expected = multivariate_normal.logpdf(z, mean=np.zeros(6), cov=np.eye(6))-log_j
            self.assertAlmostEqual(reference.evaluate(reference.decode(z)), expected, places=10)

    def test_common_frame_invariance(self):
        anchor = pose([1., 2., 3.], Rotation.from_rotvec([.2, .1, -.3]).as_matrix())
        source = pose([-.5, .3, 1.], Rotation.from_rotvec([-.1, .5, -.4]).as_matrix())
        shift = pose([4., -2., .5], Rotation.from_rotvec([.7, -.4, .2]).as_matrix())
        before = SourceDensity(spec(), source, anchor)
        after = SourceDensity(spec(), compose(shift, source), compose(shift, anchor))
        value = before.decode([.1, -.2, .3, .4, -.5, .6])
        self.assertAlmostEqual(before.evaluate(value), after.evaluate(compose(shift, value)), places=10)

    def test_full_mixture_and_uniform_limit(self):
        logs = [math.log(.01), math.log(.2), math.log(.4)]
        self.assertAlmostEqual(math.exp(mixture_log_density(*logs)), .5*.01+.25*.2+.25*.4)
        self.assertAlmostEqual(mixture_log_density(*logs, probabilities=(1., 0., 0.)), logs[0])
        self.assertAlmostEqual(math.exp(mixture_log_density(-math.inf, logs[1], logs[2])), .15)
        with self.assertRaises(ValueError):mixture_log_density(*logs, probabilities=(.5, .25, .3))
        with self.assertRaises(ValueError):mixture_log_density(math.nan, *logs[1:])

    def test_covariance_and_pose_changes_detected(self):
        s = spec(); s['covariance'][0][0] *= 2
        with self.assertRaises(ValueError):expected_covariance(s)
        p = pose([0., 0., 0.], np.eye(3))
        with self.assertRaises(ValueError):pose_error(p, pose([.1, 0., 0.], np.eye(3)))

    def test_atlas_decode_reciprocal_branches_have_expected_component_density(self):
        reference = MapDensityReference(model())
        anchor = pose([2., -.4, .8], Rotation.from_rotvec([.2, -.7, .4]).as_matrix())
        z = np.array([.5, -.3, .1, -.2, .4, .6])
        for branch in range(3):
            value = atlas_decode(reference, anchor, branch, z)
            actual = reference.evaluate([value], anchor)[1][0, branch]
            x = reference.density.mean[branch]+reference.density.lower[branch]@z
            u = x[3:]/reference.density.ell
            log_j = (reference.density.logdet[branch]-3*math.log(reference.density.ell)
                     -2*math.log(math.pi)-2*math.log1p(u@u))
            expected = multivariate_normal.logpdf(z, mean=np.zeros(6), cov=np.eye(6))-log_j
            self.assertAlmostEqual(actual, expected, places=10)

    def test_explicit_correlated_mean_and_haar_jacobian(self):
        lower = np.diag([.2, .3, .1, .15, .2, .12])
        lower[3, 0] = .08
        lower[4, 1] = -.11
        lower[5, 2] = .07
        mean = np.array([.12, -.04, .07, .03, -.02, .06])
        s = dict(angular_length=2., covariance=(lower@lower.T).tolist(),
                 explicit_gaussian=dict(schema='source-gaussian-v1', mean=mean.tolist(),
                                        provenance='synthetic fixed correlated chart'))
        anchor = pose([2., -.4, .8], Rotation.from_rotvec([.2, -.7, .4]).as_matrix())
        source = pose([1., 2., 3.], Rotation.from_rotvec([-.2, .1, .3]).as_matrix())
        reference = SourceDensity(s, source, anchor)
        np.testing.assert_allclose(reference.lower, lower, rtol=1e-14, atol=1e-16)
        for z in (np.zeros(6), np.array([.3, -.6, .1, -.2, .5, -.4])):
            x = mean+lower@z
            u = x[3:]/2.
            j = reference.logdet-3*math.log(2.)-2*math.log(math.pi)-2*math.log1p(u@u)
            wanted = multivariate_normal.logpdf(z, mean=np.zeros(6), cov=np.eye(6))-j
            self.assertAlmostEqual(reference.evaluate(reference.decode(z)), wanted, places=11)
        pose_error(reference.decode(np.linalg.solve(lower, -mean)), source)

        bad_specs = []
        for key, value in [('translation_sigma', .2), ('rotation_scale_deg', 1.)]:
            bad = copy.deepcopy(s);bad[key] = value;bad_specs.append(bad)
        for key, value in [('schema', 'unversioned'), ('provenance', '  '), ('mean', [0.]*5)]:
            bad = copy.deepcopy(s);bad['explicit_gaussian'][key] = value;bad_specs.append(bad)
        for matrix in (np.zeros((6, 6)), np.eye(6)*-1,
                       np.eye(6)+np.diag([.1]*5, 1), np.full((6, 6), np.nan)):
            bad = copy.deepcopy(s);bad['covariance'] = matrix.tolist();bad_specs.append(bad)
        for bad in bad_specs:
            with self.assertRaises(ValueError):expected_covariance(bad)


if __name__ == '__main__':
    unittest.main()
