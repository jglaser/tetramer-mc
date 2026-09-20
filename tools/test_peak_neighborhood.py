"""Original-N masked integrals, covariance, and rigid-member chart geometry."""
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_peak_neighborhood import MaskedMoments, geometry_model_audit, geometry_rows, selected_keys
from prepare_cayley_rms_cover import uniform_draws


class PeakNeighborhoodTests(unittest.TestCase):
    def test_masked_finite_law_uses_unconditional_n(self):
        density = np.array([.1, .2, .3, .4]); target = np.array([.5, 2., 7., 1.])
        radii = [.2, .7, 1.2, 1.7]; old = [10., 40., 10., 40.]
        moments = MaskedMoments(2.)
        for i, count in enumerate(np.rint(10*density).astype(int)):
            for _ in range(count):
                logw = math.log(target[i]/density[i])
                moments.add(radii[i], old[i], logw, 0., [logw, logw])
        result = moments.report()
        self.assertAlmostEqual(math.exp(result['physical']['full']['logQ']), target.sum())
        self.assertAlmostEqual(math.exp(result['physical']['old_r_le_32']['logQ']), target[[0, 2]].sum())
        self.assertAlmostEqual(math.exp(result['physical']['old_r_gt_32']['logQ']), target[[1, 3]].sum())
        self.assertEqual(result['physical']['radial_0']['draws'], 10)
        self.assertEqual(result['physical']['radial_0']['nonzero'], 1)
        self.assertAlmostEqual(math.exp(result['physical']['radial_0']['logQ']), target[0])

    def test_disjoint_variance_includes_negative_covariance(self):
        physical = np.array([2., 5., 0., 7., 3., 0.]); group = np.array([0, 1, 0, 1, 0, 1])
        moments = MaskedMoments(1.)
        for value, index in zip(physical, group):
            if value:
                logw = math.log(value); moments.add(.25+.5*index, 10.+30*index, logw, 0., [logw, logw])
            else: moments.add(.25+.5*index, 10.+30*index)
        result = moments.report(); check = result['partition_checks']['physical']['old_chart']
        expected = physical.var(ddof=1)/len(physical)
        self.assertAlmostEqual(math.exp(check['log_variance_of_mean']), expected)
        a, b = physical*(group == 0), physical*(group == 1)
        covariance = np.cov(a, b, ddof=1)[0, 1]/len(physical)
        observed = check['covariances'][0]
        self.assertEqual(observed['sign'], -1)
        self.assertAlmostEqual(-math.exp(observed['log_absolute_covariance']), covariance)
        diagonal = sum(math.exp(result['physical'][key]['log_variance_of_mean']) for key in ('old_r_le_32', 'old_r_gt_32'))
        self.assertAlmostEqual(diagonal+2*covariance, expected)
        self.assertGreater(diagonal, expected)

    def test_boundaries_and_empty_supported_pieces(self):
        self.assertIn('radial_0', selected_keys(.5, 32., 2.))
        self.assertIn('old_r_le_32', selected_keys(.5, 32., 2.))
        self.assertIn('radial_1', selected_keys(np.nextafter(.5, 1.), np.nextafter(32., 40.), 2.))
        self.assertIn('old_r_gt_32', selected_keys(.7, np.nextafter(32., 40.), 2.))
        self.assertIn('radial_2', selected_keys(2., 40., 2.))
        moments = MaskedMoments(.5)
        moments.add(.1, 40.); moments.add(.2, 40.)
        result = moments.report()
        self.assertIsNone(result['physical']['old_r_gt_32']['logQ'])
        self.assertEqual(result['physical']['old_r_gt_32']['draws'], 2)
        self.assertIn('not a mass upper bound', result['physical']['old_r_gt_32']['coverage'])
        self.assertNotIn('radial_1', result['physical'])
        with self.assertRaises(ValueError): moments.merge(MaskedMoments(1.))

    def test_non_native_center_and_anisotropic_member_geometry(self):
        def pose(position, rot):
            return dict(position=list(position), orientation=rot.as_quat()[[3, 0, 1, 2]].tolist())
        fixed = pose([4., -8., 2.], Rotation.from_rotvec([.4, -.2, .6]))
        peak = pose([-3., 1., 7.], Rotation.from_rotvec([-.2, .7, -.4]))
        members = np.array([[2., 1., 3.], [2., -1., -3.], [-2., 1., -3.], [-2., -1., 3.]])
        rf = Rotation.from_quat(np.array(fixed['orientation'])[[1, 2, 3, 0]]).as_matrix()
        rp = Rotation.from_quat(np.array(peak['orientation'])[[1, 2, 3, 0]]).as_matrix()
        relative = rf.T@rp; second = members.T@members/len(members)
        moment = np.trace(second)*np.eye(3)-second; left = relative@moment@relative.T
        sigma = np.eye(6); ell = 8.; sigma[3:, 3:] = ell**2/4*np.linalg.inv(left)
        model = dict(weights=[1.], means=[[0.]*6], covariances=[sigma.tolist()], angular_length=ell,
                     coordinate_convention='anchor-body-relative', anchors=[dict(
                         position=(rf.T@(np.array(peak['position'])-fixed['position'])).tolist(), rotation=relative.tolist())])
        region = dict(fixed_neighbor=fixed); cfg = dict(metadata=dict(rigid_members=[pose(m, Rotation.identity()) for m in members]))
        audit, geometry = geometry_model_audit(model, region, cfg, peak)
        self.assertTrue(audit['selected_peak_checked'])
        poses, latent, _, expected_j = uniform_draws(model, fixed, 2., 100, 72)
        positions = np.array([p['position'] for p in poses])
        rotations = Rotation.from_quat(np.array([p['orientation'] for p in poses])[:, [1, 2, 3, 0]]).as_matrix()
        radius, logj, errors, rms2 = geometry_rows(positions, rotations, geometry)
        np.testing.assert_allclose(radius, np.linalg.norm(latent, axis=1), atol=1e-13)
        np.testing.assert_allclose(logj, expected_j, atol=1e-13)
        self.assertLess(errors['member_rms2'], 1e-12)
        self.assertTrue(np.all(rms2 <= radius**2+1e-12))
        changed = dict(model, covariances=[(sigma*1.01).tolist()])
        with self.assertRaises(AssertionError): geometry_model_audit(changed, region, cfg, peak)


if __name__ == '__main__': unittest.main()
