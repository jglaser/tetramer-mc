"""Reference-limit and failure controls for the geometric separation audit."""
import copy
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from audit_peak_separation import geometry_summary


def pose(position, rotvec):
    q = Rotation.from_rotvec(rotvec).as_quat()
    return dict(position=list(position), orientation=q[[3, 0, 1, 2]].tolist())


class PeakSeparationTest(unittest.TestCase):
    def fixture(self):
        # Regular tetrahedron: <rr^T>=I and A=2I analytically, regardless of frame.
        vertices = [[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]]
        fixed = pose([3., -7., 2.], [.7, -.3, .2])
        selected = pose([-2., 4., 1.], [-.4, .5, .3])
        cfg = dict(metadata=dict(rigid_members=[dict(position=p) for p in vertices],
                                 native_poses=[pose([0., 0., 0.], [0., 0., 0.])],
                                 member_error_scale=2., angle_error_scale_deg=15.),
                   capture_center=[0., 0., 0.], capture_radius=18.)
        fr = Rotation.from_rotvec([.7, -.3, .2]).as_matrix()
        pr = Rotation.from_rotvec([-.4, .5, .3]).as_matrix()
        model = dict(angular_length=7., means=[[0.]*6], weights=[1.],
                     anchors=[dict(position=(fr.T @ (np.asarray(selected['position'])-fixed['position'])).tolist(),
                                   rotation=(fr.T @ pr).tolist())],
                     covariances=[np.diag([1., 1., 1., 49/8, 49/8, 49/8]).tolist()])
        return cfg, fixed, selected, model

    def test_regular_tetrahedron_has_known_metric_and_eigenvalue_bound(self):
        cfg, fixed, selected, model = self.fixture()
        report, world = geometry_summary(cfg, fixed, selected, model, [.5, 1., 2.])
        np.testing.assert_allclose(report['A_A2'], 2*np.eye(3), atol=1e-14)
        np.testing.assert_allclose(report['member_generalized_max_eigenvalues'], [1.5]*4, atol=1e-14)
        self.assertAlmostEqual(report['guarded_member_displacement_per_chart_radius'], math.sqrt(2.5), places=9)
        np.testing.assert_allclose(world.mean(axis=0), selected['position'], atol=1e-14)
        self.assertEqual(report['deterministic_checks']['count'], 76)
        self.assertLess(report['deterministic_checks']['RMS_squared_identity_error'], 1e-12)
        self.assertLess(report['deterministic_checks']['q_bound_excess'], 0.)

    def test_non_geometric_covariance_cannot_claim_the_same_containment(self):
        cfg, fixed, selected, model = self.fixture()
        bad = copy.deepcopy(model)
        bad['covariances'][0][3][3] *= 1.01
        with self.assertRaisesRegex(ValueError, 'covariance is not geometric'):
            geometry_summary(cfg, fixed, selected, bad, [.5, 1., 2.])

    def test_wrong_anchor_cannot_claim_the_selected_center(self):
        cfg, fixed, selected, model = self.fixture()
        model['anchors'][0]['position'][0] += .01
        with self.assertRaisesRegex(ValueError, 'center mismatch'):
            geometry_summary(cfg, fixed, selected, model, [.5, 1., 2.])


if __name__ == '__main__':
    unittest.main()
